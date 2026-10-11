"""
RAG на LlamaIndex (блоки 5.3–5.5): ответ на вопрос по корпусу data/<категория>/… с цитатами.

    python -m app.services.rag                          # подключиться (или проиндексировать) + вопрос по умолчанию
    python -m app.services.rag "Каков срок возврата оплаты?"
    python -m app.services.rag --rebuild "…"            # пересобрать коллекцию (то же, что scripts/ingest.py --rebuild)
    python -m app.services.rag --build                  # только подключиться или проиндексировать, без вопроса
    python -m app.services.rag --lab "…"                # учебный корпус блоков 5.3–5.4 (data/rag-block-03)

Два контура (docs/rag.md, схема):

    ingestion (app/services/ingestion.py, scripts/ingest.py):
      файлы PDF/DOCX/HTML/MD → метаданные → SentenceSplitter 512/64 → bge-m3 → QdrantVectorStore
      IngestionPipeline + SimpleDocumentStore (UPSERTS): повторный запуск — «0 changed, N unchanged»

    query (этот модуль):
      вопрос → retriever top-10 (RAG__CANDIDATES_TOP_K) → score-guard → re-ranker (RERANK__ENABLED) →
      top-5 (RAG__SIMILARITY_TOP_K) → нумерованный контекст «[1] (файл) текст» → модель → ответ с [1], [2]

- Score-guard: лучшая близость кандидатов ниже RAG__SCORE_THRESHOLD — модель не вызывается, ответ
  «По базе не нашёл, могу эскалировать.» (rag_common.NO_ANSWER), в лог — rag_score_guard. Второй
  слой — правило в промпте: нет ответа в контексте — та же фраза.
- Цитаты — нумерованный контекст и правило «после утверждения — номер фрагмента» (как в
  reference-сервисе курса): source с id=N — фрагмент [N]; cited — сослалась ли на него модель.
- Эмбеддинги — ProjectEmbedding (модуль блока 5.1: bge-m3 в Ollama, кеш); модель ответа —
  OpenAI-совместимая LLM__* (Ollama llama3.2, OpenRouter) через OpenAILike.
- Хранение — коллекция RAG__COLLECTION (kb_docs) в формате LlamaIndex (_node_content). build()
  сверяет размерность (EMBEDDING_DIM), метрику (COSINE) и модель эмбеддингов: иначе поиск вернул бы мусор.

Чат (app/chat/service.py) пользуется теми же retrieve() и контекстом, а ответ модели стримит сам.

answer(question) -> {"answer", "top_score", "confident", "sources": [{"id", "file_name", "page", "score",
"snippet", …}, …]} — контракт rag_common.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import Document, NodeWithScore, QueryBundle
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

from app.observability.logging import get_logger
from app.services import chunking, embeddings
from app.services.ingestion import IngestionService, ProjectEmbedding
from app.services.rag_common import (
    DEFAULT_QUESTION,
    USER_TEMPLATE,
    RAGError,
    RAGLLMError,
    RAGNotIndexed,
    RAGUnavailable,
    final_answer,
    llm_error,
    numbered_context,
    qdrant_error,
    refusal,
    source,
    system_prompt,
)
from app.services.vector_store import make_qdrant_client

__all__ = ["ProjectEmbedding", "RAGError", "RAGLLMError", "RAGNotIndexed", "RAGService", "RAGUnavailable",
           "Retrieval", "open_rag"]

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.llms import LLM

    from app.core.config import Settings

log = get_logger()

# Что считать «кодом конвейера» при сравнении с bare-metal (scripts/rag_compare.py, docs/rag.md):
# индексация и запрос в этом модуле. С блока 5.5 индексация — в ingestion.py (IngestionService), здесь —
# только запуск; сверка коллекции (check_collection) одинаковая в обеих версиях и не считается.
PIPELINE = {
    "ingestion": ("load_corpus", "RAGService.build"),
    "query": ("build_llm", "chat_messages", "RAGService.retrieve", "RAGService.answer", "Retrieval"),
}


def build_llm(cfg: Settings) -> LLM:
    """Модель ответа: тот же OpenAI-совместимый провайдер, что у /chat (LLM__*), с прокси,
    сертификатами ОС и заголовками OpenRouter. OpenAILike, а не OpenAI: класс OpenAI LlamaIndex
    знает только модели OpenAI и на llama3.2 упал бы, не найдя размер её контекста."""
    import httpx
    from llama_index.llms.openai_like import OpenAILike

    from app.core.config import (
        http_client_options,
        is_local_url,
        provider_headers,
        proxy_for,
    )

    proxy = proxy_for(cfg.llm.base_url, cfg.llm.proxy_url)
    options = http_client_options(proxy, cfg.llm.use_system_certs and not is_local_url(cfg.llm.base_url))
    return OpenAILike(
        model=cfg.rag.llm_model or cfg.llm.default_model,
        api_base=cfg.llm.base_url or "https://api.openai.com/v1",
        api_key=cfg.llm.openai_api_key.get_secret_value(),
        is_chat_model=True,
        context_window=cfg.context_window,
        max_tokens=cfg.rag.max_tokens,
        temperature=cfg.rag.temperature,
        timeout=cfg.llm.request_timeout,
        max_retries=cfg.llm.max_retries,
        default_headers=provider_headers(cfg.llm.base_url, "multapi"),
        http_client=httpx.Client(**options) if options else None,
    )


def chat_messages(product_name: str, question: str, context: str) -> list[ChatMessage]:
    """Промпт ответа: системные правила (только по контексту, цитаты [N], отказ) и пользовательское
    сообщение с нумерованным контекстом и вопросом — те же тексты, что у bare-metal."""
    return [ChatMessage(role=MessageRole.SYSTEM, content=system_prompt(product_name)),
            ChatMessage(role=MessageRole.USER, content=USER_TEMPLATE.format(context_str=context, query_str=question))]


def load_corpus(data_dir: Path, embed_label: str, chunking_label: str | None = None) -> list[Document]:
    """Учебный корпус блоков 5.3–5.4 одним списком документов: для сравнения нарезок
    (scripts/chunking_experiment.py), где id документа — имя файла, а не путь. Сервис индексирует
    корпус через IngestionService."""
    if not data_dir.is_dir():
        raise RAGError(f"Нет папки корпуса {data_dir}: проверьте RAG__LAB_DIR")
    reader = SimpleDirectoryReader(input_dir=str(data_dir), recursive=True,
                                   file_metadata=lambda path: {"file_name": Path(path).name})
    documents = reader.load_data()
    seen: dict[str, int] = {}
    for document in documents:
        name = document.metadata["file_name"]
        seen[name] = seen.get(name, 0) + 1
        document.id_ = name if seen[name] == 1 else f"{name}#{seen[name]}"
        document.metadata["embed_model"] = embed_label
        if chunking_label:
            document.metadata["chunking"] = chunking_label
        # В вектор — только текст, как у bare-metal; модели — ещё имя файла (и страница PDF).
        document.excluded_embed_metadata_keys = list(document.metadata)
        document.excluded_llm_metadata_keys = ["embed_model", "chunking"]
    return documents


@dataclass
class Retrieval:
    """Результат поиска: фрагменты для контекста (top_n после re-ranker) и лучшая близость среди
    кандидатов — по ней score-guard решает, звать ли модель."""

    question: str
    nodes: list[NodeWithScore]
    top_score: float
    candidates: int
    threshold: float

    @property
    def confident(self) -> bool:
        return bool(self.nodes) and self.top_score >= self.threshold

    def fragments(self) -> list[tuple[str, str | None, int | None]]:
        return [(node.node.get_content(), node.metadata.get("file_name"), node.metadata.get("page"))
                for node in self.nodes]

    def context(self) -> str:
        """«[1] (refunds_v2.docx)\\nтекст\\n\\n[2] (support-regulation_v2.3.pdf, стр. 2)\\nтекст»."""
        return numbered_context(self.fragments())

    def sources(self) -> list[dict[str, Any]]:
        return [source(number, node.node.get_content(), file_name=node.metadata.get("file_name"),
                       page=node.metadata.get("page"), score=node.score, path=node.metadata.get("source"),
                       category=node.metadata.get("category"))
                for number, node in enumerate(self.nodes, start=1)]

    def log_guard(self, **extra: Any) -> None:
        """Срабатывание score-guard — в лог: без текста вопроса (в нём бывают персональные данные)."""
        log.info("rag_score_guard", top_score=round(self.top_score, 3), threshold=self.threshold,
                 candidates=self.candidates, question_chars=len(self.question), **extra)


class RAGService:
    def __init__(self, cfg: Settings, *, client: Any = None, embed_model: BaseEmbedding | None = None,
                 llm: LLM | None = None, reranker: Any = None, data_dir: Path | None = None,
                 collection: str | None = None) -> None:
        if client is None and not cfg.qdrant_url:
            raise RAGError("QDRANT_URL не задан: RAG хранит фрагменты в Qdrant — укажите адрес в .env")
        if cfg.embedding_dim is None:
            raise RAGError("EMBEDDING_DIM не задан: укажите размерность модели эмбеддингов в .env")
        self.cfg = cfg
        self.settings = cfg.rag
        self.data_dir = data_dir or cfg.rag.data_dir
        self.collection = collection or cfg.rag.collection
        if client is None:
            key = cfg.qdrant_api_key.get_secret_value() if cfg.qdrant_api_key else None
            client = make_qdrant_client(cfg.qdrant_url, key, asynchronous=False)
        self.client = client
        self.embed_model = embed_model or ProjectEmbedding(embeddings.default_service())
        self._llm = llm
        if reranker is None:
            from app.services.reranker import build_reranker

            reranker = build_reranker(cfg)
        self.reranker = reranker or None
        self.chunking_label = chunking.label(cfg.rag.chunking, chunk_size=cfg.rag.chunk_size,
                                             chunk_overlap=cfg.rag.chunk_overlap,
                                             buffer_size=cfg.rag.semantic_buffer_size,
                                             breakpoint_percentile_threshold=cfg.rag.semantic_breakpoint)
        self.chunking_mismatch: str | None = None   # нарезка готовой коллекции, если она другая
        self.retriever: Any = None
        self.indexed = 0           # сколько фрагментов записал последний build(); 0 — подключились к готовой
        self._lock = threading.Lock()

    @property
    def llm(self) -> LLM:
        """Модель ответа создаётся при первом ответе: чату (он стримит своей моделью) и скриптам
        индексации она не нужна."""
        if self._llm is None:
            self._llm = build_llm(self.cfg)
        return self._llm

    # -------------------------------------------------------- индексация
    def ingestion(self) -> IngestionService:
        """Индексатор того же корпуса и коллекции, с теми же клиентом Qdrant и эмбеддингами."""
        return IngestionService(self.cfg, client=self.client, embed_model=self.embed_model, data_dir=self.data_dir,
                                collection=self.collection)

    def build(self, *, rebuild: bool = False, create: bool = True) -> RAGService:
        """Подключиться к коллекции или, если её нет (или rebuild), проиндексировать корпус через
        IngestionService. create=False — только подключиться (lifespan сервиса): нет коллекции —
        RAGNotIndexed."""
        try:
            exists = self.client.collection_exists(self.collection)
            points = self.client.count(self.collection, exact=True).count if exists else 0
            if rebuild or (not points and create):
                report = self.ingestion().run(rebuild=rebuild)
                self.indexed = report.chunks
                points = self.client.count(self.collection, exact=True).count \
                    if self.client.collection_exists(self.collection) else 0
            else:
                self.indexed = 0
            if not points:
                raise RAGNotIndexed(f"Коллекция {self.collection} пуста или её нет: проиндексируйте корпус — "
                                    "python scripts/ingest.py data/")
            self.check_collection()
            store = QdrantVectorStore(client=self.client, collection_name=self.collection)
            index = VectorStoreIndex.from_vector_store(store, embed_model=self.embed_model)
            self.retriever = index.as_retriever(similarity_top_k=self.settings.candidates_top_k)
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise qdrant_error(exc, self.cfg.qdrant_url) from exc
        return self

    def check_collection(self) -> None:
        """Размерность, метрика и модель эмбеддингов коллекции — те же, что сейчас в .env."""
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        if not isinstance(vectors, VectorParams):
            raise RAGError(f"Коллекция {self.collection} с именованными векторами: её создавали не этим "
                           "кодом — пересоберите: python scripts/ingest.py --rebuild")
        if vectors.size != self.cfg.embedding_dim or vectors.distance != Distance.COSINE:
            raise RAGError(f"Коллекция {self.collection}: векторы из {vectors.size} чисел, метрика "
                           f"{vectors.distance.value}, а нужно EMBEDDING_DIM={self.cfg.embedding_dim} и Cosine — "
                           "её заполняла другая модель: python scripts/ingest.py --rebuild")
        records, _ = self.client.scroll(self.collection, limit=1, with_payload=["embed_model", "chunking"])
        payload = (records[0].payload or {}) if records else {}
        model = payload.get("embed_model")
        if model and model != self.embed_model.model_name:
            raise RAGError(f"Коллекция {self.collection} проиндексирована моделью {model}, а сейчас "
                           f"{self.embed_model.model_name}: вопросы и фрагменты были бы в разных пространствах — "
                           "python scripts/ingest.py data/ (документы переиндексируются сами) или --rebuild")
        # Другая нарезка — не ошибка: поиск работает, просто не на выбранной конфигурации.
        stored = payload.get("chunking", "без метки (до блока 5.4)") if records else None
        self.chunking_mismatch = stored if stored and stored != self.chunking_label else None
        if self.chunking_mismatch:
            log.warning("rag_chunking_mismatch", collection=self.collection, stored=stored,
                        configured=self.chunking_label, fix="python scripts/ingest.py data/")

    # -------------------------------------------------------- поиск и ответ
    def retrieve(self, question: str) -> Retrieval:
        """Кандидаты из Qdrant (RAG__CANDIDATES_TOP_K) → score-guard → re-ranker → первые
        RAG__SIMILARITY_TOP_K. Не прошли порог — nodes пустой, re-ranker не вызывается."""
        if self.retriever is None:
            raise RAGError("Индекс не готов: сначала build()")
        question = question.strip()
        if not question:
            raise ValueError("Пустой вопрос")
        try:
            candidates = self.retriever.retrieve(QueryBundle(question))
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise qdrant_error(exc, self.cfg.qdrant_url) from exc
        top_score = max((node.score or 0.0 for node in candidates), default=0.0)
        threshold = self.settings.score_threshold
        top_n = self.settings.similarity_top_k
        if top_score < threshold:
            nodes: list[NodeWithScore] = []
        elif self.reranker:
            nodes = rerank_nodes(self.reranker, question, candidates, top_n)
        else:
            nodes = candidates[:top_n]
        return Retrieval(question, nodes, top_score, len(candidates), threshold)

    def answer(self, question: str) -> dict[str, Any]:
        """Ответ с цитатами. Score-guard — без вызова модели; модель отказалась по промпту — тоже
        confident=false и без источников."""
        found = self.retrieve(question)
        if not found.confident:
            found.log_guard(where="rag_query")
            return refusal(found.top_score)
        try:
            response = self.llm.chat(chat_messages(self.cfg.support.product_name, found.question, found.context()))
        except Exception as exc:
            explained = llm_error(exc, self.cfg)
            if explained is exc:
                raise
            raise explained from exc
        result = final_answer(response.message.content or "", found.top_score, found.sources())
        if not result["confident"]:
            log.info("rag_refused", reason="model", top_score=round(found.top_score, 3), where="rag_query")
        return result

    def ensure_ready(self) -> None:
        """Для сервиса: подключиться к коллекции, если на старте её ещё не было (индексация шла в
        фоне или корпус индексировали с хоста). Один раз — дальше запросы идут в готовый retriever."""
        if self.retriever is None:
            with self._lock:
                if self.retriever is None:
                    self.build(create=False)

    def close(self) -> None:
        self.client.close()


def open_rag(cfg: Settings) -> RAGService | None:
    """Для lifespan сервиса: RAGService и подключение к готовой коллекции — один раз на процесс.
    Без QDRANT_URL — None (/rag/query отвечает 503). Коллекции ещё нет или Qdrant не отвечает —
    сервис стартует, а RAG подключится при первом запросе (ensure_ready). Коллекция под другую
    модель — исключение: старт с заведомо неверной конфигурацией хуже понятной ошибки."""
    if not cfg.qdrant_url:
        log.info("rag_disabled", reason="QDRANT_URL не задан")
        return None
    service = RAGService(cfg)
    try:
        service.build(create=False)
        log.info("rag_ready", collection=service.collection, candidates=cfg.rag.candidates_top_k,
                 top_n=cfg.rag.similarity_top_k, score_threshold=cfg.rag.score_threshold,
                 model=cfg.rag.llm_model or cfg.llm.default_model, embed_model=service.embed_model.model_name,
                 chunking=service.chunking_label, reranker=getattr(service.reranker, "model", None))
    except (RAGNotIndexed, RAGUnavailable) as exc:
        log.warning("rag_not_ready", collection=service.collection, error=str(exc),
                    note="сервис работает; RAG подключится, когда коллекция появится (RAG__INGEST_ON_START "
                         "или python scripts/ingest.py data/)")
    except Exception:
        service.close()
        raise
    return service


def no_implicit_event_loop() -> None:
    """Для вызовов RAGService из главного потока (CLI, скрипты). Синхронные методы LlamaIndex
    запускают свои корутины через asyncio.get_event_loop() (async_utils.asyncio_run): в главном
    потоке без цикла Python 3.12–3.13 создаёт цикл неявно — с DeprecationWarning и без закрытия.
    После set_event_loop(None) get_event_loop падает, и LlamaIndex берёт asyncio.run — как в потоках
    сервиса (asyncio.to_thread) и как будет по умолчанию в Python 3.14."""
    asyncio.set_event_loop(None)


def rerank_nodes(reranker: Any, question: str, nodes: list[NodeWithScore], top_n: int) -> list[NodeWithScore]:
    """Переранжировать найденные узлы (app/services/reranker.py) и оставить top_n. score узла —
    по-прежнему близость из Qdrant (её показывают sources и с ней сравнивается порог), порядок —
    по оценке re-ranker."""
    from app.services.reranker import Candidate

    ranked = reranker.rerank(question, [Candidate(text=node.text, doc_id=node.metadata.get("file_name"),
                                                  score=node.score or 0.0, payload=node) for node in nodes], top_n)
    return [candidate.payload for candidate in ranked]


# ---------------------------------------------------------------- python -m app.services.rag
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG на LlamaIndex по корпусу data/ с цитатами (блоки 5.3–5.5)")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION, help="вопрос по корпусу")
    parser.add_argument("--rebuild", action="store_true", help="удалить коллекцию и проиндексировать заново")
    parser.add_argument("--build", action="store_true", help="только подключиться или проиндексировать, без вопроса")
    parser.add_argument("--lab", action="store_true", help="учебный корпус блоков 5.3–5.4 (RAG__LAB_DIR)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from pydantic import ValidationError

    from app.core.config import get_settings
    from app.observability.logging import setup_logging
    from app.services.reranker import RerankError

    setup_logging("WARNING", stream=sys.stderr)
    no_implicit_event_loop()
    try:
        cfg = get_settings()
        lab = {"data_dir": cfg.rag.lab_dir, "collection": cfg.rag.lab_collection} if args.lab else {}
        service = RAGService(cfg, **lab).build(rebuild=args.rebuild)
        print(f"Коллекция {service.collection}: " + (f"проиндексировано фрагментов — {service.indexed}"
              if service.indexed else "подключились к готовой"), file=sys.stderr)
        if service.chunking_mismatch:
            print(f"Внимание: нарезка коллекции — «{service.chunking_mismatch}», а в .env — «{service.chunking_label}». "
                  "Переиндексируйте: python scripts/ingest.py data/", file=sys.stderr)
        if not args.build:
            print(json.dumps(service.answer(args.question), ensure_ascii=False, indent=2))
        service.close()
        return 0
    except (RAGError, RerankError, embeddings.EmbeddingError, ValidationError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
