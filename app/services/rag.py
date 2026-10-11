"""
RAG на LlamaIndex (блок 5.3): ответ на вопрос по корпусу data/rag-block-03.

    python -m app.services.rag                          # индекс (подключиться или собрать) + вопрос по умолчанию
    python -m app.services.rag "Как включить вход по отпечатку пальца?"
    python -m app.services.rag --rebuild "…"            # пересобрать коллекцию: сменили корпус, нарезку или модель
    python -m app.services.rag --build                  # только индексация, без вопроса

Конвейер — компоненты LlamaIndex:

    SimpleDirectoryReader → SentenceSplitter → QdrantVectorStore → VectorStoreIndex → QueryEngine
      .md .txt .pdf .docx    512 / 64 токена    коллекция             эмбеддинги         top-3 → модель
                                                rag_block_03          блока 5.1

- Чтение — SimpleDirectoryReader(recursive=True): PDF и DOCX — ридеры llama-index-readers-file
  (DOCX — через docx2txt). Метаданные — только имя файла (и номер страницы у PDF): полный путь
  с именем пользователя Windows в Qdrant и в промпт не попадает.
- Нарезка — SentenceSplitter(chunk_size, chunk_overlap) из RAG__*: границы по предложениям.
- Хранение — своя коллекция RAG__COLLECTION в Qdrant (QDRANT_URL): LlamaIndex хранит узлы в своём
  формате (_node_content), и коллекция блока 5.2 с плоским payload ему не подходит (docs/rag.md,
  «Коллекция»). Подключаясь к готовой, build() сверяет размерность (EMBEDDING_DIM), метрику (COSINE)
  и модель эмбеддингов: иначе поиск вернул бы мусор.
- Эмбеддинги — ProjectEmbedding: модуль блока 5.1 (bge-m3 в Ollama, нормирование, кеш), тот же,
  что у версии без фреймворка и у коллекции блока 5.2.
- Ответ — OpenAI-совместимая модель LLM__* (Ollama llama3.2, OpenRouter) через OpenAILike, промпт —
  rag_common: только по фрагментам, без ответа — «В базе знаний ответа нет.».
- Порог: если близость лучшего фрагмента ниже RAG__MIN_SCORE, модель не вызывается — ответ «нет».

answer(question) -> {"answer", "top_score", "sources": [{"text", "source", "score"}, …]}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.schema import Document, NodeWithScore, QueryBundle
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import PrivateAttr
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

from app.services import embeddings
from app.services.rag_common import (
    DEFAULT_QUESTION,
    NO_ANSWER,
    REFINE_TEMPLATE,
    USER_TEMPLATE,
    RAGError,
    RAGLLMError,
    RAGNotIndexed,
    RAGUnavailable,
    llm_error,
    qdrant_error,
    result,
    source,
    system_prompt,
)
from app.services.vector_store import make_qdrant_client

__all__ = ["ProjectEmbedding", "RAGError", "RAGLLMError", "RAGNotIndexed", "RAGService", "RAGUnavailable",
           "open_rag"]

if TYPE_CHECKING:
    from llama_index.core.llms import LLM

    from app.core.config import Settings

# Что считать «кодом конвейера» при сравнении с bare-metal (scripts/rag_compare.py, docs/rag.md):
# индексация и запрос. Сверка коллекции (check_collection) — одинаковая в обеих версиях, не считается.
PIPELINE = {
    "ingestion": ("ProjectEmbedding", "RAGService.load_documents", "RAGService.build"),
    "query": ("build_llm", "prompts", "RAGService.answer", "sources"),
}


class ProjectEmbedding(BaseEmbedding):
    """Модель эмбеддингов блока 5.1 для LlamaIndex: те же векторы, префиксы E5, нормирование и
    кеш, что у bare-metal-версии и у коллекции блока 5.2."""

    _service: Any = PrivateAttr()

    def __init__(self, service: embeddings.EmbeddingService, **kwargs: Any) -> None:
        super().__init__(model_name=service.label, embed_batch_size=64, **kwargs)
        self._service = service

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._service.embed_query(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._service.embed_documents([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._service.embed_documents(list(texts))

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)


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


def prompts(product_name: str) -> tuple[ChatPromptTemplate, ChatPromptTemplate]:
    """Шаблоны ответа и уточнения на русском: у LlamaIndex по умолчанию английские."""
    system = ChatMessage(role=MessageRole.SYSTEM, content=system_prompt(product_name))
    qa = ChatPromptTemplate(message_templates=[system, ChatMessage(role=MessageRole.USER, content=USER_TEMPLATE)])
    refine = ChatPromptTemplate(message_templates=[system,
                                                   ChatMessage(role=MessageRole.USER, content=REFINE_TEMPLATE)])
    return qa, refine


class RAGService:
    def __init__(self, cfg: Settings, *, client: Any = None, embed_model: BaseEmbedding | None = None,
                 llm: LLM | None = None) -> None:
        if client is None and not cfg.qdrant_url:
            raise RAGError("QDRANT_URL не задан: RAG хранит фрагменты в Qdrant — укажите адрес в .env")
        if cfg.embedding_dim is None:
            raise RAGError("EMBEDDING_DIM не задан: укажите размерность модели эмбеддингов в .env")
        self.cfg = cfg
        self.settings = cfg.rag
        self.collection = cfg.rag.collection
        if client is None:
            key = cfg.qdrant_api_key.get_secret_value() if cfg.qdrant_api_key else None
            client = make_qdrant_client(cfg.qdrant_url, key, asynchronous=False)
        self.client = client
        self.embed_model = embed_model or ProjectEmbedding(embeddings.default_service())
        self.llm = llm or build_llm(cfg)
        self.query_engine: Any = None
        self.indexed = 0           # сколько фрагментов записал последний build(); 0 — подключились к готовой
        self._lock = threading.Lock()

    # -------------------------------------------------------- индексация
    def load_documents(self) -> list[Document]:
        """Корпус: все файлы папки RAG__DATA_DIR, PDF — по странице на документ."""
        if not self.settings.data_dir.is_dir():
            raise RAGError(f"Нет папки корпуса {self.settings.data_dir}: проверьте RAG__DATA_DIR")
        reader = SimpleDirectoryReader(input_dir=str(self.settings.data_dir), recursive=True,
                                       file_metadata=lambda path: {"file_name": Path(path).name})
        documents = reader.load_data()
        seen: dict[str, int] = {}
        for document in documents:
            name = document.metadata["file_name"]
            seen[name] = seen.get(name, 0) + 1
            document.id_ = name if seen[name] == 1 else f"{name}#{seen[name]}"
            document.metadata["embed_model"] = self.embed_model.model_name
            # В вектор — только текст, как у bare-metal; модели — ещё имя файла (и страница PDF).
            document.excluded_embed_metadata_keys = list(document.metadata)
            document.excluded_llm_metadata_keys = ["embed_model"]
        return documents

    def build(self, *, rebuild: bool = False, create: bool = True) -> RAGService:
        """Подключиться к коллекции RAG__COLLECTION или, если её нет (или rebuild), проиндексировать
        корпус. create=False — только подключиться (lifespan сервиса): нет коллекции — RAGNotIndexed."""
        try:
            exists = self.client.collection_exists(self.collection)
            if exists and rebuild:
                self.client.delete_collection(self.collection)
                exists = False
            points = self.client.count(self.collection, exact=True).count if exists else 0
            store = QdrantVectorStore(client=self.client, collection_name=self.collection)
            if points:
                self.check_collection()
                index = VectorStoreIndex.from_vector_store(store, embed_model=self.embed_model)
                self.indexed = 0
            elif not create:
                raise RAGNotIndexed(f"Коллекция {self.collection} пуста или её нет: проиндексируйте корпус — "
                                    "python -m app.services.rag --build")
            else:
                splitter = SentenceSplitter(chunk_size=self.settings.chunk_size,
                                            chunk_overlap=self.settings.chunk_overlap)
                index = VectorStoreIndex.from_documents(
                    self.load_documents(), storage_context=StorageContext.from_defaults(vector_store=store),
                    embed_model=self.embed_model, transformations=[splitter])
                self.indexed = self.client.count(self.collection, exact=True).count
                self.check_collection()
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise qdrant_error(exc, self.cfg.qdrant_url) from exc
        qa, refine = prompts(self.cfg.support.product_name)
        self.query_engine = index.as_query_engine(similarity_top_k=self.settings.similarity_top_k, llm=self.llm,
                                                  text_qa_template=qa, refine_template=refine)
        return self

    def check_collection(self) -> None:
        """Размерность, метрика и модель эмбеддингов коллекции — те же, что сейчас в .env."""
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        if not isinstance(vectors, VectorParams):
            raise RAGError(f"Коллекция {self.collection} с именованными векторами: её создавали не этим "
                           "кодом — пересоберите: python -m app.services.rag --rebuild")
        if vectors.size != self.cfg.embedding_dim or vectors.distance != Distance.COSINE:
            raise RAGError(f"Коллекция {self.collection}: векторы из {vectors.size} чисел, метрика "
                           f"{vectors.distance.value}, а нужно EMBEDDING_DIM={self.cfg.embedding_dim} и Cosine — "
                           "её заполняла другая модель: python -m app.services.rag --rebuild")
        records, _ = self.client.scroll(self.collection, limit=1, with_payload=["embed_model"])
        model = (records[0].payload or {}).get("embed_model") if records else None
        if model and model != self.embed_model.model_name:
            raise RAGError(f"Коллекция {self.collection} проиндексирована моделью {model}, а сейчас "
                           f"{self.embed_model.model_name}: вопросы и фрагменты были бы в разных пространствах — "
                           "python -m app.services.rag --rebuild")

    # -------------------------------------------------------- ответ
    def answer(self, question: str) -> dict[str, Any]:
        """Поиск top-k фрагментов и ответ модели по ним. Лучший фрагмент ниже RAG__MIN_SCORE —
        ответ NO_ANSWER без вызова модели: выдумывать не из чего."""
        if self.query_engine is None:
            raise RAGError("Индекс не готов: сначала build()")
        question = question.strip()
        if not question:
            raise ValueError("Пустой вопрос")
        bundle = QueryBundle(question)
        try:
            nodes = self.query_engine.retrieve(bundle)
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise qdrant_error(exc, self.cfg.qdrant_url) from exc
        top_score = max((node.score or 0.0 for node in nodes), default=0.0)
        if top_score < self.settings.min_score:
            return result(NO_ANSWER, top_score, sources(nodes))
        try:
            response = self.query_engine.synthesize(bundle, nodes)
        except Exception as exc:
            explained = llm_error(exc, self.cfg)
            if explained is exc:
                raise
            raise explained from exc
        return result(str(response).strip() or NO_ANSWER, top_score, sources(response.source_nodes))

    def ensure_ready(self) -> None:
        """Для сервиса: подключиться к коллекции, если на старте её ещё не было. Один раз —
        дальше запросы идут в готовый query_engine; проиндексировать корпус можно с хоста, не
        перезапуская сервис (python -m app.services.rag --build)."""
        if self.query_engine is None:
            with self._lock:
                if self.query_engine is None:
                    self.build(create=False)

    def close(self) -> None:
        self.client.close()


def open_rag(cfg: Settings) -> RAGService | None:
    """Для lifespan сервиса: RAGService и подключение к готовой коллекции — один раз на процесс.
    Без QDRANT_URL — None (/rag/query отвечает 503). Коллекции ещё нет или Qdrant не отвечает —
    сервис стартует, а RAG подключится при первом запросе (ensure_ready). Коллекция под другую
    модель — исключение: старт с заведомо неверной конфигурацией хуже понятной ошибки."""
    from app.observability.logging import get_logger

    log = get_logger()
    if not cfg.qdrant_url:
        log.info("rag_disabled", reason="QDRANT_URL не задан")
        return None
    service = RAGService(cfg)
    try:
        service.build(create=False)
        log.info("rag_ready", collection=service.collection, top_k=cfg.rag.similarity_top_k,
                 min_score=cfg.rag.min_score, model=cfg.rag.llm_model or cfg.llm.default_model,
                 embed_model=service.embed_model.model_name)
    except (RAGNotIndexed, RAGUnavailable) as exc:
        log.warning("rag_not_ready", collection=service.collection, error=str(exc),
                    note="сервис работает; RAG подключится при первом запросе к /rag/query")
    except Exception:
        service.close()
        raise
    return service


def no_implicit_event_loop() -> None:
    """Для вызовов RAGService из главного потока (CLI, scripts/rag_compare.py). Синхронные методы
    LlamaIndex запускают свои корутины через asyncio.get_event_loop() (async_utils.asyncio_run): в
    главном потоке без цикла Python 3.12–3.13 создаёт цикл неявно — с DeprecationWarning и без
    закрытия. После set_event_loop(None) get_event_loop падает, и LlamaIndex берёт asyncio.run — как
    в потоках сервиса (asyncio.to_thread) и как будет по умолчанию в Python 3.14."""
    asyncio.set_event_loop(None)


def sources(nodes: list[NodeWithScore]) -> list[dict[str, Any]]:
    return [source(node.text, node.metadata.get("file_name"), node.score) for node in nodes]


# ---------------------------------------------------------------- python -m app.services.rag
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG на LlamaIndex по корпусу data/rag-block-03 (блок 5.3)")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION, help="вопрос по корпусу")
    parser.add_argument("--rebuild", action="store_true", help="удалить коллекцию и проиндексировать заново")
    parser.add_argument("--build", action="store_true", help="только подключиться или проиндексировать, без вопроса")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from pydantic import ValidationError

    from app.core.config import get_settings
    from app.observability.logging import setup_logging

    setup_logging("WARNING", stream=sys.stderr)
    no_implicit_event_loop()
    try:
        cfg = get_settings()
        service = RAGService(cfg).build(rebuild=args.rebuild)
        print(f"Коллекция {service.collection}: " + (f"проиндексировано фрагментов — {service.indexed}"
              if service.indexed else "подключились к готовой"), file=sys.stderr)
        if not args.build:
            print(json.dumps(service.answer(args.question), ensure_ascii=False, indent=2))
        service.close()
        return 0
    except (RAGError, embeddings.EmbeddingError, ValidationError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
