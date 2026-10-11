"""
RAG без фреймворка (блок 5.3): тот же путь, что app/services/rag.py, но руками — чтобы видеть,
что именно делает LlamaIndex.

    python -m app.services.rag_baremetal                       # индекс + вопрос по умолчанию, вывод — JSON
    python -m app.services.rag_baremetal "Сколько запросов в минуту можно делать к API?"
    python -m app.services.rag_baremetal --rebuild "…"

1. Чтение — сами: .md и .txt как текст UTF-8, .pdf — pypdf (текст страниц), .docx — python-docx
   (абзацы). Другие форматы пропускаются.
2. Нарезка — наивная: слова подряд, пока фрагмент не наберёт RAG__CHUNK_SIZE токенов (cl100k_base,
   как у SentenceSplitter), следующий начинается на RAG__CHUNK_OVERLAP токенов раньше. Границы
   предложений и заголовков не учитываются, переносы строк теряются — в отличие от SentenceSplitter.
3. Эмбеддинги — тот же модуль блока 5.1 (embed_documents / embed_query): та же модель и кеш.
4. Своя коллекция RAG__BAREMETAL_COLLECTION, плоский payload {text, file_name, chunk_index,
   embed_model}, id — uuid5 от файла и номера фрагмента: повторная индексация перезаписывает те же точки.
5. Поиск — qdrant_client.query_points (search устарел), top-k = RAG__SIMILARITY_TOP_K.
6. Промпт — system + фрагменты + вопрос (тексты из rag_common, как у LlamaIndex) →
   openai.chat.completions.create к модели LLM__*. Порог RAG__MIN_SCORE — тот же.

answer(question) -> тот же словарь, что у RAGService.answer: answer, top_score, sources.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.services import embeddings
from app.services.rag_common import (
    DEFAULT_QUESTION,
    NO_ANSWER,
    USER_TEMPLATE,
    RAGError,
    RAGNotIndexed,
    llm_error,
    qdrant_error,
    result,
    source,
    system_prompt,
)
from app.services.vector_store import make_qdrant_client

if TYPE_CHECKING:
    from app.core.config import Settings

# Что считать «кодом конвейера» при сравнении с LlamaIndex (scripts/rag_compare.py, docs/rag.md):
# индексация и запрос. Сверка коллекции (check_collection) — одинаковая в обеих версиях, не считается.
PIPELINE = {
    "ingestion": ("read_file", "naive_chunks", "BareMetalRAG.build"),
    "query": ("build_openai", "BareMetalRAG.answer"),
}
NAMESPACE = uuid.UUID("4b7c9a10-3e52-4d8b-9f61-0a2d5c7e8b33")
TEXT_SUFFIXES = {".md", ".txt"}


def read_file(path: Path) -> str:
    """Текст файла корпуса; формат, который не умеем читать, — пустая строка."""
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix == ".docx":
        from docx import Document

        return "\n".join(paragraph.text for paragraph in Document(str(path)).paragraphs)
    return ""


def naive_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Слова подряд до chunk_size токенов; следующий фрагмент — с перекрытием overlap токенов."""
    words = text.split()
    sizes = [embeddings.count_tokens(word + " ") for word in words]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end, tokens = start, 0
        while end < len(words) and (end == start or tokens + sizes[end] <= chunk_size):
            tokens += sizes[end]
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        back, tokens = end, 0
        while back > start + 1 and tokens + sizes[back - 1] <= overlap:
            back -= 1
            tokens += sizes[back]
        start = back
    return chunks


class BareMetalRAG:
    def __init__(self, cfg: Settings, *, client: Any = None, embedder: Any = None, llm_client: Any = None) -> None:
        if client is None and not cfg.qdrant_url:
            raise RAGError("QDRANT_URL не задан: укажите адрес Qdrant в .env")
        if cfg.embedding_dim is None:
            raise RAGError("EMBEDDING_DIM не задан: укажите размерность модели эмбеддингов в .env")
        self.cfg = cfg
        self.settings = cfg.rag
        self.collection = cfg.rag.baremetal_collection
        if client is None:
            key = cfg.qdrant_api_key.get_secret_value() if cfg.qdrant_api_key else None
            client = make_qdrant_client(cfg.qdrant_url, key, asynchronous=False)
        self.client = client
        self.embedder = embedder or embeddings.default_service()
        self.llm = llm_client or build_openai(cfg)
        self.indexed = 0

    def build(self, *, rebuild: bool = False, create: bool = True) -> BareMetalRAG:
        """Коллекция есть и не пуста — проверить и пользоваться; нет (или rebuild) — проиндексировать."""
        try:
            exists = self.client.collection_exists(self.collection)
            if exists and rebuild:
                self.client.delete_collection(self.collection)
                exists = False
            if exists and self.client.count(self.collection, exact=True).count:
                self.check_collection()
                self.indexed = 0
                return self
            if not create:
                raise RAGNotIndexed(f"Коллекция {self.collection} пуста или её нет: "
                                     "python -m app.services.rag_baremetal --build")
            if not exists:
                self.client.create_collection(self.collection, vectors_config=VectorParams(
                    size=self.cfg.embedding_dim, distance=Distance.COSINE))
            rows = []
            for path in sorted(p for p in self.settings.data_dir.rglob("*") if p.is_file()):
                for number, chunk in enumerate(naive_chunks(read_file(path), self.settings.chunk_size,
                                                            self.settings.chunk_overlap)):
                    rows.append((path.name, number, chunk))
            vectors = self.embedder.embed_documents([chunk for _, _, chunk in rows])
            self.client.upsert(self.collection, wait=True, points=[
                PointStruct(id=str(uuid.uuid5(NAMESPACE, f"{name}#{number}")), vector=vector,
                            payload={"text": chunk, "file_name": name, "chunk_index": number,
                                     "embed_model": self.embedder.label})
                for (name, number, chunk), vector in zip(rows, vectors, strict=True)])
            self.indexed = len(rows)
            return self
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise qdrant_error(exc, self.cfg.qdrant_url) from exc

    def check_collection(self) -> None:
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        if not isinstance(vectors, VectorParams) or vectors.size != self.cfg.embedding_dim:
            raise RAGError(f"Коллекция {self.collection} не под EMBEDDING_DIM={self.cfg.embedding_dim}: "
                                 "python -m app.services.rag_baremetal --rebuild")
        records, _ = self.client.scroll(self.collection, limit=1, with_payload=["embed_model"])
        model = (records[0].payload or {}).get("embed_model") if records else None
        if model and model != self.embedder.label:
            raise RAGError(f"Коллекция {self.collection} проиндексирована моделью {model}, а сейчас "
                                 f"{self.embedder.label}: python -m app.services.rag_baremetal --rebuild")

    def answer(self, question: str) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("Пустой вопрос")
        vector = self.embedder.embed_query(question)
        try:
            hits = self.client.query_points(self.collection, query=vector, limit=self.settings.similarity_top_k,
                                            with_payload=True).points
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            raise qdrant_error(exc, self.cfg.qdrant_url) from exc
        found = [source(hit.payload["text"], hit.payload.get("file_name"), hit.score) for hit in hits]
        top_score = hits[0].score if hits else 0.0
        if top_score < self.settings.min_score:
            return result(NO_ANSWER, top_score, found)
        context = "\n\n".join(f"file_name: {hit.payload.get('file_name')}\n\n{hit.payload['text']}" for hit in hits)
        messages = [
            {"role": "system", "content": system_prompt(self.cfg.support.product_name)},
            {"role": "user", "content": USER_TEMPLATE.format(context_str=context, query_str=question)},
        ]
        try:
            completion = self.llm.chat.completions.create(
                model=self.settings.llm_model or self.cfg.llm.default_model, messages=messages,
                max_tokens=self.settings.max_tokens, temperature=self.settings.temperature)
        except Exception as exc:
            explained = llm_error(exc, self.cfg)
            if explained is exc:
                raise
            raise explained from exc
        text = (completion.choices[0].message.content or "").strip()
        return result(text or NO_ANSWER, top_score, found)

    def close(self) -> None:
        self.client.close()


def build_openai(cfg: Settings) -> Any:
    """Клиент OpenAI SDK к модели LLM__* — как у сервиса: прокси, сертификаты ОС, заголовки OpenRouter."""
    from openai import DefaultHttpxClient, OpenAI

    from app.core.config import (
        http_client_options,
        is_local_url,
        provider_headers,
        proxy_for,
    )

    proxy = proxy_for(cfg.llm.base_url, cfg.llm.proxy_url)
    options = http_client_options(proxy, cfg.llm.use_system_certs and not is_local_url(cfg.llm.base_url))
    return OpenAI(api_key=cfg.llm.openai_api_key.get_secret_value(), base_url=cfg.llm.base_url,
                  timeout=cfg.llm.request_timeout, max_retries=cfg.llm.max_retries,
                  http_client=DefaultHttpxClient(**options) if options else None,
                  default_headers=provider_headers(cfg.llm.base_url, "multapi"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG без фреймворка по корпусу data/rag-block-03 (блок 5.3)")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION, help="вопрос по корпусу")
    parser.add_argument("--rebuild", action="store_true", help="удалить коллекцию и проиндексировать заново")
    parser.add_argument("--build", action="store_true", help="только индексация, без вопроса")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from pydantic import ValidationError

    from app.core.config import get_settings
    from app.observability.logging import setup_logging

    setup_logging("WARNING", stream=sys.stderr)
    try:
        service = BareMetalRAG(get_settings()).build(rebuild=args.rebuild)
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
