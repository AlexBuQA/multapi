"""
GET /kb/search — поиск по базе знаний в Qdrant (блок 5.2).

Вопрос кодируется моделью эмбеддингов блока 5.1 (embed_query, в потоке: модуль синхронный),
затем VectorStore.search с фильтром из параметров: раздел (category), файл (source), дата
редакции не раньше created_after; архивные редакции исключены, пока не попросят
include_archived=true. Это проверка цепочки «вопрос → вектор → Qdrant» и в Docker, и
основа для RAG блока 5.3: там найденные фрагменты уйдут модели.
"""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Query

from app.chat.domain import RequestError
from app.deps.providers import VectorStoreDep
from app.schemas.kb import KBHit, KBSearchResponse
from app.services import embeddings
from app.services.vector_store import VectorStoreError, VectorStoreUnavailable, build_filter

router = APIRouter(prefix="/kb", tags=["kb"])


@router.get(
    "/search",
    response_model=KBSearchResponse,
    summary="Поиск по базе знаний",
    description="Семантический поиск фрагментов справки в Qdrant: вопрос → эмбеддинг → top_k ближайших "
                "с фильтрами по метаданным. Архивные редакции по умолчанию исключены.",
    responses={503: {"description": "Векторная база не настроена или недоступна, нет модели эмбеддингов"}},
)
async def search_kb(
    store: VectorStoreDep,
    q: Annotated[str, Query(min_length=1, max_length=1000, description="Вопрос пользователя")],
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
    category: Annotated[str | None, Query(description="Раздел справки: billing, api, account, …")] = None,
    source: Annotated[str | None, Query(description="Файл базы знаний: faq.md, release_notes.md, …")] = None,
    created_after: Annotated[date | None, Query(description="Только редакции не старше этой даты")] = None,
    include_archived: bool = False,
) -> KBSearchResponse:
    if not q.strip():
        raise RequestError(422, "empty_query", "Пустой вопрос.")
    try:
        vector = await asyncio.to_thread(embeddings.embed_query, q)
    except embeddings.EmbeddingError as exc:
        raise RequestError(503, "embeddings_unavailable", str(exc)) from exc
    query_filter = build_filter(category=category, source=source, include_archived=include_archived,
                               created_after=created_after.isoformat() if created_after else None)
    try:
        points = await store.search(vector, top_k=top_k, query_filter=query_filter)
    except VectorStoreUnavailable as exc:
        raise RequestError(503, "vector_store_unavailable", str(exc)) from exc
    except VectorStoreError as exc:
        raise RequestError(503, "vector_store_error", str(exc)) from exc
    hits = []
    for point in points:
        payload = point.payload or {}
        hits.append(KBHit(id=str(point.id), score=round(point.score, 4),
                          **{key: payload.get(key) for key in ("doc_id", "title", "source", "category", "status",
                                                               "created_at", "text")}))
    return KBSearchResponse(query=q, collection=store.collection, hits=hits)
