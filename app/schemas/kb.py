"""Схемы ответа GET /kb/search (блок 5.2)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class KBHit(BaseModel):
    id: str = Field(description="id точки в Qdrant (uuid5 от источника и ключа фрагмента)")
    score: float = Field(description="Косинусная близость вопроса и фрагмента, от −1 до 1")
    doc_id: str | None = Field(default=None, examples=["KB-030"])
    title: str | None = Field(default=None, examples=["Возврат оплаты"])
    source: str | None = Field(default=None, examples=["help_center.jsonl"])
    category: str | None = Field(default=None, examples=["billing"])
    status: str | None = Field(default=None, examples=["actual"])
    created_at: str | None = Field(default=None, examples=["2026-05-19T00:00:00Z"])
    text: str | None = Field(default=None, description="Фрагмент — то, что получит модель в RAG (блок 5.3)")


class KBSearchResponse(BaseModel):
    query: str
    collection: str
    hits: list[KBHit]
