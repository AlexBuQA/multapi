"""Схемы POST /rag/query (блок 5.3): тот же словарь, что возвращает RAGService.answer."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RagQuery(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="Вопрос по базе знаний",
                          examples=["Сколько стоит тариф «Команда» в месяц?"])


class RagSource(BaseModel):
    text: str = Field(description="Начало фрагмента (300 символов), который получила модель")
    source: str | None = Field(default=None, description="Файл корпуса", examples=["02_tariffs_and_payment.md"])
    score: float = Field(description="Косинусная близость вопроса и фрагмента")


class RagAnswer(BaseModel):
    answer: str = Field(description="Ответ модели по фрагментам или «В базе знаний ответа нет.»")
    top_score: float = Field(description="Близость лучшего фрагмента; ниже RAG__MIN_SCORE модель не вызывается")
    sources: list[RagSource] = Field(description="Фрагменты, по которым построен ответ (RAG__SIMILARITY_TOP_K)")
