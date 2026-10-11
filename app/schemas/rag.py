"""Схемы RAG: POST /rag/query (блоки 5.3–5.5) — тот же словарь, что возвращает RAGService.answer, —
и загрузка документов POST /documents/upload (блок 5.5)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RagQuery(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="Вопрос по базе знаний",
                          examples=["Каков срок возврата оплаты?"])


class RagSource(BaseModel):
    id: int = Field(description="Номер цитаты: [id] в тексте ответа", examples=[1])
    file_name: str | None = Field(default=None, description="Имя файла", examples=["refunds_v2.docx"])
    source: str | None = Field(default=None, description="Путь в корпусе (data/…)", examples=["billing/refunds_v2.docx"])
    category: str | None = Field(default=None, description="Категория — папка корпуса", examples=["billing"])
    page: int | None = Field(default=None, description="Страница PDF (у других форматов — null)")
    score: float = Field(description="Косинусная близость вопроса и фрагмента")
    snippet: str = Field(description="Начало фрагмента (200 символов), который получила модель")
    cited: bool = Field(default=False, description="Модель сослалась на фрагмент в ответе ([id])")


class RagAnswer(BaseModel):
    answer: str = Field(description="Ответ с цитатами [1], [2] или «По базе не нашёл, могу эскалировать.»")
    top_score: float = Field(description="Близость лучшего фрагмента; ниже RAG__SCORE_THRESHOLD модель не вызывается")
    confident: bool = Field(description="Ответ дан по базе: top_score не ниже порога и модель не отказалась")
    sources: list[RagSource] = Field(description="Фрагменты контекста (RAG__SIMILARITY_TOP_K); при отказе — пусто")


class UploadAccepted(BaseModel):
    status: Literal["queued"] = "queued"
    file: str = Field(description="Путь в корпусе", examples=["uploads/tariffs_2027.pdf"])
    size: int = Field(description="Размер, байт")
    replaced: bool = Field(description="Файл с таким именем уже был — новая версия заменит его фрагменты")
    status_url: str = Field(description="Где смотреть результат индексации")


class DocumentStatus(BaseModel):
    file: str
    state: Literal["queued", "new", "changed", "unchanged", "failed", "removed"]
    documents: int = Field(default=0, description="Документов LlamaIndex: страниц PDF, разделов Markdown")
    error: str | None = None
    renamed_to: str | None = Field(default=None, description="Файл не прочитался и переименован в .failed")
    indexed: bool = Field(description="Фрагменты файла в коллекции и находятся поиском")
