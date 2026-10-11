"""
Подставные модели для тестов RAG (блоки 5.3–5.4) без сети: эмбеддинги из триграмм текста и модель
ответа, которая запоминает промпт. Файл не начинается с test_ — pytest не ищет в нём тестов.
"""
from __future__ import annotations

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import Field

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "rag-block-03"
DIM = 256


class TrigramEmbeddings:
    """Заглушка модуля эмбеддингов блока 5.1: тот же интерфейс (label, embed_query,
    embed_documents), вектор — триграммы текста, нормированный. Считает вызовы."""

    def __init__(self, label: str = "openai:local:11434/trigram-test") -> None:
        self.label = label
        self.document_calls = 0
        self.query_calls = 0

    @staticmethod
    def vector(text: str) -> list[float]:
        v = [0.0] * DIM
        padded = f"  {text.lower()} "
        for i in range(len(padded) - 2):
            v[int(hashlib.md5(padded[i:i + 3].encode()).hexdigest()[:8], 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self.vector(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self.vector(text) for text in texts]


class RecordingLLM(CustomLLM):
    """Модель ответа для LlamaIndex: запоминает промпты, отвечает заданной строкой."""

    reply: str = "Лимит зависит от тарифа: «Команда» — 300 запросов в минуту."
    prompts: list[str] = Field(default_factory=list)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=4096, num_output=256, model_name="recording")

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        self.prompts.append(prompt)
        return CompletionResponse(text=self.reply)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        raise NotImplementedError


def threaded(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Вызов RAG так же, как в сервисе (asyncio.to_thread), — в потоке без цикла событий. Из главного
    потока LlamaIndex на Python 3.12–3.13 создал бы цикл неявно (rag.no_implicit_event_loop)."""
    with ThreadPoolExecutor(1) as pool:
        return pool.submit(fn, *args, **kwargs).result()
