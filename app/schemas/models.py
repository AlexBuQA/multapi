"""
Каталог моделей для GET /models (блок 3.4): статический список с ценами.

Цены OpenAI — справочные, в долларах за 1 млн токенов, по прайсу OpenAI 2025 года.
Перед расчётом бюджета сверяйте их с https://openai.com/api/pricing. Локальные модели
Ollama бесплатны: они считаются на своём компьютере. Их контекст зависит от
настройки num_ctx в Ollama, поэтому context_window не указан.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    id: str = Field(description="Имя модели для поля model в запросе")
    provider: Literal["openai", "ollama"]
    input_per_1m: float = Field(ge=0, description="Цена входных токенов, $ за 1 млн")
    output_per_1m: float = Field(ge=0, description="Цена выходных токенов, $ за 1 млн")
    context_window: int | None = Field(default=None, description="Окно контекста, токенов")
    description: str = ""


MODEL_CATALOG: list[ModelInfo] = [
    ModelInfo(id="gpt-4o-mini", provider="openai", input_per_1m=0.15, output_per_1m=0.60,
              context_window=128_000, description="Дешёвая модель для чата и tool calling"),
    ModelInfo(id="gpt-4o", provider="openai", input_per_1m=2.50, output_per_1m=10.00,
              context_window=128_000, description="Мультимодальная модель: текст и изображения"),
    ModelInfo(id="gpt-4.1-nano", provider="openai", input_per_1m=0.10, output_per_1m=0.40,
              context_window=1_047_576, description="Самая дешёвая из семейства 4.1: классификация, короткие ответы"),
    ModelInfo(id="gpt-4.1-mini", provider="openai", input_per_1m=0.40, output_per_1m=1.60,
              context_window=1_047_576, description="Длинный контекст по цене mini-модели"),
    ModelInfo(id="gpt-4.1", provider="openai", input_per_1m=2.00, output_per_1m=8.00,
              context_window=1_047_576, description="Старшая модель семейства 4.1"),
    ModelInfo(id="llama3.2", provider="ollama", input_per_1m=0.0, output_per_1m=0.0,
              description="Локально в Ollama, 3B: чат и классификатор проекта"),
    ModelInfo(id="qwen3:4b-instruct", provider="ollama", input_per_1m=0.0, output_per_1m=0.0,
              description="Локально в Ollama, 4B: Function Calling (блок 3.1)"),
    ModelInfo(id="llama3.2-vision", provider="ollama", input_per_1m=0.0, output_per_1m=0.0,
              description="Локально в Ollama, 11B: анализ изображений (вариант А)"),
]
