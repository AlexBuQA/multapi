"""
Слой 2 модерации (блок 4.4, необязательный): OpenAI Moderation API, модель omni-moderation-latest.

    await client.moderations.create(model="omni-moderation-latest", input=text)

Ответ — flagged, categories (bool) и category_scores (0..1) по категориям OpenAI: harassment,
hate, illicit, self-harm, sexual, violence и их подкатегориям. Категория блокирует текст:
- если для неё задан свой порог (MODERATION__THRESHOLDS) — когда score >= порога. Так можно и
  ужесточить проверку (violence: 0.3), и ослабить её;
- иначе — когда её отметил сам OpenAI (categories[...] = true).
Имена категорий — как у слоя ключевых слов: «self-harm/intent» -> «self_harm_intent».

Ключ — MODERATION__OPENAI_API_KEY (или AUDIO_API_KEY: оба — ключи OpenAI). Через OpenRouter
модерация не работает: у него нет /moderations. Модель Ollama на CPU тоже не умеет
модерировать, поэтому по умолчанию слой выключен и работает только слой ключевых слов.

Ошибка API (сеть, 429, ключ) текст не блокирует: слой ключевых слов уже пройден, а
недоступный OpenAI не должен останавливать поддержку. В лог — moderation_api_failed.
MODERATION__FAIL_CLOSED=true — наоборот, блокировать при ошибке (категория moderation_unavailable).
"""
from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

DEFAULT_MODEL = "omni-moderation-latest"
MAX_INPUT_CHARS = 30_000          # длинный документ — его начало: тарифицируются токены входа


def category_name(name: str) -> str:
    """«self-harm/intent» -> «self_harm_intent»: как категории слоя ключевых слов."""
    return name.replace("-", "_").replace("/", "_")


class OpenAIModerator:
    def __init__(self, client: AsyncOpenAI, *, model: str = DEFAULT_MODEL,
                 thresholds: dict[str, float] | None = None) -> None:
        self.client = client
        self.model = model
        self.thresholds = {category_name(k): float(v) for k, v in (thresholds or {}).items()}

    async def categories(self, text: str) -> list[str]:
        """Категории, по которым текст блокируется; пустой список — текст в порядке.
        Ошибки API поднимаются как есть: что с ними делать, решает ModerationService."""
        response = await self.client.moderations.create(model=self.model, input=text[:MAX_INPUT_CHARS])
        blocked: list[str] = []
        for result in response.results:
            flags: dict[str, Any] = result.categories.model_dump(by_alias=True)
            scores: dict[str, Any] = result.category_scores.model_dump(by_alias=True)
            for raw_name, flagged in flags.items():
                name = category_name(raw_name)
                threshold = self.thresholds.get(name)
                score = scores.get(raw_name) or 0.0
                hit = score >= threshold if threshold is not None else bool(flagged)
                if hit and name not in blocked:
                    blocked.append(name)
        return blocked
