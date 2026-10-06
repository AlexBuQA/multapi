"""
Классификатор обращений (бонус ДЗ 2.5): FAQ / тех. проблема / жалоба / прочее.

Использует модель из SUPPORT_CLASSIFIER_MODEL (по умолчанию совпадает с primary,
т.е. llama3.2 на локальном Ollama). Возвращает одну из категорий строкой.

Ответ модели сопоставляется с категориями БЕЗ учёта регистра и пунктуации:
«тех. проблема», «тех проблема», «Тех.проблема» и «техническая проблема»
распознаются одинаково. Сравнение идёт по целым словам, а если в ответе
упомянуто несколько категорий — выбирается та, что встречается раньше.
"""
from __future__ import annotations

import re

from .config import settings
from .robust_client import RobustLLMClient
from .utils import get_logger

logger = get_logger("classifier")

CATEGORIES = ["FAQ", "тех. проблема", "жалоба", "прочее"]
DEFAULT_CATEGORY = "прочее"

# Дополнительные формулировки, которыми модель может назвать категорию.
# Само название категории проверяется всегда; варианты сравниваются после
# нормализации, поэтому точки, кавычки и регистр здесь не важны.
_ALIASES: dict[str, tuple[str, ...]] = {
    "FAQ": ("частый вопрос", "частые вопросы"),
    "тех. проблема": ("техпроблема", "техническая проблема", "технический сбой"),
    "жалоба": (),
    "прочее": ("другое",),
}

_SYSTEM = (
    "Ты классификатор обращений в техподдержку. Отнеси сообщение пользователя "
    "строго к одной из категорий: FAQ, тех. проблема, жалоба, прочее. "
    "Ответь ТОЛЬКО названием категории, без пояснений."
)


def normalize(text: str) -> str:
    """Нижний регистр, «ё»→«е», пунктуация → пробел, схлопывание пробелов."""
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _patterns() -> list[tuple[re.Pattern[str], str]]:
    """Регулярки «целое слово/фраза» для каждой категории и её вариантов."""
    result = []
    for cat in CATEGORIES:
        for variant in (cat, *_ALIASES.get(cat, ())):
            phrase = re.escape(normalize(variant))
            result.append((re.compile(rf"(?<!\w){phrase}(?!\w)"), cat))
    return result


_PATTERNS = _patterns()


def match_category(raw: str) -> str | None:
    """
    Сопоставляет ответ модели с категорией. Возвращает None, если ни одна
    категория не распознана.
    """
    text = normalize(raw)
    best: tuple[int, str] | None = None
    for pattern, cat in _PATTERNS:
        m = pattern.search(text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), cat)
    return best[1] if best else None


def classify(text: str, *, client: RobustLLMClient | None = None) -> str:
    """Возвращает категорию обращения. При сбое или нераспознанном ответе — 'прочее'."""
    client = client or RobustLLMClient()
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": text},
    ]
    raw = client.chat(
        messages, temperature=0.0, max_tokens=10,
        label="classify", model_override=settings.classifier_model,
    )

    category = match_category(raw)
    if category is None:
        logger.info("Категория обращения: %s (нераспознано: %r)", DEFAULT_CATEGORY, raw)
        return DEFAULT_CATEGORY
    logger.info("Категория обращения: %s (ответ модели: %r)", category, raw.strip())
    return category
