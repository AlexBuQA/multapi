"""
Классификатор обращений (бонус ДЗ 2.5): FAQ / тех. проблема / жалоба / прочее.

Использует модель из SUPPORT_CLASSIFIER_MODEL (по умолчанию совпадает с primary,
т.е. llama3.2 на локальном Ollama). Возвращает одну из категорий строкой.
"""
from __future__ import annotations

from .config import settings
from .robust_client import RobustLLMClient
from .utils import get_logger

logger = get_logger("classifier")

CATEGORIES = ["FAQ", "тех. проблема", "жалоба", "прочее"]

_SYSTEM = (
    "Ты классификатор обращений в техподдержку. Отнеси сообщение пользователя "
    "строго к одной из категорий: FAQ, тех. проблема, жалоба, прочее. "
    "Ответь ТОЛЬКО названием категории, без пояснений."
)


def classify(text: str, *, client: RobustLLMClient | None = None) -> str:
    """Возвращает категорию обращения. При сбое — 'прочее'."""
    client = client or RobustLLMClient()
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": text},
    ]
    raw = client.chat(
        messages, temperature=0.0, max_tokens=10,
        label="classify", model_override=settings.classifier_model,
    ).strip().lower()

    for cat in CATEGORIES:
        if cat.split()[0].lower() in raw:
            logger.info("Категория обращения: %s", cat)
            return cat
    logger.info("Категория обращения: прочее (нераспознано: %r)", raw)
    return "прочее"
