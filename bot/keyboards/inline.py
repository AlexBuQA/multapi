"""
Inline-клавиатуры (блок 4.2).

Темы /ask — разделы руководства пользователя «Личного кабинета» (data/knowledge_base.json),
по которому отвечает ассистент техподдержки из дипломного проекта: вход и пароль,
уведомления, оплата, API, мобильное приложение. Тема уходит в сервис в начале вопроса
(«Тема: Оплата. Вопрос: …») и помогает модели понять, о чём речь.

callback_data — "topic:<slug>", отмена — "topic:cancel" (до 64 байт, как требует Telegram).
"""
from __future__ import annotations

from typing import NamedTuple
from uuid import UUID

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

TOPIC_PREFIX = "topic:"
CANCEL = "cancel"


class Topic(NamedTuple):
    slug: str
    title: str


TOPICS: dict[str, Topic] = {t.slug: t for t in (
    Topic("login", "Вход и пароль"),
    Topic("notifications", "Уведомления и письма"),
    Topic("billing", "Оплата и документы"),
    Topic("api", "API и ключи"),
    Topic("mobile", "Мобильное приложение"),
)}


def topics_kb() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for topic in TOPICS.values():
        builder.button(text=topic.title, callback_data=TOPIC_PREFIX + topic.slug)
    builder.button(text="Отмена", callback_data=TOPIC_PREFIX + CANCEL)
    builder.adjust(2, 2, 1, 1)            # темы по две в ряд, «Отмена» — отдельной строкой
    return builder.as_markup()


# Оценка ответа (блок 4.4): callback_data "fb:up:<message_id>" / "fb:down:<message_id>" —
# до 44 байт при лимите Telegram 64. message_id — id ответа в сервисе из события done.
FEEDBACK_PREFIX = "fb:"


def feedback_kb(message_id: UUID) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="👍", callback_data=f"{FEEDBACK_PREFIX}up:{message_id}")
    builder.button(text="👎", callback_data=f"{FEEDBACK_PREFIX}down:{message_id}")
    builder.adjust(2)
    return builder.as_markup()


def parse_feedback(data: str | None) -> tuple[str, UUID] | None:
    """«fb:up:<uuid>» -> ("up", UUID); всё остальное — None."""
    parts = (data or "").split(":")
    if len(parts) != 3 or parts[0] + ":" != FEEDBACK_PREFIX or parts[1] not in ("up", "down"):
        return None
    try:
        return parts[1], UUID(parts[2])
    except ValueError:
        return None
