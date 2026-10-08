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
