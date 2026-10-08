"""
Последний рубеж (блок 4.2): любое необработанное исключение в handler.

Ошибки chat-сервиса handlers ловят сами и отвечают понятным текстом (bot/texts.py). Сюда
попадает остальное — например, Telegram отклонил запрос. В лог — трассировка и номер
update, пользователю — «что-то пошло не так», без подробностей.
"""
from __future__ import annotations

import contextlib
import logging

from aiogram import Router
from aiogram.types import ErrorEvent

from bot import texts

router = Router(name="errors")
log = logging.getLogger(__name__)


@router.errors()
async def on_error(event: ErrorEvent) -> bool:
    update = event.update
    log.error("unhandled_error update_id=%s", update.update_id, exc_info=event.exception)
    with contextlib.suppress(Exception):          # Telegram недоступен — ответить некуда
        if update.callback_query is not None:
            await update.callback_query.answer(texts.UNEXPECTED, show_alert=True)
        elif update.message is not None:
            await update.message.answer(texts.UNEXPECTED)
    return True
