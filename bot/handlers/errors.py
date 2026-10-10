"""
Последний рубеж (блок 4.2): любое необработанное исключение в handler.

Ошибки chat-сервиса handlers ловят сами и отвечают понятным текстом (bot/texts.py). Сюда
попадает остальное — например, Telegram отклонил запрос. В лог — трассировка и номер
update, пользователю — «что-то пошло не так», без подробностей.

Сбой связи с Telegram (TelegramNetworkError: таймаут, обрыв через прокси) — не ошибка кода:
в лог — одна строка без трассировки, пользователю — что ответ не дошёл и команду, возможно,
стоит повторить (блок 4.4: так оборвался ответ на /clear у бота в Docker).
"""
from __future__ import annotations

import contextlib
import logging

from aiogram import Router
from aiogram.exceptions import TelegramNetworkError
from aiogram.types import ErrorEvent

from bot import texts

router = Router(name="errors")
log = logging.getLogger(__name__)


@router.errors()
async def on_error(event: ErrorEvent) -> bool:
    update = event.update
    if isinstance(event.exception, TelegramNetworkError):
        log.warning("telegram_network_error update_id=%s error=%s", update.update_id, event.exception.message)
        text = texts.TELEGRAM_NETWORK
    else:
        log.error("unhandled_error update_id=%s", update.update_id, exc_info=event.exception)
        text = texts.UNEXPECTED
    with contextlib.suppress(Exception):          # Telegram недоступен — ответить некуда
        if update.callback_query is not None:
            await update.callback_query.answer(text, show_alert=True)
        elif update.message is not None:
            await update.message.answer(text)
    return True
