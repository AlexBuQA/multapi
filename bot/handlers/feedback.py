"""
Оценки ответов 👍/👎 (блок 4.4).

Под последним сообщением ответа — кнопки с callback_data fb:up:<message_id> и
fb:down:<message_id> (bot/services/streaming.py, attach_feedback). Нажатие:
1. разбор callback_data (parse_feedback); чужой формат — тихо игнорируется;
2. чат клиента в сервисе — по chat.id сообщения (POST /chats идемпотентен);
3. POST /chats/{chat_id}/messages/{message_id}/feedback {"value": "up" | "down"} —
   сервис хранит одну оценку на пару (owner_external_id, message_id);
4. кнопки убираются: edit_reply_markup(reply_markup=None), и всплывающее «Спасибо за
   оценку!» (или «Оценка уже учтена», если это повтор).
Ошибка сервиса — кнопки остаются, чтобы можно было нажать ещё раз; ответ сервиса 404 (ответ
не найден, например чат удалён) — кнопки убираются: оценить уже нельзя.
"""
from __future__ import annotations

import logging

import httpx
from aiogram import F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.types import CallbackQuery, Message

from bot import texts
from bot.handlers.common import backend_chat_id
from bot.keyboards.inline import FEEDBACK_PREFIX, parse_feedback
from bot.services.backend_client import BACKEND_ERRORS, BackendClient

router = Router(name="feedback")
log = logging.getLogger(__name__)


async def drop_buttons(message: Message) -> None:
    try:
        await message.edit_reply_markup(reply_markup=None)
    except TelegramAPIError as exc:          # сообщение старое или кнопок уже нет
        log.info("feedback_buttons_not_removed chat=%s error=%s", message.chat.id, exc)


@router.callback_query(F.data.startswith(FEEDBACK_PREFIX))
async def on_feedback(callback: CallbackQuery, backend: BackendClient) -> None:
    parsed = parse_feedback(callback.data)
    message = callback.message if isinstance(callback.message, Message) else None
    if parsed is None or message is None:
        await callback.answer(texts.FEEDBACK_GONE)
        return
    value, message_id = parsed
    try:
        chat_id = await backend_chat_id(backend, message.chat)
        result = await backend.send_feedback(chat_id, message_id, value)
    except BACKEND_ERRORS as exc:
        log.warning("feedback_failed telegram_chat=%s error=%r", message.chat.id, exc)
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (404, 422):
            await drop_buttons(message)
            await callback.answer(texts.FEEDBACK_GONE)
        else:
            await callback.answer(texts.FEEDBACK_FAILED)
        return
    log.info("feedback telegram_chat=%s message_id=%s value=%s saved=%s", message.chat.id, message_id, value,
             result.get("saved"))
    await drop_buttons(message)
    await callback.answer(texts.FEEDBACK_THANKS if result.get("saved") else texts.FEEDBACK_ALREADY)
