"""
Обычные сообщения (блок 4.2) — роутер подключается последним.

- Текст, не начинающийся с «/», — вопрос ассистенту: в сервис через
  BackendClient.send_message(chat_id, message.text), ответ — потоком (bot/services/streaming.py).
- Неизвестная команда — подсказка /help: в сервис она не уходит.
- Фото, голос, стикеры и прочее — «пока только текст» (медиа — блок 4.3).
"""
from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.types import Message

from bot import texts
from bot.handlers.common import backend_chat_id
from bot.services.backend_client import BACKEND_ERRORS, BackendClient
from bot.services.chat_queue import ChatQueue
from bot.services.streaming import answer_with_stream

router = Router(name="text")
log = logging.getLogger(__name__)


@router.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message, backend: BackendClient, chat_queue: ChatQueue) -> None:
    async with chat_queue.turn(message.chat.id):        # второй вопрос — после ответа на первый
        try:
            chat_id = await backend_chat_id(backend, message.chat)
        except BACKEND_ERRORS as exc:
            log.warning("backend_error chat=%s during=chat error=%r", message.chat.id, exc)
            await message.answer(texts.user_message(exc))
            return
        renderer = await answer_with_stream(message, backend, chat_id, message.text or "")
    # Ни вопрос, ни ответ в лог бота не пишутся: персональные данные маскирует и хранит сервис.
    log.info("answered telegram_chat=%s chat_id=%s messages=%d edits=%d chars=%d", message.chat.id, chat_id,
             len(renderer.messages), renderer.edits, renderer.chars)


@router.message(F.text.startswith("/"))
async def on_unknown_command(message: Message) -> None:
    await message.answer(texts.UNKNOWN_COMMAND)


@router.message()
async def on_other(message: Message) -> None:
    await message.answer(texts.ONLY_TEXT)
