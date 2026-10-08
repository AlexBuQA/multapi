"""
Ответ потоком в Telegram (блок 4.2).

answer_with_stream(message, backend, chat_id, prompt):
1. Пока модель думает над первым словом (llama3.2 на CPU — до 15 с), в чате висит
   «печатает…» (ChatActionSender: sendChatAction каждые 5 с).
2. Первый фрагмент — новое сообщение бота, дальше оно дописывается message.edit_text(buffer):
   буфер копит весь текст, каждая правка показывает его целиком.
3. Правки — не чаще раза в EDIT_INTERVAL секунд. Задание говорит «после каждого SSE-чанка»,
   но фрагменты приходят по слову, а Telegram ограничивает частоту правок одного чата
   (около раза в секунду) и отвечает 429 Too Many Requests с паузой. Фрагменты
   накапливаются в буфере, а очередная правка показывает всё пришедшее; последняя правка
   после [DONE] — всегда, полный текст не теряется. Если Telegram всё же ответил
   RetryAfter, промежуточные правки не делаются, пока пауза не пройдёт; финальная правка и
   новое сообщение ждут паузу и повторяются.
4. Ответ длиннее MESSAGE_LIMIT символов (у Telegram предел 4096) продолжается новым
   сообщением; место разрыва — последний перевод строки или пробел.
5. Ошибка до первого фрагмента — сообщение об ошибке вместо ответа; посреди ответа —
   пометка «Ответ прерван: …» к уже показанному тексту. Текст ответа модели не
   разбирается как HTML/Markdown: parse_mode не задан, спецсимволы показываются как есть.

В блоке 4.3 вместо edit_text будет нативный sendMessageDraft (Bot API 10.0).
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from typing import Protocol
from uuid import UUID

from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from bot import texts
from bot.services.backend_client import BACKEND_ERRORS

log = logging.getLogger(__name__)

EDIT_INTERVAL = 1.0      # с между правками одного сообщения
MESSAGE_LIMIT = 4000     # символов в сообщении; у Telegram 4096 — с запасом на пометку об ошибке


class StreamingBackend(Protocol):
    def send_message(self, chat_id: UUID, content: str): ...  # -> AsyncIterator[str]


def split_point(text: str, limit: int) -> int:
    """Где разорвать слишком длинный текст: последний перевод строки или пробел до limit."""
    for separator in ("\n", " "):
        position = text.rfind(separator, limit // 2, limit)
        if position > 0:
            return position + 1
    return limit


class StreamRenderer:
    """Показывает растущий текст ответа: первое сообщение, затем правки с интервалом."""

    def __init__(self, source: Message, *, interval: float = EDIT_INTERVAL, limit: int = MESSAGE_LIMIT,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.source = source
        self.interval = interval
        self.limit = limit
        self.clock = clock
        self.current: Message | None = None    # сообщение бота, которое сейчас дописывается
        self.buffer = ""                       # его полный текст
        self.shown = ""                        # что из него уже видно в Telegram
        self.last_edit = 0.0
        self.messages: list[Message] = []      # все сообщения ответа
        self.edits = 0
        self.chars = 0                         # символов ответа пришло от сервиса
        self.blocked_until = 0.0               # Telegram попросил паузу (RetryAfter) до этого момента

    @property
    def started(self) -> bool:
        return bool(self.messages)

    async def feed(self, chunk: str) -> None:
        self.buffer += chunk
        self.chars += len(chunk)
        await self._overflow()
        await self._show(force=False)

    async def finish(self, note: str = "") -> None:
        self.buffer += note
        await self._overflow()
        await self._show(force=True)

    async def _overflow(self) -> None:
        while len(self.buffer) > self.limit:
            cut = split_point(self.buffer, self.limit)
            head, tail = self.buffer[:cut], self.buffer[cut:]
            self.buffer = head
            await self._show(force=True)           # дописать текущее сообщение до конца
            self.current, self.buffer, self.shown = None, tail, ""

    async def _show(self, *, force: bool) -> None:
        text = self.buffer
        if not text.strip() or text == self.shown:
            return
        if self.current is None:
            self.current = await self._send(text)
            self.messages.append(self.current)
        else:
            now = self.clock()
            if not force and (now - self.last_edit < self.interval or now < self.blocked_until):
                return
            if not await self._edit(text, retry=force):
                return
        self.shown = text
        self.last_edit = self.clock()

    async def _send(self, text: str) -> Message:
        """Новое сообщение. Его нельзя пропустить, поэтому на RetryAfter — пауза и повтор."""
        try:
            return await self.source.answer(text)
        except TelegramRetryAfter as exc:
            await asyncio.sleep(exc.retry_after)
            return await self.source.answer(text)

    async def _edit(self, text: str, *, retry: bool) -> bool:
        assert self.current is not None
        try:
            await self.current.edit_text(text)
        except TelegramRetryAfter as exc:
            if not retry:
                # Промежуточные правки — после паузы: следующие фрагменты копятся в буфере.
                self.blocked_until = self.clock() + exc.retry_after
                return False
            await asyncio.sleep(exc.retry_after)
            await self.current.edit_text(text)
        except TelegramBadRequest as exc:
            if "message is not modified" not in exc.message:
                raise
        self.edits += 1
        return True


async def answer_with_stream(message: Message, backend: StreamingBackend, chat_id: UUID, prompt: str, *,
                             interval: float = EDIT_INTERVAL) -> StreamRenderer:
    renderer = StreamRenderer(message, interval=interval)
    chunks = backend.send_message(chat_id, prompt)
    try:
        assert message.bot is not None
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            first = await anext(chunks, None)
        if first is None:
            await message.answer(texts.EMPTY_ANSWER)
            return renderer
        await renderer.feed(first)
        async for chunk in chunks:
            await renderer.feed(chunk)
        await renderer.finish()
        if not renderer.started:                       # ответ из одних пробелов и переводов строк
            await message.answer(texts.EMPTY_ANSWER)
    except BACKEND_ERRORS as exc:
        log.warning("backend_error chat=%s during=stream error=%r", message.chat.id, exc)
        reason = texts.user_message(exc)
        if renderer.started:
            await renderer.finish(texts.INTERRUPTED.format(reason=reason))
        else:
            await message.answer(reason)
    finally:
        with contextlib.suppress(Exception):
            await chunks.aclose()
    return renderer
