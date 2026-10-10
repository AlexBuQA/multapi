"""
Ответ потоком в Telegram (блоки 4.2–4.3).

answer_with_stream(message, backend, chat_id, prompt, media=..., mime=..., mode=...):
1. Пока модель думает над первым словом (llama3.2 на CPU — до 15 с, фото — дольше), в чате
   висит «печатает…» (ChatActionSender: sendChatAction каждые 5 с).
2. mode="draft" (блок 4.3, по умолчанию; BOT_STREAMING) — нативный черновик Telegram,
   DraftRenderer:
   - сразу — пустой черновик sendMessageDraft(text=""): по документации Bot API 10.0
     Telegram показывает «Thinking…». aiogram пустые поля не отправляет, так что text в
     запросе нет совсем; если Telegram такой черновик не примет, бот просто обойдётся без
     «Thinking…» — на правки сообщения он переходит, только если отклонён черновик с текстом;
   - фрагменты копятся в буфере, черновик с одним draft_id показывает весь буфер — Telegram
     анимирует рост текста. Обновления — не чаще раза в DRAFT_INTERVAL: фрагмент приходит
     на каждое слово, а запрос к Bot API на каждое слово — это десятки запросов в секунду
     и 429 Too Many Requests. Черновик живёт около 30 с: если модель надолго задумалась,
     он обновляется раз в DRAFT_KEEPALIVE тем же текстом;
   - в конце — sendMessage с полным текстом: черновик эфемерный, без этого ответ исчез бы;
   - ответ длиннее MESSAGE_LIMIT: готовая часть отправляется sendMessage, остаток — новым
     черновиком с новым draft_id.
   sendMessageDraft работает только в личных чатах. В группе, а также если Telegram
   отклонил черновик (старый Bot API), ответ показывается правками — как в блоке 4.2.
3. mode="edit" (блок 4.2) — StreamRenderer: первый фрагмент — новое сообщение, дальше
   message.edit_text(buffer) не чаще раза в EDIT_INTERVAL (Telegram ограничивает частоту
   правок одного чата). RetryAfter — промежуточные правки ждут паузу, финальная
   повторяется. Ответ длиннее MESSAGE_LIMIT продолжается новым сообщением.
4. Ошибка до первого фрагмента — сообщение об ошибке вместо ответа; посреди ответа —
   пометка «Ответ прерван: …» к уже показанному тексту, и он сохраняется сообщением. Текст
   ответа модели не разбирается как HTML/Markdown: parse_mode не задан.
5. Блок 4.4. Ответ не прошёл модерацию (событие moderation, AnswerStream.replacement) —
   renderer.replace: уже отправленные части ответа удаляются, вместо них — текст сервиса
   «Не могу показать ответ…» (тема «самоповреждение» — слова поддержки, texts.replacement_text).
   Черновик эфемерный и исчезает сам.
6. Блок 4.4. Под последним сообщением ответа — кнопки 👍/👎 (fb:up:<id>, fb:down:<id>, id —
   из события done). Под заменённым модерацией ответом и под ответом с ошибкой их нет.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal, Protocol
from uuid import UUID

import httpx
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest, TelegramNotFound, TelegramRetryAfter
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from bot import texts
from bot.keyboards.inline import feedback_kb
from bot.services.backend_client import BACKEND_ERRORS, error_body, error_code

log = logging.getLogger(__name__)

EDIT_INTERVAL = 1.0      # с между правками одного сообщения (mode="edit")
DRAFT_INTERVAL = 0.3     # с между обновлениями черновика (mode="draft")
DRAFT_KEEPALIVE = 20.0   # с: черновик живёт ~30 с — обновить, если текст давно не менялся
MESSAGE_LIMIT = 4000     # символов в сообщении; у Telegram 4096 — с запасом на пометку об ошибке
StreamMode = Literal["draft", "edit"]


class StreamingBackend(Protocol):
    def send_message(self, chat_id: UUID, content: str, *args: Any,
                     **kwargs: Any) -> AsyncIterator[str]: ...


async def delete_messages(messages: list[Message]) -> None:
    """Удалить части ответа (бот может удалять свои сообщения). Не вышло — не страшно: замена
    всё равно придёт отдельным сообщением."""
    for message in messages:
        try:
            await message.delete()
        except TelegramAPIError as exc:
            log.warning("message_delete_failed chat=%s error=%r", message.chat.id, exc)


def new_draft_id() -> int:
    """draft_id — ненулевое целое; одинаковый у всех обновлений одного черновика."""
    return secrets.randbelow(2**31 - 2) + 1


def split_point(text: str, limit: int) -> int:
    """Где разорвать слишком длинный текст: последний перевод строки или пробел до limit."""
    for separator in ("\n", " "):
        position = text.rfind(separator, limit // 2, limit)
        if position > 0:
            return position + 1
    return limit


class StreamRenderer:
    """Показывает растущий текст ответа: первое сообщение, затем правки с интервалом."""

    mode: StreamMode = "edit"

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

    @property
    def updates(self) -> int:
        return self.edits

    @property
    def all_messages(self) -> list[Message]:
        return self.messages

    async def start(self) -> None:
        """Ожидание первого фрагмента: «печатает…» показывает answer_with_stream."""

    async def close(self) -> None:
        pass

    async def feed(self, chunk: str) -> None:
        self.buffer += chunk
        self.chars += len(chunk)
        await self._overflow()
        await self._show(force=False)

    async def finish(self, note: str = "") -> None:
        self.buffer += note
        await self._overflow()
        await self._show(force=True)

    async def replace(self, text: str) -> None:
        """Ответ не прошёл модерацию: показанные части — удалить, вместо них — text."""
        await delete_messages(self.messages)
        self.messages, self.current, self.buffer, self.shown = [], None, "", ""
        self.current = await self._send(text)
        self.messages.append(self.current)

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


class DraftRenderer:
    """Растущий ответ черновиком sendMessageDraft; в конце — sendMessage с полным текстом.
    Если Telegram отклонил черновик, дальше ответ показывает StreamRenderer (правки)."""

    mode: StreamMode = "draft"

    def __init__(self, source: Message, *, interval: float = DRAFT_INTERVAL, limit: int = MESSAGE_LIMIT,
                 keepalive: float = DRAFT_KEEPALIVE, clock: Callable[[], float] = time.monotonic) -> None:
        assert source.bot is not None
        self.source = source
        self.bot = source.bot
        self.chat_id = source.chat.id
        self.thread_id = source.message_thread_id if source.is_topic_message else None
        self.interval = interval
        self.limit = limit
        self.keepalive = keepalive
        self.clock = clock
        self.draft_id = new_draft_id()
        self.buffer = ""                       # текст текущего черновика
        self.shown = ""                        # что из него уже в черновике у Telegram
        self.last_draft = 0.0
        self.blocked_until = 0.0               # Telegram попросил паузу (RetryAfter) до этого момента
        self.messages: list[Message] = []      # отправленные сообщения ответа
        self.drafts = 0                        # обновлений черновика
        self.chars = 0
        self.fallback: StreamRenderer | None = None
        self.placeholder = True                # Telegram принимает пустой черновик («Thinking…»)
        self._lock = asyncio.Lock()            # обновления черновика — по одному, по порядку
        self._keepalive_task: asyncio.Task[None] | None = None

    @property
    def started(self) -> bool:
        """Пользователь уже видит часть ответа — ошибку надо дописать к ней, а не вместо неё."""
        if self.fallback is not None:
            return bool(self.messages) or self.fallback.started
        return bool(self.messages) or bool(self.buffer.strip())

    @property
    def updates(self) -> int:
        return self.drafts + (self.fallback.edits if self.fallback else 0)

    @property
    def all_messages(self) -> list[Message]:
        return self.messages + (self.fallback.messages if self.fallback else [])

    async def start(self) -> None:
        """Пустой черновик — «Thinking…» — и поддержка его жизни, пока нет текста."""
        await self._draft("", force=True, origin="start")
        if self.fallback is None and self.keepalive > 0:
            self._keepalive_task = asyncio.create_task(self._keep_alive())

    async def close(self) -> None:
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive_task
            self._keepalive_task = None

    async def _keep_alive(self) -> None:
        """Отдельная задача: только обновляет черновик тем же текстом. На правки сообщения она
        не переключает — это делает лишь основной поток ответа (feed), иначе две задачи
        одновременно писали бы сообщения."""
        while self.fallback is None:
            await asyncio.sleep(self.keepalive / 4)
            if self.clock() - self.last_draft >= self.keepalive and (self.buffer.strip() or self.placeholder):
                await self._draft(self.buffer if self.buffer.strip() else "", force=True, origin="keepalive")

    async def feed(self, chunk: str) -> None:
        self.chars += len(chunk)
        if self.fallback is not None:
            await self.fallback.feed(chunk)
            return
        self.buffer += chunk
        await self._overflow()
        if self.buffer.strip():
            await self._draft(self.buffer, force=False, origin="feed")

    async def finish(self, note: str = "") -> None:
        await self.close()
        if self.fallback is not None:
            await self.fallback.finish(note)
            return
        self.buffer += note
        await self._overflow()
        if self.buffer.strip():
            await self._send(self.buffer)       # черновик эфемерный: ответ сохраняет sendMessage
        self.buffer = self.shown = ""

    async def replace(self, text: str) -> None:
        """Ответ не прошёл модерацию: черновик исчезнет сам, отправленные части — удалить."""
        await self.close()
        if self.fallback is not None:
            await delete_messages(self.messages)
            self.messages = []
            await self.fallback.replace(text)
            return
        await delete_messages(self.messages)
        self.messages, self.buffer, self.shown = [], "", ""
        await self._send(text)

    async def _overflow(self) -> None:
        while len(self.buffer) > self.limit:
            cut = split_point(self.buffer, self.limit)
            head, self.buffer = self.buffer[:cut], self.buffer[cut:]
            await self._send(head)
            self.draft_id, self.shown = new_draft_id(), ""   # остаток — новым черновиком

    async def _draft(self, text: str, *, force: bool, origin: str) -> None:
        """Обновить черновик. origin: start — «Thinking…», feed — основной поток ответа,
        keepalive — поддержка жизни. Черновик — только показ: любая ошибка Telegram здесь не
        роняет ответ, его всё равно сохранит финальный sendMessage."""
        async with self._lock:
            if self.fallback is not None or (text == self.shown and not force):
                return
            now = self.clock()
            if now < self.blocked_until and origin != "start":   # Telegram просил паузу — ждём её
                return
            # Первый текст — сразу, без оглядки на интервал: «Thinking…» сменяется началом ответа.
            if not force and self.shown and now - self.last_draft < self.interval:
                return
            try:
                await self.bot.send_message_draft(chat_id=self.chat_id, draft_id=self.draft_id, text=text,
                                                  message_thread_id=self.thread_id)
            except TelegramRetryAfter as exc:
                self.blocked_until = self.clock() + exc.retry_after   # следующие фрагменты копятся в буфере
                return
            except (TelegramBadRequest, TelegramNotFound) as exc:
                if origin == "feed" and text:
                    await self._switch_to_edits(exc)              # черновик с текстом не принят — правки
                else:                                             # «Thinking…» или поддержка — без них
                    log.info("draft_skipped telegram_chat=%s origin=%s error=%s", self.chat_id, origin, exc)
                    self.placeholder = self.placeholder and bool(text)
                return
            except TelegramAPIError as exc:                       # сеть, 5xx Telegram и прочее
                log.warning("draft_failed telegram_chat=%s origin=%s error=%r", self.chat_id, origin, exc)
                return
            self.shown, self.last_draft = text, self.clock()
            self.drafts += 1

    async def _switch_to_edits(self, exc: Exception) -> None:
        log.warning("draft_unavailable telegram_chat=%s error=%s — ответ покажут правки сообщения",
                    self.chat_id, exc)
        self.fallback = StreamRenderer(self.source, limit=self.limit, clock=self.clock)
        if self.buffer.strip():
            await self.fallback.feed(self.buffer)
            self.fallback.chars = 0               # их уже посчитал self.chars
        self.buffer = self.shown = ""

    async def _send(self, text: str) -> None:
        """Сообщение с частью ответа. Его нельзя пропустить: на RetryAfter — пауза и повтор."""
        try:
            message = await self.source.answer(text)
        except TelegramRetryAfter as exc:
            await asyncio.sleep(exc.retry_after)
            message = await self.source.answer(text)
        self.messages.append(message)


Renderer = StreamRenderer | DraftRenderer


def make_renderer(message: Message, mode: StreamMode = "draft") -> Renderer:
    """Черновик — только в личном чате: sendMessageDraft принимает лишь их."""
    if mode == "draft" and message.chat.type == ChatType.PRIVATE:
        return DraftRenderer(message)
    return StreamRenderer(message)


async def attach_feedback(renderer: Renderer, message_id: UUID | None) -> None:
    """Кнопки 👍/👎 под последним сообщением ответа. Не вышло — ответ всё равно показан."""
    if message_id is None or not renderer.all_messages:
        return
    last = renderer.all_messages[-1]
    try:
        await last.edit_reply_markup(reply_markup=feedback_kb(message_id))
    except TelegramAPIError as exc:
        log.warning("feedback_buttons_failed chat=%s error=%r", last.chat.id, exc)


async def answer_with_stream(message: Message, backend: StreamingBackend, chat_id: UUID, prompt: str, *,
                             media: bytes | None = None, mime: str | None = None, filename: str | None = None,
                             mode: StreamMode = "draft", renderer: Renderer | None = None) -> Renderer:
    renderer = renderer or make_renderer(message, mode)
    # Текст — send_message(chat_id, текст), как в блоке 4.2; файл — тем же методом с media.
    chunks = (backend.send_message(chat_id, prompt) if media is None
              else backend.send_message(chat_id, prompt, media=media, mime=mime, filename=filename))
    try:
        assert message.bot is not None
        async with ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
            await renderer.start()
            first = await anext(chunks, None)
        replacement = getattr(chunks, "replacement", None)
        if first is None:
            await renderer.close()
            await message.answer(texts.replacement_text(replacement or "", getattr(chunks, "categories", []))
                                 if replacement else texts.EMPTY_ANSWER)
            return renderer
        await renderer.feed(first)
        async for chunk in chunks:
            await renderer.feed(chunk)
        replacement = getattr(chunks, "replacement", None)
        if replacement:                                # блок 4.4: ответ не прошёл модерацию
            await renderer.replace(texts.replacement_text(replacement, getattr(chunks, "categories", [])))
            return renderer
        await renderer.finish()
        if not renderer.started:                       # ответ из одних пробелов и переводов строк
            await message.answer(texts.EMPTY_ANSWER)
        await attach_feedback(renderer, getattr(chunks, "message_id", None))
    except BACKEND_ERRORS as exc:
        if isinstance(exc, httpx.HTTPStatusError) and error_code(exc) == "moderation_blocked":
            # Штатный отказ модерации (блок 4.4), а не сбой сервиса: одна строка без текста вопроса.
            categories = ",".join(str(c) for c in error_body(exc).get("categories") or [])
            log.info("question_blocked chat=%s categories=%s", message.chat.id, categories)
        else:
            log.warning("backend_error chat=%s during=stream error=%r", message.chat.id, exc)
        reason = texts.user_message(exc)
        if renderer.started:
            await renderer.finish(texts.INTERRUPTED.format(reason=reason))
        else:
            await renderer.close()
            await message.answer(reason)
    finally:
        await renderer.close()
        with contextlib.suppress(Exception):
            await chunks.aclose()
    return renderer
