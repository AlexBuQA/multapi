"""
Отправка рассылок из очереди сервиса (блок 4.4).

/broadcast (или POST /chats/admin/broadcast) кладёт рассылку в таблицу broadcast_queue со
статусом pending — бот в этот момент ничего не отправляет. Отправляет фоновая задача бота
BroadcastWorker:

1. раз в BOT_BROADCAST_POLL секунд — POST /chats/admin/broadcast/claim?interface=telegram:
   сервис отдаёт одну рассылку для Telegram (pending -> sending; две копии бота одновременно
   одну и ту же не получат — FOR UPDATE SKIP LOCKED) или 204, если очередь пуста;
2. текст уходит каждому получателю — Telegram chat.id из owner_external_id чатов telegram.
   Между сообщениями — пауза SEND_PAUSE: Telegram принимает от бота около 30 сообщений в
   секунду, а на превышение отвечает RetryAfter — тогда бот ждёт, сколько сказано, и
   повторяет. Заблокировал бота, удалил чат — получатель считается недоставленным;
3. итог — POST /chats/admin/broadcast/{id}/result {sent, failed}: статус sent или failed;
4. администраторам из BOT_ADMIN_IDS — «Рассылка №… отправлена: доставлено …».

Если бот упал посреди рассылки, она остаётся в статусе sending, и через 15 минут сервис
выдаст её снова (повторно получат и те, кому она уже ушла: доставка «хотя бы один раз»).
Сервис не принимает итог за рассылку, которая не в статусе sending (409), — повтор не нужен.

Задача запускается в bot/__main__.py, только если задан ADMIN_TOKEN, и отменяется при
остановке бота. Ошибки сервиса не останавливают её: следующая попытка — через интервал, в
лог — одна строка на смену состояния, а не каждые 5 секунд.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Protocol

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter

from bot import texts
from bot.services.backend_client import BACKEND_ERRORS

log = logging.getLogger(__name__)

SEND_PAUSE = 0.05        # с между сообщениями рассылки: не больше 20 в секунду
RETRY_AFTER_MAX = 60.0   # с: дольше RetryAfter не ждём — получатель недоставлен
FINISH_ATTEMPTS = 3      # итог не дошёл до сервиса — ещё попытки, иначе рассылку выдадут снова


class BroadcastBackend(Protocol):
    async def claim_broadcast(self) -> dict[str, Any] | None: ...
    async def finish_broadcast(self, broadcast_id: int, sent: int, failed: int) -> dict[str, Any]: ...


class QuietEmptyPolls(logging.Filter):
    """httpx пишет строку INFO на каждый запрос, а очередь опрашивается раз в 5 секунд: пустой
    ответ (204) в лог бота не идёт. Забранная рассылка видна по broadcast_started."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not ("/chats/admin/broadcast/claim" in message and " 204 " in message)


def chat_id_of(recipient: Any) -> int | None:
    """owner_external_id чата telegram — chat.id строкой; что-то другое отправить нельзя."""
    try:
        return int(str(recipient).strip())
    except ValueError:
        return None


class BroadcastWorker:
    def __init__(self, bot: Bot, backend: BroadcastBackend, *, poll: float = 5.0,
                 admin_ids: list[int] | None = None, pause: float = SEND_PAUSE) -> None:
        self.bot, self.backend = bot, backend
        self.poll, self.pause = poll, pause
        self.admin_ids = list(admin_ids or [])
        self._last_error: str | None = None

    async def run(self) -> None:
        """Бесконечный цикл до отмены задачи."""
        log.info("broadcast_worker_started poll=%ss", self.poll)
        while True:
            try:
                handled = await self.run_once()
            except Exception:  # noqa: BLE001 — неожиданный ответ сервиса не должен остановить рассылки
                log.exception("broadcast_worker_error")
                handled = False
            if not handled:
                await asyncio.sleep(self.poll)

    async def run_once(self) -> bool:
        """Одна рассылка из очереди. True — была рассылка (следующую можно брать сразу)."""
        try:
            item = await self.backend.claim_broadcast()
        except BACKEND_ERRORS as exc:
            self._report_error(exc)
            return False
        if self._last_error is not None:
            log.info("broadcast_queue_available")
            self._last_error = None
        if item is None:
            return False
        broadcast_id = int(item["id"])
        recipients = list(item.get("recipients") or [])
        log.info("broadcast_started id=%s recipients=%s", broadcast_id, len(recipients))
        sent, failed = await self.deliver(str(item.get("message", "")), recipients)
        await self._finish(broadcast_id, sent, failed)
        log.info("broadcast_finished id=%s sent=%s failed=%s", broadcast_id, sent, failed)
        await self._notify_admins(texts.BROADCAST_DONE.format(id=broadcast_id, sent=sent, failed=failed))
        return True

    async def deliver(self, message: str, recipients: list[Any]) -> tuple[int, int]:
        sent = failed = 0
        for number, recipient in enumerate(recipients):
            if number:
                await asyncio.sleep(self.pause)
            chat_id = chat_id_of(recipient)
            if chat_id is not None and await self._send(chat_id, message):
                sent += 1
            else:
                failed += 1
        return sent, failed

    async def _send(self, chat_id: int, message: str) -> bool:
        for attempt in (1, 2):
            try:
                await self.bot.send_message(chat_id, message, parse_mode=None)   # текст админа — как есть
                return True
            except TelegramRetryAfter as exc:
                if attempt == 2 or exc.retry_after > RETRY_AFTER_MAX:
                    log.warning("broadcast_send_failed chat=%s error=RetryAfter(%s)", chat_id, exc.retry_after)
                    return False
                await asyncio.sleep(exc.retry_after)
            except TelegramAPIError as exc:        # заблокировал бота, чат удалён, неверный id
                log.info("broadcast_send_failed chat=%s error=%s", chat_id, type(exc).__name__)
                return False
        return False

    async def _finish(self, broadcast_id: int, sent: int, failed: int) -> None:
        for attempt in range(1, FINISH_ATTEMPTS + 1):
            try:
                await self.backend.finish_broadcast(broadcast_id, sent, failed)
                return
            except BACKEND_ERRORS as exc:
                log.warning("broadcast_result_failed id=%s attempt=%s error=%r", broadcast_id, attempt, exc)
                status = getattr(getattr(exc, "response", None), "status_code", 0)
                if 400 <= status < 500:         # 404/409: повтор ничего не изменит
                    return
                if attempt < FINISH_ATTEMPTS:
                    await asyncio.sleep(self.poll)

    async def _notify_admins(self, text: str) -> None:
        for admin_id in self.admin_ids:
            with contextlib.suppress(TelegramAPIError):   # админ не писал боту — Telegram не даст написать ему
                await self.bot.send_message(admin_id, text, parse_mode=None)

    def _report_error(self, exc: Exception) -> None:
        response = getattr(exc, "response", None)
        kind = f"{type(exc).__name__}:{getattr(response, 'status_code', '')}"
        if kind != self._last_error:
            log.warning("broadcast_queue_unavailable error=%r — следующая попытка через %s с", exc, self.poll)
            self._last_error = kind


def start_broadcasts(bot: Bot, backend: Any, *, poll: float, admin_ids: list[int],
                     enabled: bool) -> asyncio.Task[None] | None:
    if not enabled:
        log.info("broadcast_worker_disabled reason=ADMIN_TOKEN не задан — рассылки из очереди не отправляются")
        return None
    worker = BroadcastWorker(bot, backend, poll=poll, admin_ids=admin_ids)
    task = asyncio.create_task(worker.run(), name="broadcasts")
    task.add_done_callback(_worker_finished)
    return task


def _worker_finished(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and task.exception() is not None:
        log.error("broadcast_worker_crashed error=%r", task.exception())


async def stop_broadcasts(task: asyncio.Task[None] | None) -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task
