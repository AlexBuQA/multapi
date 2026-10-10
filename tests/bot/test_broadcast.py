"""
Рассылки из очереди сервиса (блок 4.4, bot/services/broadcast.py): BroadcastWorker с
FakeBackend и MockedSession — без сети и без пауз.
"""
from __future__ import annotations

import asyncio
import logging

import httpx
from aiogram.exceptions import TelegramForbiddenError, TelegramRetryAfter
from aiogram.methods import SendMessage

from bot import texts
from bot.services import broadcast
from bot.services.broadcast import BroadcastWorker, start_broadcasts, stop_broadcasts
from bot_fakes import ADMIN_ID, FakeBackend, message_update


def queued(backend: FakeBackend, *recipients: str, message: str = "Плановые работы в 23:00") -> None:
    backend.broadcasts.append({"id": len(backend.results) + len(backend.broadcasts) + 1, "message": message,
                               "interface": "telegram", "recipients": list(recipients)})


def worker(bot, backend, **kwargs) -> BroadcastWorker:
    return BroadcastWorker(bot, backend, poll=0.01, admin_ids=[ADMIN_ID], pause=0, **kwargs)


async def test_sends_to_everyone_and_reports(bot, session):
    backend = FakeBackend()
    queued(backend, "1001", "1002", "-1003")                  # личные чаты и группа
    assert await worker(bot, backend).run_once() is True
    assert {int(m.chat_id): m.text for m in session.of(SendMessage)} == {
        1001: "Плановые работы в 23:00", 1002: "Плановые работы в 23:00", -1003: "Плановые работы в 23:00",
        ADMIN_ID: texts.BROADCAST_DONE.format(id=1, sent=3, failed=0)}
    assert all(m.parse_mode is None for m in session.of(SendMessage))   # текст админа — не HTML, как есть
    assert backend.results == [(1, 3, 0)]
    assert await worker(bot, backend).run_once() is False               # очередь пуста


async def test_blocked_and_bad_recipients_are_failed(bot, session):
    backend = FakeBackend()
    queued(backend, "1001", "web-user", "1002")
    session.fail[SendMessage] = [TelegramForbiddenError(method=SendMessage(chat_id=1001, text="x"),
                                                        message="bot was blocked by the user")]
    await worker(bot, backend).run_once()
    assert backend.results == [(1, 1, 2)]
    assert session.sent_texts(1002) == ["Плановые работы в 23:00"]


async def test_retry_after_waits_and_repeats(bot, session):
    backend = FakeBackend()
    queued(backend, "1001")
    session.fail[SendMessage] = [TelegramRetryAfter(method=SendMessage(chat_id=1001, text="x"),
                                                    message="Too Many Requests", retry_after=0)]
    await worker(bot, backend).run_once()
    assert backend.results == [(1, 1, 0)]
    assert session.sent_texts(1001) == ["Плановые работы в 23:00"] * 2              # отказ и повтор


async def test_too_long_retry_after_gives_up(bot, session):
    backend = FakeBackend()
    queued(backend, "1001")
    session.fail[SendMessage] = [TelegramRetryAfter(method=SendMessage(chat_id=1001, text="x"),
                                                    message="Too Many Requests", retry_after=3600)]
    await asyncio.wait_for(worker(bot, backend).run_once(), 1)          # не ждёт час
    assert backend.results == [(1, 0, 1)]


async def test_result_is_retried_when_service_blinks(bot, session):
    class Blinking(FakeBackend):
        fails = 1

        async def finish_broadcast(self, broadcast_id, sent, failed):
            if self.fails:
                self.fails -= 1
                raise httpx.ConnectError("refused", request=httpx.Request("POST", "http://backend.test/"))
            return await super().finish_broadcast(broadcast_id, sent, failed)

    backend = Blinking()
    queued(backend, "1001")
    await worker(bot, backend).run_once()
    assert backend.results == [(1, 1, 0)]


async def test_rejected_result_is_not_retried(bot, session):
    """409 (рассылку уже завершили или забрали заново) — повтор ничего не изменит."""
    class Rejecting(FakeBackend):
        calls = 0

        async def finish_broadcast(self, broadcast_id, sent, failed):
            self.calls += 1
            request = httpx.Request("POST", "http://backend.test/chats/admin/broadcast/1/result")
            response = httpx.Response(409, json={"error": {"code": "broadcast_not_sending", "message": "..."}},
                                      request=request)
            raise httpx.HTTPStatusError("409", request=request, response=response)

    backend = Rejecting()
    queued(backend, "1001")
    await asyncio.wait_for(worker(bot, backend).run_once(), 1)
    assert backend.calls == 1


async def test_queue_errors_are_logged_once(bot, session, caplog):
    error = httpx.ConnectError("refused", request=httpx.Request("POST", "http://backend.test/"))
    backend = FakeBackend(admin_error=error)
    w = worker(bot, backend)
    with caplog.at_level(logging.INFO, logger=broadcast.__name__):
        for _ in range(3):
            assert await w.run_once() is False
        backend.admin_error = None
        assert await w.run_once() is False
    messages = [r.getMessage().split()[0] for r in caplog.records]
    assert messages == ["broadcast_queue_unavailable", "broadcast_queue_available"]


async def test_worker_survives_unexpected_answer(bot, session, caplog):
    class Weird(FakeBackend):
        calls = 0

        async def claim_broadcast(self):
            self.calls += 1
            return {"message": "без id"} if self.calls == 1 else None

    backend = Weird()
    with caplog.at_level(logging.INFO, logger=broadcast.__name__):     # уровень корня меняют другие тесты
        task = asyncio.create_task(worker(bot, backend).run())
        while backend.calls < 3:
            await asyncio.sleep(0.01)
        assert not task.done()
        await stop_broadcasts(task)
    assert task.cancelled() and "broadcast_worker_error" in caplog.text


async def test_start_only_with_admin_token(bot):
    assert start_broadcasts(bot, FakeBackend(), poll=0.01, admin_ids=[], enabled=False) is None
    task = start_broadcasts(bot, FakeBackend(), poll=0.01, admin_ids=[], enabled=True)
    assert task is not None and not task.done()
    await stop_broadcasts(task)
    assert task.cancelled()
    await stop_broadcasts(None)


async def test_admin_command_to_delivery(dp, bot, session):
    """/broadcast администратора -> очередь -> фоновая задача -> сообщения пользователям."""
    backend = dp["backend"] = FakeBackend()
    for user in (2001, 2002):
        await dp.feed_update(bot, message_update("/start", user))
    await dp.feed_update(bot, message_update("/broadcast Новая функция в кабинете", ADMIN_ID))
    assert session.sent_texts(2001)[1:] == []
    await worker(bot, backend).run_once()
    assert session.sent_texts(2001)[-1] == session.sent_texts(2002)[-1] == "Новая функция в кабинете"
    assert session.sent_texts(ADMIN_ID)[-1] == texts.BROADCAST_DONE.format(id=1, sent=2, failed=0)


def test_empty_polls_do_not_flood_the_log():
    """Пустой опрос очереди (204) каждые 5 с — не в лог бота; забранная рассылка и ошибки — в лог."""
    quiet = broadcast.QuietEmptyPolls()

    def record(message: str) -> logging.LogRecord:
        return logging.LogRecord("httpx", logging.INFO, __file__, 1, message, None, None)

    base = 'HTTP Request: POST http://127.0.0.1:8000/chats/admin/broadcast/claim?interface=telegram "HTTP/1.1 '
    assert not quiet.filter(record(base + '204 No Content"'))
    assert quiet.filter(record(base + '200 OK"'))
    assert quiet.filter(record(base + '401 Unauthorized"'))
    assert quiet.filter(record('HTTP Request: POST http://127.0.0.1:8000/chats "HTTP/1.1 200 OK"'))
