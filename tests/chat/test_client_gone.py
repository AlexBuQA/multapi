"""
Клиент ушёл, не дождавшись первого фрагмента ответа (блок 4.3, проверка на Windows).

Картинку gemma3:4b на CPU разбирала около 200 с, бот ждал 120 с (BACKEND_STREAM_TIMEOUT) и
уходил по ReadTimeout. До первого фрагмента StreamingResponse ещё нет, и разрыв никто не
слушал: запрос к модели шёл до конца и держал замок чата, поэтому /clear ждал минуту.
Теперь POST /chats/{id}/messages, пока ждёт первый фрагмент, проверяет соединение
(routes.first_chunk): клиент ушёл — запрос к модели отменяется, замок отпускается.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from chat_fakes import FakeLLM, make_settings
from log_capture import captured_logs
from log_capture import events as log_events

from app.chat import routes
from app.chat.deps import get_llm_client
from app.chat.routes import ClientGone, first_chunk
from app.chat.service import ChatLocks
from app.core.config import get_settings
from app.main import app
from app.schemas.chat import ChatDelta

PNG = (Path(__file__).resolve().parents[2] / "samples" / "chart.png").read_bytes()


# ---------------------------------------------------------------- first_chunk
class Connection:
    """Соединение клиента для first_chunk: gone() — клиент закрыл его."""

    def __init__(self) -> None:
        self.closed = False
        self.checks = 0

    async def is_disconnected(self) -> bool:
        self.checks += 1
        return self.closed

    def gone(self) -> None:
        self.closed = True


async def slow_answer(release: asyncio.Event, log: list[str]):
    """Генератор ответа: молчит до release, отмену и закрытие записывает в log."""
    try:
        await release.wait()
        yield "Первый"
        yield " ответ"
    except asyncio.CancelledError:
        log.append("cancelled")
        raise
    finally:
        log.append("closed")


async def test_waits_while_client_is_connected():
    connection, release, log = Connection(), asyncio.Event(), []
    chunks = slow_answer(release, log)
    asyncio.get_running_loop().call_later(0.05, release.set)
    assert await first_chunk(chunks, connection.is_disconnected, poll=0.01) == "Первый"
    assert connection.checks >= 2 and log == []          # проверял соединение и ждал дальше
    assert [chunk async for chunk in chunks] == [" ответ"]   # поток продолжается с того же места


async def test_client_gone_cancels_the_answer():
    connection, log = Connection(), []
    chunks = slow_answer(asyncio.Event(), log)            # модель молчит вечно
    asyncio.get_running_loop().call_later(0.05, connection.gone)
    with pytest.raises(ClientGone):
        await asyncio.wait_for(first_chunk(chunks, connection.is_disconnected, poll=0.01), 2)
    assert log == ["cancelled", "closed"]                 # генератор отменён, его finally выполнен


async def test_empty_answer_and_errors_before_first_chunk():
    async def empty():
        return
        yield  # noqa: unreachable — генератор без фрагментов

    async def failing():
        await asyncio.sleep(0.02)
        raise RuntimeError("провайдер недоступен")
        yield  # noqa: unreachable

    connection = Connection()
    assert await first_chunk(empty(), connection.is_disconnected, poll=0.01) is None
    with pytest.raises(RuntimeError, match="провайдер недоступен"):  # клиент на связи — ошибка как есть
        await first_chunk(failing(), connection.is_disconnected, poll=0.01)


async def test_outer_cancel_cancels_the_answer_too():
    """Отменили сам обработчик (остановка сервиса) — ответ модели тоже отменяется."""
    log: list[str] = []
    waiting = asyncio.create_task(first_chunk(slow_answer(asyncio.Event(), log), Connection().is_disconnected,
                                              poll=0.01))
    await asyncio.sleep(0.03)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert log == ["cancelled", "closed"]


# ---------------------------------------------------------------- через приложение
class SilentLLM:
    """Модель, которая думает над картинкой дольше, чем ждёт клиент."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False

    async def stream(self, req):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        yield ChatDelta(content="не дойдёт")


@pytest.fixture
def silent_app(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, chat_vision_model="gemma3:4b")
    llm = SilentLLM()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.state.chat_locks = ChatLocks()                     # как после lifespan: замки общие на запросы
    monkeypatch.setattr(routes, "DISCONNECT_POLL", 0.01)
    yield llm
    del app.state.chat_locks
    app.dependency_overrides.clear()


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def call_and_leave(chat_id: str, llm: SilentLLM, gone: asyncio.Event) -> asyncio.Task:
    """POST с фото напрямую через ASGI: тело отдаётся сразу, разрыв — когда gone."""
    request = httpx.Request("POST", f"http://test/chats/{chat_id}/messages", data={"content": "Что за ошибка?"},
                            files={"media": ("screen.png", PNG, "image/png")})
    body = request.read()
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": f"/chats/{chat_id}/messages",
             "raw_path": f"/chats/{chat_id}/messages".encode(), "query_string": b"", "root_path": "",
             "headers": [(name.lower(), value) for name, value in request.headers.raw],
             "server": ("test", 80), "client": ("127.0.0.1", 50000)}
    body_sent = False
    sent: list[dict] = []

    async def receive() -> dict:
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)

    task = asyncio.create_task(app(scope, receive, send))
    task.sent = sent  # type: ignore[attr-defined]
    await asyncio.wait_for(llm.started.wait(), 5)
    return task


async def test_bot_timeout_frees_chat_lock_and_model(silent_app):
    llm = silent_app
    gone = asyncio.Event()
    async with client() as http:
        chat_id = (await http.post("/chats", json={"owner_external_id": f"tg-{uuid4().hex[:8]}",
                                                   "interface": "telegram"})).json()["chat_id"]
        with captured_logs("INFO") as logs:
            request = await call_and_leave(chat_id, llm, gone)
            clear = asyncio.create_task(http.delete(f"/chats/{chat_id}/messages"))
            await asyncio.sleep(0.1)
            assert not clear.done()                         # /clear ждёт замок, пока модель думает
            gone.set()                                      # бот получил ReadTimeout и ушёл
            await asyncio.wait_for(request, 2)
            cleared = await asyncio.wait_for(clear, 2)      # замок отпущен сразу, а не через минуты
        history = (await http.get(f"/chats/{chat_id}/messages")).json()
    assert request.sent[0]["status"] == 499 and llm.cancelled
    assert cleared.status_code == 200 and history == []
    gone_event = log_events(logs, "chat_client_gone")[0]
    assert gone_event["stage"] == "before_first_chunk" and gone_event["level"] == "warning"
    assert log_events(logs, "chat_turn_finished")[0]["outcome"] == "interrupted"


async def test_question_stays_in_history_when_client_left(silent_app):
    llm = silent_app
    gone = asyncio.Event()
    async with client() as http:
        chat_id = (await http.post("/chats", json={"owner_external_id": f"tg-{uuid4().hex[:8]}",
                                                   "interface": "telegram"})).json()["chat_id"]
        request = await call_and_leave(chat_id, llm, gone)
        gone.set()
        await asyncio.wait_for(request, 2)
        history = (await http.get(f"/chats/{chat_id}/messages")).json()
    assert [(m["role"], m["content"]) for m in history] == [("user", "Что за ошибка?")]   # ответа нет
    assert history[0]["media"]["kind"] == "image"


async def test_slow_first_chunk_still_reaches_waiting_client(tmp_path, monkeypatch):
    """Пока клиент ждёт, проверки соединения ответ не обрывают (httpx.ASGITransport)."""
    settings = make_settings(tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llm_client] = lambda: FakeLLM(delay=0.05)
    monkeypatch.setattr(routes, "DISCONNECT_POLL", 0.005)
    try:
        async with client() as http:
            chat_id = (await http.post("/chats", json={"owner_external_id": "cli-slow", "interface": "cli"})).json()
            response = await http.post(f"/chats/{chat_id['chat_id']}/messages", data={"content": "Меня зовут Аня"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200 and response.text.endswith('data: {"type": "done"}\n\n')
