"""BackendClient (блок 4.2): HTTP-вызовы подменяет httpx.MockTransport — сервис не нужен."""
from __future__ import annotations

import json
from uuid import UUID, uuid4

import httpx
import pytest

from bot import texts
from bot.services.backend_client import BackendClient, BackendStreamError
from bot.services.sse import SSEEvent, iter_sse, sse_lines

CHAT = UUID("9f76d71a-fe47-4a61-8c2f-757a5a300737")


def client(handler) -> BackendClient:
    return BackendClient("http://backend.test/", timeout=15, transport=httpx.MockTransport(handler))


def sse(*frames: str) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream; charset=utf-8"},
                          content="".join(frames).encode())


async def collect(stream) -> list[str]:
    return [chunk async for chunk in stream]


async def test_get_or_create_chat_returns_uuid():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"chat_id": str(CHAT), "created": False})

    async with client(handler) as backend:
        chat_id = await backend.get_or_create_chat("123456789", "telegram")
    assert chat_id == CHAT and isinstance(chat_id, UUID)
    assert (seen[0].method, seen[0].url.path) == ("POST", "/chats")
    assert json.loads(seen[0].content) == {"owner_external_id": "123456789", "interface": "telegram"}


async def test_send_message_parses_sse_frames():
    def handler(request: httpx.Request) -> httpx.Response:
        assert (request.method, request.url.path) == ("POST", f"/chats/{CHAT}/messages")
        assert json.loads(request.content) == {"content": "Как меня зовут?"}
        assert request.headers["accept"] == "text/event-stream"
        assert request.headers["x-user-id"] == f"chat-{CHAT}"           # свой счётчик лимита у каждого чата
        return sse("data: Вас зовут \n\n", ": keep-alive\n\n", "data: Аня.\n\n",
                   "data: Первая строка\ndata: \ndata: третья\n\n", "data: [DONE]\n\n")

    async with client(handler) as backend:
        chunks = await collect(backend.send_message(CHAT, "Как меня зовут?"))
    assert chunks == ["Вас зовут ", "Аня.", "Первая строка\n\nтретья"]


async def test_send_message_error_event_raises():
    body = json.dumps({"error": {"code": "llm_timeout", "message": "Модель не ответила вовремя."}}, ensure_ascii=False)

    async with client(lambda request: sse("data: Начало\n\n", f"event: error\ndata: {body}\n\n")) as backend:
        stream = backend.send_message(CHAT, "вопрос")
        assert await anext(stream) == "Начало"
        with pytest.raises(BackendStreamError) as caught:
            await anext(stream)
    assert caught.value.code == "llm_timeout"
    assert texts.user_message(caught.value) == texts.TIMEOUT


async def test_stream_without_done_is_an_error():
    async with client(lambda request: sse("data: обрыв\n\n")) as backend:
        with pytest.raises(BackendStreamError) as caught:
            await collect(backend.send_message(CHAT, "вопрос"))
    assert caught.value.code == "stream_incomplete"


async def test_http_error_before_stream_keeps_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"code": "rate_limited", "message": "Слишком много запросов."}})

    async with client(handler) as backend:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            await collect(backend.send_message(CHAT, "вопрос"))
    assert caught.value.response.json()["error"]["code"] == "rate_limited"     # тело прочитано
    assert texts.user_message(caught.value) == texts.RATE_LIMITED


async def test_clear_messages_sends_delete_to_chat_url():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"status": "ok"})

    async with client(handler) as backend:
        assert await backend.clear_messages(CHAT) is None
    assert [(r.method, str(r.url)) for r in seen] == [("DELETE", f"http://backend.test/chats/{CHAT}/messages")]


async def test_unknown_chat_is_http_status_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": "chat_not_found", "message": "Чат не найден."}})

    async with client(handler) as backend:
        with pytest.raises(httpx.HTTPStatusError):
            await backend.clear_messages(uuid4())


async def test_connect_error_propagates():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with client(handler) as backend:
        with pytest.raises(httpx.ConnectError) as caught:
            await backend.get_or_create_chat("1", "telegram")
    assert texts.user_message(caught.value) == texts.UNAVAILABLE


async def test_timeout_and_trust_env():
    backend = BackendClient("http://backend.test", timeout=20)
    try:
        assert backend._http.timeout.read == 20 and backend._http.timeout.connect == 10
        assert backend._http._trust_env is False                   # HTTP(S)_PROXY к сервису не применяются
    finally:
        await backend.aclose()


@pytest.mark.parametrize("status, code, expected", [
    (422, "validation_error", texts.REJECTED),
    (503, "chat_storage_unavailable", texts.STORAGE_DOWN),
    (502, "llm_unavailable", texts.MODEL_DOWN),
    (504, "llm_timeout", texts.TIMEOUT),
    (500, "internal_error", texts.SERVICE_ERROR.format(status=500)),
])
def test_status_errors_become_user_messages(status, code, expected):
    request = httpx.Request("POST", "http://backend.test/chats")
    response = httpx.Response(status, json={"error": {"code": code, "message": "..."}}, request=request)
    exc = httpx.HTTPStatusError("error", request=request, response=response)
    assert texts.user_message(exc) == expected


def test_read_timeout_is_timeout_message():
    exc = httpx.ReadTimeout("timed out", request=httpx.Request("POST", "http://backend.test/chats"))
    assert texts.user_message(exc) == texts.TIMEOUT


async def test_iter_sse_edge_cases():
    async def lines(*items: str):
        for item in items:
            yield item

    events = [e async for e in iter_sse(lines("event: error", "data:{\"a\":1}", "", "data: без пустой строки"))]
    assert events == [SSEEvent("error", '{"a":1}'), SSEEvent("message", "без пустой строки")]
    windows = [e async for e in iter_sse(lines("data:  два пробела\r", "\r", "id: 7", "retry: 10", "", ""))]
    assert windows == [SSEEvent("message", " два пробела")]


async def test_unicode_line_separators_stay_inside_the_answer():
    """httpx aiter_lines() резал бы ответ по \u2028 и \x85 — кусок после них пропадал."""
    answer = "Шаг 1\u2028Шаг 2\x85Шаг 3\x0bконец"

    async with client(lambda request: sse(f"data: {answer}\n\n", "data: [DONE]\n\n")) as backend:
        assert await collect(backend.send_message(CHAT, "вопрос")) == [answer]


async def test_sse_lines_handles_crlf_split_between_chunks():
    async def chunks(*items: str):
        for item in items:
            yield item

    lines = [line async for line in sse_lines(chunks("data: раз\r", "\ndata: два\r\n\r", "\n", "data: три\r"))]
    assert lines == ["data: раз", "data: два", "", "data: три"]
    assert [line async for line in sse_lines(chunks("a\rb\n", "c"))] == ["a", "b", "c"]
