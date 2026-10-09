"""BackendClient (блоки 4.2–4.3): HTTP-вызовы подменяет httpx.MockTransport — сервис не нужен."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import httpx
import pytest
from starlette.requests import Request as StarletteRequest

from bot import texts
from bot.services.backend_client import BackendClient, BackendStreamError, make_http
from bot.services.sse import SSEEvent, iter_sse, sse_lines

CHAT = UUID("9f76d71a-fe47-4a61-8c2f-757a5a300737")
# Поток в формате задания блока 4.3.
SPEC_STREAM = (b'data: {"type":"token","delta":"\xd0\x92\xd0\xb0\xd1\x81 \xd0\xb7\xd0\xbe\xd0\xb2\xd1\x83\xd1\x82"}\n\n'
               b'data: {"type":"token","delta":" \xd0\x90\xd0\xbd\xd1\x8f."}\n\n'
               b'data: {"type":"done"}\n\n')


def client(handler, **options) -> BackendClient:
    http = make_http("http://backend.test/", timeout=15, transport=httpx.MockTransport(handler))
    return BackendClient(http, retry_delays=(0, 0), **options)


def sse(*frames: str | bytes) -> httpx.Response:
    body = b"".join(f if isinstance(f, bytes) else f.encode() for f in frames)
    return httpx.Response(200, headers={"content-type": "text/event-stream; charset=utf-8"}, content=body)


def token(delta: str) -> str:
    return f"data: {json.dumps({'type': 'token', 'delta': delta}, ensure_ascii=False)}\n\n"


DONE = 'data: {"type": "done"}\n\n'


async def collect(stream) -> list[str]:
    return [chunk async for chunk in stream]


async def form_of(request: httpx.Request) -> dict:
    """Тело запроса так, как его разберёт сервис (Starlette + python-multipart)."""
    body = request.content

    async def receive() -> dict:
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {"type": "http", "method": "POST", "path": "/", "headers": [
        (k.lower().encode(), v.encode()) for k, v in request.headers.items()]}
    form = await StarletteRequest(scope, receive).form()
    result = {}
    try:
        for key, value in form.multi_items():
            result[key] = (value.filename, value.content_type, await value.read()) if hasattr(value, "filename") else value
    finally:
        await form.close()                     # временные файлы формы
    return result


async def test_get_or_create_chat_returns_uuid():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"chat_id": str(CHAT), "created": False})

    backend = client(handler)
    chat_id = await backend.get_or_create_chat("123456789", "telegram")
    assert chat_id == CHAT and isinstance(chat_id, UUID)
    assert (seen[0].method, seen[0].url.path) == ("POST", "/chats")
    assert json.loads(seen[0].content) == {"owner_external_id": "123456789", "interface": "telegram"}


async def test_send_message_parses_sse_through_mock_transport():
    """Критерий блока 4.3: поток data: {"type":"token"} … {"type":"done"} через MockTransport."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return sse(SPEC_STREAM)

    chunks = await collect(client(handler).send_message(CHAT, "Как меня зовут?"))
    assert chunks == ["Вас зовут", " Аня."]
    request = seen[0]
    assert (request.method, request.url.path) == ("POST", f"/chats/{CHAT}/messages")
    assert await form_of(request) == {"content": "Как меня зовут?"}           # форма, не JSON
    assert request.headers["accept"] == "text/event-stream"
    assert request.headers["x-user-id"] == f"chat-{CHAT}"                     # свой счётчик лимита у чата


async def test_send_message_with_media_is_multipart():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return sse(token("Это голосовое"), DONE)

    voice = b"OggS" + bytes(range(256)) * 4
    chunks = await collect(client(handler).send_message(CHAT, "Ответь на голосовое сообщение.", media=voice,
                                                        mime="audio/ogg"))
    assert chunks == ["Это голосовое"]
    assert seen[0].headers["content-type"].startswith("multipart/form-data; boundary=")
    assert await form_of(seen[0]) == {"content": "Ответь на голосовое сообщение.",
                                      "media": ("voice.ogg", "audio/ogg", voice)}


async def test_default_user_name_goes_with_every_question():
    """BOT_DEFAULT_USER_NAME уходит полем user_name — и с текстом, и с файлом; не задано — поля нет."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return sse(DONE)

    named = client(handler, user_name="Александра")
    await collect(named.send_message(CHAT, "Как меня зовут?"))
    await collect(named.send_message(CHAT, "Что на фото?", media=b"\xff\xd8\xff" + bytes(64), mime="image/jpeg"))
    await collect(client(handler).send_message(CHAT, "Как меня зовут?"))
    forms = [await form_of(request) for request in seen]
    assert forms[0] == {"content": "Как меня зовут?", "user_name": "Александра"}
    assert forms[1]["user_name"] == "Александра" and forms[1]["media"][0] == "photo.jpg"
    assert forms[2] == {"content": "Как меня зовут?"}


async def test_document_name_in_russian_survives_multipart():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return sse(DONE)

    pdf = b"%PDF-1.4 ..."
    await collect(client(handler).send_message(CHAT, "Что в договоре?", media=pdf, mime="application/pdf",
                                               filename="Договор №1.pdf"))
    assert (await form_of(seen[0]))["media"] == ("Договор №1.pdf", "application/pdf", pdf)


async def test_send_message_error_event_raises():
    error = json.dumps({"type": "error", "code": "llm_timeout", "message": "Модель не ответила вовремя."},
                       ensure_ascii=False)
    stream = client(lambda request: sse(token("Начало"), f"data: {error}\n\n")).send_message(CHAT, "вопрос")
    assert await anext(stream) == "Начало"
    with pytest.raises(BackendStreamError) as caught:
        await anext(stream)
    assert caught.value.code == "llm_timeout"
    assert texts.user_message(caught.value) == texts.TIMEOUT


async def test_stream_without_done_is_an_error():
    with pytest.raises(BackendStreamError) as caught:
        await collect(client(lambda request: sse(token("обрыв"))).send_message(CHAT, "вопрос"))
    assert caught.value.code == "stream_incomplete"


async def test_bad_and_unknown_events():
    with pytest.raises(BackendStreamError) as caught:
        await collect(client(lambda request: sse("data: [DONE]\n\n")).send_message(CHAT, "вопрос"))
    assert caught.value.code == "bad_event"                                  # старый формат 4.2 — ошибка
    backend = client(lambda request: sse('data: {"type": "ping"}\n\n', token("ok"), DONE))
    assert await collect(backend.send_message(CHAT, "вопрос")) == ["ok"]      # новый тип события — пропуск


async def test_http_error_before_stream_keeps_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"code": "rate_limited", "message": "Слишком много запросов."}})

    with pytest.raises(httpx.HTTPStatusError) as caught:
        await collect(client(handler).send_message(CHAT, "вопрос"))
    assert caught.value.response.json()["error"]["code"] == "rate_limited"     # тело прочитано
    assert texts.user_message(caught.value) == texts.RATE_LIMITED


async def test_media_error_text_comes_from_service():
    message = "Картинка больше 5 МБ — пришлите файл поменьше."

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(413, json={"error": {"code": "media_too_large", "message": message}})

    with pytest.raises(httpx.HTTPStatusError) as caught:
        await collect(client(handler).send_message(CHAT, "фото", media=b"\xff\xd8\xff", mime="image/jpeg"))
    assert texts.user_message(caught.value) == message


async def test_clear_messages_sends_delete_to_chat_url():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"status": "ok"})

    assert await client(handler).clear_messages(CHAT) is None
    assert [(r.method, str(r.url)) for r in seen] == [("DELETE", f"http://backend.test/chats/{CHAT}/messages")]


async def test_unknown_chat_is_http_status_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": "chat_not_found", "message": "Чат не найден."}})

    with pytest.raises(httpx.HTTPStatusError):
        await client(handler).clear_messages(uuid4())


# ---------------------------------------------------------------- устойчивость
class Attempts:
    """Обработчик MockTransport: первые fail запросов — ConnectError, дальше — ответ."""

    def __init__(self, fail: int, response=lambda: httpx.Response(200, json={"chat_id": str(CHAT)}),
                 error: type[Exception] = httpx.ConnectError) -> None:
        self.fail, self.response, self.error, self.count = fail, response, error, 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.count += 1
        if self.count <= self.fail:
            raise self.error("connection refused", request=request)
        return self.response()


async def test_connect_error_is_retried():
    handler = Attempts(fail=2)
    assert await client(handler).get_or_create_chat("1", "telegram") == CHAT
    assert handler.count == 3


async def test_connect_error_after_retries_propagates():
    handler = Attempts(fail=10)
    with pytest.raises(httpx.ConnectError) as caught:
        await client(handler).get_or_create_chat("1", "telegram")
    assert handler.count == 3                                                 # 1 + 2 повтора
    assert texts.user_message(caught.value) == texts.UNAVAILABLE


async def test_connect_timeout_is_retried_but_read_timeout_is_not():
    connect = Attempts(fail=1, error=httpx.ConnectTimeout)
    assert await client(connect).get_or_create_chat("1", "telegram") == CHAT and connect.count == 2
    read = Attempts(fail=10, error=httpx.ReadTimeout)
    with pytest.raises(httpx.ReadTimeout) as caught:
        await client(read).get_or_create_chat("1", "telegram")
    assert read.count == 1                                                    # запрос мог дойти до сервиса
    assert texts.user_message(caught.value) == texts.TIMEOUT


@pytest.mark.parametrize("status", [500, 502, 503, 429, 400])
async def test_http_errors_are_not_retried(status):
    handler = Attempts(fail=0, response=lambda: httpx.Response(status, json={"error": {"code": "x", "message": "y"}}))
    with pytest.raises(httpx.HTTPStatusError):
        await collect(client(handler).send_message(CHAT, "вопрос"))
    assert handler.count == 1                                                 # 5xx мог стоить вызова модели


async def test_stream_connect_error_is_retried_with_same_file():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            raise httpx.ConnectError("refused", request=request)
        return sse(token("ответ"), DONE)

    pdf = b"%PDF-1.7 " + b"x" * 5000
    chunks = await collect(client(handler).send_message(CHAT, "Что в файле?", media=pdf, mime="application/pdf"))
    assert chunks == ["ответ"] and len(seen) == 2
    assert (await form_of(seen[1]))["media"] == ("document.pdf", "application/pdf", pdf)


class BrokenStream(httpx.AsyncByteStream):
    """Отдаёт первый фрагмент и рвётся — как сервис, упавший посреди ответа."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield token("Начало ответа").encode()
        raise httpx.ReadError("connection reset")


async def test_partial_stream_is_not_retried():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=BrokenStream())

    stream = client(handler).send_message(CHAT, "вопрос")
    assert await anext(stream) == "Начало ответа"
    with pytest.raises(httpx.ReadError):
        await anext(stream)
    assert len(calls) == 1                                                    # ответ уже шёл — повтора нет


async def test_timeouts_singleton_client_and_trust_env():
    http = make_http("http://backend.test", timeout=60)
    try:
        timeout = http.timeout
        assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (3.0, 60.0, 10.0, 5.0)
        assert http._trust_env is False                                     # HTTP(S)_PROXY к сервису не применяются
        backend = BackendClient(http)
        assert backend.http is http and backend.base_url == "http://backend.test"
    finally:
        await http.aclose()


async def test_stream_gets_its_own_read_timeout():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return sse(DONE) if request.url.path.endswith("/messages") else httpx.Response(200, json={"status": "ok"})

    backend = client(handler, stream_timeout=120)
    await collect(backend.send_message(CHAT, "вопрос"))
    await backend.health()
    assert seen[0] == {"connect": 3.0, "read": 120.0, "write": 10.0, "pool": 5.0}
    assert seen[1]["read"] == 15                                              # обычный запрос — BACKEND_TIMEOUT


# ---------------------------------------------------------------- тексты ошибок
def status_error(status: int, code: str, message: str = "...") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://backend.test/chats")
    response = httpx.Response(status, json={"error": {"code": code, "message": message}}, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


@pytest.mark.parametrize("status, code, expected", [
    (422, "validation_error", texts.REJECTED),
    (503, "chat_storage_unavailable", texts.STORAGE_DOWN),
    (502, "llm_unavailable", texts.MODEL_DOWN),
    (504, "llm_timeout", texts.TIMEOUT),
    (429, "rate_limited", texts.RATE_LIMITED),
    (500, "internal_error", texts.SERVICE_ERROR),
    (503, "something_new", texts.SERVICE_ERROR),
    (415, "unsupported_content_type", texts.CLIENT_ERROR.format(status=415)),
])
def test_status_errors_become_user_messages(status, code, expected):
    assert texts.user_message(status_error(status, code)) == expected


def test_spec_error_texts():
    """Тексты из задания блока 4.3 — понятные сообщения вместо трейсбека."""
    request = httpx.Request("POST", "http://backend.test/chats")
    assert texts.user_message(httpx.ConnectError("refused", request=request)).startswith("Сервис недоступен")
    assert texts.user_message(httpx.ReadTimeout("slow", request=request)).startswith("Ответ занимает слишком долго")
    assert texts.user_message(status_error(429, "rate_limited")).startswith("Слишком много запросов, подождите минуту")
    assert texts.user_message(status_error(500, "internal_error")).startswith("Внутренняя ошибка сервиса")


@pytest.mark.parametrize("code", ["media_too_large", "media_unsupported", "audio_not_configured",
                                  "vision_not_configured"])
def test_media_errors_show_service_text(code):
    assert texts.user_message(status_error(422, code, "Текст для пользователя.")) == "Текст для пользователя."


# ---------------------------------------------------------------- разбор SSE
async def test_iter_sse_edge_cases():
    async def lines(*items: str):
        for item in items:
            yield item

    events = [e async for e in iter_sse(lines("event: error", "data:{\"a\":1}", "", "data: без пустой строки"))]
    assert events == [SSEEvent("error", '{"a":1}'), SSEEvent("message", "без пустой строки")]
    windows = [e async for e in iter_sse(lines("data:  два пробела\r", "\r", "id: 7", "retry: 10", "", ""))]
    assert windows == [SSEEvent("message", " два пробела")]


async def test_unicode_line_separators_stay_inside_the_answer():
    """JSON с ensure_ascii=False оставляет   и \x85 как есть; httpx aiter_lines() резал бы по ним."""
    answer = "Шаг 1 Шаг 2\x85Шаг 3\x0bконец"
    assert " " in token(answer)
    assert await collect(client(lambda request: sse(token(answer), DONE)).send_message(CHAT, "вопрос")) == [answer]


async def test_multiline_delta_is_one_event():
    answer = "Шаги:\n\n1. Откройте\r\n2. Нажмите"
    assert await collect(client(lambda request: sse(token(answer), DONE)).send_message(CHAT, "вопрос")) == [answer]


async def test_sse_lines_handles_crlf_split_between_chunks():
    async def chunks(*items: str):
        for item in items:
            yield item

    lines = [line async for line in sse_lines(chunks("data: раз\r", "\ndata: два\r\n\r", "\n", "data: три\r"))]
    assert lines == ["data: раз", "data: два", "", "data: три"]
    assert [line async for line in sse_lines(chunks("a\rb\n", "c"))] == ["a", "b", "c"]
