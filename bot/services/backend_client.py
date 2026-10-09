"""
BackendClient (блоки 4.2–4.3) — тонкий async-клиент chat-сервиса на httpx.

    get_or_create_chat(owner, interface) -> UUID   POST /chats (идемпотентен на стороне сервиса)
    send_message(chat_id, content, media=None, mime=None)
                                         -> фрагменты ответа по мере генерации
                                            POST /chats/{id}/messages, форма + поток SSE
    clear_messages(chat_id)              -> None   DELETE /chats/{id}/messages
    health()                             -> dict   GET /health — для /status

Все вопросы бота — и текст, и фото, голос, документ — идут через один send_message: без
media это обычный вопрос, с media — форма multipart/form-data с полями content и media.
Сам файл бот не разбирает: картинку, расшифровку голоса и текст PDF/DOCX готовит сервис
(app/chat/media.py). Поэтому openai, pypdf и python-docx боту не нужны.

Поток ответа — события SSE, в каждом одна строка data: с JSON (блок 4.3):
    {"type": "token", "delta": "..."}   фрагмент ответа
    {"type": "done"}                    конец
    {"type": "error", "code", "message"} ошибка посреди ответа -> BackendStreamError
Строки режет sse_lines (bot/services/sse.py), а не Response.aiter_lines(): та делит текст
ещё и по \\u2028 и \\x85, а JSON с ensure_ascii=False оставляет их в строке как есть.

HTTP-клиент — один на всё приложение (singleton): его создаёт make_http() в bot/__main__.py,
закрывает await http.aclose() в finally. BackendClient только пользуется им.

Таймауты: подключение 3 с, отправка запроса 10 с, ожидание соединения из пула 5 с, ответ
на обычный запрос — BACKEND_TIMEOUT (60 с). У потока ответа модели свой предел чтения —
BACKEND_STREAM_TIMEOUT (120 с): это пауза до первого фрагмента и между фрагментами, а не
длина всего ответа. Llama на CPU думает над первым словом до минуты, над фото — дольше.

Повтор — только если до сервиса не удалось подключиться (httpx.ConnectError,
ConnectTimeout): запрос тогда точно не дошёл, и повторить его безопасно, даже POST с
файлом. Ответы 4xx и 5xx не повторяются: 5xx мог случиться уже после вызова модели, а он
стоит денег у платного провайдера. Поток, который уже начал отдавать ответ, не повторяется
тем более — пользователь получил бы ответ дважды.

Ошибки не превращаются здесь в текст — пробрасываются выше, их переводит в сообщение
пользователю bot/texts.py:
- httpx.ConnectError, httpx.TimeoutException (ReadTimeout и др.) — сеть и таймаут;
- httpx.HTTPStatusError — сервис ответил кодом 4xx/5xx; тело ответа уже прочитано, код и
  текст ошибки сервиса — в response.json()["error"];
- BackendStreamError — ошибка посреди ответа или поток оборвался без события done.

X-User-ID: chat-<chat_id> — у каждого чата свой счётчик лимита запросов сервиса.
trust_env=False: HTTP(S)_PROXY из окружения к сервису не применяются — он обычно на этом
же компьютере.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import UUID

import httpx

from bot.services.sse import iter_sse, sse_lines

log = logging.getLogger(__name__)

USER_AGENT = "multapi-telegram-bot/4.3"
CONNECT_TIMEOUT = 3.0
WRITE_TIMEOUT = 10.0
POOL_TIMEOUT = 5.0
RETRY_DELAYS = (0.5, 1.0)                 # пауза перед 2-й и 3-й попыткой подключиться
CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
# Имя файла в форме, если его не передали: сервис сверяет его расширение с MIME.
DEFAULT_FILENAMES = {
    "image/jpeg": "photo.jpg", "image/png": "image.png", "image/webp": "image.webp", "image/gif": "image.gif",
    "audio/ogg": "voice.ogg", "audio/mpeg": "audio.mp3", "audio/mp4": "audio.m4a", "audio/x-m4a": "audio.m4a",
    "audio/wav": "audio.wav", "application/pdf": "document.pdf", DOCX_MIME: "document.docx",
}


class BackendStreamError(Exception):
    """Ответ оборвался: событие error в потоке SSE или поток без события done."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# Всё, что handlers ловят и показывают пользователю понятным текстом.
BACKEND_ERRORS: tuple[type[Exception], ...] = (httpx.HTTPError, BackendStreamError)


def error_body(exc: httpx.HTTPStatusError) -> dict[str, Any]:
    """Ошибка сервиса из тела {"error": {"code", "message"}} — или пустой словарь."""
    try:
        error = exc.response.json()["error"]
    except (ValueError, KeyError, TypeError, httpx.ResponseNotRead):
        return {}
    return error if isinstance(error, dict) else {}


def error_code(exc: httpx.HTTPStatusError) -> str | None:
    code = error_body(exc).get("code")
    return str(code) if code is not None else None


def user_header(chat_id: UUID) -> dict[str, str]:
    """X-User-ID — клиент для лимита запросов сервиса (RATE_LIMIT_PER_MIN, блок 3.8). Без
    него все пользователи бота считались бы одним клиентом — по IP бота — и делили лимит."""
    return {"X-User-ID": f"chat-{chat_id}"}


def make_http(base_url: str, *, timeout: float = 60.0,
              transport: httpx.AsyncBaseTransport | None = None) -> httpx.AsyncClient:
    """HTTP-клиент сервиса — один на всё приложение (bot/__main__.py)."""
    return httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        timeout=httpx.Timeout(connect=CONNECT_TIMEOUT, read=timeout, write=WRITE_TIMEOUT, pool=POOL_TIMEOUT),
        transport=transport,
        trust_env=False,
        headers={"User-Agent": USER_AGENT},
    )


def parse_event(data: str) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except ValueError as exc:
        raise BackendStreamError("bad_event", f"событие потока не JSON: {data[:200]}") from exc
    if not isinstance(payload, dict):
        raise BackendStreamError("bad_event", f"событие потока не объект: {data[:200]}")
    return payload


class BackendClient:
    def __init__(self, http: httpx.AsyncClient, *, stream_timeout: float = 120.0,
                 retry_delays: tuple[float, ...] = RETRY_DELAYS, user_name: str | None = None) -> None:
        self.http = http
        self.base_url = str(http.base_url).rstrip("/")
        self.stream_timeout = stream_timeout
        self.retry_delays = retry_delays
        self.user_name = user_name          # BOT_DEFAULT_USER_NAME: уходит с каждым вопросом

    async def _send(self, build: Callable[[], httpx.Request], *, stream: bool = False) -> httpx.Response:
        """Запрос с повтором при ошибке подключения. Каждая попытка — новый запрос: тело
        multipart собирается заново."""
        for attempt, delay in enumerate((*self.retry_delays, None), start=1):
            request = build()
            try:
                return await self.http.send(request, stream=stream)
            except CONNECT_ERRORS as exc:
                if delay is None:
                    raise
                log.info("backend_retry attempt=%d method=%s path=%s error=%r", attempt, request.method,
                         request.url.path, exc)
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        response = await self._send(lambda: self.http.build_request(method, url, **kwargs))
        response.raise_for_status()
        return response

    async def get_or_create_chat(self, owner_external_id: str, interface: str) -> UUID:
        response = await self._request("POST", "/chats", json={"owner_external_id": owner_external_id,
                                                               "interface": interface})
        return UUID(response.json()["chat_id"])

    async def send_message(self, chat_id: UUID, content: str, media: bytes | None = None, mime: str | None = None,
                           *, filename: str | None = None) -> AsyncIterator[str]:
        """Вопрос (и необязательный файл) -> фрагменты ответа. filename — имя файла для сервиса;
        не задано — по MIME (voice.ogg, photo.jpg, …). С вопросом уходит и имя по умолчанию
        (поле user_name), если оно задано: так модель обращается к пользователю, пока он не
        представился."""
        data = {"content": content}
        if self.user_name:
            data["user_name"] = self.user_name
        files = None
        if media is not None:
            mime = mime or "application/octet-stream"
            files = {"media": (filename or DEFAULT_FILENAMES.get(mime, "file.bin"), media, mime)}
        timeout = httpx.Timeout(connect=CONNECT_TIMEOUT, read=self.stream_timeout, write=WRITE_TIMEOUT,
                                pool=POOL_TIMEOUT)
        headers = {"Accept": "text/event-stream", **user_header(chat_id)}
        response = await self._send(lambda: self.http.build_request(
            "POST", f"/chats/{chat_id}/messages", data=data, files=files, headers=headers, timeout=timeout),
            stream=True)
        done = False
        try:
            if response.is_error:
                await response.aread()          # тело ошибки нужно handlers: код и сообщение
                response.raise_for_status()
            async for event in iter_sse(sse_lines(response.aiter_text())):
                payload = parse_event(event.data)
                kind = payload.get("type")
                if kind == "token":
                    yield str(payload.get("delta", ""))
                elif kind == "done":
                    done = True
                    break
                elif kind == "error":
                    raise BackendStreamError(str(payload.get("code", "stream_error")), str(payload.get("message", "")))
                # другие типы событий пропускаются: сервис может добавить новые, старый бот не сломается
        finally:
            await response.aclose()
        if not done:
            raise BackendStreamError("stream_incomplete", "поток ответа оборвался без события done")

    async def clear_messages(self, chat_id: UUID) -> None:
        await self._request("DELETE", f"/chats/{chat_id}/messages", headers=user_header(chat_id))

    async def health(self) -> dict[str, Any]:
        response = await self._request("GET", "/health")
        return dict(response.json())
