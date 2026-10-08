"""
BackendClient (блок 4.2) — тонкий async-клиент chat-сервиса блока 4.1 на httpx.

    get_or_create_chat(owner, interface) -> UUID   POST /chats (идемпотентен на стороне сервиса)
    send_message(chat_id, content)       -> фрагменты ответа по мере генерации
                                            POST /chats/{id}/messages, поток SSE
    clear_messages(chat_id)              -> None   DELETE /chats/{id}/messages
    health()                             -> dict   GET /health — для /status

Ошибки не превращаются здесь в текст — пробрасываются выше, их переводит в сообщение
пользователю bot/texts.py:
- httpx.ConnectError, httpx.TimeoutException (ReadTimeout и др.) — сеть и таймаут;
- httpx.HTTPStatusError — сервис ответил кодом 4xx/5xx; тело ответа уже прочитано, код
  ошибки сервиса — в response.json()["error"]["code"];
- BackendStreamError — ошибка посреди ответа (событие error) или поток оборвался без
  data: [DONE].

Таймаут (BACKEND_TIMEOUT, по умолчанию 30 с) — на подключение, на ожидание первого
фрагмента (в него входит и очередь вопросов чата в сервисе) и на паузу между фрагментами, а
не на весь ответ: длинный ответ, который идёт потоком, не обрывается. Чтобы второй вопрос
не ждал в очереди сервиса, бот сам отправляет вопросы одного чата по очереди (ChatQueue).

X-User-ID: chat-<chat_id> — у каждого чата свой счётчик лимита запросов сервиса. trust_env=False: HTTP(S)_PROXY из окружения к сервису не
применяются — он обычно на этом же компьютере.

В блоке 4.3 send_message получит необязательные media: bytes | None и mime: str | None —
отдельного метода под медиа не будет.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any
from uuid import UUID

import httpx

from bot.services.sse import iter_sse, sse_lines

DONE = "[DONE]"
USER_AGENT = "multapi-telegram-bot/4.2"


class BackendStreamError(Exception):
    """Ответ оборвался: событие error в потоке SSE или поток без [DONE]."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message

    @classmethod
    def from_event(cls, data: str) -> BackendStreamError:
        try:
            error = json.loads(data)["error"]
            return cls(str(error["code"]), str(error.get("message", "")))
        except (ValueError, KeyError, TypeError):
            return cls("stream_error", data[:200])


# Всё, что handlers ловят и показывают пользователю понятным текстом.
BACKEND_ERRORS: tuple[type[Exception], ...] = (httpx.HTTPError, BackendStreamError)


def error_code(exc: httpx.HTTPStatusError) -> str | None:
    """Код ошибки сервиса из тела {"error": {"code": ...}} — или None."""
    try:
        return str(exc.response.json()["error"]["code"])
    except (ValueError, KeyError, TypeError, httpx.ResponseNotRead):
        return None


def user_header(chat_id: UUID) -> dict[str, str]:
    """X-User-ID — клиент для лимита запросов сервиса (RATE_LIMIT_PER_MIN, блок 3.8). Без
    него все пользователи бота считались бы одним клиентом — по IP бота — и делили лимит."""
    return {"X-User-ID": f"chat-{chat_id}"}


class BackendClient:
    def __init__(self, base_url: str, *, timeout: float = 30.0,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
            transport=transport,
            trust_env=False,
            headers={"User-Agent": USER_AGENT},
        )

    async def __aenter__(self) -> BackendClient:
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                        tb: TracebackType | None) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def get_or_create_chat(self, owner_external_id: str, interface: str) -> UUID:
        response = await self._http.post("/chats", json={"owner_external_id": owner_external_id,
                                                         "interface": interface})
        response.raise_for_status()
        return UUID(response.json()["chat_id"])

    async def send_message(self, chat_id: UUID, content: str) -> AsyncIterator[str]:
        done = False
        async with self._http.stream("POST", f"/chats/{chat_id}/messages", json={"content": content},
                                     headers={"Accept": "text/event-stream", **user_header(chat_id)}) as response:
            if response.is_error:
                await response.aread()          # тело ошибки нужно handlers: код и сообщение
                response.raise_for_status()
            async for event in iter_sse(sse_lines(response.aiter_text())):
                if event.event == "error":
                    raise BackendStreamError.from_event(event.data)
                if event.data == DONE:
                    done = True
                    break
                yield event.data
        if not done:
            raise BackendStreamError("stream_incomplete", "поток ответа оборвался без data: [DONE]")

    async def clear_messages(self, chat_id: UUID) -> None:
        response = await self._http.delete(f"/chats/{chat_id}/messages", headers=user_header(chat_id))
        response.raise_for_status()

    async def health(self) -> dict[str, Any]:
        response = await self._http.get("/health")
        response.raise_for_status()
        return dict(response.json())
