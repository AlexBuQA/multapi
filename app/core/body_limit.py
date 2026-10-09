"""
Предел размера тела запроса с файлом (блок 4.3): POST /chats/{chat_id}/messages.

Starlette разбирает форму multipart целиком до вызова обработчика: файл любого размера
лёг бы на диск (SpooledTemporaryFile), а проверка MEDIA__MAX_*_BYTES в app/chat/media.py
сработала бы уже после этого. Поэтому предел — на уровне ASGI, до разбора:
- заголовок Content-Length больше предела — сразу 413, тело не читается;
- без Content-Length (chunked) или с неверным — байты считаются по мере чтения, и на
  превышении разбор формы прерывается с тем же 413.

Предел — самый большой из MEDIA__MAX_*_BYTES плюс 1 МБ на поле content и разметку формы.
Настройки берутся так же, как в обработчиках: get_settings с учётом
app.dependency_overrides (тесты подменяют настройки именно так).
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import Settings, get_settings

MESSAGES_PATH = re.compile(r"^/chats/[^/]+/messages/?$")
FORM_OVERHEAD = 1024 * 1024
CODE = "request_too_large"


class BodyTooLarge(HTTPException):
    """Тело больше предела. HTTPException — чтобы FastAPI не превратил его при разборе формы
    в 400 «There was an error parsing the body»; ответ в общем формате даёт app/main.py."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(status_code=413, detail=message(limit))


def message(limit: int) -> str:
    return f"Запрос больше {limit // (1024 * 1024)} МБ — пришлите файл поменьше."


def request_limit(settings: Settings) -> int:
    media = settings.media
    return max(media.max_image_bytes, media.max_audio_bytes, media.max_document_bytes) + FORM_OVERHEAD


class BodyLimitMiddleware:
    def __init__(self, app: ASGIApp, limit: Callable[[Scope], int] | None = None) -> None:
        self.app = app
        self.limit = limit or self._limit_from_settings

    @staticmethod
    def _limit_from_settings(scope: Scope) -> int:
        app = scope.get("app")
        overrides: dict[Any, Any] = getattr(app, "dependency_overrides", {}) or {}
        return request_limit(overrides.get(get_settings, get_settings)())

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or not MESSAGES_PATH.match(scope["path"]):
            await self.app(scope, receive, send)
            return
        limit = self.limit(scope)
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and length.isdigit() and int(length) > limit:
            await _reject(send, limit)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise BodyTooLarge(limit)
            return message

        await self.app(scope, limited_receive, send)


async def _reject(send: Send, limit: int) -> None:
    body = json.dumps({"error": {"code": CODE, "message": message(limit)}}, ensure_ascii=False).encode()
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", b"application/json; charset=utf-8"),
                            (b"content-length", str(len(body)).encode()), (b"connection", b"close")]})
    await send({"type": "http.response.body", "body": body})
