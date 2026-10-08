"""
charset=utf-8 в Content-Type JSON-ответов.

FastAPI отдаёт JSON с «Content-Type: application/json» без charset. По RFC 8259 JSON
всегда в UTF-8, и браузеры, curl, httpx и Swagger так его и читают. Windows PowerShell 5.1
(Invoke-RestMethod, Invoke-WebRequest) без charset декодирует тело как ISO-8859-1, и
«Я не могу» выводится как «Ð¯ Ð½Ðµ Ð¼Ð¾Ð³Ñƒ».

Middleware дописывает charset ко всем ответам application/json: к ответам эндпоинтов, к
ошибкам FastAPI и к 429 от лимита. Схема OpenAPI не меняется — меняется только заголовок.
"""
from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

JSON_MEDIA_TYPE = "application/json"


def with_charset(content_type: str) -> str:
    """«application/json» → «application/json; charset=utf-8»; остальное без изменений."""
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != JSON_MEDIA_TYPE or "charset=" in content_type.lower():
        return content_type
    return f"{content_type}; charset=utf-8"


class JSONCharsetMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_charset(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                content_type = headers.get("content-type")
                if content_type:
                    headers["content-type"] = with_charset(content_type)
            await send(message)

        await self.app(scope, receive, send_with_charset)
