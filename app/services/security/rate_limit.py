"""
Лимит запросов к модели (блок 3.8, LLM10 — неограниченное потребление).

RATE_LIMIT_PER_MIN запросов POST /chat и /chat/stream в минуту на клиента; 0 — без
лимита. Клиент — заголовок X-User-ID, а без него — IP. Сверх лимита — 429
{"error": {"code": "rate_limited", ...}} и Retry-After: сколько секунд до конца окна.

Счётчики — в том же Redis, что и кеш: у нескольких копий сервиса лимит общий. Окно —
60 секунд от первого запроса клиента: SET ключ 0 EX 60 NX, затем INCR — одной
транзакцией, работает и на Redis до 7.0 (там нет EXPIRE NX). Redis недоступен — запрос
пропускается (fail-open) и в лог пишется rate_limit_unavailable: отказ всем клиентам из-за
упавшего кеша хуже, чем минута без лимита.

Заголовки ответа /chat при включённом лимите (по ним scripts/load_test.py отличает
причины, по которым 429 не пришёл):
- X-RateLimit-Limit — лимит, с которым запущен сервис. Нет заголовка — лимит выключен:
  RATE_LIMIT_PER_MIN читается при старте, правка .env без перезапуска не действует;
- X-RateLimit-Remaining — сколько запросов осталось в окне. Только когда счётчик в Redis
  сработал: Limit есть, а Remaining нет — Redis недоступен и лимит пропускает всех.

Ограничение: авторизации у сервиса нет, X-User-ID присылает сам клиент и может его
менять. Для анонимного злоупотребления надёжнее IP, а в рабочей системе ID брался бы из
токена.

Middleware стоит внутри CORSMiddleware: ответ 429 браузер увидит с CORS-заголовками, а
не как ошибку CORS. RequestContextMiddleware снаружи добавляет X-Request-ID и пишет
строку http_request со статусом 429.
"""
from __future__ import annotations

import json

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import get_settings
from app.observability.logging import get_logger
from app.observability.middleware import USER_ID_HEADER, valid_id

LIMITED_PATHS = frozenset({"/chat", "/chat/stream"})
WINDOW_SECONDS = 60
KEY_PREFIX = "ratelimit:"
LIMIT_HEADER = "X-RateLimit-Limit"
REMAINING_HEADER = "X-RateLimit-Remaining"

log = get_logger()


def client_key(scope: Scope) -> tuple[str, str]:
    """(ключ счётчика, вид клиента): X-User-ID, если он есть и безопасен, иначе IP."""
    user_id = valid_id(Headers(scope=scope).get(USER_ID_HEADER))
    if user_id:
        return KEY_PREFIX + "user:" + user_id, "user"
    host = (scope.get("client") or ("unknown", 0))[0]
    return KEY_PREFIX + "ip:" + str(host), "ip"


def with_headers(send: Send, extra: list[tuple[bytes, bytes]]) -> Send:
    """send, который добавляет заголовки к началу ответа (и к потоку /chat/stream)."""
    async def send_with_headers(message: Message) -> None:
        if message["type"] == "http.response.start":
            message = {**message, "headers": [*message.get("headers", []), *extra]}
        await send(message)
    return send_with_headers


async def hit(redis: object, key: str) -> tuple[int, int]:
    """Засчитывает запрос: (номер запроса в окне, секунд до конца окна)."""
    async with redis.pipeline(transaction=True) as pipe:   # type: ignore[attr-defined]
        pipe.set(key, 0, ex=WINDOW_SECONDS, nx=True)
        pipe.incr(key)
        pipe.ttl(key)
        _, count, ttl = await pipe.execute()
    return int(count), int(ttl) if ttl and int(ttl) > 0 else WINDOW_SECONDS


class RateLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        limit = get_settings().rate_limit_per_min
        if (scope["type"] != "http" or not limit or scope.get("method") != "POST"
                or scope.get("path") not in LIMITED_PATHS):
            await self.app(scope, receive, send)
            return
        redis = getattr(scope["app"].state, "cache", None)
        key, kind = client_key(scope)
        counted = False
        count, retry_after = 0, 0
        if redis is not None:
            try:
                count, retry_after = await hit(redis, key)
                counted = True
            except Exception as exc:  # noqa: BLE001 — любая ошибка Redis: пропускаем запрос
                log.warning("rate_limit_unavailable", error=repr(exc)[:200])
        headers = [(LIMIT_HEADER.lower().encode(), str(limit).encode())]
        if counted:
            headers.append((REMAINING_HEADER.lower().encode(), str(max(limit - count, 0)).encode()))
        if count <= limit:
            await self.app(scope, receive, with_headers(send, headers))
            return

        log.warning("rate_limited", client=kind, count=count, limit=limit, retry_after=retry_after)
        request_id = scope.get("state", {}).get("request_id")
        body = json.dumps({"error": {
            "code": "rate_limited",
            "message": f"Слишком много запросов: не больше {limit} в минуту. Повторите через {retry_after} с.",
            "request_id": request_id,
        }}, ensure_ascii=False).encode("utf-8")
        await send({"type": "http.response.start", "status": 429, "headers": [
            (b"content-type", b"application/json; charset=utf-8"),
            (b"content-length", str(len(body)).encode()),
            (b"retry-after", str(retry_after).encode()),
            *headers,
        ]})
        await send({"type": "http.response.body", "body": body})
