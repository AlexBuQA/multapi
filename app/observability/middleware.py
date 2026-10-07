"""
Контекст запроса для логов (блок 3.6): request_id, user_id, метод и путь в каждой
строке лога, строка http_request на каждый запрос, заголовок X-Request-ID в ответе.

Middleware написано на уровне ASGI, а не через @app.middleware("http") из стартер-кода.
Тот вариант (BaseHTTPMiddleware) выполняет эндпоинт в отдельной задаче, и это даёт
два эффекта:
- contextvars, привязанные внутри эндпоинта (user_id и session_id из тела запроса),
  не видны middleware — в строке http_request оставался user_id: null;
- call_next возвращает ответ, как только готовы заголовки. Для /chat/stream строка
  http_request появлялась раньше llm_request_completed, а latency_ms в ней было
  временем до первого фрагмента, а не всего потока.
Здесь эндпоинт выполняется в той же задаче, а строка пишется, когда ответ отправлен
целиком.
"""
from __future__ import annotations

import re
import time
import uuid

import structlog
from opentelemetry import trace
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.observability.logging import get_logger

REQUEST_ID_HEADER = "X-Request-ID"
USER_ID_HEADER = "X-User-ID"
QUIET_PATHS = frozenset({"/health", "/ready"})   # healthcheck каждые 15 с — в лог только на DEBUG
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")  # чужой ID не должен ломать строку лога

log = get_logger()


def valid_id(value: str | None) -> str | None:
    """ID из заголовка, если он безопасен для лога и заголовка ответа; иначе None."""
    return value if value and _VALID_ID.match(value) else None


class RequestContextMiddleware:
    """Добавляется последним, поэтому внешний: видит все запросы, включая CORS preflight."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_id = valid_id(headers.get(REQUEST_ID_HEADER)) or uuid.uuid4().hex[:12]
        # request.state.request_id — для обработчиков ошибок (поле request_id в теле ответа).
        scope.setdefault("state", {})["request_id"] = request_id

        # Span запроса от FastAPI (если трейсинг включён): request_id — в его атрибуты,
        # trace_id — в лог. По любой строке лога трейс находится в Phoenix, и наоборот.
        span = trace.get_current_span()
        trace_fields: dict[str, str] = {}
        if span.is_recording():
            span.set_attribute("request.id", request_id)
            trace_fields["trace_id"] = format(span.get_span_context().trace_id, "032x")

        # Контекст прошлого запроса не должен «протечь» в этот; дальше request_id и прочее
        # добавляются во все строки лога автоматически (merge_contextvars).
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id, method=scope["method"], path=scope["path"],
            user_id=valid_id(headers.get(USER_ID_HEADER)), **trace_fields,
        )

        status = 500   # если ответ так и не начался — необработанное исключение

        async def send_with_request_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        started = time.perf_counter()
        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            write = log.debug if scope["path"] in QUIET_PATHS and status < 400 else log.info
            write("http_request", status=status, latency_ms=latency_ms)
