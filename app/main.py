"""
HTTP-сервис ассистента техподдержки на FastAPI (блоки 3.4–3.8, 4.1).

Запуск из корня проекта:
    uvicorn app.main:app --reload --port 8000
В Docker (блок 3.5): docker compose up -d --build
Swagger: http://localhost:8000/docs

Сборка приложения:
- lifespan создаёт AsyncOpenAI, подключение к Redis и семафор (Bulkhead) и закрывает
  их при остановке. Если Redis не отвечает, сервис всё равно стартует: кеш
  пропускается, /ready отвечает 503, а когда Redis поднимется, клиент
  переподключится сам;
- lifespan до создания клиента OpenAI включает трейсинг в Phoenix (блок 3.6), если
  задан PHOENIX_COLLECTOR_ENDPOINT;
- если задан LLM__PROXY_URL, а base_url внешний (OpenAI), клиент ходит через прокси
  (блок 3.7); к локальному Ollama прокси не применяется;
- RequestContextMiddleware (app/observability/middleware.py) присваивает запросу
  request_id (или берёт из X-Request-ID), привязывает его к contextvars structlog — он
  попадает во все JSON-строки лога этого запроса, — пишет строку http_request и
  возвращает X-Request-ID в ответе;
- CORS — только для адресов из CORS_ORIGINS;
- защитный слой (блок 3.8, app/services/security/): lifespan создаёт канарейку
  (app.state.canary), LLMService проверяет вход и ответ модели, RateLimitMiddleware
  ограничивает число запросов к /chat (RATE_LIMIT_PER_MIN) и сообщает его в заголовках
  X-RateLimit-Limit и X-RateLimit-Remaining;
- JSONCharsetMiddleware дописывает charset=utf-8 к Content-Type JSON-ответов: без него
  Windows PowerShell 5.1 показывает кириллицу как «Ð¯…»;
- чаты с историей на сервере (блок 4.1, app/chat/): роутер /chats; lifespan проверяет
  стратегию контекста и для CHAT_REPOSITORY=postgres создаёт движок SQLAlchemy и фабрику
  сессий (init_chat_storage), а при остановке закрывает их;
- обработчики переводят доменные ошибки LLM и ошибки валидации в единый JSON
  {"error": {"code", "message", ...}}; трейсбек в ответ не попадает никогда.

Без LLM__OPENAI_API_KEY в .env приложение не стартует: get_settings() поднимает
ValidationError — так и задумано.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from redis.asyncio import Redis

from app.chat import routes as chat_routes
from app.chat.context import make_strategy
from app.chat.deps import close_chat_storage, init_chat_storage
from app.chat.domain import ChatInputError, ChatNotFoundError, ChatStorageError
from app.core.charset import JSONCharsetMiddleware
from app.core.config import (
    get_settings,
    http_client_options,
    is_local_url,
    provider_headers,
    proxy_display,
    proxy_for,
)
from app.core.exceptions import LLMError, LLMRateLimitError
from app.observability.logging import get_logger, setup_logging
from app.observability.middleware import REQUEST_ID_HEADER, USER_ID_HEADER, RequestContextMiddleware
from app.observability.pii_presidio import load_redactor
from app.observability.tracing import fastapi_telemetry, setup_tracing, shutdown_tracing
from app.routers import chat, health, models
from app.services.llm import CACHE_ERRORS
from app.services.security.canary import new_canary
from app.services.security.rate_limit import LIMIT_HEADER, REMAINING_HEADER, RateLimitMiddleware

settings = get_settings()   # без ключа здесь ValidationError — uvicorn не стартует
setup_logging(settings.log_level, log_file=settings.log_file)   # JSON-логи; LOG_LEVEL и LOG_FILE из .env
log = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg = get_settings()
    # Трейсинг — до создания клиента OpenAI, чтобы инструментация точно применилась.
    app.state.tracer_provider = setup_tracing(cfg.phoenix_project_name, cfg.phoenix_collector_endpoint)
    proxy = proxy_for(cfg.llm.base_url, cfg.llm.proxy_url)
    if proxy:
        log.info("llm_proxy_enabled", proxy=proxy_display(proxy), base_url=cfg.llm.base_url or "https://api.openai.com/v1")
    http_options = http_client_options(proxy, cfg.llm.use_system_certs and not is_local_url(cfg.llm.base_url))
    app.state.openai = AsyncOpenAI(
        api_key=cfg.llm.openai_api_key.get_secret_value(),
        base_url=cfg.llm.base_url,
        timeout=cfg.llm.request_timeout,
        max_retries=cfg.llm.max_retries,
        # DefaultAsyncHttpxClient — httpx-клиент с настройками SDK по умолчанию плюс прокси
        # и, если LLM__USE_SYSTEM_CERTS=true, проверка HTTPS по хранилищу сертификатов ОС.
        http_client=DefaultAsyncHttpxClient(**http_options) if http_options else None,
        default_headers=provider_headers(cfg.llm.base_url, "multapi"),   # только для OpenRouter
    )
    app.state.llm_limiter = asyncio.Semaphore(cfg.llm.max_concurrency)
    # Канарейка (блок 3.8): новая при каждом старте; само значение в лог не пишем.
    app.state.canary = new_canary() if cfg.security.enabled else None
    # Presidio (опционально): модель грузится несколько секунд — один раз здесь, в потоке.
    app.state.pii_redactor = await asyncio.to_thread(load_redactor) if cfg.pii_presidio else None

    # Клиент создаётся всегда: подключение ленивое, после падения Redis клиент
    # переподключается сам. Недоступный Redis замедляет запрос не больше чем на
    # таймауты ниже, а ошибки кеша LLMService только пишет в лог.
    app.state.cache = Redis.from_url(cfg.redis_url, decode_responses=True,
                                     socket_connect_timeout=1.0, socket_timeout=1.0)
    try:
        await app.state.cache.ping()
        log.info("redis_connected", redis_url=cfg.redis_url)
    except CACHE_ERRORS as exc:
        log.warning("redis_unavailable", redis_url=cfg.redis_url, error=repr(exc),
                    note="запросы идут без кеша, /ready отвечает 503, пока Redis не поднимется")

    # Чаты (блок 4.1): неподдерживаемая стратегия контекста — ошибка старта, а не 500 на запросах.
    make_strategy(cfg.chat_context_strategy, cfg.chat_context_window)
    await init_chat_storage(app.state, cfg)

    log.info("service_started", default_model=cfg.llm.default_model,
             provider=cfg.llm.base_url or "https://api.openai.com/v1",
             security_enabled=cfg.security.enabled, rate_limit_per_min=cfg.rate_limit_per_min,
             chat_repository=cfg.chat_repository)
    if not cfg.security.enabled:
        log.warning("security_disabled", note="SECURITY__ENABLED=false: проверки входа и ответа выключены "
                                              "(только для прогона garak baseline)")
    yield

    await app.state.openai.close()
    await app.state.cache.aclose()
    await close_chat_storage(app.state)
    if app.state.pii_redactor is not None:
        app.state.pii_redactor.close()
    shutdown_tracing(app.state.tracer_provider)


app = FastAPI(
    title=settings.app_name,
    version="4.1.0",
    description=(
        "Чат-ядро ассистента техподдержки: ответ целиком (`POST /chat`) и потоком "
        "(`POST /chat/stream`), кеш в Redis, каталог моделей; чаты с историей на сервере "
        "(`/chats`, ответ потоком SSE). Каждый ответ содержит "
        "заголовок `X-Request-ID`; ошибки приходят в формате "
        "`{\"error\": {\"code\": ..., \"message\": ...}}`."
    ),
    lifespan=lifespan,
    telemetry=fastapi_telemetry(),   # span на HTTP-запрос — корень трейса в Phoenix
    openapi_tags=[
        {"name": "chat", "description": "Запросы к модели"},
        {"name": "chats", "description": "Чаты с историей на сервере (блок 4.1)"},
        {"name": "models", "description": "Каталог моделей и цен"},
        {"name": "health", "description": "Проверка состояния"},
    ],
)

# Лимит запросов (блок 3.8, RATE_LIMIT_PER_MIN): добавлен первым, поэтому внутренний —
# ответ 429 проходит через CORS и RequestContextMiddleware (CORS-заголовки, X-Request-ID,
# строка http_request).
app.add_middleware(RateLimitMiddleware)

# charset=utf-8 у JSON-ответов — для Windows PowerShell 5.1 (app/core/charset.py).
app.add_middleware(JSONCharsetMiddleware)

# CORS: явный список адресов фронтенда. ["*"] вместе с allow_credentials=True запрещает
# Settings._check_cors — браузер такой ответ всё равно отверг бы.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],   # DELETE — очистка истории чата (блок 4.1)
    allow_headers=["Content-Type", "Authorization", REQUEST_ID_HEADER, USER_ID_HEADER],
    expose_headers=[REQUEST_ID_HEADER, "Retry-After", LIMIT_HEADER, REMAINING_HEADER],
)


# Добавлен последним, поэтому внешний: видит все запросы, включая CORS preflight.
app.add_middleware(RequestContextMiddleware)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error(request: Request, status_code: int, code: str, message: str,
           headers: dict[str, str] | None = None, **extra: Any) -> JSONResponse:
    request_id = _request_id(request)
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": request_id, **extra}},
        headers={**(headers or {}), **({REQUEST_ID_HEADER: request_id} if request_id else {})},
    )


@app.exception_handler(LLMError)
async def handle_llm_error(request: Request, exc: LLMError) -> JSONResponse:
    # 429 -> 429, таймаут -> 504, остальное (ключ, нет соединения, ошибка провайдера) -> 502.
    # Причина уже записана в лог строкой llm_request_failed (LLMService).
    headers = None
    if isinstance(exc, LLMRateLimitError) and exc.retry_after:
        headers = {"Retry-After": str(exc.retry_after)}
    return _error(request, exc.status_code, exc.code, exc.message, headers)


@app.exception_handler(ChatNotFoundError)
async def handle_chat_not_found(request: Request, exc: ChatNotFoundError) -> JSONResponse:
    return _error(request, 404, exc.code, str(exc))


@app.exception_handler(ChatInputError)
async def handle_chat_input(request: Request, exc: ChatInputError) -> JSONResponse:
    return _error(request, 422, exc.code, "Запрос не прошёл валидацию.",
                  fields=[{"field": exc.field, "message": exc.message}])


@app.exception_handler(ChatStorageError)
async def handle_chat_storage(request: Request, exc: ChatStorageError) -> JSONResponse:
    log.warning("chat_storage_error", error=exc.message, cause=repr(exc.__cause__)[:300])
    return _error(request, 503, exc.code, exc.message)


def _field(loc: tuple[Any, ...]) -> str:
    parts = [str(p) for p in loc[1:]] if loc and loc[0] in {"body", "query", "path", "header"} else [str(p) for p in loc]
    return ".".join(parts) or "body"


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    fields = [
        {"field": "body" if err.get("type") == "json_invalid" else _field(tuple(err.get("loc", ()))),
         "message": err.get("msg", "")}
        for err in exc.errors()
    ]
    return _error(request, 422, "validation_error", "Запрос не прошёл валидацию.", fields=fields)


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled_error", request_id=_request_id(request))
    return _error(request, 500, "internal_error", "Внутренняя ошибка сервиса.")


app.include_router(chat.router)
app.include_router(chat_routes.router)
app.include_router(models.router)
app.include_router(health.router)
