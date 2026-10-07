"""
HTTP-сервис ассистента техподдержки на FastAPI (блоки 3.4–3.5).

Запуск из корня проекта:
    uvicorn app.main:app --reload --port 8000
В Docker (блок 3.5): docker compose up -d --build
Swagger: http://localhost:8000/docs

Сборка приложения:
- lifespan создаёт AsyncOpenAI, подключение к Redis и семафор (Bulkhead) и закрывает
  их при остановке. Если Redis не отвечает, сервис всё равно стартует: кеш
  пропускается, /ready отвечает 503, а когда Redis поднимется, клиент
  переподключится сам;
- middleware присваивает запросу request_id (или берёт из X-Request-ID), пишет одну
  строку лога на запрос и возвращает X-Request-ID в ответе;
- CORS — только для адресов из CORS_ORIGINS;
- обработчики переводят доменные ошибки LLM и ошибки валидации в единый JSON
  {"error": {"code", "message", ...}}; трейсбек в ответ не попадает никогда.

Без LLM__OPENAI_API_KEY в .env приложение не стартует: get_settings() поднимает
ValidationError — так и задумано.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from redis.asyncio import Redis

from app.core.config import get_settings
from app.core.exceptions import LLMError, LLMRateLimitError
from app.routers import chat, health, models
from app.services.llm import CACHE_ERRORS

settings = get_settings()   # без ключа здесь ValidationError — uvicorn не стартует

logger = logging.getLogger("llm-service")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False
logger.setLevel(settings.log_level)   # LOG_LEVEL из .env

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")   # чужой ID не должен ломать строку лога


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg = get_settings()
    app.state.openai = AsyncOpenAI(
        api_key=cfg.llm.openai_api_key.get_secret_value(),
        base_url=cfg.llm.base_url,
        timeout=cfg.llm.request_timeout,
        max_retries=cfg.llm.max_retries,
    )
    app.state.llm_limiter = asyncio.Semaphore(cfg.llm.max_concurrency)

    # Клиент создаётся всегда: подключение ленивое, после падения Redis клиент
    # переподключается сам. Недоступный Redis замедляет запрос не больше чем на
    # таймауты ниже, а ошибки кеша LLMService только пишет в лог.
    app.state.cache = Redis.from_url(cfg.redis_url, decode_responses=True,
                                     socket_connect_timeout=1.0, socket_timeout=1.0)
    try:
        await app.state.cache.ping()
        logger.info("Redis %s доступен — кеш ответов включён", cfg.redis_url)
    except CACHE_ERRORS as exc:
        logger.warning("Redis %s недоступен (%s) — запросы идут без кеша, /ready отвечает 503, "
                       "пока Redis не поднимется", cfg.redis_url, exc)

    logger.info("Модель по умолчанию: %s, провайдер: %s", cfg.llm.default_model,
                cfg.llm.base_url or "https://api.openai.com/v1")
    yield

    await app.state.openai.close()
    await app.state.cache.aclose()


app = FastAPI(
    title=settings.app_name,
    version="3.5.0",
    description=(
        "Чат-ядро ассистента техподдержки: ответ целиком (`POST /chat`) и потоком "
        "(`POST /chat/stream`), кеш в Redis, каталог моделей. Каждый ответ содержит "
        "заголовок `X-Request-ID`; ошибки приходят в формате "
        "`{\"error\": {\"code\": ..., \"message\": ...}}`."
    ),
    lifespan=lifespan,
    openapi_tags=[
        {"name": "chat", "description": "Запросы к модели"},
        {"name": "models", "description": "Каталог моделей и цен"},
        {"name": "health", "description": "Проверка состояния"},
    ],
)

# CORS: явный список адресов фронтенда. ["*"] вместе с allow_credentials=True запрещает
# Settings._check_cors — браузер такой ответ всё равно отверг бы.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", REQUEST_ID_HEADER],
    expose_headers=[REQUEST_ID_HEADER],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Добавлен последним, поэтому внешний: видит все запросы, включая CORS preflight."""
    incoming = request.headers.get(REQUEST_ID_HEADER, "")
    request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
    request.state.request_id = request_id

    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        # Для /chat/stream это время до первого фрагмента: тело потока идёт после.
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%s",
            request_id, request.method, request.url.path, status, duration_ms,
            extra={"request_id": request_id, "method": request.method, "path": request.url.path,
                   "status": status, "duration_ms": duration_ms},
        )
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


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
    logger.warning("request_id=%s llm_error=%s cause=%r", _request_id(request), exc.code, exc.__cause__)
    headers = None
    if isinstance(exc, LLMRateLimitError) and exc.retry_after:
        headers = {"Retry-After": str(exc.retry_after)}
    return _error(request, exc.status_code, exc.code, exc.message, headers)


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
    logger.exception("request_id=%s необработанная ошибка", _request_id(request))
    return _error(request, 500, "internal_error", "Внутренняя ошибка сервиса.")


app.include_router(chat.router)
app.include_router(models.router)
app.include_router(health.router)
