"""
GET /health и GET /ready (блоки 3.4–3.5).

- /health — живость (liveness): процесс отвечает. Без зависимостей, всегда 200, даже если
  Redis или провайдер LLM недоступны — оркестратор не должен перезапускать сервис из-за
  временно лежащего Redis.
- /ready — готовность (readiness): PING в Redis с таймаутом. 200 {"status": "ok",
  "redis": "up"} или 503 {"status": "degraded", "redis": "down"}. На неё смотрят
  HEALTHCHECK в Dockerfile и healthcheck в compose.yaml.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request, Response, status

from app.services.llm import CACHE_ERRORS

router = APIRouter(tags=["health"])

READY_TIMEOUT = 1.5   # с; дольше ждать нельзя — healthcheck в compose даёт запросу 5 с


@router.get(
    "/health",
    summary="Сервис жив (liveness)",
    description="Без зависимостей: отвечает 200, даже если Redis или провайдер LLM недоступны.",
    responses={200: {"description": "Процесс работает", "content": {"application/json": {"example": {"status": "ok"}}}}},
)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="Сервис готов (readiness)",
    description=(
        f"Проверяет Redis командой PING с таймаутом {READY_TIMEOUT} с. Пока Redis недоступен — "
        "503: сервис отвечает на запросы, но без кеша. Провайдер LLM здесь не вызывается, "
        "чтобы проверка не тратила токены."
    ),
    responses={
        200: {"description": "Redis доступен", "content": {"application/json": {"example": {"status": "ok", "redis": "up"}}}},
        503: {"description": "Redis недоступен", "content": {"application/json": {"example": {"status": "degraded", "redis": "down"}}}},
    },
)
async def ready(request: Request, response: Response) -> dict[str, str]:
    cache = getattr(request.app.state, "cache", None)
    if cache is not None:
        try:
            if await asyncio.wait_for(cache.ping(), timeout=READY_TIMEOUT):
                return {"status": "ok", "redis": "up"}
        except CACHE_ERRORS:
            pass
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "degraded", "redis": "down"}
