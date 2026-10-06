"""GET /health и GET /ready (блок 3.4)."""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.services.llm import CACHE_ERRORS

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Сервис жив",
    description="Без зависимостей: отвечает 200, даже если Redis или провайдер LLM недоступны.",
    responses={200: {"description": "Процесс работает", "content": {"application/json": {"example": {"status": "ok"}}}}},
)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="Состояние зависимостей",
    description=(
        "Проверяет Redis командой PING. `degraded` — сервис отвечает, но без кеша. "
        "Провайдер LLM здесь не вызывается, чтобы проверка не тратила токены."
    ),
    responses={200: {"description": "ready или degraded", "content": {"application/json": {"example": {
        "status": "degraded", "components": {"redis": False}}}}}},
)
async def ready(request: Request) -> dict[str, object]:
    cache = getattr(request.app.state, "cache", None)
    redis_ok = False
    if cache is not None:
        try:
            redis_ok = bool(await cache.ping())
        except CACHE_ERRORS:
            redis_ok = False
    return {"status": "ready" if redis_ok else "degraded", "components": {"redis": redis_ok}}
