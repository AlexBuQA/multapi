"""
Внедрение зависимостей (блок 3.4).

Клиент OpenAI, Redis и семафор создаются один раз в lifespan (app/main.py) и лежат
в app.state. Провайдеры ниже достают их из request.app.state — глобальных
переменных-клиентов на уровне модулей нет. Ручки объявляют зависимости через
Annotated-алиасы, а тесты подменяют их через app.dependency_overrides.
"""
from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import Depends, Request
from openai import AsyncOpenAI
from redis.asyncio import Redis

from app.core.config import Settings, get_settings
from app.observability.pii_presidio import NameRedactor
from app.services.llm import LLMService

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_openai(request: Request) -> AsyncOpenAI:
    return request.app.state.openai


def get_cache(request: Request) -> Redis | None:
    """Клиент Redis. None бывает только в тестах — сервис тогда работает без кеша."""
    return request.app.state.cache


def get_limiter(request: Request) -> asyncio.Semaphore:
    return request.app.state.llm_limiter


OpenAIDep = Annotated[AsyncOpenAI, Depends(get_openai)]
CacheDep = Annotated[Redis | None, Depends(get_cache)]
def get_redactor(request: Request) -> NameRedactor | None:
    """Presidio для prompt_preview; None — маскирование только regex (по умолчанию)."""
    return getattr(request.app.state, "pii_redactor", None)


def get_canary(request: Request) -> str | None:
    """Секретная метка процесса (блок 3.8); создаётся в lifespan, в тестах может не быть."""
    return getattr(request.app.state, "canary", None)


LimiterDep = Annotated[asyncio.Semaphore, Depends(get_limiter)]
RedactorDep = Annotated[NameRedactor | None, Depends(get_redactor)]
CanaryDep = Annotated[str | None, Depends(get_canary)]


def get_llm_service(
    openai_client: OpenAIDep,
    cache: CacheDep,
    settings: SettingsDep,
    limiter: LimiterDep,
    redactor: RedactorDep,
    canary: CanaryDep,
) -> LLMService:
    return LLMService(openai_client, cache, settings, limiter=limiter, redactor=redactor, canary=canary)


LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
