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
from app.services.llm import LLMService

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_openai(request: Request) -> AsyncOpenAI:
    return request.app.state.openai


def get_cache(request: Request) -> Redis | None:
    """None — Redis не ответил при старте сервиса, работаем без кеша."""
    return request.app.state.cache


def get_limiter(request: Request) -> asyncio.Semaphore:
    return request.app.state.llm_limiter


OpenAIDep = Annotated[AsyncOpenAI, Depends(get_openai)]
CacheDep = Annotated[Redis | None, Depends(get_cache)]
LimiterDep = Annotated[asyncio.Semaphore, Depends(get_limiter)]


def get_llm_service(
    openai_client: OpenAIDep,
    cache: CacheDep,
    settings: SettingsDep,
    limiter: LimiterDep,
) -> LLMService:
    return LLMService(openai_client, cache, settings, limiter=limiter)


LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
