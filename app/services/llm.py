"""
LLMService (блок 3.4) — слой работы с моделью для HTTP-сервиса.

Сервис создаётся на каждый запрос (app/deps/providers.py), а долгоживущие объекты
получает готовыми из app.state: клиент AsyncOpenAI, подключение к Redis и семафор.

- complete() — ответ целиком с кешем в Redis: ключ "chat:" + sha256 от параметров
  запроса без user_id/session_id/stream, TTL — CACHE_TTL_SECONDS;
- stream() — фрагменты ответа по мере генерации, без кеша, с итоговым usage;
- ошибки OpenAI SDK переводятся в доменные исключения из app/core/exceptions.py.

Из блока 3.3 перенесены: повторы средствами SDK (max_retries, без tenacity поверх —
иначе попытки перемножаются), stream_options.include_usage и семафор, созданный один
раз: теперь он живёт в lifespan и ограничивает запросы всего сервиса (Bulkhead).
Ошибки Redis запрос не роняют: кеш — ускорение, а не обязательная часть ответа.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

import openai
from redis.exceptions import RedisError

from app.core.config import Settings
from app.core.exceptions import (
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.schemas.chat import ChatDelta, ChatRequest, ChatResponse, Usage

logger = logging.getLogger("llm-service")

# Поля, которые не влияют на ответ модели и не должны дробить кеш.
CACHE_KEY_EXCLUDE = {"user_id", "session_id", "stream"}
CACHE_ERRORS = (RedisError, OSError, asyncio.TimeoutError)


def _retry_after(exc: openai.APIStatusError) -> int | None:
    value = exc.response.headers.get("retry-after") if exc.response is not None else None
    try:
        return max(1, round(float(value))) if value else None
    except ValueError:
        return None


@contextlib.contextmanager
def provider_errors(model: str) -> Iterator[None]:
    """Переводит исключения OpenAI SDK в доменные. Порядок важен: таймаут —
    подкласс ошибки соединения, а все HTTP-ошибки — подклассы APIStatusError."""
    try:
        yield
    except openai.RateLimitError as exc:
        raise LLMRateLimitError(retry_after=_retry_after(exc)) from exc
    except openai.APITimeoutError as exc:
        raise LLMTimeoutError() from exc
    except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
        raise LLMAuthError() from exc
    except openai.APIConnectionError as exc:
        raise LLMUnavailableError() from exc
    except openai.NotFoundError as exc:
        raise LLMError(f"Модель «{model}» не найдена у провайдера.") from exc
    except openai.APIStatusError as exc:
        raise LLMError(f"Провайдер LLM вернул ошибку {exc.status_code}.") from exc
    except openai.OpenAIError as exc:
        raise LLMError() from exc


class LLMService:
    def __init__(
        self,
        openai_client: Any,
        cache: Any | None,
        settings: Settings,
        *,
        limiter: asyncio.Semaphore | None = None,
    ) -> None:
        self.openai = openai_client
        self.cache = cache
        self.settings = settings
        self.limiter = limiter or asyncio.Semaphore(settings.llm.max_concurrency)

    # ------------------------------------------------------------------ #
    def _resolve(self, req: ChatRequest) -> ChatRequest:
        """Подставляет модель по умолчанию — до расчёта ключа кеша, иначе после
        смены LLM__DEFAULT_MODEL из кеша вернулся бы ответ старой модели."""
        return req if req.model else req.model_copy(update={"model": self.settings.llm.default_model})

    @staticmethod
    def cache_key(req: ChatRequest) -> str:
        payload = req.model_dump(mode="json", exclude=CACHE_KEY_EXCLUDE)
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return "chat:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _params(self, req: ChatRequest) -> dict[str, Any]:
        return {
            "model": req.model,
            "messages": [m.model_dump() for m in req.messages],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }

    async def _cache_get(self, key: str) -> str | None:
        if self.cache is None:
            return None
        try:
            return await self.cache.get(key)
        except CACHE_ERRORS as exc:
            logger.warning("cache.get failed, идём в модель без кеша: %r", exc)
            return None

    async def _cache_set(self, key: str, value: str) -> None:
        if self.cache is None:
            return
        try:
            await self.cache.setex(key, self.settings.cache_ttl_seconds, value)
        except CACHE_ERRORS as exc:
            logger.warning("cache.setex failed, ответ не закеширован: %r", exc)

    # ------------------------------------------------------------------ #
    async def complete(self, req: ChatRequest) -> ChatResponse:
        req = self._resolve(req)
        key = self.cache_key(req)

        cached = await self._cache_get(key)
        if cached is not None:
            try:
                return ChatResponse.model_validate_json(cached).model_copy(update={"cached": True})
            except ValueError:
                logger.warning("В кеше по ключу %s запись старого формата — запрашиваем модель", key)

        async with self.limiter:
            with provider_errors(req.model or ""):
                raw = await self.openai.chat.completions.create(**self._params(req))
        response = ChatResponse.from_openai(raw)

        # Пустой ответ (например, модель сразу упёрлась в лимит) не кешируем.
        if response.content:
            await self._cache_set(key, response.model_dump_json())
        return response

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]:
        """Фрагменты ответа по мере генерации; последний кадр — usage. Без кеша."""
        req = self._resolve(req)
        async with self.limiter:   # слот держится, пока идёт генерация
            with provider_errors(req.model or ""):
                stream = await self.openai.chat.completions.create(
                    **self._params(req), stream=True, stream_options={"include_usage": True}
                )
                try:
                    async for chunk in stream:
                        if chunk.choices:
                            text = chunk.choices[0].delta.content
                            if text:
                                yield ChatDelta(content=text)
                        if getattr(chunk, "usage", None):
                            yield ChatDelta(usage=Usage.from_openai(chunk.usage))
                finally:
                    # Клиент ушёл или поток оборвался — закрываем соединение с провайдером,
                    # чтобы модель не генерировала ответ впустую.
                    close = getattr(stream, "close", None)
                    if close is not None:
                        await close()
