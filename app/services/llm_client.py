"""
Асинхронный LLM-клиент (блок 3.3) — версия RobustLLMClient из модуля 2 на asyncio.

Что сохранено из синхронного клиента:
- fallback-цепочка провайдеров из src/config.py (LLM_PROVIDER, LLM_FALLBACK_ORDER);
- кеш ответов LLMCache (ключ — model + messages + temperature, TTL);
- HTTP-прокси только для удалённых провайдеров, не для localhost.

Что устроено иначе:
- AsyncOpenAI вместо OpenAI: ни одного блокирующего вызова внутри async def;
- повторы — встроенные в SDK (max_retries): SDK сам повторяет 408/409/429/5xx и
  ошибки соединения с экспоненциальной задержкой и учитывает Retry-After. Поэтому
  tenacity здесь не нужен (в синхронном RobustLLMClient он остаётся);
- два уровня таймаутов: таймаут SDK — на одну HTTP-попытку (LLM_REQUEST_TIMEOUT),
  asyncio.timeout — на всю операцию complete(), включая повторы и fallback
  (LLM_CALL_TIMEOUT);
- Semaphore — атрибут экземпляра: создаётся один раз в __init__ и ограничивает
  число одновременных запросов этого клиента (complete, batch_chat, stream_chat).

Методы: complete(), batch_chat() (gather с return_exceptions), batch_chat_strict()
(TaskGroup: все ответы или ничего), stream_chat() (async-генератор токенов).
Каждый вызов пишет JSON-строку в logs/llm_calls.jsonl: события llm.call и llm.stream.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

import httpx
from openai import AsyncOpenAI

from app.config import AsyncClientSettings, client_settings
from app.logging_utils import get_event_logger, log_event
from src.cache import LLMCache
from src.config import ProviderConfig, Settings
from src.config import settings as llm_settings
from src.robust_client import AllProvidersFailedError

ClientFactory = Callable[[ProviderConfig], Any]


def _usage(response: Any) -> dict[str, int]:
    u = getattr(response, "usage", None)
    prompt = int(getattr(u, "prompt_tokens", 0) or 0)
    completion = int(getattr(u, "completion_tokens", 0) or 0)
    total = int(getattr(u, "total_tokens", 0) or 0) or prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


class AsyncLLMClient:
    """Асинхронный клиент с fallback, кешем, лимитом конкурентности и логом вызовов."""

    def __init__(
        self,
        *,
        concurrency: int | None = None,
        model: str | None = None,
        use_cache: bool = True,
        cache: LLMCache | None = None,
        cfg: Settings | None = None,
        options: AsyncClientSettings | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.cfg = cfg or llm_settings
        self.options = options or client_settings
        self.concurrency = concurrency or self.options.llm_concurrency
        # Один семафор на экземпляр: создаётся здесь и больше нигде.
        self._sem = asyncio.Semaphore(self.concurrency)
        self.call_timeout = self.options.llm_call_timeout
        self.model = model  # переопределение модели основного провайдера
        self.cache = (cache or LLMCache(ttl=self.cfg.cache_ttl)) if use_cache else None
        self.events = get_event_logger(self.options.llm_call_log_path)

        self.providers = self.cfg.chain()
        factory = client_factory or self._make_sdk_client
        # Провайдеры без ключа пропускаются (например, OpenRouter, пока ключа нет).
        self._clients: dict[str, Any] = {
            p.name: factory(p) for p in self.providers if client_factory or p.api_key
        }

    # ------------------------------------------------------------------ #
    def _make_sdk_client(self, provider: ProviderConfig) -> AsyncOpenAI:
        http_client = None
        if self.cfg.proxy and not provider.is_local:
            http_client = httpx.AsyncClient(proxy=self.cfg.proxy, timeout=self.cfg.request_timeout)
        return AsyncOpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            timeout=self.cfg.request_timeout,               # одна HTTP-попытка
            max_retries=self.options.llm_sdk_max_retries,   # 429/5xx: backoff + Retry-After
            http_client=http_client,
        )

    def _model_for(self, provider: ProviderConfig, override: str | None) -> str:
        """Переопределение модели — только для основного провайдера: имя локальной
        модели Ollama не подходит облачному fallback."""
        is_primary = provider.name == self.cfg.provider
        return ((override or self.model) if is_primary else None) or provider.chat_model

    @staticmethod
    def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": system}] if system else []
        return messages + [{"role": "user", "content": prompt}]

    def _check_concurrency(self, concurrency: int | None) -> None:
        if concurrency is not None and concurrency != self.concurrency:
            raise ValueError(
                f"Лимит конкурентности задаётся при создании клиента: семафор экземпляра "
                f"создан на {self.concurrency}. Для concurrency={concurrency} создайте "
                f"AsyncLLMClient(concurrency={concurrency})."
            )

    # ------------------------------------------------------------------ #
    async def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """Один запрос: кеш → семафор → asyncio.timeout → цепочка провайдеров."""
        messages = self._messages(prompt, system)
        primary_model = self._model_for(self.providers[0], model) if self.providers else ""
        log: dict[str, Any] = {"model": primary_model, "provider": None, "prompt_chars": len(prompt)}

        if self.cache is not None:
            cached = self.cache.get(primary_model, messages, temperature)
            if cached is not None:
                log_event(self.events, "llm.call", status="cache_hit", duration_ms=0.0,
                          queue_ms=0.0, **log)
                return cached

        queued = time.perf_counter()
        async with self._sem:
            started = time.perf_counter()
            status = "error"
            try:
                async with asyncio.timeout(self.call_timeout):   # бюджет на всю операцию
                    text, provider, used_model, usage = await self._call_chain(
                        messages, model, temperature, max_tokens
                    )
                status = "ok"
                log.update(provider=provider, model=used_model, **usage)
            except TimeoutError:
                status = "timeout"
                raise
            except asyncio.CancelledError:
                status = "cancelled"   # например, TaskGroup отменил задачу после чужой ошибки
                raise
            except Exception as exc:
                status = f"error:{type(exc).__name__}"
                raise
            finally:
                finished = time.perf_counter()
                log_event(self.events, "llm.call", status=status,
                          duration_ms=_ms(finished - started),
                          queue_ms=_ms(started - queued), **log)

        if self.cache is not None:
            self.cache.set(primary_model, messages, temperature, text)
        return text

    async def _call_chain(
        self, messages: list[dict[str, str]], model: str | None,
        temperature: float, max_tokens: int | None,
    ) -> tuple[str, str, str, dict[str, int]]:
        last_exc: Exception | None = None
        for provider in self.providers:
            client = self._clients.get(provider.name)
            if client is None:
                continue
            used_model = self._model_for(provider, model)
            kwargs: dict[str, Any] = {"model": used_model, "messages": messages,
                                      "temperature": temperature}
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            try:
                response = await client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 — переходим к следующему провайдеру
                last_exc = exc
                log_event(self.events, "llm.fallback", provider=provider.name,
                          model=used_model, error=f"{type(exc).__name__}: {exc}")
                continue
            text = response.choices[0].message.content or ""
            return text, provider.name, used_model, _usage(response)
        raise AllProvidersFailedError(str(last_exc)) from last_exc

    # ------------------------------------------------------------------ #
    async def batch_chat(
        self,
        prompts: list[str],
        concurrency: int | None = None,
        *,
        models: Sequence[str | None] | None = None,
        **kwargs: Any,
    ) -> list[str | BaseException]:
        """
        Пакет запросов: результаты в порядке prompts; упавший запрос возвращается
        исключением на своей позиции и не роняет остальные (return_exceptions=True).
        Одновременно выполняется не больше self.concurrency запросов — их ограничивает
        семафор экземпляра внутри complete().
        """
        self._check_concurrency(concurrency)
        coros = [
            self.complete(prompt, model=models[i] if models else None, **kwargs)
            for i, prompt in enumerate(prompts)
        ]
        return await asyncio.gather(*coros, return_exceptions=True)

    async def batch_chat_strict(
        self,
        prompts: list[str],
        concurrency: int | None = None,
        *,
        models: Sequence[str | None] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """
        «Все ответы или ничего»: при первой ошибке TaskGroup отменяет остальные задачи
        и поднимает ExceptionGroup — уже полученные ответы теряются.
        """
        self._check_concurrency(concurrency)
        tasks: list[asyncio.Task[str]] = []
        try:
            async with asyncio.TaskGroup() as tg:
                for i, prompt in enumerate(prompts):
                    tasks.append(tg.create_task(
                        self.complete(prompt, model=models[i] if models else None, **kwargs)
                    ))
        except* Exception as group:
            log_event(self.events, "llm.batch_strict_failed", prompts=len(prompts),
                      failed=len(group.exceptions),
                      cancelled=sum(t.cancelled() for t in tasks),
                      completed=sum(t.done() and not t.cancelled() and t.exception() is None
                                    for t in tasks))
            raise
        return [t.result() for t in tasks]

    # ------------------------------------------------------------------ #
    async def stream_chat(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Async-генератор: отдаёт фрагменты ответа по мере прихода. Fallback на следующий
        провайдер возможен только до первого токена. После закрытия стрима в лог
        пишется событие llm.stream: время до первого токена, общее время и usage.
        """
        messages = self._messages(prompt, system)
        async with self._sem:
            started = time.perf_counter()
            first_at: float | None = None
            last_at: float | None = None
            chunks = 0
            usage: dict[str, int] = {}
            status = "error"
            provider_name: str | None = None
            used_model: str | None = None
            last_exc: Exception | None = None
            try:
                for provider in self.providers:
                    client = self._clients.get(provider.name)
                    if client is None:
                        continue
                    used_model = self._model_for(provider, model)
                    kwargs: dict[str, Any] = {
                        "model": used_model, "messages": messages, "temperature": temperature,
                        "stream": True, "stream_options": {"include_usage": True},
                    }
                    if max_tokens is not None:
                        kwargs["max_tokens"] = max_tokens
                    try:
                        stream = await client.chat.completions.create(**kwargs)
                    except Exception as exc:  # noqa: BLE001 — ещё ни одного токена: fallback
                        last_exc = exc
                        log_event(self.events, "llm.fallback", provider=provider.name,
                                  model=used_model, error=f"{type(exc).__name__}: {exc}")
                        continue

                    provider_name = provider.name
                    async for chunk in stream:
                        if getattr(chunk, "usage", None):
                            usage = _usage(chunk)
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta.content
                        if delta:
                            now = time.perf_counter()
                            first_at = first_at or now
                            last_at = now
                            chunks += 1
                            yield delta
                    status = "ok"
                    return
                raise AllProvidersFailedError(str(last_exc)) from last_exc
            except (GeneratorExit, asyncio.CancelledError):
                status = "cancelled"   # клиент перестал читать (например, закрыл SSE)
                raise
            except Exception as exc:
                status = f"error:{type(exc).__name__}"
                raise
            finally:
                log_event(
                    self.events, "llm.stream", status=status, provider=provider_name,
                    model=used_model, prompt_chars=len(prompt), chunks=chunks,
                    ttft_ms=_ms(first_at - started) if first_at else None,
                    last_token_ms=_ms(last_at - started) if last_at else None,
                    duration_ms=_ms(time.perf_counter() - started), **usage,
                )

    # ------------------------------------------------------------------ #
    async def aclose(self) -> None:
        for client in self._clients.values():
            close = getattr(client, "close", None)
            if close is not None:
                await close()

    async def __aenter__(self) -> AsyncLLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
