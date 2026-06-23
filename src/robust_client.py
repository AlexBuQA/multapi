"""
Надёжный клиент LLM (ДЗ 2.3, адаптирован под Ollama + отдельный аудио-эндпоинт).

Возможности:
- обёртка над OpenAI SDK, перехват ошибок 429 / 5xx / таймаутов;
- retry с exponential backoff (1s -> 2s -> 4s -> 8s -> 16s) + jitter, до 5 попыток;
- fallback-цепочка провайдеров;
- логирование timestamp / код ошибки / номер попытки;
- трекинг usage: prompt_tokens, completion_tokens, стоимость;
- поддержка мультимодальных сообщений (vision): content может быть списком блоков;
- HTTP-прокси из настроек, который НЕ применяется к локальным endpoint (localhost),
  чтобы не ломать локальный Ollama;
- отдельный аудио-клиент (raw_client с другим провайдером) для Whisper/TTS.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Iterable

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .config import ProviderConfig, Settings, settings
from .utils import UsageTracker, get_logger

logger = get_logger("client")

RETRYABLE = (RateLimitError, APITimeoutError, InternalServerError, APIConnectionError)


class RobustLLMClient:
    """Надёжный клиент с retry + fallback. Параметры — из Settings."""

    USER_FACING_FAILURE = "Сервис временно недоступен. Попробуйте позже."

    def __init__(self, cfg: Settings | None = None, usage: UsageTracker | None = None):
        self.settings = cfg or settings
        self.usage = usage or UsageTracker()
        self._clients: dict[str, OpenAI] = {}

    # ------------------------------------------------------------------ #
    def _client_for(self, provider: ProviderConfig) -> OpenAI:
        """Ленивая инициализация и кеширование клиента по провайдеру."""
        if provider.name not in self._clients:
            if not provider.api_key:
                raise RuntimeError(
                    f"Не задан ключ {provider.api_key_env} для провайдера "
                    f"{provider.name}. Проверьте .env."
                )
            # HTTP-прокси применяем только к удалённым endpoint, не к localhost.
            http_client = None
            if self.settings.proxy and not provider.is_local:
                logger.info("Провайдер=%s: HTTP-прокси включён", provider.name)
                http_client = httpx.Client(
                    proxy=self.settings.proxy, timeout=self.settings.request_timeout
                )
            self._clients[provider.name] = OpenAI(
                api_key=provider.api_key,
                base_url=provider.base_url,
                timeout=self.settings.request_timeout,
                max_retries=0,  # ретраи делаем сами через tenacity
                http_client=http_client,
            )
        return self._clients[provider.name]

    def _call_with_retry(self, provider: ProviderConfig, fn: Callable[[OpenAI], Any]) -> Any:
        """Выполняет fn(client) с exponential backoff + jitter для данного провайдера."""
        attempt = {"n": 0}

        @retry(
            retry=retry_if_exception_type(RETRYABLE),
            wait=wait_exponential_jitter(initial=1, max=16, jitter=1),
            stop=stop_after_attempt(self.settings.max_retries),
            reraise=True,
        )
        def _runner() -> Any:
            attempt["n"] += 1
            try:
                return fn(self._client_for(provider))
            except RETRYABLE as exc:
                code = getattr(exc, "status_code", "—")
                logger.warning(
                    "Провайдер=%s попытка=%d/%d код=%s ошибка=%s",
                    provider.name, attempt["n"], self.settings.max_retries,
                    code, type(exc).__name__,
                )
                raise

        return _runner()

    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        stream: bool = False,
        label: str = "chat",
        model_override: str | None = None,
    ) -> str:
        """
        Chat-completion с retry+fallback. Возвращает текст ответа.

        model_override позволяет использовать другую модель (vision-модель для
        изображений или модель классификатора), не меняя провайдера.
        messages поддерживает мультимодальный content (списком блоков).
        """
        last_exc: Exception | None = None
        for provider in self.settings.chain():
            model = model_override or provider.chat_model
            try:
                logger.info("Запрос к провайдеру=%s модель=%s", provider.name, model)

                def _do(client: OpenAI) -> Any:
                    return client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream,
                    )

                response = self._call_with_retry(provider, _do)

                if stream:
                    return self._consume_stream(response)

                content = response.choices[0].message.content or ""
                u = getattr(response, "usage", None)
                if u is not None:
                    self.usage.add_chat(
                        getattr(u, "prompt_tokens", 0) or 0,
                        getattr(u, "completion_tokens", 0) or 0,
                        provider.price_in_per_1m, provider.price_out_per_1m,
                        label=f"{label}/{provider.name}",
                    )
                return content

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.error(
                    "Провайдер=%s исчерпан (%s: %s). Следующий.",
                    provider.name, type(exc).__name__, exc,
                )
                continue

        logger.critical("Все провайдеры недоступны. Последняя ошибка: %s", last_exc)
        return self.USER_FACING_FAILURE

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        model_override: str | None = None,
    ) -> Iterable[str]:
        """Генератор чанков ответа (стриминг в CLI)."""
        last_exc: Exception | None = None
        for provider in self.settings.chain():
            model = model_override or provider.chat_model
            try:
                def _do(client: OpenAI) -> Any:
                    return client.chat.completions.create(
                        model=model, messages=messages, temperature=temperature,
                        max_tokens=max_tokens, stream=True,
                    )

                response = self._call_with_retry(provider, _do)
                self.usage.calls += 1
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.error("Стрим: провайдер=%s упал (%s). Следующий.", provider.name, exc)
                continue
        logger.critical("Стрим: все провайдеры недоступны (%s)", last_exc)
        yield self.USER_FACING_FAILURE

    def _consume_stream(self, response: Any) -> str:
        parts: list[str] = []
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                parts.append(delta.content)
        self.usage.calls += 1
        return "".join(parts)

    # ------------------------------------------------------------------ #
    def raw_client(self, provider: ProviderConfig | None = None) -> OpenAI:
        """Прямой доступ к SDK-клиенту нужного провайдера (для audio.* и пр.)."""
        return self._client_for(provider or self.settings.primary())

    def call_with_retry(self, provider: ProviderConfig, fn: Callable[[OpenAI], Any]) -> Any:
        """Публичная обёртка над retry-логикой (используется voice.py для аудио)."""
        return self._call_with_retry(provider, fn)

    def measure(self, fn: Callable[[], Any]) -> tuple[Any, float]:
        start = time.perf_counter()
        return fn(), time.perf_counter() - start
