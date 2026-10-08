"""
Доменные исключения слоя LLM (блок 3.4).

LLMService переводит в них ошибки OpenAI SDK, а обработчик в app/main.py — в HTTP-ответ
{"error": {"code": ..., "message": ...}} с кодом status_code. Наружу уходит понятный
текст; исходная ошибка провайдера остаётся в __cause__ и пишется только в лог сервиса.
"""
from __future__ import annotations


class LLMError(Exception):
    """Провайдер ответил ошибкой, которую клиент сервиса исправить не может."""

    code = "llm_error"
    status_code = 502
    default_message = "Провайдер LLM вернул ошибку. Попробуйте позже."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


class LLMRateLimitError(LLMError):
    """429 от провайдера: превышен лимит запросов или токенов."""

    code = "llm_rate_limit"
    status_code = 429
    default_message = "Провайдер LLM ограничил частоту запросов. Повторите попытку позже."

    def __init__(self, message: str | None = None, *, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after   # секунды из заголовка Retry-After провайдера, если он был


class LLMTimeoutError(LLMError):
    """Провайдер не ответил за LLM__REQUEST_TIMEOUT (с учётом повторов SDK)."""

    code = "llm_timeout"
    status_code = 504
    default_message = "Провайдер LLM не ответил вовремя. Попробуйте позже или сократите запрос."


class LLMAuthError(LLMError):
    """401/403: неверный ключ или у ключа нет доступа."""

    code = "llm_auth"
    status_code = 502
    default_message = "Провайдер LLM отклонил ключ доступа сервиса."


class LLMUnavailableError(LLMError):
    """Нет соединения с провайдером: Ollama не запущен, неверный адрес, сеть."""

    code = "llm_unavailable"
    status_code = 502
    default_message = "Провайдер LLM недоступен. Попробуйте позже."


class LLMContentFiltered(LLMError):
    """Блок 3.8: поток остановлен — в ответе модели метка, начало системного промпта или
    роль джейлбрейка (StreamGuard). В /chat такой ответ заменяется отказом без ошибки, а
    в /chat/stream часть ответа уже у клиента: поток заканчивается кадром error."""

    code = "content_filter"
    status_code = 502
    default_message = "Ответ остановлен фильтром безопасности."
