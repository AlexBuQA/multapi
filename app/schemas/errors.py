"""
Единый формат ошибок сервиса (блок 3.4) и описания ответов для Swagger.

Любая ошибка — HTTP-код и тело {"error": {"code": ..., "message": ...}}. У ошибок
валидации дополнительно fields — список {"field", "message"}. request_id совпадает
с заголовком X-Request-ID: по нему ошибку находят в логе сервиса.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class FieldError(BaseModel):
    field: str
    message: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    fields: list[FieldError] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


_RID = "9f1c2d3e4b5a69788796a5b4c3d2e1f0"


def _example(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "request_id": _RID, **extra}}


LLM_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    422: {
        "model": ErrorResponse,
        "description": "Запрос не прошёл валидацию: поле и причина — в error.fields",
        "content": {"application/json": {"example": _example(
            "validation_error", "Запрос не прошёл валидацию.",
            fields=[{"field": "temperature", "message": "Input should be less than or equal to 2"}],
        )}},
    },
    429: {
        "model": ErrorResponse,
        "description": "Провайдер ограничил частоту запросов; если он передал Retry-After, заголовок есть и в ответе",
        "content": {"application/json": {"example": _example(
            "llm_rate_limit", "Провайдер LLM ограничил частоту запросов. Повторите попытку позже.",
        )}},
    },
    502: {
        "model": ErrorResponse,
        "description": "Ошибка на стороне провайдера: неверный ключ, провайдер недоступен или вернул ошибку",
        "content": {"application/json": {"examples": {
            "llm_auth": {"summary": "Неверный ключ провайдера", "value": _example(
                "llm_auth", "Провайдер LLM отклонил ключ доступа сервиса.")},
            "llm_unavailable": {"summary": "Нет соединения с провайдером", "value": _example(
                "llm_unavailable", "Провайдер LLM недоступен. Попробуйте позже.")},
            "llm_error": {"summary": "Провайдер вернул ошибку", "value": _example(
                "llm_error", "Модель «gpt-unknown» не найдена у провайдера.")},
        }}},
    },
    504: {
        "model": ErrorResponse,
        "description": "Провайдер не ответил за LLM__REQUEST_TIMEOUT",
        "content": {"application/json": {"example": _example(
            "llm_timeout", "Провайдер LLM не ответил вовремя. Попробуйте позже или сократите запрос.",
        )}},
    },
}
