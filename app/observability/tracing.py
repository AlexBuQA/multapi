"""
Трейсинг вызовов модели в Phoenix (блок 3.6).

setup_tracing() вызывается в lifespan до создания клиента AsyncOpenAI: регистрирует
OpenTelemetry TracerProvider с экспортом в Phoenix и включает автоинструментацию
OpenAI SDK (OpenInference). После этого каждый chat.completions.create() создаёт span
с входом, ответом, моделью и токенами.

Отличия от стартер-кода задания:
- register() получает адрес вида http://phoenix:6006/v1/traces. Если передать
  endpoint без пути, phoenix.otel отправит спаны на корень сервера, и Phoenix их не
  примет; путь /v1/traces он дописывает сам, только когда адрес берётся из переменной
  окружения;
- batch=True: спаны отправляются фоновым потоком пачками. По умолчанию register()
  отправляет каждый span синхронно, прямо в обработчике запроса;
- без PHOENIX_COLLECTOR_ENDPOINT трейсинг выключен: локальный uvicorn и тесты не шлют
  спаны в несуществующий Phoenix. В compose.yaml переменная задана.

FastAPI начиная с 0.142 сама пишет span на каждый HTTP-запрос, как только настроен
TracerProvider. Он становится корнем трейса: POST /chat -> llm.chat -> ChatCompletion.
Настройки — в fastapi_telemetry(): без healthcheck-ов (иначе каждые 15 с в Phoenix
появлялся бы трейс GET /ready) и без служебных спанов fastapi.dependencies,
fastapi.endpoint, fastapi.serialization.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

from collections.abc import MutableMapping
from typing import Any

from fastapi.telemetry import TelemetryConfig
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from phoenix.otel import register

from app.observability.logging import get_logger
from app.observability.middleware import QUIET_PATHS

log = get_logger()


def traces_endpoint(endpoint: str) -> str:
    """http://phoenix:6006 -> http://phoenix:6006/v1/traces (OTLP по HTTP)."""
    parsed = urlparse(endpoint)
    if parsed.path in ("", "/"):
        return endpoint.rstrip("/") + "/v1/traces"
    return endpoint


def _is_healthcheck(scope: MutableMapping[str, Any]) -> bool:
    return scope.get("path") in QUIET_PATHS


def fastapi_telemetry() -> TelemetryConfig:
    """Настройки встроенной трассировки FastAPI: FastAPI(telemetry=fastapi_telemetry())."""
    return {
        "exclude": _is_healthcheck,     # /health и /ready — без спанов
        "operation_spans": False,       # только span запроса, без внутренних этапов FastAPI
        "metrics": False,               # метрики OpenTelemetry сервис не собирает
        "logs": False,                  # логи — через structlog, не через OpenTelemetry
        "auto_configure": False,        # экспорт настраивает только setup_tracing()
    }


def setup_tracing(project_name: str = "diploma-fastapi", endpoint: str | None = None) -> TracerProvider | None:
    """Включает трейсинг; возвращает TracerProvider или None, если Phoenix не настроен."""
    endpoint = endpoint or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        log.info("tracing_disabled", reason="PHOENIX_COLLECTOR_ENDPOINT не задан")
        return None

    tracer_provider = register(
        project_name=project_name,
        endpoint=traces_endpoint(endpoint),
        batch=True,
        verbose=False,
    )
    instrumentor = OpenAIInstrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument(tracer_provider=tracer_provider)
    log.info("tracing_enabled", endpoint=traces_endpoint(endpoint), project=project_name)
    return tracer_provider


def shutdown_tracing(tracer_provider: TracerProvider | None) -> None:
    """Снимает инструментацию и отправляет оставшиеся спаны перед остановкой."""
    if tracer_provider is None:
        return
    OpenAIInstrumentor().uninstrument()
    tracer_provider.shutdown()
