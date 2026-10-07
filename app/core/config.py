"""
Настройки HTTP-сервиса (блок 3.4) на pydantic-settings v2.

Значения читаются из переменных окружения и из .env в корне проекта; переменные
окружения важнее .env. Секция LLM вложенная: её поля задаются с префиксом LLM__
(env_nested_delimiter="__"), например LLM__OPENAI_API_KEY, LLM__DEFAULT_MODEL.

Ключ провайдера обязателен: без LLM__OPENAI_API_KEY get_settings() поднимает
ValidationError, и uvicorn не стартует. Для локального Ollama подойдёт любое
непустое значение — Ollama ключ не проверяет.

Настройки ассистента с инструментами (блок 3.1) и скриптов блока 3.3 — в app/config.py.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class LLMSettings(BaseModel):
    """Подключение к OpenAI-совместимому провайдеру (OpenAI, OpenRouter, Ollama)."""

    # SecretStr: значение не попадает в repr, логи и трейсбеки — только .get_secret_value().
    openai_api_key: SecretStr
    # Пусто -> https://api.openai.com/v1. Локальный Ollama: http://localhost:11434/v1.
    base_url: str | None = None
    # Модель для запросов, в которых поле model не задано.
    default_model: str = "gpt-4o-mini"
    # Таймаут одной HTTP-попытки, с. Для Ollama на CPU нужен запас: 120 и больше.
    request_timeout: float = Field(default=30.0, gt=0)
    # Повторы SDK на 408/409/429/5xx и ошибки соединения (экспоненциальная задержка, Retry-After).
    max_retries: int = Field(default=3, ge=0, le=10)
    # Bulkhead: сколько запросов к модели сервис держит одновременно, остальные ждут.
    max_concurrency: int = Field(default=10, ge=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_ignore_empty=True,   # «КЛЮЧ=» в .env — то же, что ключа нет: действует значение по умолчанию
        extra="ignore",          # в .env есть переменные блоков 2–3, сервису они не нужны
    )

    app_name: str = "multapi — LLM-сервис техподдержки"
    # Уровень лога сервиса (логгер llm-service): DEBUG, INFO, WARNING, ERROR.
    log_level: str = "INFO"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = Field(default=3600, ge=1)
    # Адреса фронтенда, которым браузер разрешит обращаться к API. В .env — JSON-список.
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    # Куки и заголовок Authorization с другого origin. Сервису пока не нужны.
    cors_allow_credentials: bool = False
    # Пустой словарь проверяется при старте, поэтому без ключа ошибка указывает
    # точное поле: llm.openai_api_key — Field required.
    llm: LLMSettings = Field(default_factory=dict, validate_default=True)  # type: ignore[arg-type]

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, value: str) -> str:
        value = value.upper()
        if value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL: ожидается DEBUG, INFO, WARNING, ERROR или CRITICAL")
        return value

    @model_validator(mode="after")
    def _check_cors(self) -> Settings:
        if self.cors_allow_credentials and "*" in self.cors_origins:
            raise ValueError(
                "CORS: allow_credentials=True нельзя сочетать с allow_origins=['*'] — "
                "перечислите адреса фронтенда в CORS_ORIGINS"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Один экземпляр настроек на процесс: .env читается при первом вызове."""
    return Settings()
