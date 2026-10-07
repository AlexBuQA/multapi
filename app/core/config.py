"""
Настройки HTTP-сервиса (блок 3.4) на pydantic-settings v2.

Значения читаются из переменных окружения и из .env в корне проекта; переменные
окружения важнее .env. Секция LLM вложенная: её поля задаются с префиксом LLM__
(env_nested_delimiter="__"), например LLM__OPENAI_API_KEY, LLM__DEFAULT_MODEL.

Ключ провайдера обязателен: без LLM__OPENAI_API_KEY get_settings() поднимает
ValidationError, и uvicorn не стартует. Для локального Ollama подойдёт любое
непустое значение — Ollama ключ не проверяет.

Прокси (блок 3.7): LLM__PROXY_URL — HTTP-прокси до внешнего провайдера, например до
api.openai.com через прокси учебной группы. К локальным адресам (localhost, 127.0.0.1,
host.docker.internal, имена сервисов compose без точки) он не применяется — см.
proxy_for(): запросы к Ollama на своём компьютере на чужой сервер не уходят. Адрес
прокси содержит логин и пароль, поэтому хранится как SecretStr и в лог попадает только
без них (proxy_display).

Сертификаты (блок 3.7): LLM__USE_SYSTEM_CERTS=true — проверять HTTPS по хранилищу
сертификатов ОС (пакет truststore), а не по списку certifi. Нужно, когда корпоративная
сеть или антивирус проверяют HTTPS и подставляют свой корневой сертификат: браузер ему
доверяет, а Python — нет (CERTIFICATE_VERIFY_FAILED: self-signed certificate in
certificate chain). Проверка при этом не отключается.

Настройки ассистента с инструментами (блок 3.1) и скриптов блока 3.3 — в app/config.py.
"""
from __future__ import annotations

import ipaddress
import ssl
from functools import lru_cache
from typing import Any
from pathlib import Path
from urllib.parse import urlsplit

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
    # HTTP-прокси до внешнего провайдера (блок 3.7), например http://user:pass@host:8888.
    # К локальному base_url (Ollama) не применяется — см. proxy_for().
    proxy_url: SecretStr | None = None
    # Проверять HTTPS по хранилищу сертификатов ОС (truststore), а не по certifi (блок 3.7).
    use_system_certs: bool = False


LOCAL_HOSTS = {"localhost", "host.docker.internal"}
APP_URL = "https://github.com/AlexBuQA/multapi"


def is_local_url(url: str | None) -> bool:
    """Адрес на этом компьютере или в сети compose. None — api.openai.com, не локальный."""
    if not url:
        return False
    host = (urlsplit(url).hostname or "").lower()
    if host in LOCAL_HOSTS or "." not in host and ":" not in host:   # «ollama», «redis» в compose
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


def proxy_for(base_url: str | None, proxy_url: SecretStr | str | None) -> str | None:
    """Прокси для клиента с этим base_url: только для внешних адресов."""
    if proxy_url is None or is_local_url(base_url):
        return None
    value = proxy_url.get_secret_value() if isinstance(proxy_url, SecretStr) else proxy_url
    return value or None


def is_openrouter(url: str | None) -> bool:
    return (urlsplit(url or "").hostname or "").lower().endswith("openrouter.ai")


def provider_headers(base_url: str | None, title: str) -> dict[str, str] | None:
    """Заголовки атрибуции OpenRouter: по ним в его статистике видно, какое приложение
    тратит ключ (у учебного ключа группы это полезно). Другим провайдерам — ничего.
    X-Title — из примера преподавателя, X-OpenRouter-Title — из текущей документации."""
    if not is_openrouter(base_url):
        return None
    return {"HTTP-Referer": APP_URL, "X-Title": title, "X-OpenRouter-Title": title}


def tls_verify(use_system_certs: bool) -> ssl.SSLContext | bool:
    """Чем проверять HTTPS: True — certifi (по умолчанию httpx), SSLContext truststore —
    хранилище сертификатов Windows / macOS / Linux. Проверка не отключается никогда."""
    if not use_system_certs:
        return True
    try:
        import truststore
    except ImportError as exc:   # в Docker-образе пакета нет: там сервис ходит к Ollama без TLS
        raise RuntimeError("LLM__USE_SYSTEM_CERTS=true требует пакет truststore: "
                           "pip install -r requirements.txt") from exc
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def http_client_options(proxy: str | None, use_system_certs: bool) -> dict[str, Any]:
    """Параметры httpx-клиента до провайдера; пустой словарь — клиент SDK по умолчанию."""
    options: dict[str, Any] = {}
    if proxy:
        options["proxy"] = proxy
    if use_system_certs:
        options["verify"] = tls_verify(True)
    return options


def proxy_display(proxy_url: str) -> str:
    """Адрес прокси для лога — без логина и пароля."""
    parts = urlsplit(proxy_url)
    host = parts.hostname or ""
    return f"{parts.scheme}://{host}:{parts.port}" if parts.port else f"{parts.scheme}://{host}"


class SupportSettings(BaseModel):
    """Ассистент техподдержки в /chat (блок 3.7): системный промпт и статьи руководства.

    Если в запросе нет сообщения system, сервис сам добавляет промпт ассистента со
    статьями руководства, найденными по вопросу пользователя. Свой system в запросе
    важнее: тогда сообщения уходят модели как есть. Переменные — с префиксом SUPPORT__.
    """

    enabled: bool = True
    product_name: str = "Личный кабинет"
    knowledge_base_path: Path = ROOT / "data" / "knowledge_base.json"
    top_k: int = Field(default=3, ge=1, le=10)        # сколько статей подставлять
    max_sentences: int = Field(default=5, ge=1, le=20)  # предел длины ответа в промпте


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
    # Трейсинг в Phoenix (блок 3.6). Не задан — трейсинг выключен; в compose:
    # http://phoenix:6006, для локального uvicorn с Phoenix из compose: http://127.0.0.1:6006.
    phoenix_collector_endpoint: str | None = None
    phoenix_project_name: str = "diploma-fastapi"
    # Опционально: имена и адреса в prompt_preview маскирует Presidio (поверх regex).
    # Нужны пакеты presidio-* и модель ru_core_news_md — см. app/observability/pii_presidio.py.
    pii_presidio: bool = False
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = Field(default=3600, ge=1)
    # Адреса фронтенда, которым браузер разрешит обращаться к API. В .env — JSON-список.
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    # Куки и заголовок Authorization с другого origin. Сервису пока не нужны.
    cors_allow_credentials: bool = False
    # Пустой словарь проверяется при старте, поэтому без ключа ошибка указывает
    # точное поле: llm.openai_api_key — Field required.
    llm: LLMSettings = Field(default_factory=dict, validate_default=True)  # type: ignore[arg-type]
    support: SupportSettings = Field(default_factory=SupportSettings)

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
