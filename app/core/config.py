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

Чат с историей на сервере (блок 4.1): CHAT_REPOSITORY (json или postgres),
CHAT_STORAGE_DIR, DATABASE_URL, стратегия и окно контекста, бюджет токенов
CONTEXT_WINDOW - RESPONSE_TOKENS - SAFETY_MARGIN — см. app/chat/ и docs/chat.md.

Медиа в чатах (блок 4.3, app/chat/media.py): CHAT_VISION_MODEL — модель для запросов с
картинкой (текстовая llama3.2 картинок не видит), AUDIO_API_KEY / AUDIO_BASE_URL /
WHISPER_MODEL — расшифровка голоса (те же переменные, что у голосового пайплайна блока
2.6), MEDIA__* — пределы размера файлов и длины текста документа.

Уведомления из сервиса в Telegram (блок 4.3, app/services/notifier.py): BOT_URL — адрес
HTTP-API бота, INTERNAL_TOKEN — общий секрет сервиса и бота (заголовок X-Internal-Token).

Настройки ассистента с инструментами (блок 3.1) и скриптов блока 3.3 — в app/config.py.
"""
from __future__ import annotations

import ipaddress
import os
import re
import ssl
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Literal
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator, model_validator
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


class SecuritySettings(BaseModel):
    """Защитный слой /chat (блок 3.8): проверка входа, канарейка, проверка ответа.

    Переменные — с префиксом SECURITY__. SECURITY__ENABLED=false выключает слой целиком:
    так снимается garak baseline на «голом» сервисе. Маскирование персональных данных на
    входе модели и в логах от него не зависит — оно работает всегда.
    """

    enabled: bool = True
    # Длиннее — готовый отказ без вызова модели. Схема запроса пускает до 32 000 символов
    # (блок 3.4), но вопрос в поддержку — десятки и сотни; длинный текст — обычно джейлбрейк.
    max_input_chars: int = Field(default=4000, ge=100, le=32_000)


class MediaSettings(BaseModel):
    """Медиа в POST /chats/{id}/messages (блок 4.3). Переменные — с префиксом MEDIA__.

    Пределы — на стороне сервиса, независимо от клиента: бот сам выбирает фото до 2 МБ и
    документы до 10 МБ, но в сервис может прийти запрос и не от бота.
    """

    max_image_bytes: int = Field(default=5 * 1024 * 1024, ge=1024)
    max_audio_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)      # Whisper принимает до 25 МБ
    max_document_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    max_document_chars: int = Field(default=30_000, ge=1000, le=30_000)  # текст PDF/DOCX и расшифровки
    max_pdf_pages: int = Field(default=50, ge=1, le=500)
    # Сколько токенов считать за картинку в бюджете контекста: у разных моделей по-разному
    # (gemma3 — 256, gpt-4o — от 85 до ~1100 по размеру), берём с запасом.
    image_tokens: int = Field(default=800, ge=64, le=4000)
    audio_timeout: float = Field(default=120.0, gt=0)                   # с на расшифровку
    parse_timeout: float = Field(default=30.0, gt=0, le=600)            # с на разбор PDF/DOCX


class ModerationSettings(BaseModel):
    """Модерация в чатах (блок 4.4), app/moderation/. Переменные — с префиксом MODERATION__.

    Слой ключевых слов работает всегда, когда ENABLED=true; OpenAI Moderation — если
    OPENAI_ENABLED=true и есть ключ (OPENAI_API_KEY или AUDIO_API_KEY).
    """

    enabled: bool = True
    keywords_file: Path = ROOT / "app" / "moderation" / "moderation_keywords.yaml"
    openai_enabled: bool = False
    openai_api_key: SecretStr | None = None          # пусто — AUDIO_API_KEY (тоже ключ OpenAI)
    openai_base_url: str | None = None               # пусто — https://api.openai.com/v1
    openai_model: str = "omni-moderation-latest"
    # Свои пороги по категориям OpenAI, JSON: {"violence": 0.5, "self_harm": 0.2}. Категория
    # без порога блокирует, когда её отметил сам OpenAI (flagged).
    thresholds: dict[str, float] = Field(default_factory=dict)
    timeout: float = Field(default=10.0, gt=0, le=120)
    # Ошибка OpenAI Moderation: false — пропустить (ключевые слова уже проверены), true — блокировать.
    fail_closed: bool = False

    @field_validator("keywords_file")
    @classmethod
    def _keywords_from_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else ROOT / value

    @field_validator("thresholds")
    @classmethod
    def _thresholds(cls, value: dict[str, float]) -> dict[str, float]:
        for name, threshold in value.items():
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"MODERATION__THRESHOLDS: порог {name} — число от 0 до 1")
        return value


# Postgres из compose.yaml, опубликованный на 127.0.0.1:5433 (а не localhost: на Windows
# localhost сначала пробует IPv6 ::1, и каждое новое соединение ждёт отказа по нему).
DEFAULT_DATABASE_URL = "postgresql+asyncpg://multapi:multapi@127.0.0.1:5433/multapi"

ENV_CONFIG = SettingsConfigDict(
    env_file=ROOT / ".env",
    env_file_encoding="utf-8",
    env_nested_delimiter="__",
    env_ignore_empty=True,   # «КЛЮЧ=» в .env — то же, что ключа нет: действует значение по умолчанию
    extra="ignore",          # в .env есть переменные блоков 2–3, сервису они не нужны
)


def async_database_url(raw: str) -> str:
    """Адрес Postgres для async SQLAlchemy: postgresql://… и postgres://… (как их пишут в
    примерах и в Docker) -> postgresql+asyncpg://…. Сервис и Alembic работают через asyncpg;
    с другим драйвером create_async_engine упал бы на импорте psycopg."""
    from sqlalchemy.engine import make_url

    url = make_url(raw)
    if url.drivername in {"postgres", "postgresql"} or (
            url.drivername.startswith("postgresql+") and url.drivername != "postgresql+asyncpg"):
        url = url.set(drivername="postgresql+asyncpg")
    return url.render_as_string(hide_password=False)


class DatabaseSettings(BaseSettings):
    """Только DATABASE_URL — для Alembic (migrations/env.py): миграциям не нужен ключ
    провайдера LLM, без которого Settings не создаётся."""

    model_config = ENV_CONFIG
    database_url: SecretStr = SecretStr(DEFAULT_DATABASE_URL)


DEFAULT_CHAT_SYSTEM_PROMPT = (
    "Ты — ассистент техподдержки продукта «{product_name}». Отвечай по-русски, коротко и по делу. "
    "Помни, что пользователь сообщил о себе раньше в этом диалоге, — например, как его зовут, — "
    "и используй это в ответах. Не выдумывай сведений, которых в диалоге не было. "
    "Email, телефон и номера карт в сообщениях скрыты метками вида [EMAIL], [PHONE_RU], [CARD] — "
    "не проси прислать их снова."
)


class Settings(BaseSettings):
    model_config = ENV_CONFIG

    app_name: str = "multapi — LLM-сервис техподдержки"
    # Уровень лога сервиса (логгер llm-service): DEBUG, INFO, WARNING, ERROR.
    log_level: str = "INFO"
    # Копия JSON-лога в файл (блок 3.8), например logs/service.jsonl. Пусто — только консоль.
    log_file: Path | None = None
    # Лимит запросов к /chat и /chat/stream в минуту на X-User-ID или IP (блок 3.8).
    # 0 — без лимита. Счётчики — в Redis; Redis недоступен — запрос пропускается.
    rate_limit_per_min: int = Field(default=0, ge=0)
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
    security: SecuritySettings = Field(default_factory=SecuritySettings)

    # --- Чат с историей на сервере (блок 4.1), app/chat/ ---
    # Где хранить историю: json — файлы в CHAT_STORAGE_DIR, postgres — DATABASE_URL.
    chat_repository: Literal["json", "postgres"] = "json"
    # Относительный путь — от корня проекта, как LOG_FILE.
    chat_storage_dir: Path = Path("./var/chats")
    # sliding — последние CHAT_CONTEXT_WINDOW сообщений. hybrid (сводка + последние M) в 4.1
    # не реализована: с ней сервис не стартует (app/chat/context.py).
    chat_context_strategy: Literal["sliding", "hybrid"] = "sliding"
    # Сколько последних сообщений чата уходит модели. Не больше 48: в ChatRequest до 50
    # сообщений, ещё одно — системный промпт.
    chat_context_window: int = Field(default=10, ge=1, le=48)
    # Системный промпт чата, если при создании свой не задан; {product_name} — SUPPORT__PRODUCT_NAME.
    chat_system_prompt: str = DEFAULT_CHAT_SYSTEM_PROMPT
    # Postgres для CHAT_REPOSITORY=postgres; по умолчанию — сервис postgres из compose.yaml.
    # SecretStr: в адресе пароль, в лог он не попадает.
    database_url: SecretStr = SecretStr(DEFAULT_DATABASE_URL)
    # Бюджет токенов запроса: CONTEXT_WINDOW - RESPONSE_TOKENS - SAFETY_MARGIN. CONTEXT_WINDOW —
    # окно модели; у Ollama это num_ctx (по умолчанию 4096 в новых версиях), всё сверх него
    # Ollama молча отрезает с начала — вместе с системным промптом.
    context_window: int = Field(default=4096, ge=512)
    response_tokens: int = Field(default=1024, ge=16, le=16_000)   # max_tokens ответа
    safety_margin: int = Field(default=256, ge=0)                   # запас на неточность подсчёта

    # --- Медиа в чатах (блок 4.3), app/chat/media.py ---
    # Модель для запросов, в контексте которых есть картинка. Не задана — LLM__DEFAULT_MODEL,
    # если она видит изображения (gpt-4o-mini да, llama3.2 нет — тогда картинка отклоняется
    # с подсказкой). Для Ollama: gemma3:4b, qwen2.5vl:3b, llama3.2-vision. Читается и
    # SUPPORT_VISION_MODEL — та же модель, что у демо Vision блока 2.6.
    chat_vision_model: str | None = Field(
        default=None, validation_alias=AliasChoices("CHAT_VISION_MODEL", "SUPPORT_VISION_MODEL"))
    media: MediaSettings = Field(default_factory=MediaSettings)
    # Расшифровка голоса — OpenAI-совместимый /audio/transcriptions (Whisper). Ollama его не
    # умеет. Ключ не задан — голосовые сообщения получают понятный отказ 503.
    audio_api_key: SecretStr | None = None
    audio_base_url: str | None = None          # пусто — https://api.openai.com/v1
    whisper_model: str = "whisper-1"
    audio_language: str | None = "ru"          # подсказка языка для Whisper; auto — определит сам

    # --- Уведомления в Telegram (блок 4.3), app/services/notifier.py ---
    # HTTP-API бота (bot/web.py) и общий секрет: сервис шлёт его в X-Internal-Token, бот
    # сверяет. Секрет — только в .env. Не задан — /system-message и уведомления выключены.
    bot_url: str = "http://127.0.0.1:9000"
    internal_token: SecretStr | None = None
    bot_api_port: int = Field(default=9000, ge=1, le=65535)    # тот же порт слушает бот

    # --- Production-обвязка (блок 4.4) ---
    moderation: ModerationSettings = Field(default_factory=ModerationSettings)
    # Токен admin-эндпоинтов /chats/admin/* (заголовок X-Admin-Token); тот же — у бота для
    # /stats, /users, /broadcast и рассылки. Не задан — admin-эндпоинты отвечают 503.
    admin_token: SecretStr | None = None

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, value: str) -> str:
        value = value.upper()
        if value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL: ожидается DEBUG, INFO, WARNING, ERROR или CRITICAL")
        return value

    @field_validator("log_file")
    @classmethod
    def _log_file_from_root(cls, value: Path | None) -> Path | None:
        """Относительный путь — от корня проекта, как и .env: лог не зависит от того, из
        какой папки запущен uvicorn. os.devnull (тесты) не трогаем."""
        if value is None or value.is_absolute() or str(value) == os.devnull:
            return value
        return ROOT / value

    @field_validator("chat_storage_dir")
    @classmethod
    def _chat_dir_from_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else ROOT / value

    @field_validator("audio_language")
    @classmethod
    def _audio_language(cls, value: str | None) -> str | None:
        # Пустое значение в .env игнорируется (env_ignore_empty) — выключить подсказку можно словом auto.
        return None if value is None or value.strip().lower() in {"", "auto", "none"} else value.strip()

    @field_validator("internal_token")
    @classmethod
    def _internal_token(cls, value: SecretStr | None) -> SecretStr | None:
        # То же правило, что у бота (bot/config.py): короткий общий секрет легко подобрать.
        if value is not None and len(value.get_secret_value()) < 16:
            raise ValueError("INTERNAL_TOKEN короче 16 символов: сгенерируйте длинный, например "
                             "python -c \"import secrets; print(secrets.token_urlsafe(32))\"")
        return value

    @field_validator("admin_token")
    @classmethod
    def _admin_token(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 16:
            raise ValueError("ADMIN_TOKEN короче 16 символов: сгенерируйте длинный, например "
                             "python -c \"import secrets; print(secrets.token_urlsafe(32))\"")
        return value

    @field_validator("bot_url")
    @classmethod
    def _bot_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("BOT_URL: ожидается адрес вида http://127.0.0.1:9000")
        return value

    @property
    def context_budget(self) -> int:
        """Сколько токенов может занять запрос к модели (системный промпт + история)."""
        return self.context_window - self.response_tokens - self.safety_margin

    @model_validator(mode="after")
    def _check_context_budget(self) -> Settings:
        if self.context_budget < 256:
            raise ValueError(
                f"CONTEXT_WINDOW - RESPONSE_TOKENS - SAFETY_MARGIN = {self.context_budget}: на историю чата "
                "остаётся меньше 256 токенов — увеличьте CONTEXT_WINDOW или уменьшите RESPONSE_TOKENS"
            )
        return self

    @model_validator(mode="after")
    def _check_cors(self) -> Settings:
        if self.cors_allow_credentials and "*" in self.cors_origins:
            raise ValueError(
                "CORS: allow_credentials=True нельзя сочетать с allow_origins=['*'] — "
                "перечислите адреса фронтенда в CORS_ORIGINS"
            )
        return self


_COMMENT_VALUE = re.compile(r"#\s")


def commented_values(environ: Mapping[str, str] | None = None) -> list[str]:
    """Переменные, у которых вместо значения — комментарий из .env. docker compose (env_file)
    читает строку «КЛЮЧ=   # пусто => …» как значение «# пусто => …» (python-dotenv так не
    делает): без проверки сервис в контейнере падал на непонятной ошибке разбора, а строковые
    настройки молча получали текст комментария (блок 4.4, проверка на Windows)."""
    env = os.environ if environ is None else environ
    return sorted(name for name, value in env.items() if _COMMENT_VALUE.match(value))


@lru_cache
def get_settings() -> Settings:
    """Один экземпляр настроек на процесс: .env читается при первом вызове."""
    broken = commented_values()
    if broken:
        raise ValueError(f"Вместо значения — комментарий из .env: {', '.join(broken)}. docker compose читает "
                         "строку «КЛЮЧ=   # комментарий» как значение «# комментарий». В .env оставьте КЛЮЧ= "
                         'без комментария или напишите КЛЮЧ="" # комментарий.')
    return Settings()
