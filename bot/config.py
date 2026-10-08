"""
Настройки бота (блок 4.2) — pydantic-settings, из переменных окружения и файла .env в
корне проекта (тот же .env, что у сервиса; он в .gitignore, в репозиторий не попадает).

    BOT_TOKEN=123456:ABC...          токен от @BotFather — только в .env
    BACKEND_URL=http://127.0.0.1:8000
    BOT_ADMIN_IDS=111111111,222222222   Telegram user id администраторов: команда /status

Остальное — необязательное, со значениями по умолчанию:
    BACKEND_TIMEOUT=30           с; httpx: подключение, ожидание первого и каждого
                                 следующего фрагмента ответа
    BOT_USE_SYSTEM_CERTS=true    HTTPS до Telegram проверяется по хранилищу сертификатов ОС
                                 (truststore), а не по certifi — как LLM__USE_SYSTEM_CERTS
                                 у сервиса: в сети с проверкой HTTPS своим сертификатом
                                 certifi его не знает. Проверка не отключается никогда.
    BOT_PROXY_URL=               прокси до api.telegram.org, если напрямую он недоступен:
                                 http://user:pass@host:port или socks5://...
    BOT_PRODUCT_NAME=Личный кабинет   название продукта в приветствии
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
PROXY_SCHEMES = ("http", "socks4", "socks5")


class BotSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",            # в общем .env есть и переменные сервиса
        env_ignore_empty=True,     # BOT_PROXY_URL= (пусто) — то же, что не задано
        hide_input_in_errors=True, # в тексте ошибки не будет токена или пароля прокси
    )

    bot_token: SecretStr = Field(description="Токен от @BotFather")
    backend_url: str = Field(default="http://127.0.0.1:8000", description="Адрес chat-сервиса блока 4.1")
    bot_admin_ids: Annotated[list[int], NoDecode] = Field(default_factory=list)
    backend_timeout: float = Field(default=30.0, ge=1, le=600)
    bot_use_system_certs: bool = True
    bot_proxy_url: SecretStr | None = None
    bot_product_name: str = "Личный кабинет"

    @field_validator("backend_url")
    @classmethod
    def _url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("ожидается адрес вида http://127.0.0.1:8000")
        return value

    @field_validator("bot_admin_ids", mode="before")
    @classmethod
    def _admin_ids(cls, value: Any) -> Any:
        """BOT_ADMIN_IDS=111,222 или [111, 222] или одно число."""
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            text = value.strip().strip("[]")
            return [int(part) for part in text.replace(";", ",").split(",") if part.strip()]
        return value

    @field_validator("bot_proxy_url")
    @classmethod
    def _proxy(cls, value: SecretStr | None) -> SecretStr | None:
        """Схемы, которые понимает aiohttp-socks, и обязательный порт. Значение в ошибку не
        попадает: в адресе прокси бывает пароль."""
        if value is None:
            return None
        parts = urlsplit(value.get_secret_value())
        try:
            port = parts.port
        except ValueError:
            port = None
        if parts.scheme not in PROXY_SCHEMES or not parts.hostname or port is None:
            raise ValueError("ожидается http://[user:pass@]host:port, socks4://… или socks5://… — с портом")
        return value

    def proxy(self) -> str | None:
        return self.bot_proxy_url.get_secret_value() if self.bot_proxy_url else None
