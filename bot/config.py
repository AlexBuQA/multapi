"""
Настройки бота (блок 4.2) — pydantic-settings, из переменных окружения и файла .env в
корне проекта (тот же .env, что у сервиса; он в .gitignore, в репозиторий не попадает).

    BOT_TOKEN=123456:ABC...          токен от @BotFather — только в .env
    BACKEND_URL=http://127.0.0.1:8000
    BOT_ADMIN_IDS=111111111,222222222   Telegram user id администраторов: команда /status

Остальное — необязательное, со значениями по умолчанию:
    BACKEND_TIMEOUT=60           с; httpx: ожидание ответа на обычный запрос (/chats, /clear)
    BACKEND_STREAM_TIMEOUT=120   с; ожидание первого и каждого следующего фрагмента ответа
                                 модели (блок 4.3: фото на CPU модель разглядывает дольше).
                                 Подключение — 3 с, отправка запроса — 10 с
    BOT_USE_SYSTEM_CERTS=true    HTTPS до Telegram проверяется по хранилищу сертификатов ОС
                                 (truststore), а не по certifi — как LLM__USE_SYSTEM_CERTS
                                 у сервиса: в сети с проверкой HTTPS своим сертификатом
                                 certifi его не знает. Проверка не отключается никогда.
    BOT_PROXY_URL=               прокси до api.telegram.org, если напрямую он недоступен:
                                 http://user:pass@host:port или socks5://...
    BOT_PRODUCT_NAME=Личный кабинет   название продукта в приветствии
    BOT_STREAMING=draft          как показывать ответ (блок 4.3): draft — черновик
                                 sendMessageDraft (только личные чаты, в группах — правки),
                                 edit — правки сообщения, как в блоке 4.2
    BOT_DEFAULT_USER_NAME=Александра   как модель обращается к пользователю, пока он сам не
                                 представился (блок 4.3); «-» — не подсказывать имя

HTTP-API бота для уведомлений из сервиса (блок 4.3, bot/web.py):
    INTERNAL_TOKEN=              общий секрет с сервисом (заголовок X-Internal-Token). Не
                                 задан — API не запускается, бот работает как раньше
    BOT_API_HOST=127.0.0.1       где слушать: 127.0.0.1 — только этот компьютер; в Docker —
                                 0.0.0.0, чтобы сервис из другого контейнера достучался
    BOT_API_PORT=9000            тот же порт, что в BOT_URL сервиса
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
PROXY_SCHEMES = ("http", "socks4", "socks5")
DEFAULT_USER_NAME = "Александра"
# То же правило, что у сервиса (app/chat/service.py, USER_NAME): одно-три слова из букв, до 40
# знаков. Неверное имя — ошибка при старте, а не 422 на каждый вопрос.
_NAME_WORD = r"[^\W\d_]+(?:['’][^\W\d_]+)*"
USER_NAME = re.compile(rf"{_NAME_WORD}(?:[ -]{_NAME_WORD}){{0,2}}")


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
    backend_timeout: float = Field(default=60.0, ge=1, le=600)
    backend_stream_timeout: float = Field(default=120.0, ge=1, le=1800)
    bot_use_system_certs: bool = True
    bot_proxy_url: SecretStr | None = None
    bot_product_name: str = "Личный кабинет"
    bot_streaming: Literal["draft", "edit"] = "draft"
    bot_default_user_name: str | None = DEFAULT_USER_NAME
    internal_token: SecretStr | None = None
    bot_api_host: str = "127.0.0.1"
    bot_api_port: int = Field(default=9000, ge=1, le=65535)

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

    @field_validator("bot_default_user_name")
    @classmethod
    def _user_name(cls, value: str | None) -> str | None:
        """«-» — имя не подсказывать: модель обращается нейтрально, пока пользователь не
        представится."""
        name = " ".join(unicodedata.normalize("NFC", value or "").split())
        if name in ("", "-"):
            return None
        if len(name) > 40 or not USER_NAME.fullmatch(name):
            raise ValueError("BOT_DEFAULT_USER_NAME: ожидается имя — одно-три слова из букв, до 40 знаков; "
                             "«-» — без имени")
        return name

    @field_validator("internal_token")
    @classmethod
    def _token(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 16:
            raise ValueError("INTERNAL_TOKEN короче 16 символов: сгенерируйте длинный, например "
                             "python -c \"import secrets; print(secrets.token_urlsafe(32))\"")
        return value

    def proxy(self) -> str | None:
        return self.bot_proxy_url.get_secret_value() if self.bot_proxy_url else None
