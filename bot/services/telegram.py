"""
Соединение с Telegram Bot API (блок 4.2): проверка HTTPS и прокси.

AiohttpSession aiogram проверяет сертификат api.telegram.org по certifi. В сети, которая
проверяет HTTPS своим сертификатом (как у LLM__USE_SYSTEM_CERTS сервиса), certifi его не
знает и бот не стартует с CERTIFICATE_VERIFY_FAILED. TelegramSession при
BOT_USE_SYSTEM_CERTS=true проверяет сертификат по хранилищу ОС через truststore —
Windows, macOS, Linux. Проверка не отключается никогда.

BOT_PROXY_URL (http:// или socks5://) — прокси, если api.telegram.org напрямую недоступен.
aiogram подключает его через aiohttp-socks.

BOT_EXTRA_CA_FILE (блок 4.4) — дополнительные корневые сертификаты (PEM) к хранилищу ОС. Нужен
в Docker: у контейнера своё хранилище Linux, и сертификата, которым сеть или антивирус
подменяют HTTPS, в нём нет — бот падал с CERTIFICATE_VERIFY_FAILED, хотя на Windows работал.
Файл — выгрузка корневых сертификатов Windows (docs/bot.md, «Бот в Docker»): контейнер
доверяет тем же сертификатам, что и компьютер. Проверка по-прежнему не отключается.
"""
from __future__ import annotations

import logging
import ssl
from pathlib import Path

from aiogram.client.session.aiohttp import AiohttpSession

log = logging.getLogger(__name__)


def system_ssl_context() -> ssl.SSLContext | None:
    """SSLContext с хранилищем сертификатов ОС или None, если truststore не установлен."""
    try:
        import truststore
    except ImportError:
        log.warning("truststore не установлен: HTTPS до Telegram проверяется по certifi "
                    "(pip install -r requirements.txt)")
        return None
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def proxy_errors() -> tuple[type[Exception], ...]:
    """Ошибки прокси aiohttp-socks: aiogram их не оборачивает в TelegramNetworkError."""
    try:
        from aiohttp_socks import ProxyConnectionError, ProxyError, ProxyTimeoutError
    except ImportError:
        return ()
    return ProxyError, ProxyConnectionError, ProxyTimeoutError


class TelegramSession(AiohttpSession):
    def __init__(self, *, proxy: str | None = None, use_system_certs: bool = True,
                 extra_ca_file: Path | None = None) -> None:
        super().__init__(proxy=proxy)
        context = system_ssl_context() if use_system_certs else None
        if context is not None:
            # Параметры коннектора aiohttp, который AiohttpSession создаёт в create_session().
            # С прокси их собирает aiohttp-socks без ssl — тогда ssl добавляется сюда же.
            self._connector_init["ssl"] = context
        if extra_ca_file is not None:
            current = self._connector_init.get("ssl")
            if not isinstance(current, ssl.SSLContext):
                current = self._connector_init["ssl"] = ssl.create_default_context()
            current.load_verify_locations(cafile=str(extra_ca_file))     # к хранилищу ОС, а не вместо
            log.info("extra_ca_loaded file=%s", extra_ca_file.name)
