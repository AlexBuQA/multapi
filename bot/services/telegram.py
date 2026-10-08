"""
Соединение с Telegram Bot API (блок 4.2): проверка HTTPS и прокси.

AiohttpSession aiogram проверяет сертификат api.telegram.org по certifi. В сети, которая
проверяет HTTPS своим сертификатом (как у LLM__USE_SYSTEM_CERTS сервиса), certifi его не
знает и бот не стартует с CERTIFICATE_VERIFY_FAILED. TelegramSession при
BOT_USE_SYSTEM_CERTS=true проверяет сертификат по хранилищу ОС через truststore —
Windows, macOS, Linux. Проверка не отключается никогда.

BOT_PROXY_URL (http:// или socks5://) — прокси, если api.telegram.org напрямую недоступен.
aiogram подключает его через aiohttp-socks.
"""
from __future__ import annotations

import logging
import ssl

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
    def __init__(self, *, proxy: str | None = None, use_system_certs: bool = True) -> None:
        super().__init__(proxy=proxy)
        context = system_ssl_context() if use_system_certs else None
        if context is not None:
            # Параметры коннектора aiohttp, который AiohttpSession создаёт в create_session().
            # С прокси их собирает aiohttp-socks без ssl — тогда ssl добавляется сюда же.
            self._connector_init["ssl"] = context
