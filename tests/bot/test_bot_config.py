"""Настройки и запуск бота (блок 4.2): .env, токен, HTTPS до Telegram."""
from __future__ import annotations

import ssl
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from bot.config import BotSettings
from bot.services.telegram import TelegramSession


def test_env_file_and_admin_ids(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("BOT_TOKEN=123:abc\nBACKEND_URL=http://127.0.0.1:8000/\nBOT_ADMIN_IDS=111, 222\n"
                   "LLM__DEFAULT_MODEL=llama3.2\nBOT_PROXY_URL=\n", encoding="utf-8")
    settings = BotSettings(_env_file=env)
    assert settings.bot_token == SecretStr("123:abc")                  # при ошибке pytest покажет '**********'
    assert settings.backend_url == "http://127.0.0.1:8000"             # без «/» в конце
    assert settings.bot_admin_ids == [111, 222]
    assert settings.bot_proxy_url is None                              # пустое значение — не задано
    assert "123:abc" not in repr(settings)                             # токен не попадёт в лог


@pytest.mark.parametrize("raw, expected", [("[1, 2]", [1, 2]), ("7", [7]), ("", [])])
def test_admin_ids_formats(raw, expected, monkeypatch):
    monkeypatch.setenv("BOT_ADMIN_IDS", raw)
    assert BotSettings(bot_token="1:x", _env_file=None).bot_admin_ids == expected


def test_missing_token_is_an_error(monkeypatch):
    monkeypatch.delenv("BOT_TOKEN", raising=False)
    with pytest.raises(ValidationError) as caught:
        BotSettings(_env_file=None)
    assert caught.value.errors()[0]["loc"] == ("bot_token",)


def test_backend_url_must_be_http():
    with pytest.raises(ValidationError):
        BotSettings(bot_token="1:x", backend_url="127.0.0.1:8000", _env_file=None)


def test_main_explains_missing_token(monkeypatch, capsys):
    import bot.__main__ as entry

    monkeypatch.delenv("BOT_TOKEN", raising=False)
    monkeypatch.setattr(entry, "BotSettings", lambda: BotSettings(_env_file=None))
    assert entry.main() == 2
    assert "BOT_TOKEN" in capsys.readouterr().err


def test_telegram_session_uses_system_certificates():
    import truststore

    system = TelegramSession(use_system_certs=True)
    assert isinstance(system._connector_init["ssl"], truststore.SSLContext)
    plain = TelegramSession(use_system_certs=False)
    assert isinstance(plain._connector_init["ssl"], ssl.SSLContext)
    assert not isinstance(plain._connector_init["ssl"], truststore.SSLContext)
    assert plain._connector_init["ssl"].verify_mode == ssl.CERT_REQUIRED      # проверка не отключается


def test_telegram_session_with_proxy_keeps_system_certificates():
    pytest.importorskip("aiohttp_socks")
    import truststore

    session = TelegramSession(proxy="http://user:secret@proxy.local:3128", use_system_certs=True)
    assert session._connector_type.__name__ == "ProxyConnector"
    assert isinstance(session._connector_init["ssl"], truststore.SSLContext)


@pytest.mark.parametrize("proxy", ["http://user:secret@proxy.local", "socks5h://proxy.local:1080",
                                   "https://proxy.local:443", "proxy.local:3128"])
def test_bad_proxy_url_is_rejected_without_leaking_it(proxy):
    with pytest.raises(ValidationError) as caught:
        BotSettings(bot_token="1:x", bot_proxy_url=proxy, _env_file=None)
    assert "secret" not in str(caught.value) and "proxy.local" not in str(caught.value)


@pytest.mark.parametrize("proxy", ["http://user:secret@proxy.local:3128", "socks5://127.0.0.1:1080"])
def test_good_proxy_url(proxy):
    assert BotSettings(bot_token="1:x", bot_proxy_url=proxy, _env_file=None).proxy() == proxy


def test_main_explains_bad_token(monkeypatch, capsys):
    import bot.__main__ as entry

    monkeypatch.setattr(entry, "BotSettings", lambda: BotSettings(bot_token="не токен", _env_file=None))
    assert entry.main() == 1
    assert "BOT_TOKEN" in capsys.readouterr().err


def test_network_hint_tells_proxy_from_certificate():
    from bot.__main__ import network_hint

    direct = BotSettings(bot_token="1:x", _env_file=None)
    timeout = "ClientConnectorError: Cannot connect to host api.telegram.org:443 ssl:default [Превышен таймаут семафора]"
    assert "BOT_PROXY_URL=http://" in network_hint(timeout, direct)                 # как на Windows без прокси
    certificate = "ClientConnectorCertificateError: ... [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
    assert "BOT_USE_SYSTEM_CERTS" in network_hint(certificate, direct)
    proxied = BotSettings(bot_token="1:x", bot_proxy_url="http://u:secret@proxy.local:3128", _env_file=None)
    hint = network_hint(timeout, proxied)
    assert "Через прокси BOT_PROXY_URL" in hint and "secret" not in hint
