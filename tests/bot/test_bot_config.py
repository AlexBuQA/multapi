"""Настройки и запуск бота (блоки 4.2–4.4): .env, токен, HTTPS до Telegram, ADMIN_TOKEN."""
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


def test_admin_token_and_poll(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "admin-token-0123456789")
    settings = BotSettings(bot_token="1:x", _env_file=None)
    assert settings.admin_token == SecretStr("admin-token-0123456789") and settings.bot_broadcast_poll == 5.0
    assert "admin-token-0123456789" not in repr(settings)


def test_short_admin_token_is_rejected_without_leaking_it(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "short-secret")
    with pytest.raises(ValidationError) as caught:
        BotSettings(bot_token="1:x", _env_file=None)
    assert caught.value.errors()[0]["loc"] == ("admin_token",) and "short-secret" not in str(caught.value)


def test_make_backend_passes_admin_token():
    import httpx

    from bot.__main__ import make_backend

    settings = BotSettings(bot_token="1:x", admin_token="admin-token-0123456789", _env_file=None)
    assert make_backend(httpx.AsyncClient(), settings).admin_token == "admin-token-0123456789"
    assert make_backend(httpx.AsyncClient(), BotSettings(bot_token="1:x", _env_file=None)).admin_token is None


def test_commented_value_from_docker_env_file(monkeypatch, capsys):
    """docker compose читает «BOT_BROADCAST_POLL=   # пусто => 5 с» как значение-комментарий."""
    from bot.__main__ import main
    from bot.config import commented_values

    assert commented_values({"A": "# пусто => 5 с", "B": "", "C": "#hash-in-password", "D": "x # y"}) == ["A"]
    monkeypatch.setenv("BOT_BROADCAST_POLL", "# пусто => 5 с: как часто бот проверяет очередь рассылок")
    assert main() == 2
    err = capsys.readouterr().err
    assert "BOT_BROADCAST_POLL" in err and 'КЛЮЧ=""' in err


# ---------------------------------------------------------------- блок 4.4: бот в Docker
TEST_CA = Path(__file__).resolve().parent / "data" / "test-root-ca.pem"   # открытый сертификат, ключа нет


def test_extra_ca_is_added_to_certifi_store():
    """Без хранилища ОС (BOT_USE_SYSTEM_CERTS=false) — к набору certifi добавляется один сертификат."""
    before = TelegramSession(use_system_certs=False)._connector_init["ssl"].cert_store_stats()["x509_ca"]
    after = TelegramSession(use_system_certs=False, extra_ca_file=TEST_CA)._connector_init["ssl"]
    assert after.cert_store_stats()["x509_ca"] == before + 1 and after.verify_mode == ssl.CERT_REQUIRED


def test_extra_ca_is_added_to_system_store(monkeypatch):
    """С truststore — дополнительно к хранилищу ОС, а не вместо него."""
    import truststore

    loaded: list[str] = []
    original = truststore.SSLContext.load_verify_locations

    def spy(self, cafile=None, capath=None, cadata=None):
        loaded.append(cafile)
        return original(self, cafile=cafile, capath=capath, cadata=cadata)

    monkeypatch.setattr(truststore.SSLContext, "load_verify_locations", spy)
    session = TelegramSession(use_system_certs=True, extra_ca_file=TEST_CA)
    assert isinstance(session._connector_init["ssl"], truststore.SSLContext) and loaded == [str(TEST_CA)]


def test_extra_ca_file_setting(tmp_path):
    relative = BotSettings(bot_token="1:x", bot_extra_ca_file="tests/bot/data/test-root-ca.pem", _env_file=None)
    assert relative.bot_extra_ca_file == TEST_CA                        # путь — от корня проекта (в Docker /app)
    assert BotSettings(bot_token="1:x", _env_file=None).bot_extra_ca_file is None
    with pytest.raises(ValidationError, match="нет файла"):
        BotSettings(bot_token="1:x", bot_extra_ca_file=tmp_path / "missing.pem", _env_file=None)
    empty = tmp_path / "empty.pem"
    empty.write_text("не сертификат", encoding="utf-8")
    with pytest.raises(ValidationError, match="нет сертификатов"):
        BotSettings(bot_token="1:x", bot_extra_ca_file=empty, _env_file=None)


def test_certificate_hint_in_container():
    from bot.__main__ import network_hint

    error = "ClientOSError: [Errno 1] [SSL: CERTIFICATE_VERIFY_FAILED] self-signed certificate in certificate chain"
    settings = BotSettings(bot_token="1:x", _env_file=None)
    assert "BOT_EXTRA_CA_FILE=certs/windows-roots.pem" in network_hint(error, settings, container=True)
    assert "BOT_USE_SYSTEM_CERTS" in network_hint(error, settings, container=False)


def test_main_explains_missing_ca_file(monkeypatch, capsys):
    from bot.__main__ import main

    monkeypatch.setenv("BOT_TOKEN", "123:abc")
    monkeypatch.setenv("BOT_EXTRA_CA_FILE", "certs/no-such-file.pem")
    monkeypatch.setenv("BOT_PROXY_URL", "http://user:secret@proxy.local")      # без порта — тоже ошибка
    assert main() == 2
    err = capsys.readouterr().err
    assert "BOT_EXTRA_CA_FILE" in err and "нет файла" in err and "no-such-file.pem" in err
    assert "secret" not in err                                         # пароль прокси в текст не попал
