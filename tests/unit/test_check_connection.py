"""
Диагностика связи с судьёй (eval/check_connection.py) и первопричина ошибки судьи.

OpenAI SDK на любую сетевую проблему отвечает «Connection error.». На Windows за ним
оказалась неизвестная причина при судье OpenRouter через прокси преподавателя; здесь —
что команда различает: прокси не пускает к сайту, неверный пароль прокси, неверный ключ.
Сеть не нужна: ответы подставляет httpx.MockTransport.
"""
from __future__ import annotations

import httpx

import check_connection
from run_evaluation import describe_exception

BASE = "https://openrouter.ai/api/v1"
PROXY = "http://student:secret@proxy.example:8888"
KEY_JSON = {"data": {"label": "sk-or-v1-abc...", "is_free_tier": True, "usage": 0, "limit": None}}


def factory_for(via_proxy, direct):
    """via_proxy / direct: функция request -> Response (или исключение) для каждого пути."""
    def factory(proxy):
        handler = via_proxy if proxy else direct
        return httpx.Client(transport=httpx.MockTransport(handler))
    return factory


def ok(request: httpx.Request) -> httpx.Response:
    if request.url.host == "api.ipify.org":
        return httpx.Response(200, json={"ip": "203.0.113.10"})
    if request.url.path.endswith("/key"):
        return httpx.Response(200, json=KEY_JSON)
    return httpx.Response(200, json={"data": []})


def proxy_denies_openrouter(request: httpx.Request) -> httpx.Response:
    if request.url.host == "api.ipify.org":
        return ok(request)
    raise httpx.ProxyError("403 Forbidden")


def test_proxy_blocks_openrouter_but_direct_works():
    lines, code = check_connection.run_checks(BASE, PROXY, "sk-or-test", factory_for(proxy_denies_openrouter, ok))
    text = "\n".join(lines)
    assert code == 0
    assert "[OK]   1. прокси работает (api.ipify.org): внешний IP 203.0.113.10" in text
    assert "[FAIL] 2. через прокси до openrouter.ai: ProxyError: 403 Forbidden" in text
    assert "[OK]   3. напрямую до openrouter.ai" in text
    assert "LLM__PROXY_URL= пустым" in text
    assert "secret" not in text                                   # пароль прокси не печатается


def test_everything_through_proxy():
    lines, code = check_connection.run_checks(BASE, PROXY, "sk-or-test", factory_for(ok, ok))
    text = "\n".join(lines)
    assert code == 0 and "через прокси — .env менять не нужно" in text
    assert "is_free_tier=True" in text and "sk-or-test" not in text


def test_wrong_proxy_password_and_no_direct_route():
    def denied(request):
        raise httpx.ProxyError("407 Proxy Authentication Required")

    def unreachable(request):
        raise httpx.ConnectError("All connection attempts failed")

    lines, code = check_connection.run_checks(BASE, PROXY, "sk-or-test", factory_for(denied, unreachable))
    assert code == 1
    assert "Прокси отклонил логин или пароль (407)" in lines[-1]
    assert any(line.startswith("[SKIP] 4. ключ") for line in lines)


def test_rejected_key():
    def bad_key(request):
        if request.url.path.endswith("/key"):
            return httpx.Response(401, json={"error": {"message": "No auth credentials found"}})
        return ok(request)

    lines, code = check_connection.run_checks(BASE, None, "sk-or-bad", factory_for(ok, bad_key))
    assert code == 1
    assert "[FAIL] 4. ключ принят: HTTP 401" in lines
    assert "ключ не принят" in lines[-1]


def test_judge_error_shows_root_cause():
    try:
        try:
            raise httpx.ProxyError("407 Proxy Authentication Required")
        except httpx.ProxyError as inner:
            raise RuntimeError("Connection error.") from inner
    except RuntimeError as exc:
        assert describe_exception(exc) == "RuntimeError: Connection error. <- ProxyError: 407 Proxy Authentication Required"


# ---------------------------------------------------------------- сертификаты: хранилище ОС
SSL_FAIL = ("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
            "self-signed certificate in certificate chain (_ssl.c:1035)")


def test_certificate_interception_suggests_system_store():
    """Windows-прогон: все HTTPS-проверки — CERTIFICATE_VERIFY_FAILED, и через прокси, и
    напрямую: сеть или антивирус подставляют свой корневой сертификат. С хранилищем ОС
    (как у браузера) проверка проходит — команда советует LLM__USE_SYSTEM_CERTS=true."""
    import ssl

    def intercepted(request):
        raise httpx.ConnectError(SSL_FAIL)

    def factory_for(verify):
        handler = ok if isinstance(verify, ssl.SSLContext) else intercepted
        return lambda proxy: httpx.Client(transport=httpx.MockTransport(handler))

    lines, code = check_connection.diagnose(BASE, PROXY, "sk-or-test", False, factory_for)
    text = "\n".join(lines)
    assert code == 1                                     # с текущими настройками судья недоступен
    assert "повторяю с хранилищем сертификатов ОС" in text
    assert text.rstrip().endswith("Добавьте в .env: LLM__USE_SYSTEM_CERTS=true")
    assert "через прокси — .env менять не нужно" in text


def test_system_store_enabled_no_retry():
    import ssl

    seen = []

    def factory_for(verify):
        seen.append(verify)
        return lambda proxy: httpx.Client(transport=httpx.MockTransport(ok))

    lines, code = check_connection.diagnose(BASE, PROXY, "sk-or-test", True, factory_for)
    assert code == 0 and len(seen) == 1 and isinstance(seen[0], ssl.SSLContext)


def test_tls_verify_never_disables_checks():
    import ssl

    from app.core.config import http_client_options, tls_verify

    assert tls_verify(False) is True
    assert isinstance(tls_verify(True), ssl.SSLContext) and tls_verify(True).verify_mode == ssl.CERT_REQUIRED
    assert http_client_options(None, False) == {}
    options = http_client_options(PROXY, True)
    assert options["proxy"] == PROXY and isinstance(options["verify"], ssl.SSLContext)
