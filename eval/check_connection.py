"""
Проверка связи с судьёй eval (блок 3.7) — без вызова моделей: лимит бесплатных
запросов OpenRouter не тратится. Запуск из корня проекта:

    python eval/check_connection.py

Настройки — те же, что у run_evaluation.py: EVAL_JUDGE_BASE_URL и EVAL_JUDGE_API_KEY
(окружение или .env), прокси — LLM__PROXY_URL. Шаги:
1. через прокси — https://api.ipify.org: прокси доступен и принимает логин и пароль;
2. через прокси — {base_url}/models: прокси пускает к провайдеру судьи;
3. напрямую — {base_url}/models: доступен ли провайдер без прокси;
4. ключ: у OpenRouter — GET /key (лимит, остаток, бесплатный ли тариф), у остальных —
   GET /models с ключом.
В конце — что поставить в .env. Код выхода: 0 — судья доступен, 1 — нет.

Если HTTPS не проходит проверку сертификата (CERTIFICATE_VERIFY_FAILED — сеть или
антивирус подставляют свой корневой сертификат), а LLM__USE_SYSTEM_CERTS не включён,
проверка повторяется с хранилищем сертификатов ОС и подсказывает включить его.

Зачем: OpenAI SDK на любую сетевую проблему отвечает одинаково — «Connection error.».
За ним может быть отказ прокси в доступе к сайту (403), неверный пароль прокси (407) или
недоступный адрес; здесь видно, что именно.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
for path in (ROOT, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("LOG_LEVEL", "ERROR")

import httpx  # noqa: E402

from run_evaluation import describe_exception, env_value  # noqa: E402

IP_URL = "https://api.ipify.org?format=json"
KEY_FIELDS = ("is_free_tier", "usage", "limit", "limit_remaining", "rate_limit")

ClientFactory = Callable[[str | None], httpx.Client]


def client_factory(verify: Any = True, timeout: float = 15.0) -> ClientFactory:
    """httpx-клиенты с этой проверкой сертификатов. trust_env=False: «напрямую» значит
    напрямую, без HTTPS_PROXY из окружения Windows."""
    def make(proxy: str | None) -> httpx.Client:
        return httpx.Client(proxy=proxy, timeout=timeout, trust_env=False, follow_redirects=True,
                            verify=verify)
    return make


make_client = client_factory()


def probe(factory: ClientFactory, proxy: str | None, url: str,
          headers: dict[str, str] | None = None) -> tuple[bool, str, httpx.Response | None]:
    try:
        with factory(proxy) as client:
            response = client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        return False, describe_exception(exc), None
    return response.status_code < 400, f"HTTP {response.status_code}", response


def run_checks(base_url: str, proxy: str | None, api_key: str | None,
               factory: ClientFactory = make_client) -> tuple[list[str], int]:
    from app.core.config import is_openrouter, proxy_display

    lines: list[str] = []
    host = urlsplit(base_url).hostname or base_url
    models_url = base_url.rstrip("/") + "/models"

    def report(ok: bool | None, title: str, detail: str) -> None:
        mark = "[SKIP]" if ok is None else "[OK]  " if ok else "[FAIL]"
        lines.append(f"{mark} {title}: {detail}")

    lines.append(f"Судья: {base_url}; прокси: {proxy_display(proxy) if proxy else 'не задан'}")
    via_proxy = direct = False
    proxy_detail = ""
    if proxy:
        ok, detail, response = probe(factory, proxy, IP_URL)
        if ok and response is not None:
            try:
                detail = f"внешний IP {response.json().get('ip')}"
            except ValueError:
                pass
        report(ok, "1. прокси работает (api.ipify.org)", detail)
        via_proxy, proxy_detail, _ = probe(factory, proxy, models_url)
        report(via_proxy, f"2. через прокси до {host}", proxy_detail)
    else:
        report(None, "1–2. прокси", "LLM__PROXY_URL не задан")
    direct, direct_detail, _ = probe(factory, None, models_url)
    report(direct, f"3. напрямую до {host}", direct_detail)

    key_ok = False
    route = proxy if via_proxy else None if direct else False
    if route is False:
        report(None, "4. ключ", f"нет связи с {host}")
    elif not api_key:
        report(False, "4. ключ", "EVAL_JUDGE_API_KEY не задан")
    else:
        auth = {"Authorization": f"Bearer {api_key}"}
        if is_openrouter(base_url):
            key_ok, detail, response = probe(factory, route, base_url.rstrip("/") + "/key", auth)
            if key_ok and response is not None:
                data = response.json().get("data", {})
                detail = ", ".join(f"{name}={data[name]}" for name in KEY_FIELDS if name in data) or detail
        else:
            key_ok, detail, _ = probe(factory, route, models_url, auth)
        report(key_ok, "4. ключ принят", detail)

    lines.append("")
    if via_proxy and key_ok:
        lines.append("Итог: судья доступен через прокси — .env менять не нужно.")
    elif direct and key_ok:
        reason = f" ({proxy_detail})" if proxy else ""
        lines.append(f"Итог: напрямую {host} доступен, а через прокси — нет{reason}." if proxy
                     else "Итог: судья доступен напрямую.")
        if proxy and "407" in proxy_detail:
            lines.append("Прокси отклонил логин или пароль (407) — проверьте LLM__PROXY_URL; "
                         "или оставьте его пустым, и судья пойдёт напрямую.")
        elif proxy:
            lines.append("В .env оставьте LLM__PROXY_URL= пустым: судья пойдёт напрямую. Сервис с Ollama это не меняет.")
    elif (via_proxy or direct) and not key_ok:
        lines.append("Итог: связь есть, но ключ не принят — проверьте EVAL_JUDGE_API_KEY.")
    else:
        hint = " Прокси отклонил логин или пароль (407) — проверьте LLM__PROXY_URL." if "407" in proxy_detail else ""
        lines.append(f"Итог: нет связи с {host} ни через прокси, ни напрямую.{hint}")
    return lines, 0 if key_ok else 1


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Проверка связи с судьёй eval: прокси, провайдер, ключ")
    parser.add_argument("--judge-base-url", default=None, help="по умолчанию EVAL_JUDGE_BASE_URL из .env")
    parser.add_argument("--judge-api-key-env", default="EVAL_JUDGE_API_KEY")
    args = parser.parse_args(argv)

    from app.core.config import get_settings, is_local_url

    settings = get_settings()
    base_url = args.judge_base_url or env_value("EVAL_JUDGE_BASE_URL") or settings.llm.base_url
    if not base_url or is_local_url(base_url):
        print(f"Судья локальный ({base_url or 'не задан'}) — прокси и ключ не нужны.")
        return 0
    proxy = settings.llm.proxy_url.get_secret_value() if settings.llm.proxy_url else None
    lines, code = diagnose(base_url, proxy or None, env_value(args.judge_api_key_env),
                           settings.llm.use_system_certs)
    print("\n".join(lines))
    return code


def diagnose(base_url: str, proxy: str | None, api_key: str | None, use_system_certs: bool,
             factory_for: Callable[[Any], ClientFactory] = client_factory) -> tuple[list[str], int]:
    from app.core.config import tls_verify

    lines, code = run_checks(base_url, proxy, api_key, factory_for(tls_verify(use_system_certs)))
    if use_system_certs or not any("CERTIFICATE_VERIFY_FAILED" in line for line in lines):
        return lines, code
    lines += ["", "Сертификат HTTPS не прошёл проверку по списку certifi — повторяю с хранилищем "
                  "сертификатов ОС (как у браузера):", ""]
    retry, retry_code = run_checks(base_url, proxy, api_key, factory_for(tls_verify(True)))
    lines += retry
    if retry_code == 0:
        lines += ["", "С хранилищем ОС всё работает. Сеть или антивирус проверяют HTTPS и подставляют свой "
                      "корневой сертификат — ему доверяет Windows, но не certifi. "
                      "Добавьте в .env: LLM__USE_SYSTEM_CERTS=true"]
    return lines, 1


if __name__ == "__main__":
    sys.exit(main())
