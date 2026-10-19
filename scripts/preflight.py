"""
Готов ли стенд к демонстрации: одна команда перед записью видео или защитой (docs/demo.md).

    python scripts/preflight.py           # настройки, Ollama и модели, контейнеры, сервис, Qdrant, Phoenix
    python scripts/preflight.py --ask     # + один вопрос через /rag/query: время ответа и источники

Проверки — с хоста, по .env (те же настройки, что у сервиса):
1. .env: обязательные переменные заданы — печатаются только имена, не значения;
2. Ollama: отвечает, скачаны модели ответа (LLM__DEFAULT_MODEL), агента (AGENT__MODEL) и эмбеддингов
   (EMBEDDINGS__MODEL), а если задана — vision-модель (CHAT_VISION_MODEL);
3. Docker: контейнеры compose запущены, app — healthy, migrate завершился с кодом 0;
4. сервис: /health — 200, /ready — 200 (Redis на месте);
5. Qdrant: коллекция RAG__COLLECTION есть и в ней не меньше точек, чем фрагментов корпуса (102);
6. Phoenix: /healthz — 200 (не критично: без него сервис отвечает, трейсов нет).
Код выхода — 0, если все критичные проверки прошли, иначе 1.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

ROOT = Path(__file__).resolve().parent.parent
SERVICE = "http://127.0.0.1:8000"
PHOENIX = "http://127.0.0.1:6006"
CORPUS_FRAGMENTS = 102            # docs/data_inventory.md: 66 файлов → 102 фрагмента при 512/64
REQUIRED_ENV = ("LLM__OPENAI_API_KEY", "QDRANT_API_KEY", "ADMIN_TOKEN", "INTERNAL_TOKEN")
OPTIONAL_ENV = {"BOT_TOKEN": "без него контейнер bot не подключится к Telegram"}
SERVICES = ("app", "bot", "redis", "postgres", "qdrant", "phoenix")
ASK = "Сколько действует ссылка для сброса пароля?"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    critical: bool = True

    def line(self) -> str:
        mark = "OK  " if self.ok else ("FAIL" if self.critical else "WARN")
        return f"[{mark}] {self.name}: {self.detail}"


# ---------------------------------------------------------------- проверки
def check_env(env_file: Path = ROOT / ".env") -> list[Check]:
    if not env_file.exists():
        return [Check(".env", False, "файла нет — скопируйте .env.example в .env и заполните")]
    from dotenv import dotenv_values

    values = dotenv_values(env_file)
    missing = [name for name in REQUIRED_ENV if not (values.get(name) or "").strip()]
    checks = [Check(".env", not missing, "обязательные заданы" if not missing else f"пусто: {', '.join(missing)}")]
    for name, why in OPTIONAL_ENV.items():
        if not (values.get(name) or "").strip():
            checks.append(Check(name, False, f"пусто — {why}", critical=False))
    return checks


def ollama_root(base_url: str | None) -> str:
    """http://localhost:11434/v1 → http://localhost:11434 (API тегов Ollama — без /v1)."""
    parts = urlsplit(base_url or "http://localhost:11434/v1")
    return urlunsplit((parts.scheme, parts.netloc, parts.path.removesuffix("/v1").rstrip("/"), "", ""))


def has_model(names: set[str], model: str) -> bool:
    return model in names or f"{model}:latest" in names


def check_ollama(get: Callable[..., httpx.Response], base_url: str | None,
                 models: dict[str, str | None]) -> list[Check]:
    root = ollama_root(base_url)
    try:
        response = get(f"{root}/api/tags", timeout=5)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return [Check("Ollama", False, f"{root} не отвечает ({type(exc).__name__}) — запустите Ollama")]
    names = {m.get("name", "") for m in response.json().get("models", [])}
    checks = [Check("Ollama", True, f"{root}, моделей: {len(names)}")]
    for role, model in models.items():
        if not model:
            continue
        found = has_model(names, model)
        checks.append(Check(f"модель {role}", found, model if found else f"{model} нет — ollama pull {model}",
                            critical=role != "vision"))
    return checks


def compose_ps(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> list[dict[str, Any]]:
    """docker compose ps -a в JSON: у Compose 2.21+ — строка JSON на контейнер, у старых — массив."""
    result = run(["docker", "compose", "ps", "-a", "--format", "json"], cwd=ROOT, capture_output=True,
                 text=True, encoding="utf-8", timeout=30)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "docker compose ps завершился с ошибкой")
    text = result.stdout.strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def check_containers(rows: list[dict[str, Any]]) -> list[Check]:
    by_service = {row.get("Service"): row for row in rows}
    checks = []
    for service in SERVICES:
        row = by_service.get(service)
        if row is None:
            checks.append(Check(f"контейнер {service}", False, "не создан — docker compose up -d",
                                critical=service != "bot"))
            continue
        state, health = row.get("State", ""), row.get("Health", "")
        ok = state == "running" and (service != "app" or health == "healthy")
        detail = state + (f", {health}" if health else "")
        checks.append(Check(f"контейнер {service}", ok, detail, critical=service not in {"bot", "phoenix"}))
    migrate = by_service.get("migrate")
    if migrate is not None:
        code = migrate.get("ExitCode")
        checks.append(Check("миграции", code == 0, f"migrate: {migrate.get('State')}, код {code}"))
    return checks


def check_service(get: Callable[..., httpx.Response], url: str = SERVICE) -> list[Check]:
    checks = []
    for path, critical in (("/health", True), ("/ready", True)):
        try:
            response = get(f"{url}{path}", timeout=5)
            checks.append(Check(f"сервис {path}", response.status_code == 200,
                                f"{response.status_code} {response.text[:80]}", critical))
        except httpx.HTTPError as exc:
            checks.append(Check(f"сервис {path}", False, f"{url} не отвечает ({type(exc).__name__})", critical))
    return checks


def check_qdrant(get: Callable[..., httpx.Response], url: str | None, api_key: str | None, collection: str) -> Check:
    if not url:
        return Check("Qdrant", False, "QDRANT_URL не задан")
    try:
        response = get(f"{url.rstrip('/')}/collections/{collection}", headers={"api-key": api_key or ""}, timeout=5)
    except httpx.HTTPError as exc:
        return Check("Qdrant", False, f"{url} не отвечает ({type(exc).__name__})")
    if response.status_code == 404:
        return Check("Qdrant", False, f"коллекции {collection} нет — индексация при старте app или "
                                      "python scripts/ingest.py data/")
    if response.status_code != 200:
        return Check("Qdrant", False, f"{response.status_code} — проверьте QDRANT_API_KEY")
    points = (response.json().get("result") or {}).get("points_count") or 0
    return Check("Qdrant", points >= CORPUS_FRAGMENTS, f"{collection}: {points} точек (корпус — {CORPUS_FRAGMENTS})")


def check_phoenix(get: Callable[..., httpx.Response], url: str = PHOENIX) -> Check:
    try:
        response = get(f"{url}/healthz", timeout=5)
        return Check("Phoenix", response.status_code == 200, f"{url} — {response.status_code}", critical=False)
    except httpx.HTTPError as exc:
        return Check("Phoenix", False, f"не отвечает ({type(exc).__name__}) — трейсов не будет, сервис работает",
                     critical=False)


def ask_rag(post: Callable[..., httpx.Response], url: str = SERVICE, question: str = ASK) -> Check:
    started = time.perf_counter()
    try:
        response = post(f"{url}/rag/query", json={"question": question}, timeout=300)
    except httpx.HTTPError as exc:
        return Check("вопрос /rag/query", False, f"нет ответа ({type(exc).__name__})")
    seconds = time.perf_counter() - started
    if response.status_code != 200:
        return Check("вопрос /rag/query", False, f"{response.status_code} {response.text[:120]}")
    body = response.json()
    files = [s.get("file_name") for s in body.get("sources") or []]
    return Check("вопрос /rag/query", bool(body.get("confident")),
                 f"{seconds:.0f} с, confident={body.get('confident')}, источники: {', '.join(map(str, files[:3]))}")


# ---------------------------------------------------------------- запуск
def settings_or_error() -> tuple[Any, str | None]:
    try:
        from app.core.config import get_settings

        return get_settings(), None
    except Exception as exc:  # noqa: BLE001 — неверный .env — это результат проверки, а не падение
        errors = getattr(exc, "errors", None)
        if callable(errors):          # pydantic ValidationError: поле и причина, без значения (там бывают ключи)
            return None, "; ".join(f"{'.'.join(map(str, e['loc'])).upper()}: {e['msg']}"
                                   for e in errors(include_input=False, include_url=False))
        return None, f"{type(exc).__name__}: {str(exc).splitlines()[0]}"


def collect(cfg: Any, *, ask: bool, url: str = SERVICE, get: Callable[..., httpx.Response] = httpx.get,
            post: Callable[..., httpx.Response] = httpx.post,
            ps: Callable[[], list[dict[str, Any]]] = compose_ps) -> list[Check]:
    checks = check_env()
    secret = cfg.qdrant_api_key.get_secret_value() if cfg.qdrant_api_key else None
    checks += check_ollama(get, cfg.llm.base_url, {
        "ответа": cfg.rag.llm_model or cfg.llm.default_model, "агента": cfg.agent.model,
        "эмбеддингов": cfg.embeddings.model, "vision": cfg.chat_vision_model})
    try:
        checks += check_containers(ps())
    except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
        checks.append(Check("Docker", False, f"docker compose ps не сработал: {exc}"))
    checks += check_service(get, url)
    checks.append(check_qdrant(get, cfg.qdrant_url, secret, cfg.rag.collection))
    checks.append(check_phoenix(get))
    if ask:
        checks.append(ask_rag(post, url))
    return checks


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Проверка стенда перед демонстрацией")
    parser.add_argument("--ask", action="store_true", help=f"задать вопрос через /rag/query: «{ASK}»")
    parser.add_argument("--url", default=SERVICE, help="адрес сервиса")
    args = parser.parse_args(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    cfg, error = settings_or_error()
    if cfg is None:
        print(f"[FAIL] настройки: {error}")
        return 1
    checks = collect(cfg, ask=args.ask, url=args.url)
    for check in checks:
        print(check.line())
    failed = [c for c in checks if c.critical and not c.ok]
    print(f"\nСтенд {'готов' if not failed else 'НЕ готов'}: критичных проблем — {len(failed)}, "
          f"предупреждений — {sum(not c.ok and not c.critical for c in checks)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
