"""
Статические проверки файлов блоков 3.5–3.6 и 4.1: Dockerfile, .dockerignore, compose.yaml, .env.example.

Docker для этих тестов не нужен: они читают файлы как текст и следят, чтобы правки не
сломали требования заданий (multi-stage, non-root, exec-форма CMD, healthcheck-и,
Redis без портов наружу, .env вне образа и вне git, Phoenix для трейсов).

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import fnmatch
import json
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _service_block(compose: str, name: str) -> str:
    """Текст сервиса из compose.yaml: от «  name:» до следующего сервиса или раздела."""
    match = re.search(rf"^  {name}:\n(.*?)(?=^  \S|^\S)", compose, re.S | re.M)
    if match is None:
        raise AssertionError(f"в compose.yaml нет сервиса {name}")
    return match.group(1)


def _ignored(path: str, patterns: list[str]) -> bool:
    """Упрощённая логика .dockerignore: последнее совпавшее правило решает, «!» — исключение."""
    result = False
    for pattern in patterns:
        negate = pattern.startswith("!")
        pattern = pattern.lstrip("!").rstrip("/")
        parts = path.split("/")
        prefixes = ["/".join(parts[: i + 1]) for i in range(len(parts))]
        if pattern.startswith("**/"):
            hit = any(fnmatch.fnmatch(part, pattern[3:]) for part in parts)
        else:
            hit = any(fnmatch.fnmatch(prefix, pattern) for prefix in prefixes)
        if hit:
            result = not negate
    return result


class TestDockerfile(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _read("Dockerfile")
        self.lines = [line.strip() for line in self.text.splitlines()]

    def test_syntax_and_stages(self):
        self.assertEqual(self.lines[0], "# syntax=docker/dockerfile:1.7")
        stages = re.findall(r"^FROM (\S+) AS (\w+)$", self.text, re.M)
        self.assertEqual(stages, [("python:3.13-slim-bookworm", "builder"),
                                  ("python:3.13-slim-bookworm", "runtime")])

    def test_uv_and_layer_order(self):
        self.assertRegex(self.text, r"COPY --from=ghcr\.io/astral-sh/uv:[\d.]+ /uv /uvx /bin/")
        for env in ("UV_COMPILE_BYTECODE=1", "UV_LINK_MODE=copy", "UV_PYTHON_DOWNLOADS=0"):
            self.assertIn(env, self.text)
        self.assertIn("--mount=type=cache,target=/root/.cache/uv", self.text)
        # Зависимости ставятся до копирования кода — иначе кеш слоёв не работает.
        self.assertLess(self.text.index("uv sync --frozen"), self.text.index("COPY app/"))

    def test_runtime_is_non_root(self):
        runtime = self.text.split("AS runtime", 1)[1]
        self.assertIn("RUN useradd --create-home --uid 1000 appuser", runtime)
        self.assertIn("COPY --from=builder --chown=appuser:appuser /app /app", runtime)
        self.assertRegex(runtime, r"(?m)^USER appuser$")
        self.assertLess(runtime.index("USER appuser"), runtime.index("CMD"))

    def test_cmd_exec_form_on_all_interfaces(self):
        self.assertIn("EXPOSE 8000", self.lines)
        # Инструкция CMD — с начала строки (строка «    CMD [...]» с отступом относится к HEALTHCHECK).
        cmd = next(line for line in self.text.splitlines() if line.startswith("CMD "))
        args = json.loads(cmd[len("CMD "):])          # exec-форма — JSON-массив
        self.assertEqual(args[:2], ["uvicorn", "app.main:app"])
        self.assertEqual(args[args.index("--host") + 1], "0.0.0.0")

    def test_healthcheck_uses_readiness(self):
        self.assertRegex(self.text, r"HEALTHCHECK .*--interval=15s --timeout=5s --start-period=15s --retries=3")
        self.assertIn("http://localhost:8000/ready", self.text)


class TestDockerignore(unittest.TestCase):
    def setUp(self) -> None:
        lines = [line.strip() for line in _read(".dockerignore").splitlines()]
        self.patterns = [line for line in lines if line and not line.startswith("#")]

    def test_required_entries(self):
        required = [".git", ".gitignore", ".github/", ".venv/", "venv/", "__pycache__/", "*.py[cod]",
                    ".pytest_cache/", ".mypy_cache/", ".ruff_cache/", ".env", ".env.*",
                    "!.env.example", "tests/", "notebooks/", "data/", "*.md", "!README.md",
                    ".vscode/", ".idea/", ".DS_Store"]
        for entry in required:
            with self.subTest(entry=entry):
                self.assertIn(entry, self.patterns)

    def test_secrets_and_tests_stay_out_of_context(self):
        for path in (".env", ".env.local", ".git/config", "tests/test_service.py", "data/service_status.json",
                     "app/__pycache__/main.cpython-313.pyc", ".venv/bin/python"):
            with self.subTest(path=path):
                self.assertTrue(_ignored(path, self.patterns))
        for path in (".env.example", "app/main.py", "pyproject.toml", "uv.lock", "data/knowledge_base.json"):
            with self.subTest(path=path):
                self.assertFalse(_ignored(path, self.patterns))


class TestChatStorageFiles(unittest.TestCase):
    """Блок 4.1: миграции в образе, Postgres в compose."""

    def test_image_has_migrations(self):
        dockerfile = _read("Dockerfile")
        self.assertIn("COPY alembic.ini ./", dockerfile)
        self.assertIn("COPY migrations/ ./migrations/", dockerfile)
        self.assertIn("RUN mkdir -p /app/var/chats", dockerfile)
        patterns = [line.strip() for line in _read(".dockerignore").splitlines()
                    if line.strip() and not line.startswith("#")]
        for path in ("alembic.ini", "migrations/env.py", "migrations/versions/x_chat_tables.py"):
            with self.subTest(path=path):
                self.assertFalse(_ignored(path, patterns))
        self.assertTrue(_ignored("var/chats/chats/x/messages.jsonl", patterns))   # локальная история — не в образ

    def test_postgres_service(self):
        compose = _read("compose.yaml")
        postgres = _service_block(compose, "postgres")
        self.assertIn("image: postgres:16-alpine", postgres)
        self.assertRegex(postgres, r'ports:\n(\s+#.*\n)*\s+- "127\.0\.0\.1:5433:5432"')   # только loopback
        self.assertIn('test: ["CMD", "pg_isready", "-U", "multapi", "-d", "multapi"]', postgres)
        self.assertRegex(postgres, r"volumes:\n\s+- pg_data:/var/lib/postgresql/data")
        self.assertRegex(compose, r"(?m)^volumes:\n(  .*\n)*  pg_data:")

    def test_app_uses_postgres_by_service_name(self):
        app = _service_block(_read("compose.yaml"), "app")
        self.assertIn("DATABASE_URL: postgresql+asyncpg://multapi:${POSTGRES_PASSWORD:-multapi}@postgres:5432/multapi", app)
        self.assertIn("CHAT_STORAGE_DIR: /app/var/chats", app)
        self.assertRegex(app, r"volumes:\n\s+- chat_data:/app/var/chats")
        self.assertRegex(app, r"\n\s+postgres:\n\s+condition: service_healthy")


class TestCompose(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _read("compose.yaml")
        self.app = _service_block(self.text, "app")
        self.redis = _service_block(self.text, "redis")
        self.phoenix = _service_block(self.text, "phoenix")

    def test_app_service(self):
        self.assertRegex(self.app, r"build:\n\s+context: \.")
        self.assertRegex(self.app, r'ports:\n\s+- "8000:8000"')
        self.assertRegex(self.app, r"env_file:\n\s+- \.env")
        self.assertIn("REDIS_URL: redis://redis:6379/0", self.app)
        self.assertRegex(self.app, r"depends_on:\n\s+redis:\n\s+condition: service_healthy")
        self.assertIn("restart: unless-stopped", self.app)
        for param in ("interval: 15s", "timeout: 5s", "retries: 3", "start_period: 15s"):
            self.assertIn(param, self.app)

    def test_redis_service(self):
        self.assertIn("image: redis:7.4-alpine", self.redis)
        self.assertNotIn("ports:", self.redis)                 # Redis не торчит наружу
        self.assertRegex(self.redis, r"volumes:\n\s+- redis_data:/data")
        self.assertIn('test: ["CMD", "redis-cli", "ping"]', self.redis)
        for param in ("interval: 10s", "timeout: 3s", "retries: 3"):
            self.assertIn(param, self.redis)
        self.assertRegex(self.text, r"(?m)^volumes:\n  redis_data:")

    def test_phoenix_service(self):
        self.assertIn("image: arizephoenix/phoenix:latest", self.phoenix)
        self.assertRegex(self.phoenix, r'ports:\n\s+- "6006:6006".*\n\s+- "4317:4317"')
        self.assertIn("PHOENIX_WORKING_DIR: /data", self.phoenix)
        self.assertRegex(self.phoenix, r"volumes:\n(\s+#.*\n)*\s+- phoenix-data:/data")
        self.assertRegex(self.text, r"(?m)^volumes:\n(  .*\n)*  phoenix-data:")

    def test_phoenix_healthcheck(self):
        # без healthcheck docker compose up --wait не ждёт Phoenix
        self.assertIn('test: ["CMD", "/usr/bin/python3.13", "-c"', self.phoenix)
        self.assertIn("http://127.0.0.1:6006/healthz", self.phoenix)
        for param in ("start_period: 300s", "start_interval: 2s", "retries: 3"):
            self.assertIn(param, self.phoenix)

    def test_app_sends_traces_to_phoenix(self):
        self.assertIn("PHOENIX_COLLECTOR_ENDPOINT: http://phoenix:6006", self.app)
        depends = self.app.split("depends_on:", 1)[1].split("healthcheck:", 1)[0]
        self.assertRegex(depends, r"\n\s+phoenix:\n(\s+#.*\n)*\s+condition: service_started")


class TestSecrets(unittest.TestCase):
    def test_env_example_lists_variables(self):
        example = _read(".env.example")
        for name in ("LLM__OPENAI_API_KEY", "REDIS_URL", "LOG_LEVEL", "DOCKER_LLM_BASE_URL",
                     "PHOENIX_COLLECTOR_ENDPOINT", "PHOENIX_PROJECT_NAME", "PII_PRESIDIO",
                     # блок 3.7
                     "SUPPORT__ENABLED", "EVAL_JUDGE_MODEL", "EVAL_JUDGE_BASE_URL", "EVAL_JUDGE_API_KEY",
                     "EVAL_JUDGE_REASONING", "EVAL_JUDGE_MAX_TOKENS", "LLM__PROXY_URL", "LLM__USE_SYSTEM_CERTS",
                     # блок 3.8
                     "SECURITY__ENABLED", "SECURITY__MAX_INPUT_CHARS", "RATE_LIMIT_PER_MIN", "LOG_FILE",
                     # блок 4.1
                     "CHAT_REPOSITORY", "CHAT_STORAGE_DIR", "DATABASE_URL", "CHAT_CONTEXT_STRATEGY",
                     "CHAT_CONTEXT_WINDOW", "CHAT_SYSTEM_PROMPT", "CONTEXT_WINDOW", "RESPONSE_TOKENS", "SAFETY_MARGIN",
                     # блок 4.2
                     "BOT_TOKEN", "BACKEND_URL", "BOT_ADMIN_IDS", "BACKEND_TIMEOUT", "BOT_USE_SYSTEM_CERTS",
                     "BOT_PROXY_URL",
                     # блок 4.3
                     "CHAT_VISION_MODEL", "AUDIO_LANGUAGE", "MEDIA__MAX_IMAGE_BYTES", "MEDIA__MAX_DOCUMENT_BYTES",
                     "INTERNAL_TOKEN", "BOT_URL", "BACKEND_STREAM_TIMEOUT", "BOT_STREAMING", "BOT_API_HOST",
                     "BOT_API_PORT", "BOT_DEFAULT_USER_NAME"):
            self.assertRegex(example, rf"(?m)^{name}=")

    def test_env_example_has_no_api_keys(self):
        """Ключей API в .env.example нет: OpenRouter участвует в сканировании секретов GitHub,
        опубликованный учебный ключ отзовут для всей группы. Ключ и адрес прокси — в .env."""
        example = _read(".env.example")
        self.assertNotRegex(example, r"sk-[A-Za-z0-9_-]{20,}")
        for line in example.splitlines():
            if re.match(r"\s*(EVAL_JUDGE_API_KEY|LLM__OPENAI_API_KEY|BOT_TOKEN|BOT_PROXY_URL|INTERNAL_TOKEN|"
                        r"AUDIO_API_KEY)\s*=", line):
                self.assertRegex(line, r"=\s*(#|$)", line)        # значение пустое

    def test_env_not_tracked_by_git(self):
        if not (ROOT / ".git").exists():
            self.skipTest("не git-репозиторий (например, распакованный архив)")
        try:
            files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                                   check=True).stdout.splitlines()
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("git недоступен")
        self.assertFalse([f for f in files if re.search(r"(^|/)\.env$", f)])

    def test_env_in_gitignore_and_bot_outside_image(self):
        """Блок 4.2: токен бота живёт в .env — тот в .gitignore; бот в образ сервиса не входит."""
        self.assertIn(".env", _read(".gitignore").splitlines())
        self.assertIn("bot/", _read(".dockerignore").splitlines())
        self.assertNotIn("aiogram", _read("pyproject.toml"))


if __name__ == "__main__":
    unittest.main()
