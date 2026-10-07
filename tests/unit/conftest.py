"""
Общие настройки unit-тестов (блок 3.7): без сети и без ключей API.

    pytest tests/unit/ -m "not llm"

- Переменные окружения задаются до импорта app.main: настоящие ключи и адреса из .env
  тестам не нужны (переменные окружения важнее .env).
- Сеть запрещена: любая попытка открыть TCP-соединение падает с NetworkBlocked
  (подкласс OSError — для сервиса это выглядит как недоступный Redis или провайдер).
  Модель подменяется httpx.MockTransport, AsyncMock или фикстурой mocker.
  Перекрыты три пути: socket.connect / create_connection (синхронные клиенты и цикл
  asyncio на Linux), sock_connect цикла событий (на Windows asyncio соединяется через
  IOCP ConnectEx в обход socket.connect) и DNS-запросы к внешним именам.
- Исключение одно — внутренняя пара сокетов цикла событий asyncio. На Windows
  socket.socketpair() строит её TCP-соединением с 127.0.0.1 (на Linux это Unix-сокеты),
  и без исключения ни один async-тест не запустился бы. Соединение разрешено только
  на время вызова socketpair(), остальные — по-прежнему запрещены.
"""
from __future__ import annotations

import asyncio.proactor_events
import asyncio.selector_events
import functools
import ipaddress
import os
import socket
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "tests", ROOT / "eval"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ["LLM__OPENAI_API_KEY"] = "test-key"
os.environ["LLM__DEFAULT_MODEL"] = "test-model"
os.environ["LLM__BASE_URL"] = "http://llm.invalid/v1"
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = ""
os.environ["CORS_ORIGINS"] = '["http://localhost:3000"]'
os.environ["LOG_LEVEL"] = "CRITICAL"
os.environ["LLM__USE_SYSTEM_CERTS"] = "false"   # .env разработчика (true на рабочем ноутбуке) тестам не мешает

from log_capture import quiet_logs  # noqa: E402

quiet_logs()


class NetworkBlocked(OSError):
    """Попытка выйти в сеть из unit-теста."""


_socketpair_in_progress = threading.local()


def allow_inside_socketpair(make_pair: Any) -> Any:
    """socket.socketpair, внутри которого connect() разрешён: так пару сокетов строит
    Windows (_fallback_socketpair — TCP через 127.0.0.1)."""

    @functools.wraps(make_pair)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _socketpair_in_progress.active = True
        try:
            return make_pair(*args, **kwargs)
        finally:
            _socketpair_in_progress.active = False

    return wrapper


def _is_local_host(host: Any) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    if host == "localhost":
        return True
    try:
        ipaddress.ip_address(str(host).strip("[]"))
        return True
    except ValueError:
        return False


def _guard_sock_connect(real: Any) -> Any:
    async def sock_connect(self: Any, sock: socket.socket, address: Any) -> Any:
        if sock.family == getattr(socket, "AF_UNIX", None):
            return await real(self, sock, address)
        raise NetworkBlocked(f"сеть в unit-тестах запрещена: {address}")

    return sock_connect


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if self.family == getattr(socket, "AF_UNIX", None) or getattr(_socketpair_in_progress, "active", False):
            return real_connect(self, address)
        raise NetworkBlocked(f"сеть в unit-тестах запрещена: {address}")

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> socket.socket:
        raise NetworkBlocked(f"сеть в unit-тестах запрещена: {address}")

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_local_host(host):            # IP и localhost разрешаются без сети
            return real_getaddrinfo(host, *args, **kwargs)
        raise NetworkBlocked(f"сеть в unit-тестах запрещена: DNS {host}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket, "socketpair", allow_inside_socketpair(socket.socketpair))
    for loop_class in (asyncio.selector_events.BaseSelectorEventLoop, asyncio.proactor_events.BaseProactorEventLoop):
        monkeypatch.setattr(loop_class, "sock_connect", _guard_sock_connect(loop_class.sock_connect))


@pytest.fixture
def settings():
    """Настройки сервиса без .env: ключ-заглушка и модель test-model."""
    from app.core.config import Settings

    return Settings(llm={"openai_api_key": "k", "default_model": "test-model"}, _env_file=None)


def fake_completion(content: str = "Ответ модели.", *, prompt_tokens: int = 40,
                    completion_tokens: int = 10, model: str = "test-model") -> SimpleNamespace:
    """Объект, похожий на ответ AsyncOpenAI chat.completions.create."""
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                              total_tokens=prompt_tokens + completion_tokens),
    )


def completion_json(content: str, model: str = "test-model") -> dict[str, Any]:
    """Тело HTTP-ответа OpenAI API — для httpx.MockTransport."""
    return {"id": "c1", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50}}


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.set_calls: list[tuple[str, int | None]] = []

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self.set_calls.append((key, ex))
        self.data[key] = value
        return True
