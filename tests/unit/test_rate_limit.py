"""
Лимит запросов (блок 3.8): RateLimitMiddleware на счётчиках Redis.

Redis подменён FakeRateRedis — тот же интерфейс pipeline (SET NX EX, INCR, TTL), что у
redis.asyncio. Живая проверка против запущенного сервиса — scripts/load_test.py; здесь он
гоняется против того же приложения через TestClient. Заодно — charset=utf-8 у JSON-ответов
(app/core/charset.py).
"""
from __future__ import annotations

import importlib.util
import sys

import httpx
import pytest

from app.core.charset import with_charset
from conftest import ROOT

from app.core.config import Settings
from app.deps.providers import get_llm_service
from app.main import app
from app.schemas.chat import ChatRequest, ChatResponse, Usage
from log_capture import captured_logs, events


class FakeRateRedis:
    def __init__(self, fail: bool = False) -> None:
        self.values: dict[str, int] = {}
        self.ttls: dict[str, int] = {}
        self.fail = fail

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis: FakeRateRedis) -> None:
        self.redis, self.ops = redis, []

    async def __aenter__(self) -> FakePipeline:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def set(self, key: str, value: int, ex: int, nx: bool) -> None:
        self.ops.append(("set", key, value, ex, nx))

    def incr(self, key: str) -> None:
        self.ops.append(("incr", key))

    def ttl(self, key: str) -> None:
        self.ops.append(("ttl", key))

    async def execute(self) -> list:
        if self.redis.fail:
            raise ConnectionError("redis down")
        results = []
        for op, key, *rest in self.ops:
            if op == "set":
                value, ex, nx = rest
                if nx and key in self.redis.values:
                    results.append(None)
                else:
                    self.redis.values[key], self.redis.ttls[key] = value, ex
                    results.append(True)
            elif op == "incr":
                self.redis.values[key] += 1
                results.append(self.redis.values[key])
            else:
                results.append(self.redis.ttls.get(key, -2))
        return results


class StubService:
    async def complete(self, req: ChatRequest) -> ChatResponse:
        return ChatResponse(content="ok", model="stub", usage=Usage())


@pytest.fixture
def limited(mocker):
    """Приложение с лимитом 3 в минуту, Redis — FakeRateRedis."""
    previous = getattr(app.state, "cache", None)

    def setup(limit: int = 3, redis: FakeRateRedis | None = None) -> FakeRateRedis:
        settings = Settings(llm={"openai_api_key": "k"}, rate_limit_per_min=limit, _env_file=None)
        mocker.patch("app.services.security.rate_limit.get_settings", return_value=settings)
        fake = redis or FakeRateRedis()
        app.state.cache = fake
        app.dependency_overrides[get_llm_service] = lambda: StubService()
        return fake
    yield setup
    app.dependency_overrides.clear()
    app.state.cache = previous


async def post(n: int, headers: dict[str, str] | None = None, path: str = "/chat") -> list[httpx.Response]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        return [await http.post(path, json={"messages": [{"role": "user", "content": "Привет"}]}, headers=headers)
                for _ in range(n)]


async def test_request_over_limit_gets_429_with_retry_after(limited):
    limited(limit=3)
    with captured_logs("INFO") as logs:
        responses = await post(4, {"X-User-ID": "u-1", "Origin": "http://localhost:3000"})
    assert [r.status_code for r in responses] == [200, 200, 200, 429]
    over = responses[-1]
    assert over.json()["error"]["code"] == "rate_limited" and "не больше 3 в минуту" in over.json()["error"]["message"]
    assert over.headers["Retry-After"] == "60"
    # 429 проходит через CORS и RequestContextMiddleware: заголовки и строка http_request на месте
    assert over.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert over.json()["error"]["request_id"] == over.headers["X-Request-ID"]
    assert events(logs, "rate_limited")[0]["client"] == "user"
    assert [r["status"] for r in events(logs, "http_request")][-1] == 429


async def test_clients_are_counted_separately(limited):
    limited(limit=2)
    first = await post(3, {"X-User-ID": "u-1"})
    second = await post(1, {"X-User-ID": "u-2"})
    by_ip = await post(2)                                  # без X-User-ID — по IP
    assert [r.status_code for r in first] == [200, 200, 429]
    assert second[0].status_code == 200 and [r.status_code for r in by_ip] == [200, 200]


async def test_redis_down_lets_requests_through(limited):
    limited(limit=1, redis=FakeRateRedis(fail=True))
    with captured_logs("INFO") as logs:
        responses = await post(3, {"X-User-ID": "u-1"})
    assert [r.status_code for r in responses] == [200, 200, 200]
    assert events(logs, "rate_limit_unavailable")


async def test_zero_limit_and_other_paths_are_not_limited(limited):
    fake = limited(limit=0)
    assert [r.status_code for r in await post(5, {"X-User-ID": "u-1"})] == [200] * 5
    assert fake.values == {}
    limited(limit=1)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        assert {(await http.get("/health")).status_code for _ in range(3)} == {200}


# ---------------------------------------------------------------- заголовки X-RateLimit-*
async def test_limit_headers_count_down(limited):
    limited(limit=3)
    responses = await post(4, {"X-User-ID": "u-1", "Origin": "http://localhost:3000"})
    assert {r.headers["X-RateLimit-Limit"] for r in responses} == {"3"}
    assert [r.headers["X-RateLimit-Remaining"] for r in responses] == ["2", "1", "0", "0"]
    exposed = responses[0].headers["access-control-expose-headers"].lower()
    assert all(h in exposed for h in ("retry-after", "x-ratelimit-limit", "x-ratelimit-remaining"))


async def test_redis_down_shows_limit_without_remaining(limited):
    """Так load_test.py отличает упавший Redis: лимит включён, а счёта нет."""
    limited(limit=1, redis=FakeRateRedis(fail=True))
    response = (await post(1, {"X-User-ID": "u-1"}))[0]
    assert response.headers["X-RateLimit-Limit"] == "1" and "X-RateLimit-Remaining" not in response.headers


async def test_no_limit_headers_when_limit_is_off(limited):
    limited(limit=0)
    assert "X-RateLimit-Limit" not in (await post(1, {"X-User-ID": "u-1"}))[0].headers


# ---------------------------------------------------------------- scripts/load_test.py
def load_test_module():
    spec = importlib.util.spec_from_file_location("load_test", ROOT / "scripts" / "load_test.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["load_test"] = module
    spec.loader.exec_module(module)
    return module


async def run_load_test(expected: int | None = None) -> tuple[int, str]:
    import contextlib
    import io

    load_test, out = load_test_module(), io.StringIO()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        with contextlib.redirect_stdout(out):
            code = await load_test.run(http, "http://test/chat", expected, "Привет")
    return code, out.getvalue()


async def test_load_test_passes_and_reads_limit_from_service(limited):
    limited(limit=3)
    code, out = await run_load_test(expected=30)
    assert code == 0 and "[OK]   первые 3 — без 429, запрос 4 — 429" in out
    assert "Ожидался лимит 30" in out and "сервис работает с 3" in out      # .env правили без перезапуска


async def test_load_test_explains_service_without_limit(limited):
    limited(limit=0)
    code, out = await run_load_test(expected=30)
    assert code == 1 and "перезапустите uvicorn" in out


async def test_load_test_explains_redis_down(limited):
    """Случай с Windows: uvicorn вне Docker, Redis из compose без порта на localhost — 31 × 200."""
    limited(limit=30, redis=FakeRateRedis(fail=True))
    code, out = await run_load_test(expected=30)
    assert code == 1 and "Redis недоступен" in out and "docker start multapi-redis" in out


# ---------------------------------------------------------------- charset
async def test_json_responses_declare_utf8(limited):
    limited(limit=1)
    ok, over = await post(2, {"X-User-ID": "u-1"})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        invalid = await http.post("/chat", json={"messages": []})
        health = await http.get("/health")
    for response in (ok, over, invalid, health):
        assert response.headers["content-type"] == "application/json; charset=utf-8", response.request.url


@pytest.mark.parametrize(("given", "expected"), [
    ("application/json", "application/json; charset=utf-8"),
    ("application/json; charset=utf-8", "application/json; charset=utf-8"),
    ("text/event-stream; charset=utf-8", "text/event-stream; charset=utf-8"),
    ("application/problem+json", "application/problem+json"),
    ("text/html; charset=utf-8", "text/html; charset=utf-8"),
])
def test_with_charset(given, expected):
    assert with_charset(given) == expected
