"""
Тесты HTTP-сервиса блока 3.4 — без сети, без Redis и без ключей.

Приложение вызывается через httpx.ASGITransport. Клиент OpenAI и Redis заменены
заглушками в app.state (ASGITransport не запускает lifespan), настройки — через
app.dependency_overrides. Отдельный тест проверяет сам lifespan на TestClient.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# До импорта app.main: настройки читаются при импорте. Переменные окружения важнее .env,
# поэтому тесты не зависят от локального .env.
os.environ["LLM__OPENAI_API_KEY"] = "test-key"
os.environ["LLM__DEFAULT_MODEL"] = "test-model"
os.environ["CORS_ORIGINS"] = '["http://localhost:3000"]'
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"   # закрытый порт: Redis «выключен»

import httpx  # noqa: E402
import openai  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from redis.exceptions import ConnectionError as RedisConnectionError  # noqa: E402

from app.core.config import Settings, get_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.schemas.chat import ChatRequest  # noqa: E402
from app.schemas.models import MODEL_CATALOG  # noqa: E402
from app.services.llm import LLMService  # noqa: E402

# Строка лога на каждый запрос в выводе тестов — шум; проверки логов идут через assertLogs.
logging.getLogger("llm-service").setLevel(logging.WARNING)

API = "http://test/v1/chat/completions"


# --------------------------------------------------------------------------- #
# Заглушки
# --------------------------------------------------------------------------- #
def _response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers, request=httpx.Request("POST", API))


def auth_error() -> Exception:
    return openai.AuthenticationError("Incorrect API key", response=_response(401), body=None)


def rate_limit_error() -> Exception:
    return openai.RateLimitError("Too many requests", response=_response(429, {"retry-after": "7"}), body=None)


def timeout_error() -> Exception:
    return openai.APITimeoutError(request=httpx.Request("POST", API))


def connection_error() -> Exception:
    return openai.APIConnectionError(request=httpx.Request("POST", API))


def not_found_error() -> Exception:
    return openai.NotFoundError("model not found", response=_response(404), body=None)


def chunk(text: str | None = None, usage: dict | None = None) -> SimpleNamespace:
    choices = [SimpleNamespace(delta=SimpleNamespace(content=text))] if text is not None else []
    return SimpleNamespace(choices=choices, usage=SimpleNamespace(**usage) if usage else None)


class FakeStream:
    def __init__(self, parts: list[str], usage: dict | None, fail_after: int | None = None):
        self.parts, self.usage, self.fail_after = parts, usage, fail_after
        self.closed = False

    def __aiter__(self):
        return self._chunks()

    async def _chunks(self):
        for i, part in enumerate(self.parts):
            if self.fail_after is not None and i == self.fail_after:
                raise connection_error()
            await asyncio.sleep(0)
            yield chunk(part)
        if self.usage:
            yield chunk(usage=self.usage)

    async def close(self) -> None:
        self.closed = True


class FakeCompletions:
    """Имитация AsyncOpenAI().chat.completions."""

    def __init__(self, reply: str = "Нажмите «Забыли пароль?»", error: Exception | None = None,
                 parts: list[str] | None = None, fail_after: int | None = None, delay: float = 0.0):
        self.reply, self.error, self.delay = reply, error, delay
        self.parts = parts or ["Раз", ", два", ", три"]
        self.fail_after = fail_after
        self.calls: list[dict] = []
        self.streams: list[FakeStream] = []
        self.active = self.max_active = 0

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            stream = FakeStream(self.parts, {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
                                self.fail_after)
            self.streams.append(stream)
            return stream
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.active -= 1
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5, total_tokens=17),
        )


class FakeRedis:
    def __init__(self, down: bool = False, ping_delay: float = 0.0):
        self.down, self.ping_delay = down, ping_delay
        self.data: dict[str, str] = {}
        self.ttl: dict[str, int] = {}

    def _check(self) -> None:
        if self.down:
            raise RedisConnectionError("Error 111 connecting to localhost:6379")

    async def get(self, key: str) -> str | None:
        self._check()
        return self.data.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> bool:
        self._check()
        self.data[key], self.ttl[key] = value, ttl
        return True

    async def ping(self) -> bool:
        self._check()
        await asyncio.sleep(self.ping_delay)
        return True


HI = {"messages": [{"role": "user", "content": "hi"}]}


class ServiceTestCase(unittest.IsolatedAsyncioTestCase):
    """app.state с заглушками и httpx-клиент поверх ASGI-приложения."""

    settings = Settings(llm={"openai_api_key": "test-key", "default_model": "test-model"},
                        cache_ttl_seconds=600, _env_file=None)

    def setUp(self) -> None:
        self.completions = FakeCompletions()
        self.cache = FakeRedis()
        self.use(self.completions, self.cache)
        app.dependency_overrides[get_settings] = lambda: self.settings
        self.addCleanup(app.dependency_overrides.clear)

    def use(self, completions: FakeCompletions, cache: FakeRedis | None) -> None:
        app.state.openai = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        app.state.cache = cache
        app.state.llm_limiter = asyncio.Semaphore(4)
        self.completions, self.cache = completions, cache

    def client(self, raise_app_exceptions: bool = True) -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    async def post(self, path: str, payload: object, **kwargs) -> httpx.Response:
        async with self.client() as http:
            return await http.post(path, json=payload, **kwargs)


# --------------------------------------------------------------------------- #
class TestSettings(unittest.TestCase):
    def test_missing_api_key_stops_start(self):
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(ValidationError) as ctx:
            Settings(_env_file=None)
        self.assertEqual(ctx.exception.errors()[0]["loc"], ("llm", "openai_api_key"))

    def test_nested_env_and_secret(self):
        env = {"LLM__OPENAI_API_KEY": "sk-secret", "LLM__BASE_URL": "http://localhost:11434/v1",
               "LLM__DEFAULT_MODEL": "llama3.2", "LLM__REQUEST_TIMEOUT": "120",
               "CACHE_TTL_SECONDS": "60", "CORS_ORIGINS": '["http://localhost:5173"]',
               "LLM_REQUEST_TIMEOUT": "5"}   # переменная блоков 2–3 на сервис не влияет
        with mock.patch.dict(os.environ, env, clear=True):
            s = Settings(_env_file=None)
        self.assertEqual((s.llm.default_model, s.llm.request_timeout, s.cache_ttl_seconds),
                         ("llama3.2", 120.0, 60))
        self.assertEqual(s.cors_origins, ["http://localhost:5173"])
        self.assertEqual(s.llm.openai_api_key.get_secret_value(), "sk-secret")
        self.assertNotIn("sk-secret", repr(s))

    def test_empty_value_means_default(self):
        env = {"LLM__OPENAI_API_KEY": "k", "REDIS_URL": "", "CORS_ORIGINS": ""}
        with mock.patch.dict(os.environ, env, clear=True):
            s = Settings(_env_file=None)
        self.assertEqual(s.redis_url, "redis://localhost:6379/0")
        self.assertEqual(s.cors_origins, ["http://localhost:3000"])

    def test_cors_wildcard_with_credentials_rejected(self):
        env = {"LLM__OPENAI_API_KEY": "k", "CORS_ORIGINS": '["*"]', "CORS_ALLOW_CREDENTIALS": "true"}
        with mock.patch.dict(os.environ, env, clear=True), self.assertRaises(ValidationError):
            Settings(_env_file=None)

    def test_log_level(self):
        with mock.patch.dict(os.environ, {"LLM__OPENAI_API_KEY": "k", "LOG_LEVEL": "debug"}, clear=True):
            self.assertEqual(Settings(_env_file=None).log_level, "DEBUG")
        with mock.patch.dict(os.environ, {"LLM__OPENAI_API_KEY": "k", "LOG_LEVEL": "LOUD"}, clear=True), \
                self.assertRaises(ValidationError):
            Settings(_env_file=None)

    def test_get_settings_is_cached(self):
        self.assertIs(get_settings(), get_settings())


class TestHealthAndModels(ServiceTestCase):
    async def test_health_without_dependencies(self):
        self.use(FakeCompletions(error=connection_error()), None)
        async with self.client() as http:
            response = await http.get("/health")
        self.assertEqual((response.status_code, response.json()), (200, {"status": "ok"}))

    async def test_ready_reports_redis_state(self):
        async with self.client() as http:
            up = await http.get("/ready")
            self.use(self.completions, FakeRedis(down=True))
            down = await http.get("/ready")
            self.use(self.completions, None)
            missing = await http.get("/ready")
            health = await http.get("/health")
        self.assertEqual((up.status_code, up.json()), (200, {"status": "ok", "redis": "up"}))
        for response in (down, missing):
            self.assertEqual((response.status_code, response.json()),
                             (503, {"status": "degraded", "redis": "down"}))
        self.assertEqual(health.status_code, 200)   # liveness от Redis не зависит

    async def test_ready_times_out_on_hanging_redis(self):
        self.use(self.completions, FakeRedis(ping_delay=1.0))
        with mock.patch("app.routers.health.READY_TIMEOUT", 0.05):
            async with self.client() as http:
                response = await http.get("/ready")
        self.assertEqual(response.status_code, 503)

    async def test_models_catalog(self):
        async with self.client() as http:
            data = (await http.get("/models")).json()
        self.assertEqual(len(data), len(MODEL_CATALOG))
        self.assertIn("gpt-4o-mini", {m["id"] for m in data})
        self.assertTrue(all(m["input_per_1m"] >= 0 and m["output_per_1m"] >= 0 for m in data))


class TestChat(ServiceTestCase):
    async def test_chat_ok(self):
        response = await self.post("/chat", HI)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["content"], "Нажмите «Забыли пароль?»")
        self.assertEqual((body["model"], body["finish_reason"], body["cached"]), ("test-model", "stop", False))
        self.assertEqual(body["usage"], {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17})
        call = self.completions.calls[0]
        self.assertEqual((call["model"], call["temperature"], call["max_tokens"]), ("test-model", 0.3, 1024))

    async def test_repeat_request_served_from_cache(self):
        first = (await self.post("/chat", HI)).json()
        second = (await self.post("/chat", HI)).json()
        self.assertEqual((first["cached"], second["cached"]), (False, True))
        self.assertEqual(first["content"], second["content"])
        self.assertEqual(len(self.completions.calls), 1)
        key = next(iter(self.cache.data))
        self.assertRegex(key, r"^chat:[0-9a-f]{64}$")
        self.assertEqual(self.cache.ttl[key], 600)

    def test_cache_key_ignores_ids_but_not_parameters(self):
        base = ChatRequest(**HI, model="m")
        same = ChatRequest(**HI, model="m", user_id="u-1", session_id="s-1")
        hotter = ChatRequest(**HI, model="m", temperature=1.0)
        self.assertEqual(LLMService.cache_key(base), LLMService.cache_key(same))
        self.assertNotEqual(LLMService.cache_key(base), LLMService.cache_key(hotter))

    async def test_redis_down_does_not_break_chat(self):
        self.use(self.completions, FakeRedis(down=True))
        with self.assertLogs("llm-service", "WARNING"):
            response = await self.post("/chat", HI)
        self.assertEqual((response.status_code, response.json()["cached"]), (200, False))

    async def test_provider_errors_mapped(self):
        cases = [
            (auth_error(), 502, "llm_auth"),
            (rate_limit_error(), 429, "llm_rate_limit"),
            (timeout_error(), 504, "llm_timeout"),
            (connection_error(), 502, "llm_unavailable"),
            (not_found_error(), 502, "llm_error"),
        ]
        for error, status, code in cases:
            with self.subTest(code=code):
                self.use(FakeCompletions(error=error), FakeRedis())
                with self.assertLogs("llm-service", "WARNING"):
                    response = await self.post("/chat", HI)
                self.assertEqual(response.status_code, status)
                body = response.json()
                self.assertEqual(body["error"]["code"], code)
                self.assertEqual(body["error"]["request_id"], response.headers["X-Request-ID"])
                self.assertNotIn("Traceback", response.text)
                if code == "llm_rate_limit":
                    self.assertEqual(response.headers["Retry-After"], "7")
                if code == "llm_error":
                    self.assertIn("test-model", body["error"]["message"])

    async def test_unexpected_error_is_json_500(self):
        self.use(FakeCompletions(error=RuntimeError("boom")), FakeRedis())
        with self.assertLogs("llm-service", "ERROR"):
            async with self.client(raise_app_exceptions=False) as http:
                response = await http.post("/chat", json=HI)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "internal_error")
        self.assertNotIn("boom", response.text)

    async def test_validation_errors_have_field_and_message(self):
        cases = [
            ({"messages": []}, "messages"),
            ({**HI, "temperature": 5}, "temperature"),
            ({**HI, "max_tokens": 16_001}, "max_tokens"),
            ({"messages": [{"role": "assistant", "content": "hi"}]}, "body"),
        ]
        for payload, field in cases:
            with self.subTest(field=field):
                response = await self.post("/chat", payload)
                self.assertEqual(response.status_code, 422)
                error = response.json()["error"]
                self.assertEqual(error["code"], "validation_error")
                self.assertEqual(error["fields"][0]["field"], field)
                self.assertTrue(error["fields"][0]["message"])
        async with self.client() as http:
            broken = await http.post("/chat", content=b"{oops", headers={"Content-Type": "application/json"})
        self.assertEqual(broken.json()["error"]["fields"][0]["field"], "body")
        self.assertEqual(self.completions.calls, [])


class TestStream(ServiceTestCase):
    async def frames(self, payload: object) -> tuple[httpx.Response, list[str]]:
        async with self.client() as http, http.stream("POST", "/chat/stream", json=payload) as response:
            lines = [line async for line in response.aiter_lines()]
        return response, [line[len("data: "):] for line in lines if line.startswith("data: ")]

    async def test_stream_frames_usage_and_done(self):
        response, frames = await self.frames(HI)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertEqual(frames[-1], "[DONE]")
        events = [json.loads(f) for f in frames[:-1]]
        self.assertEqual("".join(e["content"] for e in events if "content" in e), "Раз, два, три")
        self.assertEqual(events[-1], {"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}})
        call = self.completions.calls[0]
        self.assertTrue(call["stream"])
        self.assertEqual(call["stream_options"], {"include_usage": True})
        self.assertTrue(self.completions.streams[0].closed)

    async def test_error_before_first_token_is_json(self):
        self.use(FakeCompletions(error=auth_error()), FakeRedis())
        with self.assertLogs("llm-service", "WARNING"):
            response = await self.post("/chat/stream", HI)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["code"], "llm_auth")

    async def test_error_mid_stream_sends_error_frame(self):
        self.use(FakeCompletions(fail_after=2), FakeRedis())
        response, frames = await self.frames(HI)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(frames[-1], "[DONE]")
        self.assertEqual(json.loads(frames[-2])["error"]["code"], "llm_unavailable")

    async def test_stream_is_not_cached(self):
        await self.frames(HI)
        self.assertEqual(self.cache.data, {})


class TestMiddlewareAndCors(ServiceTestCase):
    async def test_request_id_generated_propagated_and_logged(self):
        with self.assertLogs("llm-service", "INFO") as logs:
            generated = await self.post("/chat", HI)
            own = await self.post("/chat", HI, headers={"X-Request-ID": "rid-abc-123"})
            hostile = await self.post("/chat", HI, headers={"X-Request-ID": "bad id\r\nstatus=200"})
        self.assertRegex(generated.headers["X-Request-ID"], r"^[0-9a-f]{32}$")
        self.assertEqual(own.headers["X-Request-ID"], "rid-abc-123")
        self.assertRegex(hostile.headers["X-Request-ID"], r"^[0-9a-f]{32}$")
        line = next(m for m in logs.output if "rid-abc-123" in m)
        self.assertRegex(line, r"request_id=rid-abc-123 method=POST path=/chat status=200 duration_ms=[\d.]+")

    async def test_cors_allows_only_configured_origin(self):
        preflight = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
        async with self.client() as http:
            ok = await http.options("/chat", headers={"Origin": "http://localhost:3000", **preflight})
            foreign = await http.options("/chat", headers={"Origin": "http://evil.example", **preflight})
            simple = await http.get("/models", headers={"Origin": "http://localhost:3000"})
        self.assertEqual(ok.headers["access-control-allow-origin"], "http://localhost:3000")
        self.assertNotIn("access-control-allow-origin", foreign.headers)
        self.assertIn("X-Request-ID", simple.headers["access-control-expose-headers"])


class TestOpenAPI(unittest.TestCase):
    def test_swagger_examples_summaries_and_responses(self):
        spec = app.openapi()
        for path in ("/chat", "/chat/stream"):
            with self.subTest(path=path):
                operation = spec["paths"][path]["post"]
                self.assertTrue(operation["summary"])
                self.assertEqual(set(operation["responses"]), {"200", "422", "429", "502", "504"})
                body = operation["requestBody"]["content"]["application/json"]
                self.assertEqual(len(body["examples"]), 2)
        self.assertEqual(len(spec["components"]["schemas"]["ChatRequest"]["examples"]), 2)
        for path in ("/health", "/ready", "/models"):
            self.assertTrue(spec["paths"][path]["get"]["summary"])
        self.assertEqual(set(spec["paths"]["/ready"]["get"]["responses"]), {"200", "503"})
        example = spec["paths"]["/chat"]["post"]["requestBody"]["content"]["application/json"]["examples"]
        for item in example.values():           # примеры проходят валидацию: «Try it out» не даст 422
            ChatRequest.model_validate(item["value"])


class TestBulkhead(unittest.IsolatedAsyncioTestCase):
    async def test_limiter_caps_parallel_calls(self):
        completions = FakeCompletions(delay=0.05)
        settings = Settings(llm={"openai_api_key": "k", "default_model": "m"}, _env_file=None)
        service = LLMService(SimpleNamespace(chat=SimpleNamespace(completions=completions)), None,
                             settings, limiter=asyncio.Semaphore(2))
        requests = [ChatRequest(messages=[{"role": "user", "content": f"q{i}"}]) for i in range(6)]
        await asyncio.gather(*(service.complete(r) for r in requests))
        self.assertEqual(completions.max_active, 2)


class TestLifespan(unittest.TestCase):
    def test_starts_without_redis(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from fastapi.testclient import TestClient
        get_settings.cache_clear()
        self.addCleanup(get_settings.cache_clear)
        with self.assertLogs("llm-service", "WARNING") as logs, TestClient(app) as http:
            # Клиент Redis создан, хотя Redis не отвечает: поднимется — переподключится сам.
            self.assertIsNotNone(app.state.cache)
            self.assertIsInstance(app.state.openai, openai.AsyncOpenAI)
            self.assertEqual(http.get("/health").status_code, 200)
            self.assertEqual(http.get("/ready").status_code, 503)
        self.assertTrue(any(re.search("Redis .* недоступен", m) for m in logs.output))


if __name__ == "__main__":
    unittest.main()
