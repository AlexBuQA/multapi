"""
Тесты наблюдаемости (блок 3.6): JSON-логи с request_id, отсутствие сырых PII в логе,
спаны с атрибутами gen_ai.* и связь span OpenAI SDK с нашим span llm.chat.

Сеть и Phoenix не нужны: спаны собирает InMemorySpanExporter, ответы модели отдаёт
httpx.MockTransport — настоящий AsyncOpenAI, поэтому автоинструментация OpenInference
срабатывает так же, как в сервисе.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "tests"):          # tests — для общего помощника log_capture
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ["LLM__OPENAI_API_KEY"] = "test-key"
os.environ["LLM__DEFAULT_MODEL"] = "test-model"
os.environ["CORS_ORIGINS"] = '["http://localhost:3000"]'
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"
# Блок 3.8: .env разработчика не меняет поведение тестов — ни выключенный на время
# garak baseline защитный слой, ни лимит запросов, ни файл лога.
os.environ["SECURITY__ENABLED"] = "true"
os.environ["RATE_LIMIT_PER_MIN"] = "0"
os.environ["LOG_FILE"] = os.devnull

import httpx  # noqa: E402
import structlog  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from openinference.instrumentation.openai import OpenAIInstrumentor  # noqa: E402
from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from app.core.config import Settings, get_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.observability.logging import get_logger, setup_logging  # noqa: E402
from app.observability.tracing import setup_tracing, traces_endpoint  # noqa: E402
from log_capture import captured_logs, events, quiet_logs  # noqa: E402

quiet_logs()

# Один провайдер на процесс: глобальный TracerProvider в OpenTelemetry задаётся один раз.
EXPORTER = InMemorySpanExporter()
PROVIDER = TracerProvider()
PROVIDER.add_span_processor(SimpleSpanProcessor(EXPORTER))
trace.set_tracer_provider(PROVIDER)

PII_PROMPT = "Мой email ivan@mail.ru, тел +7 (999) 123-45-67, карта 4111 1111 1111 1111. Не приходит письмо."
# Фрагменты с разделителями: голые «999» и «4111» встречались в случайных trace_id и
# временных метках, и тест падал без утечки (например, trace_id …29996d…).
PII_FRAGMENTS = ("ivan@mail.ru", "(999)", "123-45-67", "4111 1111")


def completion_json(model: str) -> dict:
    return {"id": "c1", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "Проверьте папку «Спам»."}}],
            "usage": {"prompt_tokens": 21, "completion_tokens": 7, "total_tokens": 28}}


def stream_body(model: str) -> bytes:
    def chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> str:
        data = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": model,
                "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage:
            data["usage"] = usage
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    parts = [chunk({"role": "assistant", "content": "Раз"}), chunk({"content": ", два"}),
             chunk({}, "stop"), chunk({}, usage={"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}),
             "data: [DONE]\n\n"]
    return "".join(parts).encode()


def mock_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if body.get("stream"):
        return httpx.Response(200, content=stream_body(body["model"]),
                              headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=completion_json(body["model"]))


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self.data[key] = value
        return True

    async def ping(self) -> bool:
        return True


class ObservabilityTestCase(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        OpenAIInstrumentor().instrument(tracer_provider=PROVIDER)

    @classmethod
    def tearDownClass(cls) -> None:
        OpenAIInstrumentor().uninstrument()

    async def asyncSetUp(self) -> None:
        EXPORTER.clear()
        self.cache = FakeRedis()
        app.state.openai = AsyncOpenAI(api_key="k", base_url="http://llm.test/v1", max_retries=0,
                                       http_client=httpx.AsyncClient(transport=httpx.MockTransport(mock_handler)))
        app.state.cache = self.cache
        app.state.llm_limiter = asyncio.Semaphore(4)
        settings = Settings(llm={"openai_api_key": "k", "default_model": "test-model"}, _env_file=None)
        app.dependency_overrides[get_settings] = lambda: settings
        self.addCleanup(app.dependency_overrides.clear)

    async def asyncTearDown(self) -> None:
        await app.state.openai.close()

    async def post(self, path: str, payload: dict, **kwargs) -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            return await http.post(path, json=payload, **kwargs)


class TestJsonLogs(ObservabilityTestCase):
    async def test_request_id_shared_by_http_and_llm_lines(self):
        payload = {"messages": [{"role": "user", "content": PII_PROMPT}], "user_id": "u-42", "session_id": "s-1"}
        with captured_logs("INFO") as logs:
            response = await self.post("/chat", payload, headers={"X-Request-ID": "req-123"})
        self.assertEqual(response.headers["x-request-id"], "req-123")
        http_line = events(logs, "http_request")[0]
        llm_line = events(logs, "llm_request_completed")[0]
        self.assertEqual(http_line["request_id"], llm_line["request_id"])
        self.assertEqual(llm_line["request_id"], "req-123")
        self.assertTrue(all(record.get("request_id") == "req-123" for record in logs))
        for field in ("model", "input_tokens", "output_tokens", "latency_ms", "finish_reason"):
            self.assertIn(field, llm_line)
        self.assertEqual((llm_line["model"], llm_line["input_tokens"], llm_line["output_tokens"],
                          llm_line["finish_reason"]), ("test-model", 21, 7, "stop"))
        self.assertEqual((llm_line["user_id"], llm_line["session_id"]), ("u-42", "s-1"))
        # user_id и session_id из тела видны и в строке middleware (ASGI-middleware, одна задача)
        self.assertEqual((http_line["user_id"], http_line["session_id"]), ("u-42", "s-1"))
        self.assertRegex(llm_line["timestamp"], r"^\d{4}-\d\d-\d\dT.*(Z|\+00:00)$")

    async def test_no_raw_pii_in_logs(self):
        buffer = io.StringIO()
        setup_logging("DEBUG", stream=buffer)
        self.addCleanup(quiet_logs)
        await self.post("/chat", {"messages": [{"role": "user", "content": PII_PROMPT}]})
        text = buffer.getvalue()
        for fragment in PII_FRAGMENTS:
            self.assertNotIn(fragment, text)
        line = next(json.loads(row) for row in text.splitlines() if '"llm_request_completed"' in row)
        self.assertTrue(line["prompt_preview"].startswith("Мой email [EMAIL], тел [PHONE_RU], карта [CARD]"))
        self.assertRegex(line["prompt_hash"], r"^sha256:[0-9a-f]{16}$")

    async def test_cache_hit_and_stream_lines(self):
        payload = {"messages": [{"role": "user", "content": "Как сбросить пароль?"}]}
        with captured_logs("INFO") as logs:
            await self.post("/chat", payload)
            await self.post("/chat", payload)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                async with http.stream("POST", "/chat/stream", json=payload) as response:
                    [line async for line in response.aiter_lines()]
        self.assertEqual(len(events(logs, "llm_cache_hit")), 1)
        stream_line = [r for r in events(logs, "llm_request_completed") if r["stream"]][0]
        self.assertEqual((stream_line["input_tokens"], stream_line["output_tokens"],
                          stream_line["finish_reason"]), (9, 4, "stop"))
        self.assertIsNotNone(stream_line["ttft_ms"])
        # строка http_request потока — после конца потока и с его полной длительностью
        http_line = [r for r in events(logs, "http_request") if r["path"] == "/chat/stream"][0]
        self.assertGreater(logs.index(http_line), logs.index(stream_line))
        self.assertGreaterEqual(http_line["latency_ms"], stream_line["latency_ms"])
        self.assertEqual(http_line["request_id"], stream_line["request_id"])

    async def test_user_id_header_and_invalid_request_id(self):
        with captured_logs("INFO") as logs:
            response = await self.post("/chat", {"messages": [{"role": "user", "content": "hi"}]},
                                       headers={"X-Request-ID": "bad id\n{}", "X-User-ID": "u-7"})
        request_id = response.headers["x-request-id"]
        self.assertRegex(request_id, r"^[0-9a-f]{12}$")        # небезопасный ID заменён своим
        for record in logs:
            self.assertEqual((record["request_id"], record["user_id"]), (request_id, "u-7"))

    async def test_healthchecks_only_at_debug(self):
        with captured_logs("INFO") as info_logs:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                await http.get("/ready")
        with captured_logs("DEBUG") as debug_logs:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                await http.get("/ready")
        self.assertEqual(events(info_logs, "http_request"), [])
        self.assertEqual(events(debug_logs, "http_request")[0]["path"], "/ready")

    def test_contextvars_reach_any_log_line(self):
        buffer = io.StringIO()
        setup_logging("INFO", stream=buffer)
        self.addCleanup(quiet_logs)
        structlog.contextvars.bind_contextvars(request_id="abc123def456")
        self.addCleanup(structlog.contextvars.clear_contextvars)
        get_logger().info("custom_event", value=1)
        record = json.loads(buffer.getvalue())
        self.assertEqual((record["event"], record["request_id"], record["level"]), ("custom_event", "abc123def456", "info"))


class TestTracing(ObservabilityTestCase):
    def spans(self) -> dict[str, list]:
        result: dict[str, list] = {}
        for span in EXPORTER.get_finished_spans():
            result.setdefault(span.name, []).append(span)
        return result

    async def test_chat_span_has_gen_ai_attributes_and_child_llm_span(self):
        payload = {"messages": [{"role": "user", "content": "Как сбросить пароль?"}],
                   "user_id": "u-42", "session_id": "s-1"}
        with captured_logs("INFO") as logs:
            response = await self.post("/chat", payload)
        spans = self.spans()
        ours = spans["llm.chat"][0]
        attributes = dict(ours.attributes)
        self.assertEqual(attributes["gen_ai.request.model"], "test-model")
        self.assertEqual((attributes["gen_ai.usage.input_tokens"], attributes["gen_ai.usage.output_tokens"]), (21, 7))
        self.assertEqual(attributes["request.id"], response.headers["x-request-id"])
        self.assertEqual((attributes["user.id"], attributes["session.id"]), ("u-42", "s-1"))
        self.assertFalse(attributes["cache.hit"])
        # корень трейса — span HTTP-запроса от FastAPI, llm.chat — его прямой потомок
        http_span = spans["POST /chat"][0]
        self.assertIsNone(http_span.parent)
        self.assertEqual(ours.parent.span_id, http_span.context.span_id)
        self.assertEqual(http_span.attributes["request.id"], response.headers["x-request-id"])
        self.assertEqual(http_span.attributes["http.response.status_code"], 200)
        # вход и выход для списков Phoenix — маскированные, той же длины, что в логе
        self.assertEqual(http_span.attributes["input.value"], "Как сбросить пароль?")
        self.assertEqual(http_span.attributes["output.value"], "Проверьте папку «Спам».")
        self.assertFalse([name for name in spans if name.startswith("fastapi.")])
        # span автоинструментации OpenAI SDK — дочерний, в нём вход, ответ и токены
        sdk_span = spans["ChatCompletion"][0]
        self.assertEqual(sdk_span.parent.span_id, ours.context.span_id)
        self.assertEqual(sdk_span.attributes["llm.model_name"], "test-model")
        self.assertEqual(sdk_span.attributes["llm.token_count.prompt"], 21)
        # trace_id в логе совпадает со span — по нему трейс находится в Phoenix
        trace_id = format(ours.context.trace_id, "032x")
        self.assertEqual(events(logs, "llm_request_completed")[0]["trace_id"], trace_id)
        self.assertEqual(events(logs, "http_request")[0]["trace_id"], trace_id)

    async def test_healthchecks_are_not_traced(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            await http.get("/health")
            await http.get("/ready")
        self.assertEqual(EXPORTER.get_finished_spans(), ())

    async def test_stream_span_is_parent_of_sdk_span(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            async with http.stream("POST", "/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]}) as r:
                [line async for line in r.aiter_lines()]
        spans = self.spans()
        ours = spans["llm.chat"][0]
        self.assertEqual(dict(ours.attributes)["gen_ai.usage.output_tokens"], 4)
        sdk_span = spans["ChatCompletion"][0]
        self.assertEqual(sdk_span.parent.span_id, ours.context.span_id)
        self.assertEqual(spans["POST /chat/stream"][0].attributes["output.value"], "Раз, два")

    async def test_root_span_has_no_raw_pii(self):
        await self.post("/chat", {"messages": [{"role": "user", "content": PII_PROMPT}]})
        root = self.spans()["POST /chat"][0]
        self.assertTrue(root.attributes["input.value"].startswith("Мой email [EMAIL], тел [PHONE_RU], карта [CARD]"))
        for fragment in PII_FRAGMENTS:
            self.assertNotIn(fragment, root.attributes["input.value"])

    async def test_cache_hit_has_no_sdk_span(self):
        payload = {"messages": [{"role": "user", "content": "повтор"}]}
        await self.post("/chat", payload)
        EXPORTER.clear()
        await self.post("/chat", payload)
        spans = self.spans()
        self.assertTrue(dict(spans["llm.chat"][0].attributes)["cache.hit"])
        self.assertNotIn("ChatCompletion", spans)


class TestTraceRedaction(ObservabilityTestCase):
    """Span ChatCompletion по умолчанию хранит текст промпта; OPENINFERENCE_HIDE_INPUTS его скрывает."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        OpenAIInstrumentor().uninstrument()
        self.addCleanup(OpenAIInstrumentor().instrument, tracer_provider=PROVIDER)

    async def test_hide_inputs_env(self):
        with mock.patch.dict(os.environ, {"OPENINFERENCE_HIDE_INPUTS": "true"}):
            OpenAIInstrumentor().instrument(tracer_provider=PROVIDER)   # настройки читаются здесь
        self.addCleanup(OpenAIInstrumentor().uninstrument)
        await self.post("/chat", {"messages": [{"role": "user", "content": PII_PROMPT}]})
        sdk_span = [s for s in EXPORTER.get_finished_spans() if s.name == "ChatCompletion"][0]
        self.assertEqual(sdk_span.attributes["input.value"], "__REDACTED__")
        dump = json.dumps({s.name: dict(s.attributes) for s in EXPORTER.get_finished_spans()}, ensure_ascii=False)
        for fragment in PII_FRAGMENTS:
            self.assertNotIn(fragment, dump)


class TestTracingSetup(unittest.TestCase):
    def test_traces_endpoint(self):
        self.assertEqual(traces_endpoint("http://phoenix:6006"), "http://phoenix:6006/v1/traces")
        self.assertEqual(traces_endpoint("http://phoenix:6006/"), "http://phoenix:6006/v1/traces")
        self.assertEqual(traces_endpoint("http://phoenix:6006/v1/traces"), "http://phoenix:6006/v1/traces")

    def test_disabled_without_endpoint(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PHOENIX_COLLECTOR_ENDPOINT", None)
            with captured_logs("INFO") as logs:
                self.assertIsNone(setup_tracing())
        self.assertEqual(len(events(logs, "tracing_disabled")), 1)

    def test_lifespan_sets_up_tracing_before_openai_client(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from fastapi.testclient import TestClient
        order: list[str] = []
        real_client = AsyncOpenAI

        def fake_setup(*args, **kwargs):
            order.append("setup_tracing")
            return None

        def fake_client(*args, **kwargs):
            order.append("AsyncOpenAI")
            return real_client(*args, **kwargs)

        with mock.patch("app.main.setup_tracing", fake_setup), mock.patch("app.main.AsyncOpenAI", fake_client), \
                TestClient(app):
            pass
        self.assertEqual(order, ["setup_tracing", "AsyncOpenAI"])


if __name__ == "__main__":
    unittest.main()
