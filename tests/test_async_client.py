"""
Тесты блока 3.3: AsyncLLMClient. Сеть не используется — вместо
AsyncOpenAI подставляется асинхронная заглушка с задержкой через asyncio.sleep.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.services.llm_client as llm_module  # noqa: E402
from app.config import AsyncClientSettings  # noqa: E402
from app.logging_utils import close_event_logger  # noqa: E402
from app.services.llm_client import AsyncLLMClient  # noqa: E402
from src.config import ProviderConfig  # noqa: E402
from src.robust_client import AllProvidersFailedError  # noqa: E402


class FakeCompletions:
    """Имитация AsyncOpenAI().chat.completions: задержка, учёт одновременных вызовов."""

    def __init__(self, delay: float = 0.05, fail_models: tuple[str, ...] = (), fail: bool = False):
        self.delay, self.fail_models, self.fail = delay, fail_models, fail
        self.calls: list[dict] = []
        self.active = self.max_active = 0

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail or kwargs["model"] in self.fail_models:
            raise RuntimeError("provider down")
        if kwargs.get("stream"):
            return self._stream()
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.active -= 1
        text = "ответ: " + kwargs["messages"][-1]["content"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def _stream(self):
        for part in ("Event loop ", "— это ", "цикл событий."):
            await asyncio.sleep(self.delay)
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=part))],
                                  usage=None)
        yield SimpleNamespace(choices=[],
                              usage=SimpleNamespace(prompt_tokens=8, completion_tokens=6,
                                                    total_tokens=14))


def fake_sdk(completions: FakeCompletions) -> SimpleNamespace:
    async def close():
        return None
    return SimpleNamespace(chat=SimpleNamespace(completions=completions), close=close)


def two_providers_cfg() -> SimpleNamespace:
    primary = ProviderConfig(name="ollama", api_key_env="X", base_url="http://localhost:11434/v1",
                             chat_model="local-model")
    backup = ProviderConfig(name="openrouter", api_key_env="Y", base_url="https://example/v1",
                            chat_model="cloud-model")
    return SimpleNamespace(provider="ollama", chain=lambda: [primary, backup],
                           request_timeout=30.0, proxy="", cache_ttl=3600)


class AsyncClientTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.log_path = Path(self._tmp.name) / "llm_calls.jsonl"
        self.clients: list[AsyncLLMClient] = []

    def tearDown(self) -> None:
        for client in self.clients:
            close_event_logger(client.events)
        self._tmp.cleanup()

    def make_client(self, fakes: dict[str, FakeCompletions], *, concurrency: int = 5,
                    call_timeout: float = 5.0, use_cache: bool = False) -> AsyncLLMClient:
        options = AsyncClientSettings(llm_concurrency=concurrency, llm_call_timeout=call_timeout,
                                      llm_call_log_path=self.log_path)
        client = AsyncLLMClient(cfg=two_providers_cfg(), options=options, use_cache=use_cache,
                                client_factory=lambda p: fake_sdk(fakes[p.name]))
        self.clients.append(client)
        return client

    def events(self, name: str) -> list[dict]:
        with open(self.log_path, encoding="utf-8") as f:
            return [r for r in map(json.loads, f) if r["event"] == name]


# --------------------------------------------------------------------------- #
class TestStaticChecks(unittest.TestCase):
    def test_no_blocking_calls(self):
        source = inspect.getsource(llm_module)
        self.assertIsNone(re.search(r"(?<!Async)OpenAI\(", source), "синхронный OpenAI")
        self.assertNotIn("time.sleep", source)
        self.assertNotIn("import requests", source)

    def test_semaphore_created_once_in_init(self):
        self.assertIn("self._sem = asyncio.Semaphore(", inspect.getsource(AsyncLLMClient.__init__))
        for method in (AsyncLLMClient.complete, AsyncLLMClient.batch_chat,
                       AsyncLLMClient.batch_chat_strict, AsyncLLMClient.stream_chat):
            with self.subTest(method=method.__name__):
                self.assertNotIn("Semaphore(", inspect.getsource(method))


class TestComplete(AsyncClientTestCase):
    async def test_answer_and_call_log(self):
        client = self.make_client({"ollama": FakeCompletions(), "openrouter": FakeCompletions()})
        self.assertEqual(await client.complete("Привет"), "ответ: Привет")
        call = self.events("llm.call")[0]
        self.assertEqual(call["status"], "ok")
        self.assertEqual((call["model"], call["provider"], call["prompt_chars"]),
                         ("local-model", "ollama", 6))
        self.assertGreater(call["duration_ms"], 0)
        self.assertEqual(call["total_tokens"], 15)

    async def test_fallback_to_second_provider(self):
        backup = FakeCompletions()
        client = self.make_client({"ollama": FakeCompletions(fail=True), "openrouter": backup})
        self.assertEqual(await client.complete("x"), "ответ: x")
        self.assertEqual(backup.calls[0]["model"], "cloud-model")
        self.assertEqual(self.events("llm.fallback")[0]["provider"], "ollama")
        self.assertEqual(self.events("llm.call")[0]["provider"], "openrouter")

    async def test_call_timeout(self):
        client = self.make_client({"ollama": FakeCompletions(delay=0.5),
                                   "openrouter": FakeCompletions()}, call_timeout=0.05)
        with self.assertRaises(TimeoutError):
            await client.complete("медленно")
        self.assertEqual(self.events("llm.call")[0]["status"], "timeout")

    async def test_cache_hit(self):
        primary = FakeCompletions()
        client = self.make_client({"ollama": primary, "openrouter": FakeCompletions()},
                                  use_cache=True)
        await client.complete("одно и то же")
        await client.complete("одно и то же")
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual([c["status"] for c in self.events("llm.call")], ["ok", "cache_hit"])


class TestBatch(AsyncClientTestCase):
    async def test_concurrency_limited_and_order_kept(self):
        primary = FakeCompletions(delay=0.05)
        client = self.make_client({"ollama": primary, "openrouter": FakeCompletions()},
                                  concurrency=3)
        prompts = [f"q{i}" for i in range(9)]
        results = await client.batch_chat(prompts, concurrency=3)
        self.assertEqual(results, [f"ответ: q{i}" for i in range(9)])
        self.assertLessEqual(primary.max_active, 3)
        self.assertEqual(primary.max_active, 3)

    async def test_parallel_faster_than_sequential(self):
        client = self.make_client({"ollama": FakeCompletions(delay=0.1),
                                   "openrouter": FakeCompletions()}, concurrency=10)
        started = time.perf_counter()
        await client.batch_chat([f"q{i}" for i in range(10)], concurrency=10)
        self.assertLess(time.perf_counter() - started, 0.5)   # последовательно было бы ≥ 1 с

    async def test_failed_request_returned_in_place(self):
        both_fail = ("bad",)
        client = self.make_client({"ollama": FakeCompletions(fail_models=both_fail),
                                   "openrouter": FakeCompletions(fail=True)})
        results = await client.batch_chat(["a", "b", "c", "d"], models=[None, None, "bad", None])
        self.assertIsInstance(results[2], AllProvidersFailedError)
        self.assertEqual([results[i] for i in (0, 1, 3)], ["ответ: a", "ответ: b", "ответ: d"])

    async def test_concurrency_mismatch_rejected(self):
        client = self.make_client({"ollama": FakeCompletions(), "openrouter": FakeCompletions()},
                                  concurrency=5)
        with self.assertRaises(ValueError):
            await client.batch_chat(["a"], concurrency=10)

    async def test_strict_batch_all_or_nothing(self):
        client = self.make_client({"ollama": FakeCompletions(delay=0.2, fail_models=("bad",)),
                                   "openrouter": FakeCompletions(fail=True)}, concurrency=5)
        caught = []
        try:
            await client.batch_chat_strict(["a", "b", "c", "d", "e"],
                                           models=[None, None, "bad", None, None])
        except* AllProvidersFailedError as group:
            caught = list(group.exceptions)
        self.assertEqual(len(caught), 1)
        failed = self.events("llm.batch_strict_failed")[0]
        self.assertEqual((failed["failed"], failed["cancelled"]), (1, 4))


class TestStream(AsyncClientTestCase):
    async def test_ttft_before_total_and_usage_logged(self):
        client = self.make_client({"ollama": FakeCompletions(delay=0.05),
                                   "openrouter": FakeCompletions()})
        started = time.perf_counter()
        parts, first = [], None
        async for delta in client.stream_chat("Что такое event loop?"):
            first = first or time.perf_counter()
            parts.append(delta)
        total = time.perf_counter() - started
        self.assertEqual("".join(parts), "Event loop — это цикл событий.")
        self.assertLess(first - started, total)
        stream = self.events("llm.stream")[0]
        self.assertEqual((stream["status"], stream["chunks"], stream["total_tokens"]),
                         ("ok", 3, 14))
        self.assertLess(stream["ttft_ms"], stream["duration_ms"])

    async def test_stream_fallback_before_first_token(self):
        client = self.make_client({"ollama": FakeCompletions(fail=True),
                                   "openrouter": FakeCompletions(delay=0.01)})
        parts = [d async for d in client.stream_chat("x")]
        self.assertTrue(parts)
        self.assertEqual(self.events("llm.stream")[0]["provider"], "openrouter")


if __name__ == "__main__":
    unittest.main()
