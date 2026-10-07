"""
Опциональный Presidio (блок 3.6, задача 5).

Тесты на подменах идут всегда: маскирование выполняется фоновой задачей параллельно с
вызовом модели, ошибка Presidio не роняет запрос и не оставляет в логе имён, без
пакетов сервис работает на regex. Тесты с настоящей моделью запускаются, только если
установлены presidio-analyzer, presidio-anonymizer и ru_core_news_md.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "tests"):          # tests — для общего помощника log_capture
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ["LLM__OPENAI_API_KEY"] = "test-key"
os.environ["LLM__DEFAULT_MODEL"] = "test-model"
os.environ["CORS_ORIGINS"] = '["http://localhost:3000"]'
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"

import httpx  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from app.core.config import Settings, get_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.observability.pii import PREVIEW_CHARS, redact_pii  # noqa: E402
from app.observability.pii_presidio import WINDOW_CHARS, load_redactor  # noqa: E402
from log_capture import captured_logs, events, quiet_logs  # noqa: E402

quiet_logs()

PROMPT = "Меня зовут Иван Петров, живу в Казани. Email ivan@mail.ru. Не приходит письмо для сброса пароля."


def presidio_installed() -> bool:
    if not all(importlib.util.find_spec(name) for name in ("presidio_analyzer", "presidio_anonymizer", "spacy")):
        return False
    import spacy
    return spacy.util.is_package("ru_core_news_md")


class FakeRedactor:
    """Вместо Presidio: «находит» имя и город, работает заданное время в потоке."""

    def __init__(self, delay: float = 0.0, fail: bool = False) -> None:
        self.delay, self.fail = delay, fail
        self.started_at: float | None = None
        self.seen: str | None = None

    def preview_sync(self, raw: str) -> str:
        self.started_at = time.perf_counter()
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("NER упал")
        self.seen = redact_pii(raw)[:WINDOW_CHARS]
        return self.seen.replace("Иван Петров", "[PERSON]").replace("Казани", "[LOCATION]")[:PREVIEW_CHARS]

    async def preview(self, raw: str) -> str:
        return await asyncio.to_thread(self.preview_sync, raw)


class ServiceCase(unittest.IsolatedAsyncioTestCase):
    llm_delay = 0.0

    async def asyncSetUp(self) -> None:
        self.llm_called_at: float | None = None

        async def handler(request: httpx.Request) -> httpx.Response:
            self.llm_called_at = time.perf_counter()
            await asyncio.sleep(self.llm_delay)
            body = json.loads(request.content)
            return httpx.Response(200, json={
                "id": "c1", "object": "chat.completion", "created": 1, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "Проверьте папку «Спам»."}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 6, "total_tokens": 26}})

        app.state.openai = AsyncOpenAI(api_key="k", base_url="http://llm.test/v1", max_retries=0,
                                       http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        app.state.cache = None
        app.state.llm_limiter = asyncio.Semaphore(4)
        settings = Settings(llm={"openai_api_key": "k", "default_model": "test-model"}, _env_file=None)
        app.dependency_overrides[get_settings] = lambda: settings
        self.addCleanup(app.dependency_overrides.clear)
        self.addCleanup(setattr, app.state, "pii_redactor", None)

    async def asyncTearDown(self) -> None:
        await app.state.openai.close()

    async def chat(self) -> list[dict]:
        with captured_logs("INFO") as logs:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
                response = await http.post("/chat", json={"messages": [{"role": "user", "content": PROMPT}]})
        self.assertEqual(response.status_code, 200)
        return logs


class TestBackgroundMasking(ServiceCase):
    llm_delay = 0.3

    async def test_masking_runs_in_parallel_with_llm_call(self):
        redactor = FakeRedactor(delay=0.3)
        app.state.pii_redactor = redactor
        started = time.perf_counter()
        logs = await self.chat()
        elapsed = time.perf_counter() - started
        line = events(logs, "llm_request_completed")[0]
        self.assertTrue(line["prompt_preview"].startswith("Меня зовут [PERSON], живу в [LOCATION]. Email [EMAIL]."))
        # Маскирование началось до ответа модели и шло одновременно с ним: 0,3 + 0,3 с
        # последовательно дали бы больше 0,6 с.
        self.assertLess(redactor.started_at, self.llm_called_at + self.llm_delay)
        self.assertLess(elapsed, 0.55)

    async def test_presidio_failure_drops_preview_but_keeps_request(self):
        app.state.pii_redactor = FakeRedactor(fail=True)
        logs = await self.chat()
        line = events(logs, "llm_request_completed")[0]
        self.assertIsNone(line["prompt_preview"])          # текст только после regex мог бы оставить имя
        self.assertRegex(line["prompt_hash"], r"^sha256:")
        self.assertEqual(len(events(logs, "presidio_failed")), 1)
        self.assertNotIn("Иван", json.dumps(logs, ensure_ascii=False))

    async def test_long_prompt_is_cut_before_ner(self):
        redactor = FakeRedactor()
        app.state.pii_redactor = redactor
        global PROMPT
        original, PROMPT = PROMPT, PROMPT + " Подробности." * 2000
        try:
            await self.chat()
        finally:
            PROMPT = original
        self.assertLessEqual(len(redactor.seen), WINDOW_CHARS)   # NER не видит хвост в 26 000 символов


class TestLoadRedactor(unittest.TestCase):
    def test_without_packages_falls_back_to_regex(self):
        with mock.patch.dict(sys.modules, {"presidio_analyzer": None}), captured_logs("INFO") as logs:
            self.assertIsNone(load_redactor())
        self.assertEqual(len(events(logs, "presidio_unavailable")), 1)

    @unittest.skipUnless(presidio_installed(), "Presidio или ru_core_news_md не установлены")
    def test_without_model_falls_back_to_regex(self):
        with captured_logs("INFO") as logs:
            self.assertIsNone(load_redactor("ru_core_news_nonexistent"))
        self.assertIn("python -m spacy download", events(logs, "presidio_unavailable")[0]["error"])


@unittest.skipUnless(presidio_installed(), "Presidio или ru_core_news_md не установлены")
class TestRealPresidio(ServiceCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.redactor = load_redactor()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.redactor.close()

    def test_names_and_places(self):
        text = self.redactor.redact("Добрый день! Я Мария Сергеевна Кузнецова из Новосибирска, не могу войти.")
        self.assertEqual(text, "Добрый день! Я [PERSON] из [LOCATION], не могу войти.")

    async def test_service_log_line(self):
        app.state.pii_redactor = self.redactor
        logs = await self.chat()
        preview = events(logs, "llm_request_completed")[0]["prompt_preview"]
        self.assertTrue(preview.startswith("Меня зовут [PERSON], живу в [LOCATION]. Email [EMAIL]."), preview)
        for fragment in ("Иван", "Петров", "Казан", "ivan@"):
            self.assertNotIn(fragment, preview)


if __name__ == "__main__":
    unittest.main()
