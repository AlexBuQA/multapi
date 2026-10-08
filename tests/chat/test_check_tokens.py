"""scripts/check_tokens.py: отчёт по двум запросам и понятные ошибки провайдера — без сети."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletion

from app.chat.context import count_tokens

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("check_tokens", ROOT / "scripts" / "check_tokens.py")
check_tokens = importlib.util.module_from_spec(_spec)
sys.modules["check_tokens"] = check_tokens   # dataclass ищет модуль в sys.modules
_spec.loader.exec_module(check_tokens)


class FakeProvider:
    """Считает prompt_tokens как count_tokens плюс постоянная шапка модели."""

    def __init__(self, header: int = 0, usage: bool = True, provider: str | None = "Fake") -> None:
        self.header, self.usage, self.provider = header, usage, provider
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs) -> ChatCompletion:
        self.requests.append(kwargs)
        data = {"id": "x", "object": "chat.completion", "created": 0, "model": kwargs["model"],
                "choices": [{"index": 0, "finish_reason": "length",
                             "message": {"role": "assistant", "content": "Здр"}}]}
        if self.usage:
            prompt = count_tokens(kwargs["messages"]) + self.header
            data["usage"] = {"prompt_tokens": prompt, "completion_tokens": 1, "total_tokens": prompt + 1}
        if self.provider:
            data["provider"] = self.provider
        return ChatCompletion.model_validate(data)


async def test_exact_provider_passes(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeProvider()
    assert await check_tokens.run(fake, "m", "https://openrouter.ai/api/v1") == 0
    out = capsys.readouterr().out
    assert "+0.0 %" in out and "[OK]" in out and "провайдер Fake" in out
    assert len(fake.requests) == 2
    assert all(r["max_tokens"] == check_tokens.MAX_TOKENS for r in fake.requests)


async def test_constant_header_is_measured_and_excluded(capsys: pytest.CaptureFixture[str]) -> None:
    """Модель добавляет к каждому запросу свою шапку (шаблон harmony у gpt-oss — около 60 токенов)."""
    assert await check_tokens.run(FakeProvider(header=61), "m", None) == 1
    out = capsys.readouterr().out
    assert "[!] вне допуска" in out
    assert "+61 токенов в каждом запросе" in out
    assert "Диалог без неё" in out and "+0.0 %  [OK]" in out


async def test_labels_differ_between_requests() -> None:
    fake = FakeProvider()
    await check_tokens.run(fake, "m", None)
    first, second = (r["messages"][0]["content"] for r in fake.requests)
    assert "Метка запроса:" in first and "Метка запроса:" in second
    assert first.rsplit(":", 1)[1] != second.rsplit(":", 1)[1]


async def test_no_usage_means_nothing_to_compare(capsys: pytest.CaptureFixture[str]) -> None:
    assert await check_tokens.run(FakeProvider(usage=False), "m", None) == 2
    assert "не вернул usage.prompt_tokens" in capsys.readouterr().out


def status_error(code: int, message: str) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    body = {"message": message, "code": code}
    return openai.APIStatusError(f"Error code: {code} - {body}", response=httpx.Response(code, request=request),
                                 body=body)


def test_402_explains_paid_model() -> None:
    text = check_tokens.describe_error(status_error(402, "Insufficient credits."))
    assert text.startswith("Провайдер ответил 402: Insufficient credits.")
    assert "python scripts/check_tokens.py" in text


def test_429_suggests_retry() -> None:
    assert "Повторите через минуту" in check_tokens.describe_error(status_error(429, "Rate limit exceeded"))


def test_connection_error_points_to_proxy_settings() -> None:
    exc = openai.APIConnectionError(request=httpx.Request("POST", "https://openrouter.ai/api/v1"))
    assert "LLM__PROXY_URL" in check_tokens.describe_error(exc)


async def test_measure_reports_402_without_traceback(monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    """Ответ OpenRouter на платную модель при пустом счёте — как в прогоне на Windows."""
    def reply(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": {"message": "Insufficient credits. This account never "
                                                              "purchased credits.", "code": 402}})

    real = openai.AsyncOpenAI

    def client(**kwargs):
        kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(reply))
        return real(**kwargs)

    monkeypatch.setattr(openai, "AsyncOpenAI", client)
    assert await check_tokens.measure("openai/gpt-4o-mini") == 2
    out = capsys.readouterr().out
    assert "Провайдер ответил 402: Insufficient credits." in out
    assert "Модель платная, а у ключа нет кредитов" in out
