"""
Повторы вызова судьи (eval/run_evaluation.py) — без сети и без настоящих пауз.

Windows-прогон с судьёй google/gemma-4-31b-it:free: все три вызова — 429 «temporarily
rate-limited upstream» (бесплатный канал Google AI Studio общий для всех пользователей
OpenRouter). SDK повторял через доли секунды и сдавался. Теперь: пауза 20, 40, 60 с
(или Retry-After), дневной лимит ключа не повторяется, бесплатные модели — не чаще
20 запросов в минуту.
"""
from __future__ import annotations

import httpx
import openai
import pytest

import run_evaluation
from conftest import fake_completion

VERDICT = '{"reasoning": "...", "scores": {"relevance": 5, "correctness": 5, "completeness": 5}, "explanation": "ок"}'
ITEM = {"id": "faq_001", "question": "q", "expected_answer": "a", "expected_keywords": []}
ANSWER = {"answer": "ответ"}


def rate_limited(message: str = "google/gemma-4-31b-it:free is temporarily rate-limited upstream",
                 retry_after: str | None = None) -> openai.RateLimitError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    headers = {"retry-after": retry_after} if retry_after else {}
    return openai.RateLimitError(message, response=httpx.Response(429, request=request, headers=headers), body=None)


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(round(seconds, 1))


async def test_upstream_429_waits_and_retries(mocker):
    client = mocker.Mock()
    client.chat.completions.create = mocker.AsyncMock(
        side_effect=[rate_limited(), rate_limited(), fake_completion(VERDICT)])
    sleeps = Sleeps()
    [judged] = await run_evaluation.judge_all([ITEM], [ANSWER], [[]], client, "judge-model", sleep=sleeps)
    assert judged["verdict"].scores.correctness == 5 and judged["error"] is None
    assert sleeps.calls == [20.0, 40.0]


async def test_retry_after_header_is_respected(mocker):
    client = mocker.Mock()
    client.chat.completions.create = mocker.AsyncMock(
        side_effect=[rate_limited(retry_after="7"), fake_completion(VERDICT)])
    sleeps = Sleeps()
    await run_evaluation.judge_all([ITEM], [ANSWER], [[]], client, "judge-model", sleep=sleeps)
    assert sleeps.calls == [7.0]


async def test_gives_up_after_three_waits(mocker):
    client = mocker.Mock()
    client.chat.completions.create = mocker.AsyncMock(side_effect=rate_limited())
    sleeps = Sleeps()
    [judged] = await run_evaluation.judge_all([ITEM], [ANSWER], [[]], client, "judge-model", sleep=sleeps)
    assert sleeps.calls == [20.0, 40.0, 60.0] and client.chat.completions.create.await_count == 4
    assert judged["error"].startswith("judge_call: RateLimitError")


async def test_daily_limit_stops_judging(mocker):
    client = mocker.Mock()
    client.chat.completions.create = mocker.AsyncMock(
        side_effect=rate_limited("Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day"))
    sleeps = Sleeps()
    items = [ITEM, {**ITEM, "id": "faq_002"}, {**ITEM, "id": "faq_003"}]
    judged = await run_evaluation.judge_all(items, [ANSWER] * 3, [[]] * 3, client, "judge-model", sleep=sleeps)
    client.chat.completions.create.assert_awaited_once()      # дальше не тратим запросы
    assert sleeps.calls == []
    assert all(j["error"].startswith("judge_daily_limit") for j in judged)


async def test_free_models_are_paced(mocker):
    client = mocker.Mock()
    client.chat.completions.create = mocker.AsyncMock(return_value=fake_completion(VERDICT))
    sleeps = Sleeps()
    items = [ITEM, {**ITEM, "id": "faq_002"}, {**ITEM, "id": "faq_003"}]
    await run_evaluation.judge_all(items, [ANSWER] * 3, [[]] * 3, client, "google/gemma-4-31b-it:free", sleep=sleeps)
    assert len(sleeps.calls) == 2 and all(2.5 <= s <= run_evaluation.FREE_MODEL_INTERVAL for s in sleeps.calls)
    sleeps.calls.clear()
    await run_evaluation.judge_all(items, [ANSWER] * 3, [[]] * 3, client, "qwen3:4b-instruct", sleep=sleeps)
    assert sleeps.calls == []                                  # локальный судья — без пауз


async def test_connection_error_retried_briefly(mocker):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    client = mocker.Mock()
    client.chat.completions.create = mocker.AsyncMock(
        side_effect=[openai.APIConnectionError(request=request), fake_completion(VERDICT)])
    sleeps = Sleeps()
    [judged] = await run_evaluation.judge_all([ITEM], [ANSWER], [[]], client, "judge-model", sleep=sleeps)
    assert judged["verdict"] is not None and sleeps.calls == [5.0]


async def test_congested_provider_stops_after_two_items(mocker):
    """Канал модели перегружен надолго: после двух вопросов подряд без ответа остальные не
    отправляются — иначе 25 вопросов × 4 попытки сожгли бы дневной лимит ключа."""
    client = mocker.Mock()
    client.chat.completions.create = mocker.AsyncMock(side_effect=rate_limited())
    sleeps = Sleeps()
    items = [{**ITEM, "id": f"faq_00{i}"} for i in range(1, 6)]
    judged = await run_evaluation.judge_all(items, [ANSWER] * 5, [[]] * 5, client, "judge-model", sleep=sleeps)
    assert client.chat.completions.create.await_count == 8          # 2 вопроса × 4 попытки
    assert [j["error"].split(":", 1)[0] for j in judged] == ["judge_call"] * 2 + ["judge_unavailable"] * 3


async def test_single_rate_limited_item_does_not_stop_run(mocker):
    client = mocker.Mock()
    client.chat.completions.create = mocker.AsyncMock(
        side_effect=[rate_limited()] * 4 + [fake_completion(VERDICT)] * 2)
    sleeps = Sleeps()
    items = [ITEM, {**ITEM, "id": "faq_002"}, {**ITEM, "id": "faq_003"}]
    judged = await run_evaluation.judge_all(items, [ANSWER] * 3, [[]] * 3, client, "judge-model", sleep=sleeps)
    assert judged[0]["error"].startswith("judge_call") and judged[1]["verdict"] and judged[2]["verdict"]


def test_judge_client_has_no_sdk_retries():
    assert run_evaluation.make_judge_client("https://openrouter.ai/api/v1", "sk-or-test", 60).max_retries == 0


@pytest.mark.parametrize(("text", "daily"), [
    ("Rate limit exceeded: free-models-per-day", True),
    ("google/gemma-4-31b-it:free is temporarily rate-limited upstream", False),
])
def test_daily_limit_detection(text, daily):
    assert run_evaluation.is_daily_limit(rate_limited(text)) is daily
