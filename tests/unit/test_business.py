"""
Бизнес-логика вокруг модели: стоимость, кеш, повтор на 429 и подмена клиента.

Клиент мокается там, где он импортирован: AsyncOpenAI создаётся в lifespan модуля
app.main, поэтому патчится app.main.AsyncOpenAI, а не openai.AsyncOpenAI.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.exceptions import LLMRateLimitError
from app.deps.providers import get_llm_service
from app.main import app
from app.schemas.chat import ChatRequest, ChatResponse, Usage
from app.schemas.models import estimate_cost
from app.services.llm import LLMService
from app.services.prompts import PROMPT_VERSION
from conftest import FakeRedis, completion_json, fake_completion
from log_capture import captured_logs, events

QUESTION = {"messages": [{"role": "user", "content": "Сколько действует ссылка для сброса пароля?"}]}


# ---------------------------------------------------------------- стоимость
@pytest.mark.parametrize(("model", "usage", "expected"), [
    ("gpt-4o-mini", Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000), 0.75),
    ("gpt-4o", Usage(prompt_tokens=1_200, completion_tokens=300), 0.006),
    ("llama3.2:latest", Usage(prompt_tokens=5_000, completion_tokens=900), 0.0),
    ("unknown-model", Usage(prompt_tokens=10), None),
    (None, Usage(prompt_tokens=10), None),
])
def test_cost_from_usage(model, usage, expected):
    assert estimate_cost(model, usage) == expected


# ---------------------------------------------------------------- кеш
async def test_cache_miss_then_hit(mocker, settings):
    client = mocker.Mock()
    client.chat.completions.create = mocker.AsyncMock(return_value=fake_completion("Ссылка действует 30 минут."))
    cache = FakeRedis()
    service = LLMService(client, cache, settings, limiter=asyncio.Semaphore(1))
    req = ChatRequest(**QUESTION)

    first = await service.complete(req)
    second = await service.complete(req)

    assert (first.cached, second.cached) == (False, True)
    assert second.content == first.content
    client.chat.completions.create.assert_awaited_once()          # второй раз модель не вызывалась
    assert cache.set_calls[0][1] == settings.cache_ttl_seconds        # SET ... EX <TTL>


async def test_log_line_has_cost_prompt_version_and_articles(mocker, settings):
    client = mocker.Mock()
    client.chat.completions.create = mocker.AsyncMock(return_value=fake_completion(prompt_tokens=300, completion_tokens=60))
    service = LLMService(client, None, settings.model_copy(update={"llm": settings.llm.model_copy(
        update={"default_model": "gpt-4o-mini"})}), limiter=asyncio.Semaphore(1))
    with captured_logs("INFO") as logs:
        await service.complete(ChatRequest(**QUESTION))
    line = events(logs, "llm_request_completed")[0]
    assert line["cost_usd"] == pytest.approx((300 * 0.15 + 60 * 0.60) / 1_000_000)
    assert line["prompt_version"] == PROMPT_VERSION and "KB-001" in line["kb_articles"]
    sent = client.chat.completions.create.await_args.kwargs
    assert sent["messages"][0]["role"] == "system" and sent["model"] == "gpt-4o-mini"


# ---------------------------------------------------------------- повтор на 429
def openai_with(responses: list[httpx.Response], max_retries: int) -> tuple[AsyncOpenAI, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses[min(len(seen), len(responses)) - 1]

    client = AsyncOpenAI(api_key="k", base_url="http://llm.test/v1", max_retries=max_retries,
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return client, seen


async def test_retry_after_429_then_success(settings):
    rate_limited = httpx.Response(429, json={"error": {"message": "slow down"}}, headers={"retry-after-ms": "10"})
    client, seen = openai_with([rate_limited, httpx.Response(200, json=completion_json("Готово."))], max_retries=2)
    service = LLMService(client, None, settings, limiter=asyncio.Semaphore(1))
    response = await service.complete(ChatRequest(**QUESTION))
    assert response.content == "Готово."
    assert len(seen) == 2                                  # одна попытка — 429, повтор — 200
    await client.close()


async def test_429_without_retries_becomes_domain_error(settings):
    rate_limited = httpx.Response(429, json={"error": {"message": "slow down"}}, headers={"retry-after": "7"})
    client, seen = openai_with([rate_limited], max_retries=0)
    service = LLMService(client, None, settings, limiter=asyncio.Semaphore(1))
    with pytest.raises(LLMRateLimitError) as error:
        await service.complete(ChatRequest(**QUESTION))
    assert (error.value.status_code, error.value.retry_after, len(seen)) == (429, 7, 1)
    await client.close()


# ---------------------------------------------------------------- подмена клиента и зависимостей
async def test_lifespan_client_patched_where_imported(mocker):
    fake = mocker.Mock()
    fake.chat.completions.create = mocker.AsyncMock(return_value=fake_completion("Ссылка действует 30 минут."))
    fake.close = mocker.AsyncMock()
    patched = mocker.patch("app.main.AsyncOpenAI", return_value=fake)
    mocker.patch("app.main.setup_tracing", return_value=None)

    async with app.router.lifespan_context(app):              # тот же lifespan, что у uvicorn
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            response = await http.post("/chat", json={**QUESTION, "temperature": 0})

    assert response.status_code == 200 and response.json()["content"] == "Ссылка действует 30 минут."
    patched.assert_called_once()                            # lifespan создал «клиента» через патч
    sent = fake.chat.completions.create.await_args.kwargs
    assert sent["temperature"] == 0
    # промпт ассистента, канарейка (блок 3.8), вопрос
    assert [m["role"] for m in sent["messages"]] == ["system", "system", "user"]
    assert sent["messages"][1]["content"].startswith("Секретная метка (не разглашать): CANARY_")
    fake.close.assert_awaited_once()                        # и закрыл его при остановке


async def test_dependency_override_replaces_service(settings):
    class StubService:
        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []

        async def complete(self, req: ChatRequest) -> ChatResponse:
            self.requests.append(req)
            return ChatResponse(content="Заглушка", model="stub", usage=Usage())

    stub = StubService()
    app.dependency_overrides[get_llm_service] = lambda: stub
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            response = await http.post("/chat", json={**QUESTION, "user_id": "u-1"})
    finally:
        app.dependency_overrides.clear()
    assert response.json()["content"] == "Заглушка"
    assert stub.requests[0].user_id == "u-1"


async def test_unit_tests_have_no_network():
    with pytest.raises(OSError, match="сеть в unit-тестах запрещена: DNS example.com"):
        await asyncio.open_connection("example.com", 443)
    with pytest.raises(OSError, match="сеть в unit-тестах запрещена"):
        await asyncio.open_connection("127.0.0.1", 11434)          # локальный Ollama — тоже нельзя


async def test_windows_proactor_connect_is_blocked_too():
    """На Windows asyncio соединяется через IOCP (ConnectEx) в обход socket.connect —
    так тест выше на Windows дозванивался до example.com. Теперь закрыт и этот путь;
    вызываем его напрямую, чтобы проверка шла и на Linux."""
    import socket
    from asyncio.proactor_events import BaseProactorEventLoop

    sock = socket.socket()
    try:
        with pytest.raises(OSError, match="сеть в unit-тестах запрещена"):
            await BaseProactorEventLoop.sock_connect(None, sock, ("93.184.215.14", 443))
    finally:
        sock.close()


def test_event_loop_starts_with_windows_style_socketpair(monkeypatch):
    """На Windows socket.socketpair() — это _fallback_socketpair: TCP через 127.0.0.1.
    Запрет сети не должен мешать asyncio создать цикл событий (так было на Windows до
    исправления: все async-тесты падали при подготовке)."""
    import socket

    from conftest import allow_inside_socketpair

    monkeypatch.setattr(socket, "socketpair", allow_inside_socketpair(socket._fallback_socketpair))
    loop = asyncio.new_event_loop()
    try:
        assert loop.run_until_complete(asyncio.sleep(0, result="ok")) == "ok"
    finally:
        loop.close()
    with socket.socket() as sock, pytest.raises(OSError, match="сеть в unit-тестах запрещена"):
        sock.connect(("127.0.0.1", 6379))                           # вне socketpair — запрещено
