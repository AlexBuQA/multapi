"""
Персистентный агент блока 6.4 без сети: граф app/services/agent_persistent.py на подставной модели,
checkpointer — AsyncSqliteSaver(":memory:") и файл SQLite; внешний API отправки — AsyncMock.

    pytest tests/test_agent_persistent.py

Обязательный минимум задания — первые три теста: граф доходит до __interrupt__ (snapshot.next — узел
подтверждения), Command(resume=True) выставляет sent=True, Command(resume=False) — sent=False без вызова API.
Дальше — рестарт процесса на файле SQLite, нюанс повторного resume, роли, смешанные вызовы, POST /agent/stream
(SSE), lifespan сервиса, фильтр Alembic. Тесты с Postgres (setup() один раз, таблицы checkpoint*, autogenerate)
идут, если запущен Postgres из compose (127.0.0.1:5433), иначе пропускаются — как tests/chat/.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "tests", ROOT / "tests" / "chat"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Как tests/test_service.py: настройки до импорта app.main, .env разработчика на тесты не влияет.
os.environ["LLM__OPENAI_API_KEY"] = "test-key"
os.environ["LLM__DEFAULT_MODEL"] = "test-model"
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"
os.environ["SECURITY__ENABLED"] = "true"
os.environ["RATE_LIMIT_PER_MIN"] = "0"
os.environ["LOG_FILE"] = os.devnull

import httpx  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402
from log_capture import quiet_logs  # noqa: E402
from pydantic import PrivateAttr  # noqa: E402

from app.chat.repositories.pg_models import (  # noqa: E402
    LANGGRAPH_TABLES,
    Base,
    alembic_include_name,
)
from app.core.config import Settings, get_settings, psycopg_database_url  # noqa: E402
from app.services import agent_persistent as ap  # noqa: E402
from app.services.agent_persistent import (  # noqa: E402
    CONFIRM,
    PREPARE,
    SEND,
    build_agent,
    new_turn,
    thread_config,
)

quiet_logs()

TASK = "Отправь клиенту в Telegram-чат 12345 напоминание: письмо об автопродлении приходит за 3 дня."
TEXT = "Напоминаем: письмо об автопродлении приходит за 3 дня до окончания периода."
SENT = "Сообщение отправлено в 12345"


# ---------------------------------------------------------------- подставная модель
def call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def ask(*calls: dict[str, Any]) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls),
                     usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})


def say(text: str) -> AIMessage:
    return AIMessage(content=text, usage_metadata={"input_tokens": 120, "output_tokens": 5, "total_tokens": 125})


SEND_CALL = call(SEND, {"chat_id": "12345", "text": TEXT}, "call_send_1")


class Scripted(BaseChatModel):
    """Ответ — по числу ответов модели уже в истории thread: n-й вызов в thread получает script[n]. Без
    счётчиков в объекте — разные thread не мешают друг другу. Каждый ответ — копия (add_messages)."""

    script: list[AIMessage]
    _calls: list[list[Any]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Scripted:
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self._calls.append(list(messages))
        done = sum(isinstance(m, AIMessage) for m in messages)
        reply = self.script[min(done, len(self.script) - 1)]
        return ChatResult(generations=[ChatGeneration(message=reply.model_copy())])


def send_then_answer() -> Scripted:
    return Scripted(script=[ask(SEND_CALL), say("Готово.")])


def api() -> AsyncMock:
    """Внешний API отправки: AsyncMock — по нему видно, был ли побочный эффект."""
    return AsyncMock(return_value=SENT)


# ---------------------------------------------------------------- три теста задания
async def test_graph_stops_at_interrupt_with_approve_node_next():
    sender = api()
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        graph = build_agent(saver, model=send_then_answer(), sender=sender)
        config = thread_config("t-interrupt")
        result = await graph.ainvoke(new_turn([HumanMessage(TASK)]), config)

        [pause] = result["__interrupt__"]
        assert pause.value["type"] == "approve_send_message"
        assert pause.value["preview"] == {"chat_id": "12345", "text": TEXT,
                                          "request_id": ap.request_id("t-interrupt", "call_send_1")}
        snapshot = await graph.aget_state(config)
        assert snapshot.next == (CONFIRM,)
        assert snapshot.values["draft"]["text"] == TEXT and snapshot.values["sent"] is False
    sender.assert_not_called()                       # до решения человека — ни одного вызова API


async def test_resume_true_sends_and_sets_sent():
    sender = api()
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        graph = build_agent(saver, model=send_then_answer(), sender=sender)
        config = thread_config("t-approve")
        await graph.ainvoke(new_turn([HumanMessage(TASK)]), config)
        result = await graph.ainvoke(Command(resume=True), config)

    assert result["sent"] is True and result["draft"] is None
    sender.assert_awaited_once_with(chat_id="12345", text=TEXT, request_id=ap.request_id("t-approve", "call_send_1"))
    tool = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool.content == SENT and tool.tool_call_id == "call_send_1"
    assert result["messages"][-1].content == "Готово."


async def test_resume_false_does_not_call_the_api():
    sender = api()
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        graph = build_agent(saver, model=send_then_answer(), sender=sender)
        config = thread_config("t-reject")
        await graph.ainvoke(new_turn([HumanMessage(TASK)]), config)
        result = await graph.ainvoke(Command(resume={"approved": False, "comment": "не тот клиент"}), config)

    assert result["sent"] is False
    sender.assert_not_called()
    tool = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool.content.startswith("Не отправлено: оператор отклонил отправку (не тот клиент)")


# ---------------------------------------------------------------- персистентность и time travel
async def test_state_survives_process_restart(tmp_path):
    """Первый процесс доходит до паузы и закрывается; второй открывает тот же файл и резюмирует."""
    path = str(tmp_path / "checkpoints.sqlite")
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        await saver.setup()
        await build_agent(saver, model=send_then_answer(), sender=api()).ainvoke(
            new_turn([HumanMessage(TASK)]), thread_config("tg-12345"))

    sender = api()
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        graph = build_agent(saver, model=send_then_answer(), sender=sender)
        snapshot = await graph.aget_state(thread_config("tg-12345"))
        assert snapshot.next == (CONFIRM,) and snapshot.interrupts
        result = await graph.ainvoke(Command(resume=True), thread_config("tg-12345"))
    assert result["sent"] is True
    sender.assert_awaited_once()


async def test_repeated_resume_keeps_the_first_decision_and_outbox_blocks_a_second_send():
    """Resume — доставка пропущенного значения, а не переигровка: тот же interrupt-чек-пойнт с другим
    решением получает первое. Узел отправки выполняется заново — второй отправки нет только из-за request_id."""
    stub = Mock(return_value=SENT)
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        graph = build_agent(saver, model=send_then_answer(), sender=ap.TelegramOutbox(send=stub))
        config = thread_config("t-replay")
        await graph.ainvoke(new_turn([HumanMessage(TASK)]), config)
        history = [s async for s in graph.aget_state_history(config)]
        past = history[0].config["configurable"]["checkpoint_id"]
        assert history[0].next == (CONFIRM,)
        await graph.ainvoke(Command(resume=True), config)

        replay = await graph.ainvoke(Command(resume=False), thread_config("t-replay", checkpoint_id=past))

    assert replay["sent"] is True                     # первое решение
    stub.assert_called_once_with("12345", TEXT)       # и одна отправка на два прогона узла
    last_tool = next(m for m in reversed(replay["messages"]) if isinstance(m, ToolMessage))
    assert last_tool.content.startswith("Уже отправлено ранее")


async def test_past_checkpoint_shows_draft_before_sending():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        graph = build_agent(saver, model=send_then_answer(), sender=api())
        config = thread_config("t-history")
        await graph.ainvoke(new_turn([HumanMessage(TASK)]), config)
        await graph.ainvoke(Command(resume=True), config)
        history = [s async for s in graph.aget_state_history(config)]
        nexts = [s.next for s in reversed(history)]
        assert nexts == [("__start__",), ("call_model",), (PREPARE,), (CONFIRM,), ("call_model",), ("force_finish",), ()]
        past = next(s for s in history if s.next == (CONFIRM,)).config["configurable"]["checkpoint_id"]
        old = await graph.aget_state(thread_config("t-history", checkpoint_id=past))
    assert old.values["sent"] is False and old.values["draft"]["chat_id"] == "12345" and old.next == (CONFIRM,)


def test_request_id_is_deterministic():
    assert ap.request_id("tg-1", "call_a") == ap.request_id("tg-1", "call_a")
    assert len({ap.request_id("tg-1", "call_a"), ap.request_id("tg-2", "call_a"), ap.request_id("tg-1", "call_b")}) == 3


async def test_new_turn_in_the_same_thread_resets_counters_and_keeps_history():
    model = Scripted(script=[say("Первый ответ."), say("Второй ответ.")])
    graph = build_agent(InMemorySaver(), model=model, sender=api())
    config = thread_config("t-turns")
    await graph.ainvoke(new_turn([HumanMessage("раз")]), config)
    result = await graph.ainvoke(new_turn([HumanMessage("два")]), config)
    assert result["iteration_count"] == 1
    assert [m.content for m in result["messages"]] == ["раз", "Первый ответ.", "два", "Второй ответ."]


# ---------------------------------------------------------------- роли и разбор вызовов
async def test_read_only_role_refuses_without_interrupt():
    sender = api()
    graph = build_agent(InMemorySaver(), model=send_then_answer(), sender=sender)
    result = await graph.ainvoke(new_turn([HumanMessage(TASK)]), thread_config("t-ro", user_role="read-only"))
    assert "__interrupt__" not in result and result["sent"] is False
    sender.assert_not_called()
    assert next(m for m in result["messages"] if isinstance(m, ToolMessage)).content == ap.READ_ONLY_REFUSAL


async def test_full_role_sends_without_interrupt():
    sender = api()
    graph = build_agent(InMemorySaver(), model=send_then_answer(), sender=sender)
    result = await graph.ainvoke(new_turn([HumanMessage(TASK)]), thread_config("t-full", user_role="full"))
    assert "__interrupt__" not in result and result["sent"] is True
    sender.assert_awaited_once()


async def test_read_only_cannot_approve_on_resume():
    sender = api()
    graph = build_agent(InMemorySaver(), model=send_then_answer(), sender=sender)
    await graph.ainvoke(new_turn([HumanMessage(TASK)]), thread_config("t-ro-resume"))
    result = await graph.ainvoke(Command(resume=True), thread_config("t-ro-resume", user_role="read-only"))
    assert result["sent"] is False
    sender.assert_not_called()


def test_capped_role():
    assert ap.capped_role(None, "write-with-approve") == "write-with-approve"
    assert ap.capped_role("full", "write-with-approve") == "write-with-approve"     # выше границы — нельзя
    assert ap.capped_role("read-only", "full") == "read-only"                       # сузить — можно
    assert ap.capped_role("full", "full") == "full"


async def test_safe_tools_run_first_and_extra_sends_are_closed(monkeypatch):
    """Поиск и две отправки в одном ответе: поиск выполняется сразу, первая отправка — на подтверждение,
    вторая закрыта ToolMessage: за одно подтверждение — одно сообщение. История валидна."""
    from app.tools import naive_tools

    monkeypatch.setattr(naive_tools, "search_knowledge_base", lambda query: "[faq] за 3 дня")
    second = call(SEND, {"chat_id": "777", "text": "ещё"}, "call_send_2")
    model = Scripted(script=[ask(call("search_knowledge_base", {"query": "автопродление"}, "call_kb"), SEND_CALL, second),
                     say("Готово.")])
    sender = api()
    graph = build_agent(InMemorySaver(), model=model, sender=sender)
    config = thread_config("t-mixed")
    result = await graph.ainvoke(new_turn([HumanMessage(TASK)]), config)
    assert result["__interrupt__"][0].value["preview"]["chat_id"] == "12345"
    tools = {m.tool_call_id: m.content for m in result["messages"] if isinstance(m, ToolMessage)}
    assert tools == {"call_kb": "[faq] за 3 дня", "call_send_2": ap.ONE_PER_STEP}
    result = await graph.ainvoke(Command(resume=True), config)
    asked = {c["id"] for m in result["messages"] if isinstance(m, AIMessage) for c in m.tool_calls}
    answered = {m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)}
    assert asked == answered and sender.await_count == 1


async def test_invalid_send_arguments_go_back_to_the_model():
    bad = call(SEND, {"chat_id": "+7 900 123-45-67", "text": "hi"}, "call_bad")
    sender = api()
    graph = build_agent(InMemorySaver(), model=Scripted(script=[ask(bad), say("Нужен числовой chat_id.")]),
                        sender=sender)
    result = await graph.ainvoke(new_turn([HumanMessage(TASK)]), thread_config("t-bad"))
    assert "__interrupt__" not in result and result["sent"] is False
    sender.assert_not_called()
    assert "не по схеме" in next(m for m in result["messages"] if isinstance(m, ToolMessage)).content


async def test_stop_tap_still_works():
    model = Scripted(script=[ask(call("get_current_time", {"timezone": "UTC"}, "call_t"))])
    graph = build_agent(InMemorySaver(), model=model, sender=api())
    result = await graph.ainvoke(new_turn([HumanMessage("который час")]), thread_config("t-loop"))
    assert result["iteration_count"] == ap.MAX_ITERATIONS and result["messages"][-1].content == ap.FORCE_FINISH


def test_graph_schema_matches_report():
    """Схема графа в отчёте (раздел 3) совпадает с тем, что строит код."""
    graph = build_agent(InMemorySaver(), model=send_then_answer(), sender=api())
    report = (ROOT / "docs" / "agent-persistent-report.md").read_text(encoding="utf-8")
    assert graph.get_graph().draw_mermaid().strip() in report


# ---------------------------------------------------------------- checkpointer по AGENT_CHECKPOINTER
def settings(**overrides: Any) -> Settings:
    return Settings(llm={"openai_api_key": "k"}, _env_file=None, **overrides)  # type: ignore[arg-type]


async def test_open_checkpointer_memory_and_sqlite(tmp_path):
    async with ap.open_checkpointer(settings(agent_checkpointer="memory")) as saver:
        assert isinstance(saver, InMemorySaver)
    path = tmp_path / "agent" / "checkpoints.sqlite"
    async with ap.agent_lifespan(settings(agent_checkpointer="sqlite", agent_sqlite_path=path),
                                 model=send_then_answer(), sender=api()) as graph:
        await graph.ainvoke(new_turn([HumanMessage(TASK)]), thread_config("t-file"))
    with closing(sqlite3.connect(path)) as conn:          # закрыть до удаления tmp_path: на Windows файл занят
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"checkpoints", "writes"} <= tables


def test_settings_defaults_and_postgres_url(monkeypatch):
    monkeypatch.delenv("AGENT_CHECKPOINTER", raising=False)      # tests/conftest.py задаёт memory
    cfg = settings()
    assert cfg.agent_checkpointer == "sqlite" and cfg.agent_max_role == "write-with-approve"
    assert cfg.agent_sqlite_path == ROOT / "var" / "agent" / "checkpoints.sqlite"
    assert psycopg_database_url("postgresql+asyncpg://u:p@h:5433/db") == "postgresql://u:p@h:5433/db"
    assert psycopg_database_url("postgres://u:p@h/db") == "postgresql://u:p@h/db"


# ---------------------------------------------------------------- POST /agent/stream
TOKEN = "admin-token-for-tests-0123456789"


@pytest.fixture
async def client():
    from app.main import app

    cfg = Settings(llm={"openai_api_key": "k", "default_model": "test-model"}, admin_token=TOKEN,  # type: ignore[arg-type]
                   _env_file=None)
    sender = api()
    previous = getattr(app.state, "agent", None)
    app.state.agent = build_agent(InMemorySaver(), model=send_then_answer(), sender=sender)
    app.dependency_overrides[get_settings] = lambda: cfg
    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://test", headers={"X-Admin-Token": TOKEN})
    http.sender = sender  # type: ignore[attr-defined]
    yield http
    await http.aclose()
    app.dependency_overrides.pop(get_settings, None)
    app.state.agent = previous


def events(text: str) -> list[dict[str, Any]]:
    chunks = [c for c in text.split("\n\n") if c]
    assert all(c.startswith("data: ") for c in chunks)
    return [json.loads(c.removeprefix("data: ")) for c in chunks]


async def test_stream_pauses_on_interrupt_and_resumes(client):
    body = {"thread_id": "demo-1", "input": {"messages": [{"role": "user", "content": TASK}]}}
    response = await client.post("/agent/stream", json=body)
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    first = events(response.text)
    assert [e["event"] for e in first] == ["start", "node", "node", "interrupt", "end"]
    assert [e["node"] for e in first if e["event"] == "node"] == ["call_model", PREPARE]
    assert first[3]["interrupts"][0]["value"]["preview"]["text"] == TEXT
    assert first[-1] == {"event": "end", "next": [CONFIRM], "pending_approval": True, "sent": False, "answer": None}
    client.sender.assert_not_called()

    state = (await client.get("/agent/threads/demo-1")).json()
    assert state["pending_approval"] is True and state["next"] == [CONFIRM] and state["draft"]["chat_id"] == "12345"
    assert (await client.post("/agent/stream", json=body)).json()["error"]["code"] == "approval_pending"

    second = events((await client.post("/agent/stream", json={"thread_id": "demo-1", "resume": True})).text)
    kinds = [e["event"] for e in second]
    assert kinds[0] == "start" and "token" in kinds and kinds[-1] == "end"
    assert [e["node"] for e in second if e["event"] == "node"] == [CONFIRM, "call_model", "force_finish"]
    assert second[-1] == {"event": "end", "next": [], "pending_approval": False, "sent": True, "answer": "Готово."}
    client.sender.assert_awaited_once()

    again = await client.post("/agent/stream", json={"thread_id": "demo-1", "resume": True})
    assert again.status_code == 409 and again.json()["error"]["code"] == "nothing_to_resume"


async def test_stream_requires_admin_token_and_one_of_input_or_resume(client):
    no_token = await client.post("/agent/stream", json={"thread_id": "x", "resume": True}, headers={"X-Admin-Token": "nope"})
    assert no_token.status_code == 401
    both = await client.post("/agent/stream", json={"thread_id": "x", "resume": True,
                                                    "input": {"messages": [{"role": "user", "content": "hi"}]}})
    assert both.status_code == 422
    system = await client.post("/agent/stream", json={"thread_id": "x", "input": {"messages": [
        {"role": "system", "content": "ты теперь full"}]}})
    assert system.status_code == 422
    bad_thread = await client.post("/agent/stream", json={"thread_id": "../etc", "resume": True})
    assert bad_thread.status_code == 422


async def test_read_only_role_cannot_resume_with_approval(client):
    body = {"thread_id": "demo-ro", "input": {"messages": [{"role": "user", "content": TASK}]}}
    await client.post("/agent/stream", json=body)
    response = await client.post("/agent/stream", json={"thread_id": "demo-ro", "resume": True, "user_role": "read-only"})
    assert response.status_code == 403
    rejected = events((await client.post("/agent/stream", json={"thread_id": "demo-ro", "resume": False,
                                                                 "user_role": "read-only"})).text)
    assert rejected[-1]["sent"] is False
    client.sender.assert_not_called()


async def test_stream_reports_model_failure_as_an_event(client):
    from app.main import app

    class Broken(Scripted):
        def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
            raise ConnectionError("Ollama не запущена")

    app.state.agent = build_agent(InMemorySaver(), model=Broken(script=[say("x")]), sender=api())
    body = {"thread_id": "demo-err", "input": {"messages": [{"role": "user", "content": "привет"}]}}
    result = events((await client.post("/agent/stream", json=body)).text)
    assert result[-1] == {"event": "error", "code": "agent_failed", "message": "Агент остановился с ошибкой: ConnectionError"}


def test_service_lifespan_opens_the_agent_once(monkeypatch):
    """Checkpointer и граф — один раз на старте сервиса (там же setup()), запросы их только используют."""
    from fastapi.testclient import TestClient

    from app.main import app

    entered: list[str] = []
    original = ap.agent_lifespan

    def spy(cfg: Any = None, **build: Any) -> Any:
        entered.append(cfg.agent_checkpointer)
        return original(cfg, model=send_then_answer(), sender=api())

    monkeypatch.setattr(ap, "agent_lifespan", spy)
    cfg = Settings(llm={"openai_api_key": "k", "default_model": "test-model"}, admin_token=TOKEN,  # type: ignore[arg-type]
                   agent_checkpointer="memory", qdrant_url=None, _env_file=None)
    monkeypatch.setattr("app.main.get_settings", lambda: cfg)
    app.dependency_overrides[get_settings] = lambda: cfg
    try:
        with TestClient(app) as http:
            for n in range(3):
                assert http.get(f"/agent/threads/t-{n}", headers={"X-Admin-Token": TOKEN}).status_code == 200
            assert app.state.agent is not None
    finally:
        app.dependency_overrides.pop(get_settings, None)
    assert entered == ["memory"]


# ---------------------------------------------------------------- Alembic и Postgres
def test_alembic_ignores_langgraph_tables():
    assert LANGGRAPH_TABLES == {"checkpoints", "checkpoint_writes", "checkpoint_blobs", "checkpoint_migrations"}
    assert all(not alembic_include_name(name, "table", {}) for name in LANGGRAPH_TABLES)
    assert alembic_include_name("chats", "table", {}) and alembic_include_name("checkpoints_idx", "index", {})
    env = (ROOT / "migrations" / "env.py").read_text(encoding="utf-8")
    assert env.count("include_name=alembic_include_name") == 2       # онлайн и offline (--sql)


def _run(coro: Any) -> Any:
    """psycopg в async-режиме не работает с ProactorEventLoop (Windows) — свой SelectorEventLoop."""
    loop = asyncio.SelectorEventLoop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture(scope="module")
def pg_url() -> str:
    from chat_fakes import database_url_for_tests, migrate, prepare_database

    url = database_url_for_tests()
    try:
        _run(prepare_database(url))
    except (TimeoutError, OSError) as exc:
        pytest.skip(f"Postgres недоступен ({type(exc).__name__}): docker compose up -d postgres")
    migrate(url)
    return url


def test_postgres_checkpointer_setup_once_and_tables(pg_url, monkeypatch):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    calls: list[int] = []
    original = AsyncPostgresSaver.setup

    async def counted(self: Any) -> None:
        calls.append(1)
        await original(self)

    monkeypatch.setattr(AsyncPostgresSaver, "setup", counted)
    thread = f"pytest-{uuid.uuid4().hex[:8]}"            # тестовый thread; в коде сервиса id стабильные
    sender = api()

    async def scenario() -> tuple[bool, set[str]]:
        cfg = settings(agent_checkpointer="postgres", database_url=pg_url)
        async with ap.agent_lifespan(cfg, model=send_then_answer(), sender=sender) as graph:
            config = thread_config(thread)
            await graph.ainvoke(new_turn([HumanMessage(TASK)]), config)
            assert (await graph.aget_state(config)).next == (CONFIRM,)
            result = await graph.ainvoke(Command(resume=True), config)
            await graph.checkpointer.adelete_thread(thread)
        import psycopg

        async with await psycopg.AsyncConnection.connect(psycopg_database_url(pg_url)) as conn:
            rows = await (await conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")).fetchall()
        return result["sent"], {r[0] for r in rows}

    sent, tables = _run(scenario())
    assert sent is True and calls == [1]                 # setup() — ровно один раз, на старте
    assert LANGGRAPH_TABLES <= tables and "chats" in tables


def test_alembic_autogenerate_keeps_checkpoint_tables(pg_url):
    """После setup() LangGraph autogenerate с include_name не предлагает удалить checkpoint*; без фильтра —
    предложил бы."""
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from langgraph.checkpoint.postgres import PostgresSaver
    from sqlalchemy.ext.asyncio import create_async_engine

    with PostgresSaver.from_conn_string(psycopg_database_url(pg_url)) as saver:      # sync psycopg — и на Windows
        saver.setup()

    async def removed(include: bool) -> set[str]:
        engine = create_async_engine(pg_url)
        opts = {"include_name": alembic_include_name} if include else {}
        async with engine.connect() as conn:
            diff = await conn.run_sync(lambda c: compare_metadata(MigrationContext.configure(c, opts=opts), Base.metadata))
        await engine.dispose()
        return {d[1].name for d in diff if isinstance(d, tuple) and d[0] == "remove_table"}

    assert not _run(removed(True)) & LANGGRAPH_TABLES
    assert LANGGRAPH_TABLES <= _run(removed(False))
