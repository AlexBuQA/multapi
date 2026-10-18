"""
Агент на LangGraph (блок 6.3) без сети: кастомный StateGraph и prebuilt-агент create_agent на
подставной чат-модели (ответы по сценарию), контракт AgentState, router, stop-кран на сломанном
инструменте, инструменты @tool со строгими схемами, Mermaid-схемы и бенчмарк scripts/bench_agents.py.
"""
from __future__ import annotations

import asyncio
import copy
import importlib.metadata
import json
import operator
import re
import sys
import typing
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph.message import add_messages
from pydantic import PrivateAttr

from app.core.config import Settings
from app.services import agent_graph, agent_react
from app.services.agent_graph import (
    FORCE_FINISH,
    MAX_ITERATIONS,
    NOT_EXECUTED,
    AgentState,
    build_custom_graph,
    build_prebuilt_graph,
    initial_state,
    route_after_model,
    run_graph,
    summarize,
)
from app.tools import graph_tools, naive_tools, react_tools
from app.tools.graph_tools import TOOLS

ROOT = Path(__file__).resolve().parents[2]
TIME_TASK = "Который сейчас час в Екатеринбурге?"


@pytest.fixture(autouse=True)
def no_agent_env(monkeypatch):
    for name in ("MODEL", "TEMPERATURE", "MAX_ITERATIONS", "TIMEOUT_PER_ITERATION_SEC", "FINAL_CHECK"):
        monkeypatch.delenv(f"AGENT__{name}", raising=False)


# ---------------------------------------------------------------- подставная модель
def tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
                     usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})


def answer(text: str) -> AIMessage:
    return AIMessage(content=text, usage_metadata={"input_tokens": 120, "output_tokens": 5, "total_tokens": 125})


class FakeChat(BaseChatModel):
    """Чат-модель по сценарию: ответы по очереди (последний повторяется), запоминает входные сообщения
    и аргументы bind_tools. Каждый ответ — копия: у одного объекта add_messages совпали бы id."""

    script: list[AIMessage]
    _calls: list[list[Any]] = PrivateAttr(default_factory=list)
    _bound: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeChat:
        self._bound.append({"tools": [t.name for t in tools], **kwargs})
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self._calls.append(list(messages))
        reply = self.script[min(len(self._calls) - 1, len(self.script) - 1)]
        return ChatResult(generations=[ChatGeneration(message=reply.model_copy())])


class LoopingChat(FakeChat):
    """Всегда просит инструмент — с новым id вызова: без stop-крана это бесконечный цикл."""

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self._calls.append(list(messages))
        n = len(self._calls)
        return ChatResult(generations=[ChatGeneration(
            message=tool_call("search_knowledge_base", {"query": f"срок возврата, попытка {n}"}, f"c{n}"))])


def run(graph: Any, task: str, **kwargs: Any) -> dict[str, Any]:
    return asyncio.run(run_graph(graph, task, thread_id="test", **kwargs))


@pytest.fixture
def kb(monkeypatch):
    queries: list[str] = []

    def fake_search(query: str) -> str:
        queries.append(query)
        return "[api/faq-api.md, близость 0.76] «Старт» — 60, «Команда» — 300, «Бизнес» — 1 200 запросов в минуту."

    monkeypatch.setattr(naive_tools, "search_knowledge_base", fake_search)
    return queries


# ---------------------------------------------------------------- state и router
def test_agent_state_contract():
    hints = typing.get_type_hints(AgentState, include_extras=True)
    assert set(hints) == {"messages", "iteration_count", "tool_results"}
    messages, results = typing.get_args(hints["messages"]), typing.get_args(hints["tool_results"])
    assert messages[1] is add_messages and results[1] is operator.add
    assert hints["iteration_count"] is int                    # без Annotated: reducer по умолчанию — замена
    source = Path(agent_graph.__file__).read_text(encoding="utf-8")
    state_src = source[source.index("class AgentState"):source.index("def initial_state")]
    assert not re.search(r"OpenAI|Client|Session|api_key|BaseChatModel", state_src.split('"""')[-1])
    assert initial_state("x") == {"messages": [HumanMessage("x")], "iteration_count": 0, "tool_results": []}


@pytest.mark.parametrize(("messages", "count", "expected"), [
    ([HumanMessage("x"), tool_call("get_current_time", {"timezone": "UTC"})], 1, "execute_tool"),
    ([HumanMessage("x"), answer("Готово")], 1, "force_finish"),
    ([HumanMessage("x"), tool_call("get_current_time", {"timezone": "UTC"})], MAX_ITERATIONS, "force_finish"),
    ([HumanMessage("x"), tool_call("get_current_time", {"timezone": "UTC"})], MAX_ITERATIONS + 3, "force_finish"),
    ([HumanMessage("x"), ToolMessage(content="r", tool_call_id="c1")], 1, "force_finish"),
])
def test_route_after_model(messages, count, expected):
    state = {"messages": messages, "iteration_count": count, "tool_results": []}
    before = copy.deepcopy(state)
    assert route_after_model(state) == expected
    assert state == before                                    # router только читает state


def test_router_is_typed_with_literal():
    hint = typing.get_type_hints(route_after_model)["return"]
    assert typing.get_origin(hint) is typing.Literal and set(typing.get_args(hint)) == {"execute_tool", "force_finish"}
    assert MAX_ITERATIONS == 6


# ---------------------------------------------------------------- кастомный граф
def test_custom_graph_tool_then_answer(capsys):
    model = FakeChat(script=[tool_call("get_current_time", {"timezone": "Asia/Yekaterinburg"}), answer("Сейчас 19:46.")])
    graph = build_custom_graph(model)

    state = asyncio.run(graph.ainvoke(initial_state(TIME_TASK), config={"configurable": {"thread_id": "t"}}))

    assert state["messages"][-1].content == "Сейчас 19:46." and state["iteration_count"] == 2
    assert [type(m).__name__ for m in state["messages"]] == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    (result,) = state["tool_results"]
    assert result["name"] == "get_current_time" and result["args"] == {"timezone": "Asia/Yekaterinburg"}
    assert result["result"].endswith("+05:00") and result["iteration"] == 1 and isinstance(result["ms"], float)
    # Системный промпт — во входе модели, но не в state; strict=True — в bind_tools.
    first_input = model._calls[0]
    assert isinstance(first_input[0], SystemMessage) and first_input[0].content == agent_react.SYSTEM_PROMPT
    assert not any(isinstance(m, SystemMessage) for m in state["messages"])
    assert model._bound == [{"tools": [t.name for t in TOOLS], "strict": True}]


def test_unknown_tool_bad_arguments_and_tool_failure_are_messages_not_crashes(monkeypatch):
    def broken(query: str) -> str:
        raise ConnectionError("Qdrant не отвечает")

    monkeypatch.setattr(naive_tools, "search_knowledge_base", broken)
    script = [AIMessage(content="", tool_calls=[
        {"name": "get_user_balance", "args": {"user_id": "42"}, "id": "a", "type": "tool_call"},
        {"name": "send_telegram_message", "args": {"chat_id": "+7 900 123-45-67", "text": "Одобрено"}, "id": "b",
         "type": "tool_call"},
        {"name": "get_current_time", "args": {"timezone": "UTC", "format": "iso"}, "id": "c", "type": "tool_call"},
        {"name": "search_knowledge_base", "args": {"query": "срок возврата"}, "id": "d", "type": "tool_call"}]),
        answer("Не получилось.")]
    result = run(build_custom_graph(FakeChat(script=script)), "x")

    by_name = {r["name"]: r["result"] for r in result["tool_results"]}
    assert by_name["get_user_balance"].startswith("Ошибка: инструмента 'get_user_balance' нет")
    assert by_name["send_telegram_message"].startswith("Ошибка: аргументы send_telegram_message не по схеме — chat_id")
    assert "format" in by_name["get_current_time"] and "не по схеме" in by_name["get_current_time"]
    assert by_name["search_knowledge_base"] == \
        "Ошибка инструмента search_knowledge_base: ConnectionError: Qdrant не отвечает"
    assert result["answer"] == "Не получилось." and not result["force_finish"]


def test_stop_tap_on_a_broken_tool(monkeypatch):
    """Модель на сломанном поиске просит его снова и снова — граф останавливается на 6-м вызове."""
    monkeypatch.setattr(naive_tools, "search_knowledge_base", lambda query: (_ for _ in ()).throw(TimeoutError("нет")))
    model = LoopingChat(script=[])

    state = asyncio.run(build_custom_graph(model).ainvoke(initial_state("Найди срок возврата")))

    assert len(model._calls) == MAX_ITERATIONS and state["iteration_count"] == MAX_ITERATIONS
    assert len(state["tool_results"]) == MAX_ITERATIONS - 1      # шестой вызов не выполнен
    final = state["messages"][-1]
    assert final.content == FORCE_FINISH == "Превышен лимит итераций (6): агент не дал финального ответа."
    assert final.name == "force_finish"
    # История остаётся валидной для checkpointer: у каждого tool_call есть ToolMessage.
    answered = {m.tool_call_id for m in state["messages"] if isinstance(m, ToolMessage)}
    asked = {c["id"] for m in state["messages"] if isinstance(m, AIMessage) for c in m.tool_calls}
    assert asked == answered and state["messages"][-2].content == NOT_EXECUTED
    assert summarize(state["messages"])["force_finish"] is True


def test_force_finish_passes_a_normal_answer_through():
    state = asyncio.run(build_custom_graph(FakeChat(script=[answer("Не могу позвонить.")])).ainvoke(initial_state("x")))
    assert [type(m).__name__ for m in state["messages"]] == ["HumanMessage", "AIMessage"]
    assert state["messages"][-1].content == "Не могу позвонить." and state["iteration_count"] == 1


def test_same_reply_object_is_merged_by_add_messages():
    """Почему FakeChat отдаёт копию ответа: один и тот же AIMessage add_messages не добавит второй раз, а
    заменит по id — история не растёт, и router видит последним ToolMessage, а не новый вызов модели."""
    reply = tool_call("get_current_time", {"timezone": "UTC"})
    first = add_messages([HumanMessage("x")], [reply])
    again = add_messages(first + [ToolMessage(content="t", tool_call_id="c1")], [first[-1]])
    assert len(again) == 3 and isinstance(again[-1], ToolMessage)


# ---------------------------------------------------------------- prebuilt
def test_prebuilt_graph_same_task_same_answer():
    script = [tool_call("get_current_time", {"timezone": "Asia/Yekaterinburg"}), answer("Сейчас 19:46.")]
    custom, prebuilt = FakeChat(script=script), FakeChat(script=script)

    a = run(build_custom_graph(custom), TIME_TASK)
    b = run(build_prebuilt_graph(prebuilt), TIME_TASK, recursion_limit=agent_graph.PREBUILT_RECURSION_LIMIT)

    assert a["answer"] == b["answer"] == "Сейчас 19:46."
    for r in (a, b):
        assert r["steps"] == 2 and [c["name"] for c in r["tool_calls"]] == ["get_current_time"]
        assert (r["prompt_tokens"], r["completion_tokens"], r["total_tokens"]) == (220, 15, 235)
    assert b["tool_results"] is None                          # у prebuilt своего tool_results нет — разбор по messages
    assert prebuilt._calls[0][0].content == agent_react.SYSTEM_PROMPT
    assert prebuilt._bound[0]["strict"] is True               # strict — через middleware strict_tools


def test_prebuilt_stops_after_six_model_calls(monkeypatch):
    monkeypatch.setattr(naive_tools, "search_knowledge_base", lambda query: "Ошибка: база знаний не ответила")
    model = LoopingChat(script=[])
    result = run(build_prebuilt_graph(model), "x", recursion_limit=agent_graph.PREBUILT_RECURSION_LIMIT)
    assert len(model._calls) == MAX_ITERATIONS and result["steps"] == MAX_ITERATIONS
    assert result["force_finish"] is True and result["answer"].startswith(agent_graph.PREBUILT_LIMIT_PREFIX)


def test_run_graph_passes_thread_id():
    seen: dict[str, Any] = {}

    class Spy:
        async def ainvoke(self, state: Any, config: Any = None) -> dict[str, Any]:
            seen.update(config)
            return {"messages": [*state["messages"], answer("ok")], "iteration_count": 1, "tool_results": []}

    asyncio.run(run_graph(Spy(), "x", thread_id="bench-simple_kb"))
    assert seen == {"configurable": {"thread_id": "bench-simple_kb"}}
    asyncio.run(run_graph(Spy(), "x", thread_id="t", recursion_limit=50))
    assert seen["recursion_limit"] == 50


def test_module_level_graphs_are_lazy(monkeypatch):
    built = {"model": "m", "custom_graph": "custom", "prebuilt_graph": "prebuilt"}
    monkeypatch.setattr(agent_graph, "_BUILT", built)
    from app.services.agent_graph import custom_graph, prebuilt_graph

    assert (custom_graph, prebuilt_graph) == ("custom", "prebuilt")
    with pytest.raises(AttributeError):
        agent_graph.no_such_graph  # noqa: B018


def test_build_model_uses_service_settings():
    cfg = Settings(_env_file=None, llm={"openai_api_key": "k", "base_url": "http://127.0.0.1:11434/v1",
                                        "request_timeout": 240}, agent={"model": "qwen3:4b-instruct"})
    model = agent_graph.build_model(cfg)
    assert model.model_name == "qwen3:4b-instruct" and model.temperature == 0.0
    assert model.openai_api_base == "http://127.0.0.1:11434/v1" and model.request_timeout == 240
    assert model.use_responses_api is False
    assert agent_graph.build_model(cfg, temperature=0.7).temperature == 0.7


# ---------------------------------------------------------------- инструменты
@pytest.mark.parametrize("name", ["search_knowledge_base", "get_current_time", "send_telegram_message"])
def test_tool_description_is_the_62_description(name):
    """Docstring @tool — description для модели; текст тот же, что у ReAct-агента блока 6.2, байт в байт.
    Сравнение без нормализации пробелов: прежняя проверка склеивала пробелы и пропустила переносы строк
    docstring — в запросе к модели они шли как «\\n» (раздел 7.2 отчёта)."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    graph_tool = {t.name: t for t in TOOLS}[name]
    react = {t["function"]["name"]: t["function"]["description"] for t in react_tools.TOOLS}[name]
    sent = convert_to_openai_tool(graph_tool, strict=True)["function"]      # так bind_tools отдаёт его модели
    assert graph_tool.description == sent["description"] == react and "\n" not in react


def test_tool_schemas_are_strict():
    for t in TOOLS:
        schema = t.args_schema.model_json_schema()
        assert schema["additionalProperties"] is False and set(schema["required"]) == set(schema["properties"])
        assert all(p["pattern"].startswith("^") for p in schema["properties"].values())
    assert [t.name for t in TOOLS] == ["search_knowledge_base", "get_current_time", "send_telegram_message"]


def test_tools_call_the_baseline_functions(kb, capsys):
    assert graph_tools.search_knowledge_base.invoke({"query": "лимит API"}).startswith("[api/faq-api.md")
    assert kb == ["лимит API"]
    assert graph_tools.send_telegram_message.invoke({"chat_id": "12345", "text": "Срок — 3 дня"}) == \
        "Сообщение отправлено в 12345"
    assert capsys.readouterr().out == "[TELEGRAM → 12345] Срок — 3 дня\n"
    assert graph_tools.get_current_time.invoke({"timezone": "Mars/Phobos"}).startswith(
        "Ошибка инструмента get_current_time: ZoneInfoNotFoundError")


# ---------------------------------------------------------------- схемы и зависимости
def test_mermaid_files_match_the_code(tmp_path, monkeypatch):
    model = FakeChat(script=[answer("ok")])
    monkeypatch.setattr(agent_graph, "_BUILT", {"model": model, "custom_graph": build_custom_graph(model),
                                                "prebuilt_graph": build_prebuilt_graph(model)})
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import visualize_graph

        saved = visualize_graph.save(tmp_path)
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    assert [p.name for p in saved] == ["agent-graph-custom.mmd", "agent-graph-prebuilt.mmd"]
    custom = (tmp_path / "agent-graph-custom.mmd").read_text(encoding="utf-8")
    for edge in ("__start__ --> call_model;", "call_model -.-> execute_tool;", "call_model -.-> force_finish;",
                 "execute_tool --> call_model;", "force_finish --> __end__;"):
        assert edge in custom
    assert "ModelCallLimitMiddleware" in (tmp_path / "agent-graph-prebuilt.mmd").read_text(encoding="utf-8")
    # Файлы в docs/ и схема в отчёте — те же, что строит код сейчас.
    for name in ("agent-graph-custom.mmd", "agent-graph-prebuilt.mmd"):
        assert (ROOT / "docs" / name).read_text(encoding="utf-8") == (tmp_path / name).read_text(encoding="utf-8")
    report = (ROOT / "docs" / "agent-graph-report.md").read_text(encoding="utf-8")
    body = re.search(r"```mermaid\n(.*?)```", report, re.DOTALL).group(1)
    assert body.strip() == custom.strip()


def test_langgraph_and_langchain_are_1x():
    for package in ("langgraph", "langchain", "langchain-core", "langchain-openai"):
        assert importlib.metadata.version(package).split(".")[0] == "1", package
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for spec in ('"langgraph>=1.0,<2"', '"langchain>=1.0,<2"', '"langchain-openai>=1.0', '"langchain-core>=1.0'):
        assert spec in pyproject


def test_react_temperature_passthrough():
    from types import SimpleNamespace

    class Client:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def with_options(self, **kwargs: Any) -> Client:
            return self

        def _create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            message = SimpleNamespace(role="assistant", content="Готово.", tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                                   usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))

    client = Client()
    agent_react.run_react_with_reflection("x", client=client, temperature=0)
    agent_react.run_react_with_reflection("x", client=client)
    assert client.calls[0]["temperature"] == 0 and "temperature" not in client.calls[1]


# ---------------------------------------------------------------- бенчмарк
@pytest.fixture
def bench():
    sys.path.insert(0, str(ROOT / "scripts"))
    import bench_agents

    yield bench_agents
    sys.path.remove(str(ROOT / "scripts"))


def fake_graphs(script: list[AIMessage]) -> dict[str, Any]:
    return {"model": FakeChat(script=script), "custom_graph": build_custom_graph(FakeChat(script=script)),
            "prebuilt_graph": build_prebuilt_graph(FakeChat(script=script))}


def test_bench_series_journal_resume_and_report(bench, tmp_path, monkeypatch, kb):
    script = [tool_call("search_knowledge_base", {"query": "лимит API Команда"}), answer("300 запросов в минуту.")]
    react_calls: list[str] = []

    def fake_react(task: str, **kwargs: Any) -> dict[str, Any]:
        react_calls.append(task)
        return {"answer": "300 запросов в минуту.", "steps": 2, "revisions": 0,
                "usage": {"prompt_tokens": 300, "completion_tokens": 20, "total_tokens": 320, "calls": 3,
                          "critic": {"total_tokens": 80}},
                "trace": [{"tool_name": "search_knowledge_base", "tool_args": '{"query": "лимит"}',
                           "observation": "[api/faq-api.md] 300"}, {"tool_name": None}]}

    monkeypatch.setattr(bench, "run_react", fake_react)
    events: list[str] = []
    run_once = bench.run_once

    def spy(task: Any, impl: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append(impl)
        return run_once(task, impl, *args, **kwargs)

    monkeypatch.setattr(bench, "run_once", spy)
    tasks = [t for t in bench.TASKS if t.id == "simple_kb"]
    graphs = fake_graphs(script)

    records = bench.run_series(tasks, bench.IMPLS, 2, model="m", temperature=0, out_dir=tmp_path, graphs=graphs,
                               before_run=lambda: events.append("reset"))

    assert len(records) == 6 and len(react_calls) == 2
    assert events == ["reset", "b62_loop", "reset", "custom", "reset", "prebuilt"] * 2   # сброс — перед каждым
    assert all(r["cache_reset"] for r in records)
    assert {(r["impl"], r["repeat"]) for r in records} == {(i, n) for i in bench.IMPLS for n in (1, 2)}
    custom = next(r for r in records if r["impl"] == "custom")
    assert (custom["steps"], custom["prompt_tokens"], custom["completion_tokens"]) == (2, 220, 15)
    assert custom["check"]["correct"] and custom["tools"] == ["search_knowledge_base"]
    b62 = next(r for r in records if r["impl"] == "b62_loop")
    assert b62["llm_calls"] == 3 and b62["critic_tokens"] == 80 and b62["total_tokens"] == 320

    # --resume: готовые прогоны не повторяются, добавляется только недостающий третий.
    records = bench.run_series(tasks, ("b62_loop",), 3, model="m", temperature=0, out_dir=tmp_path,
                               resume=True, graphs=graphs)
    assert len(react_calls) == 3 and len(records) == 7

    report = tmp_path / "report.md"
    report.write_text(f"# Отчёт\n\n{bench.MARK_START}\nстарое\n{bench.MARK_END}\n\n## Дальше\n", encoding="utf-8")
    meta = {"model": "m", "temperature": 0.0, "repeats": 3}
    text = bench.finish(records, meta, tmp_path, report)
    written = report.read_text(encoding="utf-8")
    assert "старое" not in written and written.endswith("## Дальше\n") and text in written
    assert "| `simple_kb` | b62_loop | " in text and "3 из 3 |" in text and "| custom | " in text
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert {r["impl"]: r["runs"] for r in summary["rows"]} == {"b62_loop": 3, "custom": 2, "prebuilt": 2}
    # Третий прогон шёл без сброса — сводка по журналу не заявляет сброс для всей серии.
    assert summary["meta"]["cache_reset"] is False and "сброс кеша" not in text
    with_reset = [r for r in records if r["cache_reset"]]
    assert "перед каждым прогоном — сброс кеша промпта" in bench.finish(with_reset, meta, tmp_path, report)


def test_bench_reset_failure_does_not_stop_the_series(bench, tmp_path, monkeypatch, kb):
    def broken_reset() -> None:
        raise ConnectionError("Ollama не запущена")

    tasks = [t for t in bench.TASKS if t.id == "simple_kb"]
    script = [tool_call("search_knowledge_base", {"query": "лимит API Команда"}), answer("300 запросов в минуту.")]
    records = bench.run_series(tasks, ("custom",), 2, model="m", temperature=0, out_dir=tmp_path,
                               graphs=fake_graphs(script), before_run=broken_reset)
    assert len(records) == 2 and all(r["check"]["correct"] for r in records)


def test_bench_reset_cache_is_a_short_request_without_system_prompt(bench):
    """Сброс кеша промпта: в слоте Ollama остаётся запрос, который с промптом агента не делит префикс."""
    model = FakeChat(script=[answer("готово")])
    bench.reset_cache(model)
    [messages] = model._calls
    assert [type(m).__name__ for m in messages] == ["HumanMessage"] and messages[0].content == bench.RESET_PROMPT


def test_bench_records_limits_and_errors(bench, monkeypatch, tmp_path):
    monkeypatch.setattr(naive_tools, "search_knowledge_base", lambda query: "Ошибка: база знаний не ответила")
    graphs = {"custom_graph": build_custom_graph(LoopingChat(script=[])),
              "prebuilt_graph": build_prebuilt_graph(LoopingChat(script=[]))}
    task = bench.TASKS[0]
    for impl in ("custom", "prebuilt"):
        record = bench.run_once(task, impl, 1, model="m", temperature=0, graphs=graphs)
        assert record["force_finish"] is True and record["steps"] == MAX_ITERATIONS
        assert record["check"]["correct"] is False and record["check"]["why"].startswith("остановка")

    def down(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("Ollama не запущена")

    monkeypatch.setattr(bench, "run_react", down)
    record = bench.run_once(task, "b62_loop", 1, model="m", temperature=0)
    assert record["check"] == {"correct": False, "why": "ConnectionError: Ollama не запущена"}


def test_bench_stop_check(bench, tmp_path):
    resets: list[int] = []
    result = bench.stop_check(LoopingChat(script=[]), tmp_path, before_run=lambda: resets.append(1))
    assert len(resets) == 4                                       # 2 сценария × 2 графа
    for scenario in ("broken", "useless"):
        custom, prebuilt = result[scenario]["custom"], result[scenario]["prebuilt"]
        assert custom["force_finish"] and prebuilt["force_finish"]
        assert custom["steps"] == prebuilt["steps"] == MAX_ITERATIONS
        # Кастомный граф шестой вызов не выполняет (закрыт ToolMessage «Не выполнено»), prebuilt — выполняет.
        assert (custom["searches_requested"], custom["searches_executed"]) == (MAX_ITERATIONS, MAX_ITERATIONS - 1)
        assert (prebuilt["searches_requested"], prebuilt["searches_executed"]) == (MAX_ITERATIONS, MAX_ITERATIONS)
    saved = json.loads((tmp_path / "stop_check.json").read_text(encoding="utf-8"))
    assert saved["broken"]["task"] == bench.STOP_TASK and saved["useless"]["task"] == bench.USELESS_TASK
    assert saved["max_iterations"] == MAX_ITERATIONS
    assert bench.useless_search("срок возврата").startswith("В базе знаний ничего не найдено")
