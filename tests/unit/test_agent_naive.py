"""
Наивный агент (блок 6.1) без сети: цикл app/services/agent_naive.py с подставным клиентом OpenAI SDK
(ответы модели по очереди), инструменты app/tools/naive_tools.py (поиск — настоящий RAGService над
учебным корпусом в Qdrant :memory:), allowlist, CLI и прогон пяти задач scripts/run_naive_agent.py.
Поведение настоящих моделей — логи docs/agent-naive-traces/ и разбор в docs/agent_naive.md.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIConnectionError
from qdrant_client import QdrantClient
from rag_fakes import CORPUS, DIM, RecordingLLM, TrigramEmbeddings

from app.core.config import Settings
from app.services import agent_naive
from app.services.agent_naive import DEFAULT_MODEL, PREVIEW, call_tool, run_agent
from app.services.rag import ProjectEmbedding, RAGService
from app.tools import naive_tools
from app.tools.naive_tools import (
    DISPATCH,
    TOOLS,
    get_current_time,
    search_knowledge_base,
    send_telegram_message,
)

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
ROOT = Path(__file__).resolve().parents[2]
TRACE_KEYS = {"step", "tool_name", "tool_args", "tool_result", "llm_input_tokens", "llm_output_tokens", "duration_ms"}


# ---------------------------------------------------------------- подставная модель
def tool_call(name: str, arguments: str, call_id: str = "call_1") -> SimpleNamespace:
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def reply(content: str | None = None, *calls: SimpleNamespace, usage: bool = True) -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content=content, tool_calls=list(calls) or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                           usage=SimpleNamespace(prompt_tokens=100, completion_tokens=7) if usage else None)


class ScriptedClient:
    """client.chat.completions.create — следующий ответ из списка; запоминает аргументы и копию messages."""

    def __init__(self, *replies: SimpleNamespace) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.replies.pop(0)


@pytest.fixture
def kb(monkeypatch):
    """База знаний инструмента — подставная: запоминает запросы, отвечает строкой."""
    queries: list[str] = []

    def fake_search(query: str) -> str:
        queries.append(query)
        return f"[billing/auto-renewal.docx, близость 0.81] Автопродление отключается в разделе «Тариф». {query}"

    monkeypatch.setitem(DISPATCH, "search_knowledge_base", fake_search)
    return queries


# ---------------------------------------------------------------- цикл
def test_happy_path_search_send_answer(kb, capsys):
    client = ScriptedClient(
        reply(None, tool_call("search_knowledge_base", '{"query": "отключить автопродление"}', "c1")),
        reply(None, tool_call("send_telegram_message", '{"chat_id": "12345", "text": "Раздел «Тариф»"}', "c2")),
        reply("Инструкция отправлена клиенту в чат 12345."))

    result = run_agent("Найди, как отключить автопродление, и отправь клиенту в чат 12345", client=client)

    assert result == {"answer": "Инструкция отправлена клиенту в чат 12345.", "steps": 3, "trace": result["trace"]}
    assert kb == ["отключить автопродление"]
    assert "[TELEGRAM → 12345] Раздел «Тариф»" in capsys.readouterr().out
    assert [e["step"] for e in result["trace"]] == [0, 1, 2]
    assert [e["tool_name"] for e in result["trace"]] == ["search_knowledge_base", "send_telegram_message", None]
    first, _, last = result["trace"]
    assert first["tool_args"] == '{"query": "отключить автопродление"}'          # исходный JSON модели
    assert first["tool_result"].startswith("[billing/auto-renewal.docx")
    assert (first["llm_input_tokens"], first["llm_output_tokens"]) == (100, 7)
    assert last["tool_args"] is None and last["tool_result"] is None
    for entry in result["trace"]:
        assert set(entry) == TRACE_KEYS and isinstance(entry["duration_ms"], float) and entry["duration_ms"] >= 0

    # Канон: модель gpt-5.4-mini, tools=TOOLS, ответ модели и role="tool" — обратно в messages.
    assert [c["model"] for c in client.calls] == [DEFAULT_MODEL] * 3 == ["gpt-5.4-mini"] * 3
    assert all(c["tools"] is TOOLS for c in client.calls)
    messages = client.calls[2]["messages"]
    assert [m["role"] if isinstance(m, dict) else m.role for m in messages] == \
        ["user", "assistant", "tool", "assistant", "tool"]
    assert messages[2] == {"role": "tool", "tool_call_id": "c1", "content": messages[2]["content"]}
    assert messages[4] == {"role": "tool", "tool_call_id": "c2", "content": "Сообщение отправлено в 12345"}


def test_answer_without_tools_is_one_step():
    client = ScriptedClient(reply("Здравствуйте! Чем помочь?"))
    result = run_agent("Привет", client=client, model="qwen3:4b-instruct")
    assert result["answer"] == "Здравствуйте! Чем помочь?" and result["steps"] == 1 and "error" not in result
    assert client.calls[0]["model"] == "qwen3:4b-instruct"
    assert client.calls[0]["messages"] == [{"role": "user", "content": "Привет"}]


def test_unknown_tool_is_an_error_for_the_model_not_a_crash():
    client = ScriptedClient(reply(None, tool_call("get_user_balance", '{"user_id": "42"}')),
                            reply("Инструмента для баланса нет — проверить не могу."))

    result = run_agent("Проверь баланс пользователя 42", client=client)

    assert result["answer"] == "Инструмента для баланса нет — проверить не могу." and result["steps"] == 2
    hallucinated = result["trace"][0]
    assert hallucinated["tool_name"] == "get_user_balance"
    assert hallucinated["tool_result"].startswith("Ошибка: инструмента 'get_user_balance' нет")
    tool_message = client.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool" and all(name in tool_message["content"] for name in DISPATCH)


@pytest.mark.parametrize(("name", "arguments", "expected"), [
    ("get_current_time", '{"timezone": "Europe/Moscow"', "Ошибка: аргументы get_current_time — не JSON"),
    ("get_current_time", '["Europe/Moscow"]', "Ошибка инструмента get_current_time: TypeError"),
    ("get_current_time", '{"timezone": "Mars/Phobos"}', "Ошибка инструмента get_current_time: ZoneInfoNotFoundError"),
    ("send_telegram_message", '{"chat_id": "1"}', "Ошибка инструмента send_telegram_message: TypeError"),
    ("eval", '{"source": "1+1"}', "Ошибка: инструмента 'eval' нет"),
    ("__import__", '{"name": "os"}', "Ошибка: инструмента '__import__' нет"),
])
def test_call_tool_turns_every_failure_into_a_string(name, arguments, expected):
    assert call_tool(name, arguments).startswith(expected)


def test_call_tool_empty_arguments_and_exception_in_search(monkeypatch):
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT.*\+03:00", call_tool("get_current_time", ""))

    def broken(query: str) -> str:
        raise RuntimeError("Qdrant не отвечает")

    monkeypatch.setitem(DISPATCH, "search_knowledge_base", broken)
    assert call_tool("search_knowledge_base", '{"query": "x"}') == \
        "Ошибка инструмента search_knowledge_base: RuntimeError: Qdrant не отвечает"


def test_tool_failure_goes_back_to_the_model_and_the_loop_continues():
    client = ScriptedClient(reply(None, tool_call("get_current_time", '{"timezone": "Mars/Phobos"}')),
                            reply(None, tool_call("get_current_time", '{"timezone": "Europe/Moscow"}', "call_2")),
                            reply("Сейчас в Москве день."))
    result = run_agent("Который час?", client=client)
    assert result["steps"] == 3 and result["answer"] == "Сейчас в Москве день."
    assert result["trace"][0]["tool_result"].startswith("Ошибка инструмента")
    assert result["trace"][1]["tool_result"].endswith("+03:00")


def test_parallel_tool_calls_in_one_step(kb):
    client = ScriptedClient(
        reply(None, tool_call("get_current_time", "{}", "a"), tool_call("search_knowledge_base", '{"query": "возврат"}', "b")),
        reply("Готово."))

    result = run_agent("Время и правила возврата", client=client)

    step0 = [e for e in result["trace"] if e["step"] == 0]
    assert [e["tool_name"] for e in step0] == ["get_current_time", "search_knowledge_base"]
    assert step0[0]["duration_ms"] <= step0[1]["duration_ms"]      # от начала шага: вторая запись не раньше первой
    tools = [m for m in client.calls[1]["messages"] if isinstance(m, dict) and m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools] == ["a", "b"]


def test_max_steps_stops_the_endless_search():
    client = ScriptedClient(*[reply(None, tool_call("get_current_time", "{}", f"c{i}")) for i in range(3)])

    result = run_agent("Найди то, чего нет", max_steps=3, client=client)

    assert result["answer"] is None and result["steps"] == 3
    assert result["error"].startswith("max_steps")
    assert len(client.calls) == 3 and len(result["trace"]) == 3 and not client.replies


def test_default_max_steps_is_six():
    assert inspect.signature(run_agent).parameters["max_steps"].default == 6
    client = ScriptedClient(*[reply(None, tool_call("get_current_time", "{}")) for _ in range(6)])
    assert run_agent("бесконечно", client=client)["steps"] == 6 and len(client.calls) == 6


def test_tool_result_is_truncated_in_trace_but_not_for_the_model(monkeypatch):
    long = "Правило. " * 100
    monkeypatch.setitem(DISPATCH, "search_knowledge_base", lambda query: long)
    client = ScriptedClient(reply(None, tool_call("search_knowledge_base", '{"query": "правила"}')), reply("Ок."))

    result = run_agent("Правила", client=client)

    assert result["trace"][0]["tool_result"] == long[:PREVIEW] and PREVIEW == 200
    assert client.calls[1]["messages"][-1]["content"] == long


def test_missing_usage_gives_none_tokens():
    result = run_agent("Привет", client=ScriptedClient(reply("Привет!", usage=False)))
    assert result["trace"][0]["llm_input_tokens"] is None and result["trace"][0]["llm_output_tokens"] is None


def test_model_from_settings_when_client_is_built(monkeypatch):
    client = ScriptedClient(reply("ok"))
    monkeypatch.setattr(agent_naive, "build_openai", lambda cfg: client)
    monkeypatch.setattr(agent_naive, "get_settings",
                        lambda: Settings(_env_file=None, llm={"openai_api_key": "k"}, agent={"model": "qwen3:4b-instruct"}))
    run_agent("Привет")
    assert client.calls[0]["model"] == "qwen3:4b-instruct"
    monkeypatch.setattr(agent_naive, "get_settings", lambda: Settings(_env_file=None, llm={"openai_api_key": "k"}))
    client.replies.append(reply("ok"))
    run_agent("Привет")
    assert client.calls[1]["model"] == "gpt-5.4-mini"


# ---------------------------------------------------------------- инструменты
def test_tools_and_dispatch_describe_the_same_functions():
    names = [tool["function"]["name"] for tool in TOOLS]
    assert names == list(DISPATCH) == ["search_knowledge_base", "get_current_time", "send_telegram_message"]
    for tool in TOOLS:
        assert tool["type"] == "function" and set(tool["function"]) == {"name", "description", "parameters"}
        schema = tool["function"]["parameters"]
        signature = inspect.signature(DISPATCH[tool["function"]["name"]])
        assert schema["type"] == "object" and set(schema["properties"]) == set(signature.parameters)
        assert set(schema["required"]) == {p for p, v in signature.parameters.items() if v.default is inspect.Parameter.empty}
        assert all(prop["type"] == "string" and prop["description"] for prop in schema["properties"].values())


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool["function"]["name"])
def test_every_description_has_at_least_two_sentences(tool):
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", tool["function"]["description"].strip()) if len(s) > 10]
    assert len(sentences) >= 2
    assert "Вызывай" in tool["function"]["description"]          # не только что делает, но и когда звать


def test_get_current_time_is_iso_in_the_zone():
    assert get_current_time().endswith("+03:00")
    assert get_current_time("Asia/Yekaterinburg").endswith("+05:00")
    assert "T" in get_current_time("UTC") and get_current_time("UTC").endswith("+00:00")


def test_send_telegram_message_is_only_a_print(capsys):
    assert send_telegram_message("777", "Аккаунт заблокирован") == "Сообщение отправлено в 777"
    assert capsys.readouterr().out == "[TELEGRAM → 777] Аккаунт заблокирован\n"


def test_send_telegram_message_never_touches_the_network(monkeypatch, capsys):
    """Заглушка не открывает соединений: любой HTTP-запрос в тесте упал бы."""
    def no_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("заглушка Telegram пошла в сеть")

    monkeypatch.setattr(httpx.Client, "send", no_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", no_network)
    send_telegram_message("1", "текст")
    source = inspect.getsource(naive_tools)
    assert "api.telegram.org" not in source and "aiogram" not in source and "bot_token" not in source.lower()
    imports = {node.names[0].name.split(".")[0] if isinstance(node, ast.Import) else (node.module or "").split(".")[0]
               for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.Import, ast.ImportFrom))}
    assert imports <= {"__future__", "collections", "datetime", "typing", "zoneinfo", "app"}


@pytest.fixture(scope="module")
def corpus_rag():
    """Настоящий RAGService блока 5.5 над учебным корпусом: Qdrant :memory:, эмбеддинги — триграммы."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="agent61-") as docstore:
        cfg = Settings(_env_file=None, qdrant_url="http://127.0.0.1:6333", embedding_dim=DIM,
                       llm={"openai_api_key": "test-key", "base_url": "http://127.0.0.1:11434/v1"},
                       rag={"data_dir": CORPUS, "collection": "agent61", "score_threshold": 0.0, "docstore_dir": docstore})
        service = RAGService(cfg, client=QdrantClient(":memory:"), embed_model=ProjectEmbedding(TrigramEmbeddings()),
                             llm=RecordingLLM()).build()
        yield service
        service.close()


def test_search_knowledge_base_returns_top1_fragment(corpus_rag, monkeypatch):
    monkeypatch.setattr(naive_tools, "_RAG", [corpus_rag])
    question = "Сколько запросов в минуту можно делать к API?"
    top = corpus_rag.retrieve(question).nodes[0]

    found = search_knowledge_base(question)

    assert found == f"[{top.metadata['source']}, близость {top.score:.2f}] {top.node.get_content()}"
    assert re.match(r"\[[\w./-]+\.(md|html|docx|pdf|txt), близость 0\.\d\d\] ", found)


def test_search_knowledge_base_reports_nothing_found(corpus_rag, monkeypatch):
    monkeypatch.setattr(corpus_rag.settings, "score_threshold", 0.99)
    monkeypatch.setattr(naive_tools, "_RAG", [corpus_rag])
    found = search_knowledge_base("Какая комиссия у партнёрской программы?")
    assert found.startswith("В базе знаний ничего не найдено по запросу «Какая комиссия у партнёрской программы?»")
    assert "ниже порога 0.99" in found


def test_knowledge_base_connects_once(monkeypatch):
    built: list[dict[str, Any]] = []

    class FakeService:
        def __init__(self, cfg: Any) -> None:
            pass

        def build(self, **kwargs: Any) -> FakeService:
            built.append(kwargs)
            return self

    monkeypatch.setattr("app.services.rag.RAGService", FakeService)
    monkeypatch.setattr(naive_tools, "_RAG", [])
    assert naive_tools.knowledge_base() is naive_tools.knowledge_base()
    assert built == [{"create": False}]           # только подключиться к готовой коллекции, не индексировать


# ---------------------------------------------------------------- требования задания к коду
@pytest.mark.parametrize("path", ["app/services/agent_naive.py", "app/tools/naive_tools.py", "scripts/run_naive_agent.py"])
def test_no_eval_exec_getattr_globals(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not called & {"eval", "exec", "getattr", "globals", "__import__", "compile"}
    assert not any(isinstance(node, ast.Subscript) and isinstance(node.value, ast.Call)
                   and getattr(node.value.func, "id", None) in {"globals", "locals", "vars"} for node in ast.walk(tree))


def test_dispatch_is_a_plain_dict_of_functions():
    assert type(DISPATCH) is dict and all(inspect.isfunction(fn) for fn in DISPATCH.values())


def test_agent_module_stays_small():
    source = (ROOT / "app/services/agent_naive.py").read_text(encoding="utf-8")
    run_agent_lines = inspect.getsource(run_agent).splitlines()
    assert len(run_agent_lines) <= 30, "чистая логика цикла — около 25 строк"
    assert len(source.splitlines()) <= 100
    assert "langgraph" not in source.lower() and "langchain" not in source.lower()
    assert "for step in range(max_steps)" in source and "client.chat.completions.create(" in source


# ---------------------------------------------------------------- CLI
def fake_run(result: dict[str, Any], seen: list[tuple] | None = None):
    def run(task: str, max_steps: int, *, model: str | None = None) -> dict[str, Any]:
        if seen is not None:
            seen.append((task, max_steps, model))
        return result
    return run


@pytest.fixture(autouse=True)
def keep_test_logging(monkeypatch):
    """main() настраивает лог сервиса (JSON в stderr) — в тестах не перенастраиваем глобальный logging."""
    monkeypatch.setattr(agent_naive, "setup_logging", lambda *args, **kwargs: None)


def test_cli_prints_answer(monkeypatch, capsys):
    seen: list[tuple] = []
    monkeypatch.setattr(agent_naive, "run_agent", fake_run({"answer": "Готовый ответ", "steps": 1, "trace": []}, seen))
    assert agent_naive.main(["Привет", "--max-steps", "3", "--model", "m"]) == 0
    out = capsys.readouterr().out
    assert out.strip() == "Готовый ответ" and seen == [("Привет", 3, "m")]


def test_cli_prints_stop_reason_and_trace(monkeypatch, capsys):
    entry = {"step": 0, "tool_name": "search_knowledge_base", "tool_args": '{"query": "x"}', "tool_result": "нет"}
    monkeypatch.setattr(agent_naive, "run_agent", fake_run(
        {"answer": None, "steps": 6, "trace": [entry], "error": "max_steps: 6 шагов без ответа"}))
    assert agent_naive.main(["задача", "--trace"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("Остановка: max_steps: 6 шагов без ответа\n") and "--- trace ---" in out
    assert json.loads(out.strip().splitlines()[-1]) == entry


def test_cli_max_steps_from_settings(monkeypatch, capsys):
    seen: list[tuple] = []
    monkeypatch.setattr(agent_naive, "run_agent", fake_run({"answer": "ok", "steps": 1, "trace": []}, seen))
    monkeypatch.setattr(agent_naive, "get_settings",
                        lambda: Settings(_env_file=None, llm={"openai_api_key": "k"}, agent={"max_steps": 4}))
    agent_naive.main(["задача"])
    assert seen == [("задача", 4, None)]


def test_cli_reports_provider_error(monkeypatch, capsys):
    def down(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise APIConnectionError(request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"))

    monkeypatch.setattr(agent_naive, "run_agent", down)
    assert agent_naive.main(["задача"]) == 2
    assert capsys.readouterr().out.startswith("Остановка: APIConnectionError: Connection error.")


def test_cli_module_runs():
    # PYTHONIOENCODING: на Windows дочерний процесс пишет в канал в cp1251, а справка — по-русски.
    done = subprocess.run([sys.executable, "-m", "app.services.agent_naive", "--help"], cwd=ROOT,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"}, capture_output=True, text=True,
                          encoding="utf-8", timeout=120, check=False)
    assert done.returncode == 0 and "--trace" in done.stdout and "--max-steps" in done.stdout


# ---------------------------------------------------------------- пять задач: scripts/run_naive_agent.py
@pytest.fixture
def runner():
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_naive_agent

    yield run_naive_agent
    sys.path.remove(str(ROOT / "scripts"))


def test_five_tasks_cover_the_required_modes(runner):
    ids = [task.id for task in runner.TASKS]
    assert ids == ["success", "no_data", "tool_hallucination", "long_compound", "write_provocation"]
    texts = {task.id: task.text for task in runner.TASKS}
    assert "чат 12345" in texts["success"] and "баланс" in texts["tool_hallucination"]
    assert "Не уточняй" in texts["write_provocation"] and texts["long_compound"].count(",") >= 3


def test_suite_writes_a_log_per_task(runner, tmp_path, kb, monkeypatch):
    replies = [reply(None, tool_call("search_knowledge_base", '{"query": "автопродление"}')),
               reply(None, tool_call("send_telegram_message", '{"chat_id": "12345", "text": "Раздел «Тариф»"}')),
               reply("Отправил."),                                                         # success: 3 шага
               *[reply(None, tool_call("search_knowledge_base", '{"query": "комиссия"}')) for _ in range(3)],  # no_data
               reply(None, tool_call("get_user_balance", '{"user_id": "42"}')),
               reply('{"name": "get_user_balance", "arguments": {"user_id": 42}}')]          # вызов текстом
    client = ScriptedClient(*replies)
    tasks = [t for t in runner.TASKS if t.id in {"success", "no_data", "tool_hallucination"}]

    logs = runner.run_suite(tasks, model="test-model", max_steps=3, out_dir=tmp_path, client=client)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["01_success.json", "02_no_data.json", "03_tool_hallucination.json"]
    success, no_data, hallucination = (json.loads(p.read_text(encoding="utf-8")) for p in sorted(tmp_path.iterdir()))
    assert logs[0] == success
    assert not client.replies
    assert success["model"] == "test-model" and success["max_steps"] == 3
    assert success["answer"] == "Отправил." and success["error"] is None
    assert success["telegram"] == ["[TELEGRAM → 12345] Раздел «Тариф»"]
    assert success["diagnose"]["messages_sent"] == 1 and success["diagnose"]["outcome"] == "answer"
    assert success["diagnose"]["steps"] == 3 and success["diagnose"]["input_tokens_total"] == 300
    assert no_data["error"] == "max_steps: 3 шагов без ответа" and no_data["diagnose"]["outcome"] == "max_steps"
    assert no_data["diagnose"]["repeated_calls"] == 2 and no_data["diagnose"]["tools"] == ["search_knowledge_base"] * 3
    assert hallucination["diagnose"]["unknown_tools"] == ["get_user_balance"]
    assert hallucination["diagnose"]["tool_errors"] == 1 and hallucination["diagnose"]["tool_call_as_text"] is True
    assert hallucination["diagnose"]["input_tokens_total"] == 200 and hallucination["diagnose"]["outcome"] == "answer"
    assert set(success["trace"][0]) == TRACE_KEYS


def test_suite_survives_a_provider_error(runner, tmp_path, monkeypatch):
    def down(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise APIConnectionError(request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"))

    monkeypatch.setattr(runner, "run_agent", down)
    logs = runner.run_suite(runner.TASKS[:2], model="m", max_steps=6, out_dir=tmp_path)
    assert [log["diagnose"]["outcome"] for log in logs] == ["exception", "exception"]
    assert logs[0]["error"] == "APIConnectionError: Connection error."


def test_scrub_hides_the_home_folder(runner):
    home = "C:\\Users\\someone"
    text = json.dumps({"error": "нет файла C:\\Users\\someone\\multapi\\var\\x и C:/Users/someone/y"}, ensure_ascii=False)
    assert json.loads(runner.scrub(text, home)) == {"error": "нет файла ~\\multapi\\var\\x и ~/y"}


def test_suite_rejects_unknown_task_ids(runner):
    with pytest.raises(SystemExit) as exc:
        runner.suite_main(["--suite", "--only", "success,nope"])
    assert exc.value.code == 2
