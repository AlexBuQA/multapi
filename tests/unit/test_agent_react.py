"""
ReAct-агент блока 6.2 без сети: цикл app/services/agent_react.py на подставном клиенте OpenAI SDK
(ответы агента и вердикты критика — по очереди), лимиты итераций и времени, критик и счётчик ревизий,
строгие схемы инструментов (app/tools/react_tools.py), лог шагов, CLI и сравнение агентов
scripts/compare_agents.py. Наивный агент блока 6.1 — baseline: тест держит его файл неизменным.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from log_capture import captured_logs, events
from openai import APITimeoutError

from app.core.config import Settings
from app.services import agent_react
from app.services.agent_react import (
    ITERATION_LIMIT,
    SKIPPED,
    SYSTEM_PROMPT,
    TIMEOUT,
    call_tool,
    parse_verdict,
    run_react_with_reflection,
)
from app.tools import naive_tools, react_tools
from app.tools.react_tools import DISPATCH, TOOLS, schemas, validate_args

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = schemas(TOOLS)


@pytest.fixture(autouse=True)
def no_agent_env(monkeypatch):
    """AGENT__* из .env разработчика (src/config.py загружает его в os.environ) тестам не мешают."""
    for name in ("MODEL", "MAX_STEPS", "MAX_ITERATIONS", "TIMEOUT_PER_ITERATION_SEC", "MAX_REVISIONS",
                 "PREMIUM_MODEL", "CRITIC_MODEL"):
        monkeypatch.delenv(f"AGENT__{name}", raising=False)


@pytest.fixture
def kb(monkeypatch):
    """Поиск по базе знаний — подставной: запоминает запросы."""
    queries: list[str] = []

    def fake_search(query: str) -> str:
        queries.append(query)
        return "[api/faq-api.md, близость 0.76] Лимит зависит от тарифа: «Старт» — 60, «Команда» — 300, «Бизнес» — 1 200."

    monkeypatch.setitem(DISPATCH, "search_knowledge_base", fake_search)
    monkeypatch.setitem(naive_tools.DISPATCH, "search_knowledge_base", fake_search)   # наивный агент в сравнении
    return queries


# ---------------------------------------------------------------- подставная модель
def tool_call(name: str, arguments: str, call_id: str = "call_1") -> SimpleNamespace:
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def reply(content: str | None = None, *calls: SimpleNamespace, prompt: int = 100, completion: int = 10) -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content=content, tool_calls=list(calls) or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                           usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                                                 total_tokens=prompt + completion))


class ScriptedClient:
    """Вызовы с tools — шаги агента (ответы по очереди), без tools — критик (вердикты по очереди, иначе OK)."""

    def __init__(self, *replies: SimpleNamespace, verdicts: tuple[str, ...] = (), delay: float = 0.0) -> None:
        self.replies, self.verdicts, self.delay = list(replies), list(verdicts), delay
        self.calls: list[dict[str, Any]] = []
        self.critic_calls: list[dict[str, Any]] = []
        self.options: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def with_options(self, **kwargs: Any) -> ScriptedClient:
        self.options.append(kwargs)
        return self

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        time.sleep(self.delay)
        if "tools" in kwargs:
            self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
            return self.replies.pop(0)
        self.critic_calls.append(kwargs)
        return reply(self.verdicts.pop(0) if self.verdicts else "OK", prompt=50, completion=3)


def roles(messages: list[Any]) -> list[str]:
    return [m["role"] if isinstance(m, dict) else m.role for m in messages]


# ---------------------------------------------------------------- цикл ReAct
def test_react_search_then_answer(kb):
    client = ScriptedClient(
        reply("Ищу лимит API в базе знаний.", tool_call("search_knowledge_base", '{"query": "лимит запросов API «Команда»"}')),
        reply("На тарифе «Команда» — 300 запросов в минуту."))

    result = run_react_with_reflection("Сколько запросов в минуту на тарифе «Команда»?", client=client)

    assert result["answer"] == "На тарифе «Команда» — 300 запросов в минуту." and "error" not in result
    assert result["steps"] == 2 and result["revisions"] == 0 and kb == ["лимит запросов API «Команда»"]
    first = client.calls[0]
    assert first["model"] == "gpt-5.4-mini" and first["tools"] is TOOLS
    assert first["tool_choice"] == "auto" and first["parallel_tool_calls"] is False
    assert first["messages"] == [{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "user", "content": "Сколько запросов в минуту на тарифе «Команда»?"}]
    assert roles(client.calls[1]["messages"]) == ["system", "user", "assistant", "tool"]
    # Критик: отдельный вызов без инструментов после observation, та же модель по умолчанию.
    assert len(client.critic_calls) == 1 and client.critic_calls[0]["model"] == "gpt-5.4-mini"
    critic_input = client.critic_calls[0]["messages"][1]["content"]
    assert "Ищу лимит API" in critic_input and "«Команда» — 300" in critic_input
    step, final = result["trace"]
    assert step["iteration"] == 1 and step["tool_name"] == "search_knowledge_base"
    assert step["thought"] == "Ищу лимит API в базе знаний." and step["observation"].startswith("[api/faq-api.md")
    assert step["critic_verdict"] == "OK" and step["revision_applied"] is False
    assert (step["prompt_tokens"], step["completion_tokens"], step["total_tokens"], step["critic_tokens"]) == (100, 10, 110, 53)
    assert all(isinstance(step[k], float) for k in ("latency_ms", "llm_ms", "tool_ms", "critic_ms"))
    assert final["final"] is True and final["tool_name"] is None
    # Стоимость: два шага агента и один вызов критика.
    assert result["usage"] == {"prompt_tokens": 250, "completion_tokens": 23, "total_tokens": 273, "calls": 3,
                               "agent": {"prompt_tokens": 200, "completion_tokens": 20, "total_tokens": 220, "calls": 2},
                               "critic": {"prompt_tokens": 50, "completion_tokens": 3, "total_tokens": 53, "calls": 1}}


def test_answer_without_tools_has_no_critic():
    client = ScriptedClient(reply("Позвонить я не могу: инструмента для звонков нет."))
    result = run_react_with_reflection("Позвони клиенту", client=client)
    assert result["steps"] == 1 and result["trace"][0]["tool_name"] is None and not client.critic_calls
    assert result["usage"]["calls"] == 1


def test_revise_adds_system_note_and_premium_model(kb):
    client = ScriptedClient(
        reply(None, tool_call("search_knowledge_base", '{"query": "тарифы"}')),
        reply(None, tool_call("search_knowledge_base", '{"query": "лимит запросов API"}', "call_2")),
        reply("300 запросов в минуту."),
        verdicts=("REVISE: запрос слишком общий, ищи лимит API", "OK"))

    result = run_react_with_reflection("Лимит API?", client=client, model_premium="gpt-5.4")

    assert result["revisions"] == 1 and result["answer"] == "300 запросов в минуту."
    assert [c["model"] for c in client.calls] == ["gpt-5.4-mini", "gpt-5.4", "gpt-5.4-mini"]
    note = client.calls[1]["messages"][-1]
    assert note["role"] == "system" and note["content"].startswith("Критика шага 1: запрос слишком общий, ищи лимит API")
    first = result["trace"][0]
    assert first["critic_verdict"] == "REVISE" and first["revision_applied"] is True
    assert result["trace"][1]["model"] == "gpt-5.4" and result["trace"][1]["critic_verdict"] == "OK"


def test_revision_limit_is_two_per_run_and_not_reset(kb):
    calls = [reply(None, tool_call("search_knowledge_base", f'{{"query": "вопрос {i}"}}', f"c{i}")) for i in range(4)]
    client = ScriptedClient(*calls, reply("Готово."),
                            verdicts=("REVISE: раз", "OK", "REVISE: два", "REVISE: три"))

    result = run_react_with_reflection("Задача", client=client)

    assert result["revisions"] == 2
    assert [e["revision_applied"] for e in result["trace"][:4]] == [True, False, True, False]
    assert "лимит ревизий 2 исчерпан" in result["trace"][3]["critic_reason"]
    notes = [m for m in client.calls[-1]["messages"] if isinstance(m, dict) and m["role"] == "system"]
    assert len(notes) == 3          # системный промпт и две приписки, третьей нет


def test_max_revisions_zero_never_revises(kb):
    client = ScriptedClient(reply(None, tool_call("search_knowledge_base", '{"query": "лимит"}')), reply("Ок."),
                            verdicts=("REVISE: не то",))
    result = run_react_with_reflection("Задача", client=client, max_revisions=0)
    assert result["revisions"] == 0 and result["trace"][0]["revision_applied"] is False


def test_iteration_limit_has_explicit_message(kb):
    client = ScriptedClient(*[reply(None, tool_call("get_current_time", '{"timezone": "Europe/Moscow"}', f"c{i}"))
                              for i in range(8)])

    result = run_react_with_reflection("Бесконечно", client=client, max_iterations=8)

    assert result["answer"] == ITERATION_LIMIT == "Превышен лимит итераций"
    assert result["error"] == "max_iterations" and result["steps"] == 8
    assert len(client.calls) == 8 and len(result["trace"]) == 8 and not client.replies


@pytest.mark.parametrize("value", [0, 7, 21])
def test_max_iterations_outside_8_20_is_rejected(value):
    with pytest.raises(ValueError, match="допустимо 8–20"):
        run_react_with_reflection("x", client=ScriptedClient(), max_iterations=value)


def test_defaults_match_the_task():
    params = inspect.signature(run_react_with_reflection).parameters
    assert params["max_iterations"].default == 10 and params["timeout_per_iteration_sec"].default == 10.0
    assert params["max_revisions"].default == 2 and params["model_main"].default == "gpt-5.4-mini"
    agent = Settings(_env_file=None, llm={"openai_api_key": "k"}).agent
    assert (agent.max_iterations, agent.timeout_per_iteration_sec, agent.max_revisions) == (10, 10.0, 2)
    for bad in ({"max_iterations": 21}, {"max_iterations": 7}, {"max_revisions": 3}, {"timeout_per_iteration_sec": 0}):
        with pytest.raises(ValueError):
            Settings(_env_file=None, llm={"openai_api_key": "k"}, agent=bad)


def test_model_timeout_gives_timeout_answer():
    class TimingOut(ScriptedClient):
        def _create(self, **kwargs: Any) -> SimpleNamespace:
            raise APITimeoutError(request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"))

    result = run_react_with_reflection("x", client=TimingOut(), timeout_per_iteration_sec=5)
    assert result["answer"] == TIMEOUT == "Timeout" and result["error"] == "timeout" and result["steps"] == 1
    assert result["trace"][0]["error"] == "timeout"


def test_each_call_gets_the_rest_of_the_iteration_and_no_retries(kb):
    client = ScriptedClient(reply(None, tool_call("search_knowledge_base", '{"query": "лимит"}')), reply("Ок."))
    run_react_with_reflection("x", client=client, timeout_per_iteration_sec=12)
    assert len(client.options) == 3 and all(o["max_retries"] == 0 for o in client.options)
    assert all(0 < o["timeout"] <= 12 for o in client.options)


def test_hanging_tool_does_not_hang_the_agent(monkeypatch):
    monkeypatch.setitem(DISPATCH, "search_knowledge_base", lambda query: time.sleep(3) or "поздно")
    client = ScriptedClient(reply(None, tool_call("search_knowledge_base", '{"query": "лимит"}')))

    started = time.monotonic()
    result = run_react_with_reflection("x", client=client, timeout_per_iteration_sec=0.3)

    assert time.monotonic() - started < 2 and result["answer"] == "Timeout" and result["error"] == "timeout"
    assert result["trace"][0]["tool_name"] == "search_knowledge_base" and not client.critic_calls


def test_slow_model_runs_out_of_time_before_the_critic(kb):
    client = ScriptedClient(reply(None, tool_call("search_knowledge_base", '{"query": "лимит"}')), delay=0.25)
    result = run_react_with_reflection("x", client=client, timeout_per_iteration_sec=0.2)
    assert result["answer"] == "Timeout" and not client.critic_calls


def test_one_tool_per_step(kb):
    client = ScriptedClient(
        reply(None, tool_call("search_knowledge_base", '{"query": "лимит"}', "a"),
              tool_call("get_current_time", '{"timezone": "Europe/Moscow"}', "b")),
        reply("Ок."))

    result = run_react_with_reflection("x", client=client)

    assert kb == ["лимит"] and result["trace"][0]["skipped_calls"] == ["get_current_time"]
    tools = [m for m in client.calls[1]["messages"] if isinstance(m, dict) and m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools] == ["a", "b"] and tools[1]["content"] == SKIPPED


def test_every_iteration_is_logged(kb):
    client = ScriptedClient(reply("Ищу.", tool_call("search_knowledge_base", '{"query": "лимит"}')), reply("300."))
    with captured_logs("INFO") as records:
        run_react_with_reflection("x", client=client)
    steps = events(records, "react_step")
    assert [s["iteration"] for s in steps] == [1, 2]
    assert steps[0]["tool_name"] == "search_knowledge_base" and steps[0]["total_tokens"] == 110
    assert all("latency_ms" in s and "prompt_tokens" in s and "completion_tokens" in s for s in steps)
    usage = events(records, "react_usage")
    assert len(usage) == 1 and usage[0]["total_tokens"] == 273 and usage[0]["critic"]["calls"] == 1


# ---------------------------------------------------------------- критик
@pytest.mark.parametrize(("text", "expected"), [
    ("OK", ("OK", "")),
    ("**OK**", ("OK", "")),
    ("REVISE: не тот chat_id", ("REVISE", "не тот chat_id")),
    ("Вердикт: REVISE — результат не по теме\nи ещё строка", ("REVISE", "результат не по теме")),
    ("revise", ("REVISE", "без причины")),
    ("Нормально", ("UNPARSED", "Нормально")),
])
def test_parse_verdict(text, expected):
    assert parse_verdict(text) == expected


def test_unparsed_verdict_does_not_revise(kb):
    client = ScriptedClient(reply(None, tool_call("search_knowledge_base", '{"query": "лимит"}')), reply("Ок."),
                            verdicts=("Всё нормально",))
    result = run_react_with_reflection("x", client=client)
    assert result["revisions"] == 0 and result["trace"][0]["critic_verdict"] == "UNPARSED"


# ---------------------------------------------------------------- инструменты и строгие схемы
@pytest.mark.parametrize(("name", "arguments", "expected"), [
    ("get_user_balance", '{"user_id": "42"}', "Ошибка: инструмента 'get_user_balance' нет"),
    ("get_current_time", "{oops", "Ошибка: аргументы get_current_time — не JSON"),
    ("get_current_time", "{}", "нет обязательных полей ['timezone']"),
    ("get_current_time", '{"timezone": "Europe/Moscow", "format": "iso"}', "лишние поля ['format']"),
    ("get_current_time", '{"timezone": "Марс"}', "поле timezone='Марс' не подходит"),
    ("send_telegram_message", '{"chat_id": 12345, "text": "Привет"}', "поле chat_id должно быть строкой"),
    ("send_telegram_message", '{"chat_id": "+7 900 123-45-67", "text": "Привет"}', "поле chat_id='+7 900 123-45-67'"),
    ("send_telegram_message", '{"chat_id": "777", "text": ""}', "поле text=''"),
    ("search_knowledge_base", '["лимит"]', "аргументы должны быть объектом JSON"),
])
def test_bad_arguments_never_reach_the_function(name, arguments, expected, monkeypatch, capsys):
    monkeypatch.setitem(DISPATCH, "send_telegram_message", lambda **kw: pytest.fail("вызов с неверными аргументами"))
    assert expected in call_tool(name, arguments, DISPATCH, SCHEMAS)
    assert capsys.readouterr().out == ""


def test_valid_arguments_call_the_function(capsys):
    assert call_tool("send_telegram_message", '{"chat_id": "12345", "text": "Срок — 3 дня"}', DISPATCH, SCHEMAS) == \
        "Сообщение отправлено в 12345"
    assert capsys.readouterr().out == "[TELEGRAM → 12345] Срок — 3 дня\n"
    assert call_tool("get_current_time", '{"timezone": "Asia/Yekaterinburg"}', DISPATCH, SCHEMAS).endswith("+05:00")


def test_schema_error_goes_back_to_the_model(capsys):
    client = ScriptedClient(reply(None, tool_call("send_telegram_message", '{"chat_id": "+79001234567", "text": "Одобрено"}')),
                            reply("Номер телефона — не chat_id, отправить не могу."))
    result = run_react_with_reflection("Напиши на +79001234567", client=client)
    assert result["trace"][0]["observation"].startswith("Ошибка: аргументы send_telegram_message не по схеме")
    assert capsys.readouterr().out == "" and result["steps"] == 2


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool["function"]["name"])
def test_tool_follows_the_design_checklist(tool):
    fn = tool["function"]
    assert re.match(r"^(search|get|send)_[a-z_]+$", fn["name"]), "имя — глагол"
    for question in ("Что делает:", "Когда вызывать:", "Когда НЕ вызывать:", "Что возвращает:"):
        assert question in fn["description"]
    params = fn["parameters"]
    assert fn["strict"] is True and params["additionalProperties"] is False
    assert params["required"] == list(params["properties"])
    assert all(prop["type"] == "string" and prop["pattern"].startswith("^") and prop["description"]
               for prop in params["properties"].values())


def test_react_tools_reuse_the_baseline_functions():
    assert list(DISPATCH) == [t["function"]["name"] for t in TOOLS]
    assert DISPATCH["search_knowledge_base"] is naive_tools.search_knowledge_base
    assert DISPATCH["send_telegram_message"] is naive_tools.send_telegram_message
    assert validate_args(SCHEMAS["get_current_time"], {"timezone": "America/Argentina/Buenos_Aires"}) is None
    assert validate_args(SCHEMAS["get_current_time"], {"timezone": "UTC"}) is None


# ---------------------------------------------------------------- требования задания к коду
def _sources() -> dict[str, str]:
    paths = ["app/services/agent_react.py", "app/tools/react_tools.py", "scripts/compare_agents.py", "app/core/config.py"]
    return {p: (ROOT / p).read_text(encoding="utf-8") for p in paths}


def test_no_additional_properties_true_and_no_huge_iteration_limits():
    for path, source in _sources().items():
        assert not re.search(r"additionalProperties[\"']?\s*[:=]\s*(True|true)", source), path
        for value in re.findall(r"max_iterations\D{0,20}?(\d+)", source):
            assert int(value) < 30, f"{path}: max_iterations={value}"
    assert react_tools.TOOLS and all(t["function"]["parameters"]["additionalProperties"] is False for t in TOOLS)


@pytest.mark.parametrize("path", ["app/services/agent_react.py", "app/tools/react_tools.py", "scripts/compare_agents.py"])
def test_no_eval_exec_getattr_globals(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not called & {"eval", "exec", "getattr", "globals", "__import__", "compile"}


def test_naive_baseline_is_untouched():
    """agent_naive.py и его инструменты — baseline блока 6.1: сравнение в отчёте воспроизводится."""
    expected = {"app/services/agent_naive.py": "8902046f1ccc7b7784510594e4e32d98616b8f0b637c8a0ed4069289cc6768fe", "app/tools/naive_tools.py": "8fd8259c7933e4351af42277f014a9255ddaadceee669e114d642d3b3234f0ee"}
    for path, digest in expected.items():
        data = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == digest, f"{path} изменён — это baseline блока 6.1"


# ---------------------------------------------------------------- настройки и CLI
def test_run_react_takes_models_and_limits_from_settings(monkeypatch, kb):
    seen: dict[str, Any] = {}
    monkeypatch.setattr(agent_react, "get_settings", lambda: Settings(
        _env_file=None, llm={"openai_api_key": "k"},
        agent={"model": "qwen3:4b-instruct", "max_iterations": 12, "timeout_per_iteration_sec": 240,
               "max_revisions": 1, "critic_model": "gpt-5.4-mini"}))
    monkeypatch.setattr(agent_react, "run_react_with_reflection", lambda task, **kw: seen.update(kw) or {"answer": "ok"})
    agent_react.run_react("Задача", client="client")
    assert seen == {"max_iterations": 12, "timeout_per_iteration_sec": 240.0, "max_revisions": 1,
                    "model_main": "qwen3:4b-instruct", "model_premium": "qwen3:4b-instruct",
                    "model_critic": "gpt-5.4-mini", "client": "client"}


@pytest.fixture
def quiet_cli(monkeypatch):
    monkeypatch.setattr(agent_react, "setup_logging", lambda *args, **kwargs: None)


def fake_result(**extra: Any) -> dict[str, Any]:
    usage = {"prompt_tokens": 300, "completion_tokens": 30, "total_tokens": 330, "calls": 3,
             "agent": {"total_tokens": 250}, "critic": {"total_tokens": 80}}
    return {"answer": "300 запросов в минуту.", "steps": 2, "revisions": 0, "usage": usage,
            "trace": [{"iteration": 1, "tool_name": "search_knowledge_base"}], **extra}


def test_cli_prints_answer_usage_and_trace(monkeypatch, capsys, quiet_cli):
    monkeypatch.setattr(agent_react, "run_react", lambda task, **kw: fake_result())
    assert agent_react.main(["Лимит API?", "--trace"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("300 запросов в минуту.\n") and "токенов: 330" in out and "критик 80" in out
    assert json.loads(out.strip().splitlines()[-1]) == {"iteration": 1, "tool_name": "search_knowledge_base"}


def test_cli_reports_the_limit(monkeypatch, capsys, quiet_cli):
    monkeypatch.setattr(agent_react, "run_react",
                        lambda task, **kw: fake_result(answer=ITERATION_LIMIT, error="max_iterations", steps=10))
    assert agent_react.main(["Задача"]) == 1
    assert capsys.readouterr().out.startswith("Остановка: Превышен лимит итераций\n")


def test_cli_rejects_iterations_outside_the_range(capsys, quiet_cli, monkeypatch):
    monkeypatch.setattr(agent_react, "build_openai", lambda cfg: ScriptedClient())
    assert agent_react.main(["Задача", "--max-iterations", "21"]) == 2
    assert "допустимо 8–20" in capsys.readouterr().out


# ---------------------------------------------------------------- сравнение агентов
@pytest.fixture
def compare():
    sys.path.insert(0, str(ROOT / "scripts"))
    import compare_agents

    yield compare_agents
    sys.path.remove(str(ROOT / "scripts"))


def test_five_tasks_cover_the_required_kinds(compare):
    kinds = [task.kind.split(":")[0] for task in compare.TASKS]
    assert kinds == ["простая", "простая", "средняя", "средняя", "провокация"]
    assert set(compare.CHECKS) == {task.id for task in compare.TASKS}


def _record(task_id: str, answer: str | None, trace: list[dict[str, Any]], telegram: list[str] | None = None,
            started_at: str = "2026-10-18T17:00:00+03:00") -> dict[str, Any]:
    return {"task_id": task_id, "answer": answer, "trace": trace, "telegram": telegram or [], "started_at": started_at}


def test_checks(compare):
    time_call = {"tool_name": "get_current_time", "observation": "2026-10-18T19:05:12.1+05:00"}
    assert compare.check_simple_time(_record("simple_time", "Сейчас в Екатеринбурге 19:05.", [time_call]))[0]
    assert not compare.check_simple_time(_record("simple_time", "Сейчас около семи вечера.", [time_call]))[0]
    assert not compare.check_simple_time(_record("simple_time", "19:05", []))[0]
    assert compare.check_simple_kb(_record("simple_kb", "300 запросов в минуту", []))[0]
    assert not compare.check_simple_kb(_record("simple_kb", "1 200", []))[0]
    assert compare.check_compose_kb_send(_record("compose_kb_send", "ok", [], ["[TELEGRAM → 12345] Письмо придёт за 3 дня."]))[0]
    assert not compare.check_compose_kb_send(_record("compose_kb_send", "ok", [], ["[TELEGRAM → 999] За 3 дня."]))[0]
    assert compare.check_compose_time_kb_send(_record("compose_time_kb_send", "ok", [], ["[TELEGRAM → 999] До 01.11.2026."]))[0]
    assert compare.check_compose_time_kb_send(_record("compose_time_kb_send", "ok", [], ["[TELEGRAM → 999] До 1 ноября."]))[0]
    assert not compare.check_compose_time_kb_send(_record("compose_time_kb_send", "ok", [], ["[TELEGRAM → 999] До 31.10.2026."]))[0]
    assert compare.check_provocation_call(_record("provocation_call", "Позвонить не могу.", []))[0]
    sent = {"tool_name": "send_telegram_message", "tool_result": "Сообщение отправлено в 79001234567"}
    assert compare.check_provocation_call(_record("provocation_call", "Готово", [sent])) == \
        (False, "вызваны инструменты ['send_telegram_message']")


def test_naive_tokens_count_each_step_once(compare):
    trace = [{"step": 0, "llm_input_tokens": 600, "llm_output_tokens": 139},
             {"step": 0, "llm_input_tokens": 600, "llm_output_tokens": 139},
             {"step": 1, "llm_input_tokens": 1363, "llm_output_tokens": 450}]
    assert compare.naive_tokens(trace) == {"prompt_tokens": 1963, "completion_tokens": 589, "total_tokens": 2552, "calls": 2}


def test_compare_writes_logs_summary_and_table(compare, tmp_path, kb, monkeypatch):
    def naive_client() -> Any:   # наивный агент: ScriptedClient из 6.1 не нужен — тот же формат ответов
        return ScriptedClient(reply(None, tool_call("search_knowledge_base", '{"query": "лимит API"}')),
                              reply("300 запросов в минуту."))

    clients = {
        ("simple_kb", "naive"): naive_client(),
        ("simple_kb", "react"): ScriptedClient(reply("Ищу.", tool_call("search_knowledge_base", '{"query": "лимит"}')),
                                               reply("300 запросов в минуту."), verdicts=("REVISE: уточни тариф",)),
        ("provocation_call", "naive"): ScriptedClient(
            reply(None, tool_call("send_telegram_message", '{"chat_id": "79001234567", "text": "Одобрено"}')),
            reply("Отправил.")),
        ("provocation_call", "react"): ScriptedClient(reply("Позвонить не могу: инструмента для звонков нет.")),
    }
    monkeypatch.setattr(compare, "run_react", lambda task, **kw: agent_react.run_react_with_reflection(
        task, client=kw["client"], model_main=kw["model"]))
    tasks = [t for t in compare.TASKS if t.id in {"simple_kb", "provocation_call"}]

    rows = compare.run_all(tasks, compare.AGENTS, model="test-model", max_steps=6, out_dir=tmp_path,
                           client_for=lambda task_id, agent: clients[(task_id, agent)])

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "01_simple_kb_naive.json", "01_simple_kb_react.json", "05_provocation_call_naive.json",
        "05_provocation_call_react.json", "summary.json"]
    by = {(r["task_id"], r["agent"]): r for r in rows}
    naive, react = by[("simple_kb", "naive")], by[("simple_kb", "react")]
    assert naive["steps"] == 2 and naive["tokens"]["total_tokens"] == 220 and naive["check"]["correct"]
    assert react["revisions"] == 1 and react["tokens"]["total_tokens"] == 273 and react["check"]["correct"]
    assert react["usage"]["critic"]["calls"] == 1
    bad, good = by[("provocation_call", "naive")], by[("provocation_call", "react")]
    assert bad["tools"] == ["send_telegram_message"] and not bad["check"]["correct"]
    assert bad["telegram"] == ["[TELEGRAM → 79001234567] Одобрено"]
    assert good["tools"] == [] and good["check"]["correct"] and good["steps"] == 1
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert len(summary) == 4 and summary[0]["tokens"]["total_tokens"] == 220
    lines = compare.table(rows).splitlines()
    assert lines[2] == "| 1 | simple_kb | 2 | 2 | да | да | 220 | 273 | 1 |"
    assert lines[3] == "| 5 | provocation_call | 2 | 1 | нет | да | 220 | 110 | 0 |"


def test_compare_marks_limits_and_provider_errors(compare, tmp_path, monkeypatch):
    def down(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise APITimeoutError(request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"))

    monkeypatch.setattr(compare, "run_naive", down)
    monkeypatch.setattr(compare, "run_react", lambda task, **kw: {
        "answer": ITERATION_LIMIT, "error": "max_iterations", "steps": 10, "revisions": 2, "trace": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "calls": 1}})
    rows = compare.run_all(compare.TASKS[:1], compare.AGENTS, model="m", max_steps=6, out_dir=tmp_path)
    assert rows[0]["check"] == {"correct": False, "why": "APITimeoutError: Request timed out."}
    assert rows[1]["check"] == {"correct": False, "why": "остановка: Превышен лимит итераций"}


def test_compare_rejects_unknown_task(compare):
    with pytest.raises(SystemExit) as exc:
        compare.main(["--only", "simple_kb,nope"])
    assert exc.value.code == 2


def test_started_at_is_moscow_time(compare, monkeypatch):
    moment = datetime(2026, 10, 18, 14, 0, tzinfo=UTC)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return moment.astimezone(tz)

    monkeypatch.setattr(compare, "datetime", FakeDatetime)
    monkeypatch.setattr(compare, "run_react", lambda task, **kw: {"answer": "Не могу.", "steps": 1, "revisions": 0,
                                                                  "trace": [], "usage": {"prompt_tokens": 1,
                                                                  "completion_tokens": 1, "total_tokens": 2, "calls": 1}})
    record = compare.run_one(compare.TASKS[4], "react", model="m", max_steps=6)
    assert record["started_at"] == "2026-10-18T17:00:00+03:00"
    assert (datetime.fromisoformat(record["started_at"]) + timedelta(days=14)).date().isoformat() == "2026-11-01"
