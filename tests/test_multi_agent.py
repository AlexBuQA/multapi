"""
Эксперимент блока 6.5 (experiments/): мультиагент и single-agent без сети — модель-сценарий по system prompt,
поиск по базе знаний подменён. Проверяется то, на чём держится честность сравнения: один объект инструмента,
одинаковые бюджеты, подсчёт вызовов модели, токенов и передач управления, чистый контекст агентов и запись
results.json.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from experiments import common, compare
from experiments import multi_agent_langgraph as mal
from experiments import single_agent_baseline as sab

FRAGMENT = ("[api/faq-api.md, близость 0.80] Лимит зависит от тарифа: «Старт» — 60 запросов в минуту, «Команда» — "
            "300. При превышении API отвечает кодом 429 и заголовком Retry-After.")
ANSWER = ("На тарифе «Команда» — 300 запросов в минуту, при превышении API вернёт код 429 [1].\n"
          "Источники:\n[1] api/faq-api.md")
Q2 = next(q for q in common.QUESTIONS if q.id == "q2")


class TeamChat(BaseChatModel):
    """Модель-сценарий: роль — по началу system prompt. supervisor: normal — researcher, потом writer;
    always_research — всегда researcher; text — без вызова инструмента; double — два вызова за ход.
    researcher и single: поиск, пока нет результата (endless — всегда), затем факты или ответ."""

    supervisor: str = "normal"
    research: str = "once"
    usage: bool = True
    seen: list[tuple[str, list[Any]]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "team-scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> TeamChat:
        return self

    def _reply(self, role: str, messages: list[Any]) -> AIMessage:
        tools = [m for m in messages if isinstance(m, ToolMessage)]
        if role == "supervisor":
            if self.supervisor == "text":
                return AIMessage("Сейчас разберёмся.")
            target = "transfer_to_researcher" if self.supervisor == "always_research" or not tools else \
                "transfer_to_writer"
            calls = [{"name": target, "args": {"task": "лимит API на тарифе «Команда»"}, "id": f"sv{len(tools)}"}]
            if self.supervisor == "double":
                calls.append({"name": "transfer_to_writer", "args": {"task": "ответь"}, "id": f"sv{len(tools)}b"})
            return AIMessage("", tool_calls=calls)
        if role in ("researcher", "single") and (self.research == "endless" or not tools):
            return AIMessage("", tool_calls=[{"name": "search_knowledge_base", "id": f"s{len(tools)}",
                                              "args": {"query": f"лимит API Команда {len(tools)}"}}])
        if role == "researcher":
            return AIMessage("- «Команда» — 300 запросов в минуту [api/faq-api.md]\n- при превышении — код 429 "
                             "[api/faq-api.md]")
        if role in ("writer", "single"):
            return AIMessage(ANSWER)
        return AIMessage("готово")

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        system = messages[0].content if messages and isinstance(messages[0], SystemMessage) else ""
        role = next((name for prefix, name in (("Ты — supervisor", "supervisor"), ("Ты — researcher", "researcher"),
                                               ("Ты — writer", "writer"), ("Ты — агент", "single"))
                     if system.startswith(prefix)), "other")
        self.seen.append((role, list(messages)))
        reply = self._reply(role, messages)
        if self.usage:
            reply.usage_metadata = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
        return ChatResult(generations=[ChatGeneration(message=reply)])


@pytest.fixture
def kb(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    queries: list[str] = []

    def fake(query: str) -> str:
        queries.append(query)
        return FRAGMENT

    monkeypatch.setattr(common, "kb_search", fake)
    return queries


def team(model: TeamChat) -> Any:
    return mal.build_team(model, checkpointer=InMemorySaver())


def ask_team(app: Any, model: TeamChat, question: common.Question = Q2) -> dict[str, Any]:
    return mal.ask(app, question, info=common.model_info(model), model=model, cache_reset=False)


def roles(model: TeamChat) -> list[str]:
    return [role for role, _ in model.seen]


# ---------------------------------------------------------------- честность сравнения
def test_both_implementations_use_the_same_tool_object() -> None:
    model = TeamChat()
    researcher = mal.build_agents(model)[mal.RESEARCHER]
    baseline = sab.build_baseline(model)
    for agent in (researcher, baseline):
        assert agent.nodes["tools"].bound.tools_by_name["search_knowledge_base"] is common.search_knowledge_base
    assert mal.search_knowledge_base is sab.search_knowledge_base is common.search_knowledge_base


def test_prompts_share_search_and_answer_rules() -> None:
    assert common.SEARCH_RULES in common.RESEARCHER_PROMPT and common.SEARCH_RULES in common.SINGLE_PROMPT
    assert common.ANSWER_RULES in common.WRITER_PROMPT and common.ANSWER_RULES in common.SINGLE_PROMPT
    assert "НЕ пиши" in common.RESEARCHER_PROMPT and "[1], [2]" in common.ANSWER_RULES


def test_test_set_is_three_corpus_one_multistep_one_out_of_base() -> None:
    kinds = [q.kind for q in common.QUESTIONS]
    assert kinds.count("corpus") == 3 and kinds.count("multi_step") == 1 and kinds.count("out_of_base") == 1
    assert len({q.id for q in common.QUESTIONS}) == 5
    assert [q.id for q in common.pick_questions("q4,q1")] == ["q4", "q1"]
    with pytest.raises(SystemExit):
        common.pick_questions("q9")


# ---------------------------------------------------------------- мультиагент
def test_team_route_metrics_and_citations(kb: list[str]) -> None:
    model = TeamChat()
    rec = ask_team(team(model), model)
    assert rec["route"] == ["supervisor", "researcher", "supervisor", "writer"]
    assert rec["handoff_count"] == 3 and rec["fallback_routes"] == 0 and rec["forced_routes"] == 0
    assert rec["llm_calls"] == 5 == len(model.seen)
    assert {name: s["calls"] for name, s in rec["by_agent"].items()} == {"supervisor": 2, "researcher": 2, "writer": 1}
    assert rec["total_tokens"] == 5 * 110 and rec["input_tokens"] == 500 and rec["llm_calls_without_usage"] == 0
    assert rec["answer"] == ANSWER and rec["checks"]["citations"] == [1]
    assert rec["checks"]["facts_hit"] == 2 and rec["checks"]["expected"] is True
    assert rec["queries"] == kb == ["лимит API Команда 0"] and rec["contexts"] == [FRAGMENT]
    assert "300 запросов" in rec["facts"] and rec["error"] is None and rec["latency_ms"] >= 0


def test_agents_get_clean_context_and_supervisor_sees_only_facts(kb: list[str]) -> None:
    model = TeamChat()
    ask_team(team(model), model)
    researcher_in = next(msgs for role, msgs in model.seen if role == "researcher")
    assert [type(m) for m in researcher_in] == [SystemMessage, HumanMessage]
    assert researcher_in[1].content.startswith(f"Вопрос клиента: {Q2.text}\nЗадание supervisor: лимит API")
    writer_in = next(msgs for role, msgs in model.seen if role == "writer")
    assert "Факты от researcher:\n- «Команда» — 300" in writer_in[1].content
    second_supervisor = [msgs for role, msgs in model.seen if role == "supervisor"][1]
    facts = second_supervisor[-1]
    assert isinstance(facts, ToolMessage) and facts.tool_call_id == "sv0" and facts.name == "transfer_to_researcher"
    assert not any(isinstance(m, ToolMessage) and m.name == "search_knowledge_base" for m in second_supervisor)


def test_one_thread_for_all_questions_but_clean_supervisor_context(kb: list[str]) -> None:
    model = TeamChat()
    app = team(model)
    q1 = common.QUESTIONS[0]
    ask_team(app, model, q1)
    model.seen.clear()
    ask_team(app, model, Q2)
    first_supervisor = next(msgs for role, msgs in model.seen if role == "supervisor")
    assert [m.content for m in first_supervisor[1:]] == [Q2.text]
    state = app.get_state({"configurable": {"thread_id": mal.THREAD_ID}})
    assert state.values["messages"][0].content == Q2.text and state.values["route"][-1] == "writer"
    assert len(list(app.get_state_history({"configurable": {"thread_id": mal.THREAD_ID}}))) > 10


def test_budget_reserves_last_handoff_for_writer(kb: list[str]) -> None:
    model = TeamChat(supervisor="always_research")
    rec = ask_team(team(model), model)
    assert rec["route"] == ["supervisor", "researcher", "supervisor", "researcher", "supervisor", "writer"]
    assert rec["handoff_count"] == mal.MAX_HANDOFFS and rec["forced_routes"] == 1
    assert rec["by_agent"]["supervisor"]["calls"] == 2           # третий выбор — без вызова модели
    assert len(kb) == 2 and rec["answer"] == ANSWER


def test_fallback_route_when_supervisor_does_not_call_a_tool(kb: list[str]) -> None:
    model = TeamChat(supervisor="text")
    app = team(model)
    rec = ask_team(app, model)
    assert rec["route"] == ["supervisor", "researcher", "supervisor", "writer"] and rec["fallback_routes"] == 2
    messages = app.get_state({"configurable": {"thread_id": mal.THREAD_ID}}).values["messages"]
    assert any(isinstance(m, AIMessage) and m.name == "researcher" for m in messages)
    assert not any(isinstance(m, ToolMessage) for m in messages)


def test_extra_supervisor_calls_get_tool_messages(kb: list[str]) -> None:
    model = TeamChat(supervisor="double")
    app = team(model)
    rec = ask_team(app, model)
    assert rec["route"][:2] == ["supervisor", "researcher"]
    messages = app.get_state({"configurable": {"thread_id": mal.THREAD_ID}}).values["messages"]
    calls = {c["id"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    assert calls == answered
    assert any(m.content == mal.NOT_EXECUTED for m in messages if isinstance(m, ToolMessage))


def test_architecture_file_has_supervisor_graph(tmp_path: Any) -> None:
    model = TeamChat()
    agents = mal.build_agents(model)
    text = mal.write_architecture(mal.build_team(model, agents=agents), agents[mal.RESEARCHER], tmp_path / "a.md")
    for edge in ("__start__ --> supervisor", "supervisor -.-> researcher", "supervisor -.-> writer",
                 "researcher -.-> supervisor", "writer --> __end__"):
        assert edge in text
    assert text.count("```mermaid") == 2 and b"\r\n" not in (tmp_path / "a.md").read_bytes()
    assert "ToolCallLimitMiddleware.after_model" in text and "Middleware[" not in text   # «[» в узле ломает Mermaid


# ---------------------------------------------------------------- single-agent и бюджеты
def test_baseline_record(kb: list[str]) -> None:
    model = TeamChat()
    agent = sab.build_baseline(model, checkpointer=InMemorySaver())
    rec = sab.ask(agent, Q2, info=common.model_info(model), model=model, cache_reset=False)
    assert rec["implementation"] == "single_agent" and rec["handoff_count"] == 0
    assert rec["llm_calls"] == 2 and rec["by_agent"] == {"support_agent": {"calls": 2, "input_tokens": 200,
                                                                           "output_tokens": 20, "total_tokens": 220}}
    assert rec["answer"] == ANSWER and rec["searches"] == 1 and rec["contexts"] == [FRAGMENT]


def test_create_agent_budgets_cap_searches_and_model_calls(kb: list[str]) -> None:
    model = TeamChat(research="endless")
    agent = sab.build_baseline(model, checkpointer=InMemorySaver())
    rec = sab.ask(agent, Q2, info=common.model_info(model), model=model, cache_reset=False)
    assert len(kb) == common.MAX_SEARCHES                         # лишние поиски заблокированы
    assert rec["llm_calls"] == common.MAX_MODEL_CALLS and rec["error"] is None


def test_reset_cache_is_not_metered(kb: list[str]) -> None:
    model = TeamChat()
    rec = mal.ask(team(model), Q2, info=common.model_info(model), model=model, cache_reset=True)
    assert roles(model)[0] == "other" and rec["llm_calls"] == 5 == len(model.seen) - 1


def test_meter_counts_calls_without_usage(kb: list[str]) -> None:
    model = TeamChat(usage=False)
    rec = ask_team(team(model), model)
    assert rec["llm_calls"] == 5 and rec["llm_calls_without_usage"] == 5 and rec["total_tokens"] == 0


# ---------------------------------------------------------------- инструмент и проверки
def _node(source: str, score: float, text: str) -> Any:
    return SimpleNamespace(metadata={"source": source}, score=score, node=SimpleNamespace(get_content=lambda: text))


def test_search_tool_returns_top3_with_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.tools.naive_tools as naive

    nodes = [_node(f"doc{n}.md", 0.9 - n / 10, f"текст\n{n}") for n in range(5)]
    found = SimpleNamespace(nodes=nodes, top_score=0.9, threshold=0.5)
    monkeypatch.setattr(naive, "knowledge_base", lambda: SimpleNamespace(retrieve=lambda q: found))
    out = common.search_knowledge_base.invoke({"query": "лимит"})
    assert out.split("\n\n") == ["[doc0.md, близость 0.90] текст 0", "[doc1.md, близость 0.80] текст 1",
                                 "[doc2.md, близость 0.70] текст 2"]
    found.nodes, found.top_score = [], 0.31
    assert common.search_knowledge_base.invoke({"query": "amoCRM"}).startswith(
        "В базе знаний ничего не найдено по запросу «amoCRM»: лучшая близость 0.31 ниже порога 0.5")

    def broken() -> Any:
        raise ConnectionError("Qdrant недоступен")

    monkeypatch.setattr(naive, "knowledge_base", broken)
    assert common.search_knowledge_base.invoke({"query": "лимит"}).startswith(
        "Ошибка инструмента search_knowledge_base: ConnectionError")


def test_check_answer() -> None:
    by_id = {q.id: q for q in common.QUESTIONS}
    q4 = common.check_answer(by_id["q4"], "Проверьте баланс [1], оплатите через СБП [1]; поддержка ответит за 1 час [2].")
    assert q4 == {"facts_hit": 3, "facts_total": 3, "citations": [1, 2], "refused": False, "expected": True}
    q5 = common.check_answer(by_id["q5"], "В базе знаний нет информации об интеграции с amoCRM.")
    assert q5["refused"] is True and q5["expected"] is True and q5["facts_total"] == 0
    assert common.check_answer(by_id["q5"], "Откройте «Интеграции» → amoCRM.")["expected"] is False
    q3 = common.check_answer(by_id["q3"], "Письмо придёт за 3 дня, спишут в последний день.")
    assert q3["facts_hit"] == 2 and q3["citations"] == []
    assert common.check_answer(by_id["q5"], "Ничего не нашёл в справке.")["refused"] is True


def test_save_records_replaces_only_same_implementation_and_question(tmp_path: Any) -> None:
    path = tmp_path / "results.json"
    rec = {"implementation": "multi_agent", "question_id": "q1", "answer": "старый"}
    common.save_records([rec, {**rec, "question_id": "q2"}], path)
    common.save_records([{**rec, "implementation": "single_agent"}], path)
    merged = common.save_records([{**rec, "answer": "новый"}], path)
    assert [(r["implementation"], r["question_id"], r["answer"]) for r in merged] == [
        ("single_agent", "q1", "старый"), ("multi_agent", "q1", "новый"), ("multi_agent", "q2", "старый")]
    assert b"\r\n" not in path.read_bytes() and json.loads(path.read_text(encoding="utf-8")) == merged


def test_print_update_shows_agent_namespace(capsys: pytest.CaptureFixture[str]) -> None:
    call = AIMessage("", tool_calls=[{"name": "search_knowledge_base", "args": {"query": "лимит"}, "id": "1"}])
    common.print_update((("researcher:abc",), {"model": {"messages": [call]}}))
    common.print_update(((), {"ModelCallLimitMiddleware.after_model": {"run_model_call_count": 1}}))
    common.print_update(((), {"supervisor": {"route": ["supervisor", "writer"], "task": mal.FORCED_TASK}}))
    out = capsys.readouterr().out.splitlines()
    assert out[0] == '  [researcher/model] → search_knowledge_base({"query": "лимит"})'
    assert out[1].startswith("  [supervisor] → writer без вызова модели: Бюджет") and len(out) == 2


# ---------------------------------------------------------------- судья и сводка
def _rec(impl: str, qid: str, tokens: int, calls: int, latency: float, handoffs: int, **judge: Any) -> dict[str, Any]:
    question = next(q for q in common.QUESTIONS if q.id == qid)
    return {"implementation": impl, "question_id": qid, "kind": question.kind, "question": question.text,
            "total_tokens": tokens, "input_tokens": tokens - 10, "output_tokens": 10, "llm_calls": calls,
            "latency_ms": latency, "handoff_count": handoffs, "searches": 1, "error": None, "answer": "ответ [1]",
            "by_agent": {"x": {"calls": calls}}, "contexts": [FRAGMENT], "model": "m",
            "checks": common.check_answer(question, "ответ [1]"), **({"judge": judge} if judge else {})}


def test_summarize_and_main_table() -> None:
    records = [_rec("single_agent", q, 1000, 2, lat, 0, correctness=4, faithfulness=5)
               for q, lat in (("q1", 10), ("q2", 30), ("q3", 20))]
    records += [_rec("multi_agent", q, 3000, 5, lat, 3, correctness=5, faithfulness=5)
                for q, lat in (("q1", 40), ("q2", 60), ("q3", 50))]
    summary = compare.summarize(records)
    assert summary["single_agent"]["latency_p50_ms"] == 20 and summary["multi_agent"]["latency_p50_ms"] == 50
    assert summary["delta"]["tokens_x"] == 3.0 and summary["delta"]["latency_p50_x"] == 2.5
    assert summary["delta"]["handoffs_diff"] == 3.0 and summary["delta"]["correctness_diff"] == 1.0
    table = compare.main_table(summary)
    assert "| Токены на запрос | 1 000 | 3 000 | ×3.00 (+200 %) |" in table
    assert "| Передач управления на запрос | 0.0 | 3.0 | +3.0 |" in table
    assert "| Качество: LLM-судья, correctness (1–5) | 4.00 | 5.00 | +1.00 |" in table
    assert "q2 (corpus) | Multi-agent | 3 000 | 5 (x 5)" in compare.per_question_table(records)


def test_parse_verdict() -> None:
    assert compare.parse_verdict('Оценка: {"correctness": 4, "faithfulness": 7, "reason": "ок"}') == {
        "correctness": 4, "faithfulness": 5, "reason": "ок"}
    assert "error" in compare.parse_verdict("не знаю")
    assert "error" in compare.parse_verdict('{"correctness": "высоко"}')


def test_judge_records_skips_judged_answers() -> None:
    class Judge:
        calls = 0

        def invoke(self, messages: Any) -> AIMessage:
            Judge.calls += 1
            assert "Эталон: 300 запросов в минуту" in messages[1].content and FRAGMENT in messages[1].content
            return AIMessage('{"correctness": 5, "faithfulness": 4, "reason": "верно"}')

    records = [_rec("single_agent", "q2", 1, 1, 1, 0), _rec("multi_agent", "q2", 1, 1, 1, 3, correctness=3,
                                                              faithfulness=3)]
    assert compare.judge_records(records, Judge(), info={"judge_model": "j"}, rejudge=False) == 1
    assert records[0]["judge"] == {"correctness": 5, "faithfulness": 4, "reason": "верно", "judge_model": "j"}
    assert compare.judge_records(records, Judge(), info={}, rejudge=True) == 2 and Judge.calls == 3
