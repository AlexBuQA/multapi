"""
Мультиагент блока 6.5: supervisor + researcher + writer на LangGraph 1.x.

    python experiments/multi_agent_langgraph.py                  # 5 вопросов, модель AGENT__MODEL из .env
    python experiments/multi_agent_langgraph.py --questions q4   # только многошаговый вопрос

Supervisor ручной: Command(goto=..., update=...), без langgraph_supervisor.create_supervisor (обоснование —
docs/multi-agent-report.md, раздел 3). Устройство:
- supervisor — вызов модели с двумя handoff-инструментами: transfer_to_researcher(task) и
  transfer_to_writer(task). Инструменты не исполняются: узел читает tool_call и возвращает
  Command(goto=<агент>, update=...). Если модель не вызвала инструмент, срабатывает запасной маршрут: researcher,
  если фактов ещё нет, иначе writer (счётчик fallback_routes);
- researcher — create_agent(name="researcher", tools=[search_knowledge_base]) с чистым контекстом: вопрос
  клиента и задание супервизора, без переписки супервизора. Супервизору возвращается только список фактов —
  ToolMessage на его вызов transfer_to_researcher; поиски и фрагменты остаются внутри researcher;
- writer — create_agent(name="writer", tools=[]): вопрос и факты → ответ с цитатами [1], [2]. Writer — последний
  шаг: его ответ идёт сразу в END, без лишнего вызова супервизора. В create_supervisor с output_mode=
  "last_message" управление вернулось бы супервизору, и он ещё раз вызвал бы модель;
- бюджет MAX_HANDOFFS передач управления. Последняя передача зарезервирована за writer: когда остаётся одна,
  supervisor отдаёт управление writer без вызова модели (счётчик forced_routes).

handoff_count — переходы между агентами. Обычно их 3: supervisor→researcher, researcher→supervisor,
supervisor→writer; каждый дополнительный раунд поиска добавляет ещё 2.

Чекпоинтер — InMemorySaver, thread_id="exp-langgraph" — один thread на все вопросы. Каждый вопрос начинается
с RemoveMessage(REMOVE_ALL_MESSAGES): история чек-пойнтов общая, а в контексте супервизора только текущий
вопрос. Стрим — app.stream(..., stream_mode="updates", subgraphs=True): шаги супервизора и шаги внутри агентов.
В конце — app.get_graph().draw_mermaid() → docs/architecture-multi-agent.md.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain.agents import create_agent  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages  # noqa: E402
from langgraph.types import Command  # noqa: E402

from experiments.common import (  # noqa: E402
    MAX_MODEL_CALLS,
    MAX_SEARCHES,
    RESEARCHER_PROMPT,
    WRITER_PROMPT,
    Meter,
    Question,
    last_text,
    limits,
    model_info,
    parser,
    pick_questions,
    print_result,
    print_update,
    record,
    reset_cache,
    save_records,
    search_knowledge_base,
    searches,
    warmup,
)

IMPLEMENTATION = "multi_agent"
THREAD_ID = "exp-langgraph"
SUPERVISOR, RESEARCHER, WRITER = "supervisor", "researcher", "writer"
MAX_HANDOFFS = 5          # переходов на вопрос: researcher — не больше двух раундов, writer — всегда
ARCHITECTURE = ROOT / "docs" / "architecture-multi-agent.md"

SUPERVISOR_PROMPT = (
    "Ты — supervisor команды техподдержки «Личного кабинета». Сам ты не ищешь и не отвечаешь клиенту — только "
    "передаёшь управление агентам инструментами: transfer_to_researcher — researcher ищет факты в базе знаний и "
    "возвращает их списком с источниками; transfer_to_writer — writer пишет клиенту итоговый ответ по собранным "
    "фактам, после него работа закончена. Порядок: сначала transfer_to_researcher с заданием, что найти (для "
    "вопроса из нескольких частей перечисли все части). Когда придут факты: если они покрывают все части вопроса "
    "или researcher сообщил, что в базе ничего нет, — transfer_to_writer; если какой-то части не хватает — ещё раз "
    "transfer_to_researcher с заданием только по недостающей части. За ход — ровно один вызов инструмента.")
NOT_EXECUTED = "Не выполнено: supervisor передаёт управление одному агенту за ход."
FORCED_TASK = "Бюджет передач управления исчерпан: напиши ответ по уже собранным фактам."


class TeamState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]   # контекст supervisor: вопрос, его вызовы, итоги агентов
    question: str
    task: str                 # задание supervisor текущему агенту
    call_id: str | None       # вызов transfer_to_*, на который агент ответит ToolMessage
    facts: str                # факты researcher за все раунды — вход writer
    answer: str
    route: list[str]          # путь управления: supervisor, researcher, supervisor, writer
    queries: list[str]        # запросы researcher к search_knowledge_base
    contexts: list[str]       # что вернул search_knowledge_base — для судьи
    fallback_routes: int
    forced_routes: int


@tool("transfer_to_researcher", description="Передать управление researcher: он найдёт факты в базе знаний и "
      "вернёт их списком с источниками. task — что именно найти, по-русски; для вопроса из нескольких частей — "
      "все части.")
def transfer_to_researcher(task: str) -> str:
    return task


@tool("transfer_to_writer", description="Передать управление writer: он напишет клиенту итоговый ответ по "
      "собранным фактам с цитатами источников; после него работа команды закончена. task — что учесть в ответе.")
def transfer_to_writer(task: str) -> str:
    return task


HANDOFFS = {transfer_to_researcher.name: RESEARCHER, transfer_to_writer.name: WRITER}


def build_agents(model: BaseChatModel, *, tools: list[BaseTool] | None = None) -> dict[str, Any]:
    """Два create_agent. Тот же объект search_knowledge_base, что у single_agent_baseline.py."""
    return {
        RESEARCHER: create_agent(model, tools=tools if tools is not None else [search_knowledge_base],
                                 system_prompt=RESEARCHER_PROMPT, name=RESEARCHER, middleware=limits()),
        WRITER: create_agent(model, tools=[], system_prompt=WRITER_PROMPT, name=WRITER),
    }


def reply_to(state: TeamState, agent: str, content: str) -> AnyMessage:
    """Итог агента — ответ на вызов transfer_to_<agent> супервизора; после запасного маршрута вызова нет."""
    if state.get("call_id"):
        return ToolMessage(content, tool_call_id=state["call_id"], name=f"transfer_to_{agent}")
    return AIMessage(content, name=agent)


def build_team(model: BaseChatModel, *, agents: dict[str, Any] | None = None, checkpointer: Any = None,
               supervisor_model: BaseChatModel | None = None) -> Any:
    agents = agents or build_agents(model)
    router = (supervisor_model or model).bind_tools([transfer_to_researcher, transfer_to_writer])

    def supervisor(state: TeamState) -> Command[Literal["researcher", "writer"]]:
        route = state.get("route") or [SUPERVISOR]
        if len(route) >= MAX_HANDOFFS:      # передач было len(route) - 1, осталась одна — она за writer
            return Command(goto=WRITER, update={"route": [*route, WRITER], "task": FORCED_TASK, "call_id": None,
                                                "forced_routes": state.get("forced_routes", 0) + 1})
        reply = router.invoke([SystemMessage(SUPERVISOR_PROMPT), *state["messages"]])
        chosen = next((c for c in reply.tool_calls if c["name"] in HANDOFFS), None)
        skipped = [ToolMessage(NOT_EXECUTED, tool_call_id=c["id"], name=c["name"])
                   for c in reply.tool_calls if c is not chosen]
        if chosen:
            goto, task = HANDOFFS[chosen["name"]], str(chosen["args"].get("task") or state["question"])
        else:
            goto, task = (WRITER if state.get("facts") else RESEARCHER), state["question"]
        return Command(goto=goto, update={
            "messages": [reply, *skipped], "route": [*route, goto], "task": task,
            "call_id": chosen["id"] if chosen else None,
            "fallback_routes": state.get("fallback_routes", 0) + (chosen is None)})

    def researcher(state: TeamState) -> Command[Literal["supervisor"]]:
        brief = f"Вопрос клиента: {state['question']}\nЗадание supervisor: {state.get('task') or state['question']}"
        result = agents[RESEARCHER].invoke({"messages": [HumanMessage(brief)]})
        facts = last_text(result["messages"]) or "- researcher не вернул фактов"
        queries, contexts = searches(result["messages"])
        return Command(goto=SUPERVISOR, update={
            "messages": [reply_to(state, RESEARCHER, f"Факты от researcher:\n{facts}")],
            "facts": "\n".join(part for part in (state.get("facts"), facts) if part),
            "queries": [*state.get("queries", []), *queries], "contexts": [*state.get("contexts", []), *contexts],
            "route": [*state["route"], SUPERVISOR], "call_id": None})

    def writer(state: TeamState) -> dict[str, Any]:
        brief = (f"Вопрос клиента: {state['question']}\n\nФакты от researcher:\n"
                 f"{state.get('facts') or '- researcher не нашёл фактов'}\n\n"
                 f"Задание supervisor: {state.get('task') or 'ответь клиенту'}")
        result = agents[WRITER].invoke({"messages": [HumanMessage(brief)]})
        answer = last_text(result["messages"])
        done = [reply_to(state, WRITER, "Ответ клиенту готов.")] if state.get("call_id") else []
        return {"answer": answer, "messages": [*done, AIMessage(answer, name=WRITER)], "call_id": None}

    builder = StateGraph(TeamState)
    builder.add_node(SUPERVISOR, supervisor, destinations=(RESEARCHER, WRITER))
    builder.add_node(RESEARCHER, researcher, destinations=(SUPERVISOR,))
    builder.add_node(WRITER, writer)
    builder.add_edge(START, SUPERVISOR)
    builder.add_edge(WRITER, END)
    return builder.compile(checkpointer=checkpointer, name="support_team")


def new_question(text: str) -> dict[str, Any]:
    """Вход вопроса в общий thread: прежняя переписка supervisor удаляется, счётчики — с нуля."""
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), HumanMessage(text)], "question": text, "task": "",
            "call_id": None, "facts": "", "answer": "", "route": [SUPERVISOR], "queries": [], "contexts": [],
            "fallback_routes": 0, "forced_routes": 0}


def ask(app: Any, question: Question, *, info: dict[str, Any], model: Any, cache_reset: bool = True,
        thread_id: str = THREAD_ID) -> dict[str, Any]:
    if cache_reset:
        reset_cache(model)
    meter = Meter()
    thread = {"configurable": {"thread_id": thread_id}}
    error = None
    started = time.perf_counter()
    try:
        for chunk in app.stream(new_question(question.text), {**thread, "callbacks": [meter], "recursion_limit": 100},
                                stream_mode="updates", subgraphs=True):
            print_update(chunk)
    except Exception as exc:  # noqa: BLE001 — сбой одного вопроса пишется в results.json, прогон идёт дальше
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - started) * 1000
    values = app.get_state(thread).values
    route = values.get("route") or [SUPERVISOR]
    return record(IMPLEMENTATION, question, info=info, answer=values.get("answer", ""), meter=meter,
                  latency_ms=latency_ms, handoff_count=len(route) - 1, queries=values.get("queries", []),
                  contexts=values.get("contexts", []), cache_reset=cache_reset, error=error, route=route,
                  fallback_routes=values.get("fallback_routes", 0), forced_routes=values.get("forced_routes", 0),
                  facts=values.get("facts", ""))


ARCHITECTURE_TEMPLATE = """# Архитектура мультиагента (блок 6.5)

Файл пишет `python experiments/multi_agent_langgraph.py` в конце прогона: `app.get_graph().draw_mermaid()`
supervisor-графа и `get_graph().draw_mermaid()` агента researcher. Руками не править — перезапишется.
Устройство, замеры и решение — [multi-agent-report.md](multi-agent-report.md).

## Supervisor-граф

Пунктир — переходы через `Command(goto=...)`. Supervisor выбирает агента handoff-инструментом
`transfer_to_researcher(task)` или `transfer_to_writer(task)`; researcher всегда возвращает управление
supervisor, writer — последний шаг (сплошная стрелка в `__end__`). Бюджет — {handoffs} передач управления
на вопрос; последняя зарезервирована за writer.

```mermaid
{team}
```

## Researcher изнутри

`create_agent(name="researcher", tools=[search_knowledge_base])` с бюджетами: `ToolCallLimitMiddleware` —
не больше {searches} поисков, `ModelCallLimitMiddleware` — не больше {calls} вызовов модели. Writer — тот же
`create_agent` без инструментов, один вызов модели. Single-agent baseline
(`experiments/single_agent_baseline.py`) устроен как researcher, с теми же бюджетами, но с промптом обеих
ролей.

```mermaid
{researcher}
```
"""


def write_architecture(app: Any, researcher: Any, path: Path = ARCHITECTURE) -> str:
    text = ARCHITECTURE_TEMPLATE.format(team=app.get_graph().draw_mermaid().strip(),
                                        researcher=researcher.get_graph().draw_mermaid().strip(),
                                        handoffs=MAX_HANDOFFS, searches=MAX_SEARCHES, calls=MAX_MODEL_CALLS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return text


def main(argv: list[str] | None = None) -> int:
    args = parser("Мультиагент 6.5: supervisor + researcher + writer на 5 вопросах → experiments/results.json"
                  ).parse_args(argv)
    from experiments.common import console

    console()
    from app.core.config import get_settings
    from app.services.agent_graph import build_model

    model = build_model(get_settings(), model=args.model)
    agents = build_agents(model)
    app = build_team(model, agents=agents, checkpointer=InMemorySaver())
    info = model_info(model)
    print(f"Мультиагент: модель {info['model']} @ {info['provider_host']}, thread_id={THREAD_ID}, "
          f"бюджет {MAX_HANDOFFS} передач управления; прогрев модели и поиска…")
    warmup(model)
    records = []
    for question in pick_questions(args.questions):
        print(f"\n=== {question.id} ({question.kind}): {question.text}")
        records.append(ask(app, question, info=info, model=model, cache_reset=not args.no_cache_reset))
        print_result(records[-1])
        print(f"route: {' → '.join(records[-1]['route'])}")
    save_records(records, args.out)
    history = list(app.get_state_history({"configurable": {"thread_id": THREAD_ID}}))
    write_architecture(app, agents[RESEARCHER])
    print(f"\nthread {THREAD_ID}: {len(history)} чек-пойнтов за {len(records)} вопросов")
    print(f"Записано: {args.out.name} ({len(records)} записей {IMPLEMENTATION}), "
          f"{ARCHITECTURE.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
