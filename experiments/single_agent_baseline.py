"""
Single-agent baseline блока 6.5: один create_agent с тем же инструментом search_knowledge_base.

    python experiments/single_agent_baseline.py                  # 5 вопросов, модель AGENT__MODEL из .env
    python experiments/single_agent_baseline.py --questions q4   # только многошаговый вопрос

Точка отсчёта для мультиагента (experiments/multi_agent_langgraph.py). Общее у них — всё из
experiments/common.py: вопросы, ОДИН объект инструмента search_knowledge_base, бюджеты create_agent (limits()),
правила поиска (SEARCH_RULES, как у researcher) и формата ответа (ANSWER_RULES, как у writer). Промпт
SINGLE_PROMPT объединяет обе роли: найти факты и оформить ответ с цитатами [1], [2]. handoff_count = 0.

Каждый вопрос — свой thread exp-single-<id> в InMemorySaver: контекст чистый, как у мультиагента, а итоговое
состояние после стрима читается через get_state. Стрим — agent.stream(..., stream_mode="updates").
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain.agents import create_agent  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.tools import BaseTool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from experiments.common import (  # noqa: E402
    SINGLE_PROMPT,
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

IMPLEMENTATION = "single_agent"
AGENT_NAME = "support_agent"


def build_baseline(model: BaseChatModel, *, tools: list[BaseTool] | None = None, checkpointer: Any = None) -> Any:
    return create_agent(model, tools=tools if tools is not None else [search_knowledge_base],
                        system_prompt=SINGLE_PROMPT, name=AGENT_NAME, middleware=limits(),
                        checkpointer=checkpointer)


def ask(agent: Any, question: Question, *, info: dict[str, Any], model: Any, cache_reset: bool = True) -> dict[str, Any]:
    if cache_reset:
        reset_cache(model)
    meter = Meter()
    thread = {"configurable": {"thread_id": f"exp-single-{question.id}"}}
    error = None
    started = time.perf_counter()
    try:
        for chunk in agent.stream({"messages": [HumanMessage(question.text)]},
                                  {**thread, "callbacks": [meter], "recursion_limit": 100}, stream_mode="updates"):
            print_update(chunk)
    except Exception as exc:  # noqa: BLE001 — сбой одного вопроса пишется в results.json, прогон идёт дальше
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - started) * 1000
    messages = agent.get_state(thread).values.get("messages", [])
    queries, contexts = searches(messages)
    return record(IMPLEMENTATION, question, info=info, answer=last_text(messages), meter=meter,
                  latency_ms=latency_ms, handoff_count=0, queries=queries, contexts=contexts,
                  cache_reset=cache_reset, error=error)


def main(argv: list[str] | None = None) -> int:
    args = parser("Single-agent 6.5: один create_agent с search_knowledge_base на 5 вопросах → "
                  "experiments/results.json").parse_args(argv)
    from experiments.common import console

    console()
    from app.core.config import get_settings
    from app.services.agent_graph import build_model

    model = build_model(get_settings(), model=args.model)
    agent = build_baseline(model, checkpointer=InMemorySaver())
    info = model_info(model)
    print(f"Single-agent: модель {info['model']} @ {info['provider_host']}; прогрев модели и поиска…")
    warmup(model)
    records = []
    for question in pick_questions(args.questions):
        print(f"\n=== {question.id} ({question.kind}): {question.text}")
        records.append(ask(agent, question, info=info, model=model, cache_reset=not args.no_cache_reset))
        print_result(records[-1])
    save_records(records, args.out)
    print(f"\nЗаписано: {args.out.name} ({len(records)} записей {IMPLEMENTATION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
