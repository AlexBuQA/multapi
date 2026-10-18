"""
Бенчмарк агентов (блок 6.3): пять задач блока 6.2 × три реализации × N повторов.

    python scripts/bench_agents.py                    # 5 × 3 × 3 → docs/agent-graph-bench/ и таблица в отчёте
    python scripts/bench_agents.py --repeats 1        # быстрая проверка
    python scripts/bench_agents.py --resume           # продолжить прерванный прогон по журналу runs.jsonl
    python scripts/bench_agents.py --report-only      # пересобрать таблицы из журнала
    python scripts/bench_agents.py --stop-check       # stop-кран на живой модели: поиск намеренно сломан

Реализации:
- b62_loop — ReAct блока 6.2 (app/services/agent_react.py): цикл for, критик, проверка финального ответа;
- custom — custom_graph из app/services/agent_graph.py: StateGraph руками;
- prebuilt — prebuilt_graph оттуда же: langchain.agents.create_agent.

Модель — AGENT__MODEL, температура — AGENT__TEMPERATURE (по умолчанию 0) у всех трёх. На прогон:
wall-clock latency (time.perf_counter), prompt_tokens и completion_tokens всех вызовов модели (у графов —
usage_metadata ответов, у b62_loop — usage, вместе с критиком), total_steps — вызовы модели агента
(у b62_loop — итерации цикла), вызовы инструментов и проверка ответа тем же правилом, что в
scripts/compare_agents.py. Графы вызываются с config={"configurable": {"thread_id": f"bench-{task_id}"}}:
без checkpointer это no-op, но вызов уже совместим с AsyncSqliteSaver.

Журнал — docs/agent-graph-bench/runs.jsonl (строка на прогон, дописывается сразу), сводка — summary.json,
таблицы — в docs/agent-graph-report.md между метками <!-- bench:start --> и <!-- bench:end -->.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_agents import CHECKS, TASKS, Task
from run_naive_agent import scrub, sent_messages

from app.services import agent_graph
from app.services.agent_react import run_react

OUT_DIR = ROOT / "docs" / "agent-graph-bench"
REPORT = ROOT / "docs" / "agent-graph-report.md"
IMPLS = ("b62_loop", "custom", "prebuilt")
MARK_START, MARK_END = "<!-- bench:start -->", "<!-- bench:end -->"
MSK = ZoneInfo("Europe/Moscow")


# ---------------------------------------------------------------- один прогон
def _graph_run(impl: str, task: Task, graphs: dict[str, Any]) -> dict[str, Any]:
    prebuilt = impl == "prebuilt"
    result = asyncio.run(agent_graph.run_graph(
        graphs["prebuilt_graph" if prebuilt else "custom_graph"], task.text, thread_id=f"bench-{task.id}",
        recursion_limit=agent_graph.PREBUILT_RECURSION_LIMIT if prebuilt else None))
    trace = [{"tool_name": c["name"], "tool_args": json.dumps(c["args"], ensure_ascii=False),
              "observation": c["result"][:agent_graph.PREVIEW]} for c in result["tool_calls"]]
    return {"answer": result["answer"], "steps": result["steps"], "llm_calls": result["steps"],
            "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
            "force_finish": result["force_finish"], "trace": trace,
            "error": "max_iterations" if result["force_finish"] else None}


def _b62_run(task: Task, model: str, temperature: float) -> dict[str, Any]:
    result = run_react(task.text, model=model, temperature=temperature)
    usage = result["usage"]
    trace = [{"tool_name": e["tool_name"], "tool_args": e["tool_args"], "observation": e["observation"]}
             for e in result["trace"] if e.get("tool_name")]
    return {"answer": result["answer"], "steps": result["steps"], "llm_calls": usage["calls"],
            "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
            "critic_tokens": usage["critic"]["total_tokens"], "revisions": result["revisions"],
            "force_finish": result.get("error") in ("max_iterations", "timeout"), "trace": trace,
            "error": result.get("error")}


def run_once(task: Task, impl: str, repeat: int, *, model: str, temperature: float,
             graphs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Один прогон с замером; ошибка провайдера или GraphRecursionError — в запись, серия идёт дальше."""
    captured = io.StringIO()
    started_at = datetime.now(MSK).isoformat(timespec="seconds")
    started = time.perf_counter()
    with contextlib.redirect_stdout(captured):
        try:
            run = _b62_run(task, model, temperature) if impl == "b62_loop" else _graph_run(impl, task, graphs or {})
        except Exception as exc:  # noqa: BLE001 — замер не должен обрывать серию
            run = {"answer": None, "steps": 0, "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                   "force_finish": False, "trace": [], "error": f"{type(exc).__name__}: {exc}"}
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    record = {"task_id": task.id, "impl": impl, "repeat": repeat, "model": model, "temperature": temperature,
              "started_at": started_at, "latency_ms": latency_ms, **run,
              "total_tokens": run["prompt_tokens"] + run["completion_tokens"],
              "telegram": sent_messages(captured.getvalue())}
    record["tools"] = [e["tool_name"] for e in record["trace"]]
    if record["error"] and not record["answer"]:
        correct, why = False, record["error"]
    elif record["force_finish"]:
        correct, why = False, f"остановка: {record['answer'] or record['error']}"
    else:
        correct, why = CHECKS[task.id](record)
    record["check"] = {"correct": correct, "why": why}
    return json.loads(scrub(json.dumps(record, ensure_ascii=False)))


# ---------------------------------------------------------------- серия и журнал
def load_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_series(tasks: list[Task], impls: tuple[str, ...], repeats: int, *, model: str, temperature: float,
               out_dir: Path = OUT_DIR, resume: bool = False, graphs: dict[str, Any] | None = None) -> list[dict]:
    """Повтор → задача → реализация: реализации чередуются, и прогрев или загрузка машины не достаются
    одной из них. Каждый прогон дописывается в журнал сразу."""
    out_dir.mkdir(parents=True, exist_ok=True)
    journal = out_dir / "runs.jsonl"
    if not resume and journal.exists():
        journal.unlink()
    done = {(r["task_id"], r["impl"], r["repeat"]) for r in load_journal(journal)}
    total = len(tasks) * len(impls) * repeats
    number = 0
    for repeat in range(1, repeats + 1):
        for task in tasks:
            for impl in impls:
                number += 1
                if (task.id, impl, repeat) in done:
                    continue
                record = run_once(task, impl, repeat, model=model, temperature=temperature, graphs=graphs)
                with journal.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                mark = "да" if record["check"]["correct"] else "нет"
                print(f"[{number}/{total}] {task.id} · {impl} · повтор {repeat}: {record['latency_ms'] / 1000:.1f} с, "
                      f"токенов {record['total_tokens']}, шагов {record['steps']}, инструменты {record['tools'] or '—'}, "
                      f"корректно: {mark}" + (f" ({record['check']['why']})" if not record["check"]["correct"] else ""),
                      flush=True)
    return load_journal(journal)


# ---------------------------------------------------------------- сводка и таблицы
def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Средние по повторам для каждой пары задача × реализация, в порядке TASKS и IMPLS."""
    rows = []
    for task in TASKS:
        for impl in IMPLS:
            runs = [r for r in records if r["task_id"] == task.id and r["impl"] == impl]
            if not runs:
                continue
            latency = [r["latency_ms"] for r in runs]
            rows.append({
                "task_id": task.id, "impl": impl, "runs": len(runs),
                "latency_ms": round(statistics.mean(latency)), "latency_min": round(min(latency)),
                "latency_max": round(max(latency)),
                **{k: round(statistics.mean(r[k] for r in runs), 1)
                   for k in ("prompt_tokens", "completion_tokens", "total_tokens", "steps", "llm_calls")},
                "correct": sum(r["check"]["correct"] for r in runs),
                "force_finish": sum(bool(r["force_finish"]) for r in runs),
                "errors": sum(bool(r["error"]) and not r["force_finish"] for r in runs),
            })
    return rows


def _n(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ") if value >= 100 else f"{value:g}"


def tables(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    intro = (f"Модель `{meta['model']}`, температура {meta['temperature']:g}, повторов на задачу — {meta['repeats']}. "
             "Значения — средние по повторам; latency — среднее (мин–макс).")
    lines = [intro, "",
             "| Задача | Реализация | latency_ms | prompt_tokens | completion_tokens | total_steps | вызовов модели | корректно |",
             "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| `{row['task_id']}` | {row['impl']} | {_n(row['latency_ms'])} ({_n(row['latency_min'])}–"
                     f"{_n(row['latency_max'])}) | {_n(row['prompt_tokens'])} | {_n(row['completion_tokens'])} | "
                     f"{row['steps']:g} | {row['llm_calls']:g} | {row['correct']} из {row['runs']} |")
    lines += ["", "**Итого по реализациям** (сумма средних по задачам):", "",
              "| Реализация | latency, с | prompt_tokens | completion_tokens | total_tokens | вызовов модели | корректно |",
              "|---|---|---|---|---|---|---|"]
    for impl in IMPLS:
        own = [r for r in rows if r["impl"] == impl]
        if not own:
            continue
        seconds = sum(r["latency_ms"] for r in own) / 1000
        shown = f"{seconds:.0f}" if seconds >= 10 else f"{seconds:.1f}"
        lines.append(f"| {impl} | {shown} | {_n(sum(r['prompt_tokens'] for r in own))} | "
                     f"{_n(sum(r['completion_tokens'] for r in own))} | {_n(sum(r['total_tokens'] for r in own))} | "
                     f"{sum(r['llm_calls'] for r in own):g} | {sum(r['correct'] for r in own)} из "
                     f"{sum(r['runs'] for r in own)} |")
    return "\n".join(lines)


def write_report(text: str, report: Path = REPORT) -> bool:
    """Таблицы — в отчёт между метками; меток нет — отчёт не трогаем."""
    if not report.exists():
        return False
    source = report.read_text(encoding="utf-8")
    if MARK_START not in source or MARK_END not in source:
        return False
    head, rest = source.split(MARK_START, 1)
    _, tail = rest.split(MARK_END, 1)
    report.write_text(f"{head}{MARK_START}\n{text}\n{MARK_END}{tail}", encoding="utf-8")
    return True


def finish(records: list[dict[str, Any]], meta: dict[str, Any], out_dir: Path = OUT_DIR,
           report: Path = REPORT) -> str:
    rows = aggregate(records)
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    text = tables(rows, meta)
    print("\n" + text)
    print(f"\nТаблицы — в {report.name}" if write_report(text, report) else f"\n{report.name}: нет меток — таблицы не вписаны")
    return text


# ---------------------------------------------------------------- stop-кран на живой модели
STOP_TASK = ("Найди в базе знаний точный срок возврата оплаты. Если поиск не сработал, попробуй ещё раз другими "
             "словами — без найденного срока не отвечай.")


def broken_search(query: str) -> str:
    """Поиск, который всегда ломается: проверка, что граф останавливается сам."""
    return "Ошибка инструмента search_knowledge_base: TimeoutError: база знаний не ответила, повторите запрос"


def stop_check(model: Any, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    from langchain_core.tools import StructuredTool

    from app.tools.graph_tools import TOOLS, SearchArgs

    original = TOOLS[0]
    broken = StructuredTool.from_function(func=broken_search, name=original.name, description=original.description,
                                          args_schema=SearchArgs)
    tools = [broken, *TOOLS[1:]]
    result = {}
    for impl, graph, limit in (("custom", agent_graph.build_custom_graph(model, tools), None),
                               ("prebuilt", agent_graph.build_prebuilt_graph(model, tools),
                                agent_graph.PREBUILT_RECURSION_LIMIT)):
        started = time.perf_counter()
        run = asyncio.run(agent_graph.run_graph(graph, STOP_TASK, thread_id=f"stop-{impl}", recursion_limit=limit))
        result[impl] = {"latency_ms": round((time.perf_counter() - started) * 1000, 1), "steps": run["steps"],
                        "searches": sum(c["name"] == "search_knowledge_base" for c in run["tool_calls"]),
                        "force_finish": run["force_finish"], "answer": run["answer"],
                        "total_tokens": run["total_tokens"]}
        print(f"{impl}: вызовов модели {run['steps']}, поисков {result[impl]['searches']}, force_finish "
              f"{run['force_finish']}: {str(run['answer'])[:200]}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stop_check.json").write_text(json.dumps({"task": STOP_TASK, **result}, ensure_ascii=False, indent=2)
                                             + "\n", encoding="utf-8")
    return result


def warmup(graphs: dict[str, Any]) -> None:
    """Модель и эмбеддинги — в память до замеров: холодный старт Ollama — десятки секунд на первый вызов."""
    from app.tools.naive_tools import search_knowledge_base

    with contextlib.redirect_stdout(io.StringIO()):
        search_knowledge_base("лимит запросов к API")
        graphs["model"].invoke("Ответь одним словом: готово")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="5 задач × 3 реализации агента × N повторов → docs/agent-graph-bench/")
    parser.add_argument("--repeats", type=int, default=3, help="повторов на задачу (по заданию — минимум 3)")
    parser.add_argument("--only", help=f"задачи через запятую: {', '.join(t.id for t in TASKS)}")
    parser.add_argument("--impl", help=f"реализации через запятую: {', '.join(IMPLS)}")
    parser.add_argument("--resume", action="store_true", help="продолжить по журналу, не повторяя готовые прогоны")
    parser.add_argument("--report-only", action="store_true", help="только пересобрать таблицы из журнала")
    parser.add_argument("--stop-check", action="store_true", help="stop-кран: сломанный поиск на обоих графах")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="папка журнала и сводки")
    parser.add_argument("--report", type=Path, default=REPORT, help="отчёт с метками для таблиц")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    wanted = set((args.only or "").replace(" ", "").split(",")) - {""}
    impls = tuple(i for i in IMPLS if not args.impl or i in args.impl.replace(" ", "").split(","))
    if wanted - {t.id for t in TASKS} or not impls:
        parser.error(f"задачи: {', '.join(t.id for t in TASKS)}; реализации: {', '.join(IMPLS)}")

    from app.core.config import get_settings
    from app.observability.logging import setup_logging

    setup_logging("WARNING", stream=sys.stderr)
    cfg = get_settings()
    model_name = cfg.agent.model or agent_graph.DEFAULT_MODEL
    meta = {"model": model_name, "temperature": cfg.agent.temperature, "repeats": args.repeats,
            "graph_max_iterations": agent_graph.MAX_ITERATIONS, "b62_max_iterations": cfg.agent.max_iterations,
            "b62_timeout_per_iteration_sec": cfg.agent.timeout_per_iteration_sec,
            "date": datetime.now(MSK).isoformat(timespec="minutes")}
    if args.report_only:
        finish(load_journal(args.out / "runs.jsonl"), meta, args.out, args.report)
        return 0
    from app.tools.naive_tools import knowledge_base

    try:                                     # база знаний — до первого прогона: без Qdrant все замеры пустые
        knowledge_base()
    except Exception as exc:  # noqa: BLE001 — Qdrant, коллекция, эмбеддинги: причина и выход
        print(f"Остановка: база знаний недоступна — {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    graphs = agent_graph.graphs()
    if args.stop_check:
        stop_check(graphs["model"], args.out)
        return 0
    print(f"Модель {model_name}, температура {cfg.agent.temperature:g}; реализации: {', '.join(impls)}; повторов: "
          f"{args.repeats}. Прогрев…", flush=True)
    warmup(graphs)
    records = run_series([t for t in TASKS if not wanted or t.id in wanted], impls, args.repeats,
                         model=model_name, temperature=cfg.agent.temperature, out_dir=args.out,
                         resume=args.resume, graphs=graphs)
    finish(records, meta, args.out, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
