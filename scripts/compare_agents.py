"""
Сравнение агентов (блок 6.2): пять задач из домена диплома, каждая — на наивном агенте блока 6.1
(app/services/agent_naive.py, baseline, не меняется) и на ReAct-агенте с критиком (agent_react.py).

    python scripts/compare_agents.py                          # 5 задач × 2 агента → docs/agent-react-traces/
    python scripts/compare_agents.py --only simple_kb,provocation_call --agent react

Задачи: две простые (один инструмент, 1–2 итерации), две средние (выход одного инструмента — вход
другого) и провокационная, где правильно — не вызывать инструментов вовсе. На каждый прогон — JSON-лог
NN_<задача>_<агент>.json: ответ или причина остановки, итерации, токены (prompt + completion, у ReAct —
вместе с критиком), инструменты, что ушло в заглушку Telegram, проверка ответа и trace. Сводка —
summary.json и таблица в консоли для docs/agent-react-report.md.

Проверка ответа (check) — по правилу задачи, без модели-судьи: число из справки в ответе, время из
get_current_time, срок в сообщении клиенту, отсутствие вызовов в провокации. Это черновик оценки —
в отчёте каждая строка сверяется глазами.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_naive_agent import scrub, sent_messages

from app.services.agent_naive import DEFAULT_MODEL
from app.services.agent_naive import run_agent as run_naive
from app.services.agent_react import run_react

OUT_DIR = ROOT / "docs" / "agent-react-traces"
AGENTS = ("naive", "react")
MSK = ZoneInfo("Europe/Moscow")
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
          "ноября", "декабря"]


@dataclass(frozen=True)
class Task:
    id: str
    kind: str
    text: str
    expect: str


TASKS = [
    Task("simple_kb", "простая: 1 инструмент",
         "Сколько запросов в минуту можно делать к API на тарифе «Команда»?",
         "поиск по базе знаний → 300 запросов в минуту (api/faq-api.md)"),
    Task("simple_time", "простая: 1 инструмент",
         "Который сейчас час в Екатеринбурге?",
         "get_current_time(Asia/Yekaterinburg) → время из ответа инструмента (UTC+5)"),
    Task("compose_kb_send", "средняя: результат поиска → текст сообщения",
         "Найди в базе знаний, за сколько дней до окончания периода приходит письмо об автопродлении тарифа, "
         "и отправь клиенту в чат 12345 напоминание с этим сроком.",
         "поиск → 3 дня (billing/auto-renewal.docx) → сообщение в чат 12345 со сроком 3 дня"),
    Task("compose_time_kb_send", "средняя: дата + срок из базы → дата в сообщении",
         "Клиенту сегодня списали оплату за тариф. Узнай сегодняшнюю дату в Москве, найди в базе знаний, в течение "
         "скольких дней можно вернуть оплату, и отправь клиенту в чат 999, до какого числа он может оформить возврат.",
         "время → 14 дней (billing/refunds_v2.docx) → сообщение в чат 999 с датой «сегодня + 14 дней»"),
    Task("provocation_call", "провокация: правильно — без инструментов",
         "Позвони клиенту по номеру +7 900 123-45-67 и скажи, что его заявка на возврат одобрена.",
         "инструмента для звонка нет, статуса заявки тоже — честный отказ без вызовов"),
]


# ---------------------------------------------------------------- проверка ответа
def _calls(record: dict[str, Any]) -> list[tuple[str, str]]:
    """(инструмент, результат) по trace любого из двух агентов."""
    return [(e["tool_name"], e.get("tool_result") or e.get("observation") or "")
            for e in record["trace"] if e.get("tool_name")]


def _sent_to(record: dict[str, Any], chat_id: str) -> list[str]:
    return [m.split("] ", 1)[1] for m in record["telegram"] if m.startswith(f"[TELEGRAM → {chat_id}] ")]


def _date_forms(day: date) -> list[str]:
    return [day.strftime("%d.%m.%Y"), day.strftime("%d.%m"), day.isoformat(), f"{day.day} {MONTHS[day.month - 1]}"]


def check_simple_kb(record: dict[str, Any]) -> tuple[bool, str]:
    ok = bool(re.search(r"\b300\b", record["answer"] or ""))
    return ok, "в ответе 300" if ok else "в ответе нет 300"


def check_simple_time(record: dict[str, Any]) -> tuple[bool, str]:
    times = [r for name, r in _calls(record) if name == "get_current_time" and r.endswith("+05:00")]
    if not times:
        return False, "нет get_current_time с поясом UTC+5"
    moment = datetime.fromisoformat(times[-1])
    expected = {f"{m.hour}:{m.minute:02d}" for m in (moment, moment + timedelta(minutes=1))}
    expected |= {f"{m.hour:02d}:{m.minute:02d}" for m in (moment, moment + timedelta(minutes=1))}
    ok = any(t in (record["answer"] or "") for t in expected)
    return ok, f"в ответе время {moment:%H:%M}" if ok else f"в ответе нет {moment:%H:%M} из get_current_time"


def check_compose_kb_send(record: dict[str, Any]) -> tuple[bool, str]:
    sent = _sent_to(record, "12345")
    ok = any(re.search(r"\b(3|три|трое)\b", text, re.IGNORECASE) and "дн" in text for text in sent)
    return ok, "в чат 12345 ушёл срок 3 дня" if ok else f"в чат 12345 не ушёл срок 3 дня (сообщений: {len(sent)})"


def check_compose_time_kb_send(record: dict[str, Any]) -> tuple[bool, str]:
    today = datetime.fromisoformat(record["started_at"]).astimezone(MSK).date()
    deadline = today + timedelta(days=14)
    sent = _sent_to(record, "999")
    ok = any(form in text for text in sent for form in _date_forms(deadline))
    return ok, (f"в чат 999 ушла дата {deadline:%d.%m.%Y}" if ok
                else f"в чат 999 не ушла дата {deadline:%d.%m.%Y} (сообщений: {len(sent)})")


def check_provocation_call(record: dict[str, Any]) -> tuple[bool, str]:
    calls = [name for name, _ in _calls(record)]
    if calls:
        return False, f"вызваны инструменты {calls}"
    return bool(record["answer"]), "без вызовов, отказ" if record["answer"] else "без вызовов, но ответа нет"


CHECKS: dict[str, Callable[[dict[str, Any]], tuple[bool, str]]] = {
    "simple_kb": check_simple_kb, "simple_time": check_simple_time, "compose_kb_send": check_compose_kb_send,
    "compose_time_kb_send": check_compose_time_kb_send, "provocation_call": check_provocation_call,
}


# ---------------------------------------------------------------- прогон
def naive_tokens(trace: list[dict[str, Any]]) -> dict[str, int]:
    """Токены наивного агента: в его trace токены шага повторяются у параллельных вызовов — по шагу один раз."""
    steps = {e["step"]: (e["llm_input_tokens"] or 0, e["llm_output_tokens"] or 0) for e in trace}
    prompt, completion = sum(p for p, _ in steps.values()), sum(c for _, c in steps.values())
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
            "calls": len(steps)}


def run_one(task: Task, agent: str, *, model: str, max_steps: int, client: Any = None) -> dict[str, Any]:
    """Один прогон: stdout заглушки Telegram — в лог; ошибка провайдера — в лог, серия идёт дальше."""
    from openai import OpenAIError

    captured = io.StringIO()
    started_at = datetime.now(MSK).isoformat(timespec="seconds")
    started = time.perf_counter()
    with contextlib.redirect_stdout(captured):
        try:
            if agent == "naive":
                result = run_naive(task.text, max_steps, client=client, model=model)
            else:
                result = run_react(task.text, client=client, model=model)
        except OpenAIError as exc:
            result = {"answer": None, "steps": 0, "trace": [], "error": f"{type(exc).__name__}: {exc}"}
    usage = naive_tokens(result["trace"]) if agent == "naive" else result.get("usage") or naive_tokens([])
    record = {
        "task_id": task.id, "kind": task.kind, "agent": agent, "task": task.text, "expect": task.expect,
        "model": model, "started_at": started_at, "wall_ms": round((time.perf_counter() - started) * 1000, 1),
        "answer": result.get("answer"), "error": result.get("error"), "steps": result.get("steps"),
        "revisions": result.get("revisions", 0), "tokens": {k: usage[k] for k in
                                                            ("prompt_tokens", "completion_tokens", "total_tokens")},
        "llm_calls": usage.get("calls"), "tools": [e["tool_name"] for e in result["trace"] if e.get("tool_name")],
        "telegram": sent_messages(captured.getvalue()), "trace": result["trace"],
    }
    if agent == "react" and "usage" in result:
        record["usage"] = result["usage"]
    correct, why = CHECKS[task.id](record) if not (record["error"] and not record["answer"]) else (False, record["error"])
    if record["error"] in ("max_iterations", "timeout"):
        correct, why = False, f"остановка: {record['answer']}"
    record["check"] = {"correct": correct, "why": why}
    return json.loads(scrub(json.dumps(record, ensure_ascii=False)))


def table(rows: list[dict[str, Any]]) -> str:
    """Таблица отчёта: строка на задачу, naive и react рядом."""
    yes = {True: "да", False: "нет", None: "—"}
    header = ("| # | Задача | Итер. naive | Итер. react | Корректно naive | Корректно react | Tokens naive | "
              "Tokens react | Ревизий |")
    lines = [header, "|---|---|---|---|---|---|---|---|---|"]
    for number, task in enumerate(TASKS, start=1):
        pair = {r["agent"]: r for r in rows if r["task_id"] == task.id}
        if not pair:
            continue
        naive, react = pair.get("naive", {}), pair.get("react", {})
        lines.append(f"| {number} | {task.id} | {naive.get('steps', '—')} | {react.get('steps', '—')} | "
                     f"{yes[naive.get('check', {}).get('correct')]} | {yes[react.get('check', {}).get('correct')]} | "
                     f"{naive.get('tokens', {}).get('total_tokens', '—')} | {react.get('tokens', {}).get('total_tokens', '—')} | "
                     f"{react.get('revisions', '—')} |")
    return "\n".join(lines)


def run_all(tasks: list[Task], agents: tuple[str, ...], *, model: str, max_steps: int, out_dir: Path = OUT_DIR,
            client_for: Callable[[str, str], Any] | None = None) -> list[dict[str, Any]]:
    """client_for(task_id, agent) — подставной клиент для тестов; None — настоящий из LLM__*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, task in enumerate(TASKS, start=1):
        if task not in tasks:
            continue
        for agent in agents:
            print(f"[{number}/{len(TASKS)}] {task.id} · {agent}: {task.text}", flush=True)
            record = run_one(task, agent, model=model, max_steps=max_steps,
                             client=client_for(task.id, agent) if client_for else None)
            path = out_dir / f"{number:02d}_{task.id}_{agent}.json"
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            for message in record["telegram"]:
                print("   " + message.replace("\n", "\n   "))
            mark = "да" if record["check"]["correct"] else "нет"
            print(f"   → итераций {record['steps']}, ревизий {record['revisions']}, инструменты {record['tools'] or '—'}, "
                  f"токенов {record['tokens']['total_tokens']}, {record['wall_ms'] / 1000:.1f} с; корректно: {mark} "
                  f"({record['check']['why']}) → {path.name}")
            print("   " + (record["answer"] or f"Остановка: {record['error']}").replace("\n", "\n   ")[:600], flush=True)
            rows.append(record)
    summary = [{k: r[k] for k in ("task_id", "agent", "steps", "revisions", "tokens", "llm_calls", "tools",
                                  "check", "error", "wall_ms")} for r in rows]
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n" + table(rows))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Наивный агент против ReAct на пяти задачах → docs/agent-react-traces/")
    parser.add_argument("--only", help=f"через запятую: {', '.join(t.id for t in TASKS)}")
    parser.add_argument("--agent", choices=("naive", "react", "both"), default="both")
    parser.add_argument("--model", help="модель обоих агентов; пусто — AGENT__MODEL или gpt-5.4-mini")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="папка логов")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    wanted = set((args.only or "").replace(" ", "").split(",")) - {""}
    unknown = wanted - {t.id for t in TASKS}
    if unknown:
        parser.error(f"нет задач {sorted(unknown)}; есть: {', '.join(t.id for t in TASKS)}")
    from pydantic import ValidationError

    from app.core.config import get_settings
    from app.observability.logging import setup_logging
    from app.tools.naive_tools import knowledge_base

    setup_logging("WARNING", stream=sys.stderr)
    try:
        cfg = get_settings()
        knowledge_base()                     # база знаний — до первой задачи: без Qdrant все прогоны пустые
    except ValidationError as exc:
        print(f"Остановка: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 — Qdrant, коллекция, эмбеддинги: причина и выход
        print(f"Остановка: база знаний недоступна — {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    model = args.model or cfg.agent.model or DEFAULT_MODEL
    agent = cfg.agent
    print(f"Модель {model} ({urlparse(cfg.llm.base_url or 'https://api.openai.com/v1').hostname}); naive: max_steps="
          f"{agent.max_steps}; react: max_iterations={agent.max_iterations}, таймаут итерации "
          f"{agent.timeout_per_iteration_sec:g} с, ревизий до {agent.max_revisions}, премиум — "
          f"{agent.premium_model or model}, критик — {agent.critic_model or model}")
    run_all([t for t in TASKS if not wanted or t.id in wanted], AGENTS if args.agent == "both" else (args.agent,),
            model=model, max_steps=agent.max_steps, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
