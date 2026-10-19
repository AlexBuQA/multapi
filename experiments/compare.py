"""
LLM-судья и сравнительные таблицы по experiments/results.json (блок 6.5).

    python experiments/compare.py              # оценить ещё не оценённые ответы и напечатать таблицы
    python experiments/compare.py --rejudge    # оценить все ответы заново
    python experiments/compare.py --no-judge   # только таблицы: качество — по детерминированным проверкам

Судья — модель RAGAS__JUDGE_* из .env, тот же, что у оценки RAG блока 5.6 (пусто — qwen3:4b-instruct в той же
Ollama), температура 0. Один вызов на ответ, две оценки 1–5:
- correctness — совпадает ли ответ по сути с эталоном вопроса (для вопроса вне базы: честный отказ — 5,
  выдуманная инструкция — 1);
- faithfulness — всё ли в ответе опирается на фрагменты, которые вернул search_knowledge_base (тот же смысл,
  что у RAGAS faithfulness, но одной оценкой судьи, а не разбором на утверждения).
Оценки пишутся в записи results.json (поле judge), сводка — в experiments/summary.json, таблицы для
docs/multi-agent-report.md — в stdout (markdown).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import IMPLEMENTATIONS, QUESTIONS, RESULTS, load_records, write_json  # noqa: E402

SUMMARY = ROOT / "experiments" / "summary.json"
TITLES = {"single_agent": "Single-agent", "multi_agent": "Multi-agent"}
JUDGE_PROMPT = (
    "Ты — строгий проверяющий ответов техподдержки. Даны вопрос клиента, эталонный ответ, фрагменты базы знаний, "
    "которые нашёл агент, и ответ агента клиенту. Поставь две оценки от 1 до 5.\n"
    "correctness — совпадает ли ответ по сути с эталоном: 5 — все факты эталона и ничего неверного; 4 — главное "
    "верно, мелкий пропуск; 3 — верна только часть; 2 — в основном неверно или не по вопросу; 1 — неверно. Если "
    "эталон говорит, что в базе нет информации: 5 — ответ честно сообщает, что информации нет, и ничего не "
    "выдумывает; 1 — ответ выдумывает инструкцию.\n"
    "faithfulness — опирается ли ответ только на фрагменты: 5 — каждое утверждение есть во фрагментах; 3 — часть "
    "утверждений без опоры; 1 — ответ в основном выдуман. Честный отказ, когда во фрагментах ответа нет, — 5.\n"
    'Верни только JSON без пояснений вокруг: {"correctness": <1-5>, "faithfulness": <1-5>, "reason": "<одно '
    'предложение по-русски>"}')
SCORES = ("correctness", "faithfulness")


# ---------------------------------------------------------------- судья
def build_judge(cfg: Any) -> Any:
    """ChatOpenAI к судье RAGAS__JUDGE_* (пусто — LLM__*): прокси и сертификаты ОС — как у build_model."""
    from langchain_openai import ChatOpenAI
    from openai import DefaultHttpxClient

    from app.core.config import http_client_options, is_local_url, provider_headers, proxy_for

    base_url = cfg.ragas.judge_base_url or cfg.llm.base_url
    key = (cfg.ragas.judge_api_key or cfg.llm.openai_api_key).get_secret_value()
    options = http_client_options(proxy_for(base_url, cfg.llm.proxy_url),
                                  cfg.llm.use_system_certs and not is_local_url(base_url))
    return ChatOpenAI(model=cfg.ragas.judge_model, temperature=0, api_key=key, base_url=base_url,
                      timeout=cfg.ragas.judge_timeout, max_retries=3, max_tokens=cfg.ragas.judge_max_tokens,
                      default_headers=provider_headers(base_url, "multapi-judge"),
                      http_client=DefaultHttpxClient(**options) if options else None, use_responses_api=False)


def judge_input(rec: dict[str, Any]) -> str:
    reference = next(q.reference for q in QUESTIONS if q.id == rec["question_id"])
    contexts = "\n\n".join(rec.get("contexts") or []) or "— (агент ничего не нашёл)"
    return (f"Вопрос клиента: {rec['question']}\n\nЭталон: {reference}\n\nФрагменты базы знаний:\n{contexts}\n\n"
            f"Ответ агента:\n{rec.get('answer') or '— (пустой ответ)'}")


def parse_verdict(text: str) -> dict[str, Any]:
    """JSON из ответа судьи: первая {...}; оценки — целые 1–5. Не разобралось — error и сырой ответ."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
        verdict = {name: min(5, max(1, round(float(data[name])))) for name in SCORES}
    except (ValueError, KeyError, TypeError):
        return {"error": "не JSON с оценками", "raw": text[:500]}
    return {**verdict, "reason": str(data.get("reason", ""))[:500]}


def judge(model: Any, rec: dict[str, Any]) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        reply = model.invoke([SystemMessage(JUDGE_PROMPT), HumanMessage(judge_input(rec))])
    except Exception as exc:  # noqa: BLE001 — сбой судьи на одном ответе не останавливает остальные
        return {"error": f"{type(exc).__name__}: {exc}"}
    return parse_verdict(reply.text)


def judge_records(records: list[dict[str, Any]], model: Any, *, info: dict[str, Any], rejudge: bool) -> int:
    done = 0
    for rec in records:
        if rec.get("judge") and "error" not in rec["judge"] and not rejudge:
            continue
        rec["judge"] = {**judge(model, rec), **info}
        done += 1
        verdict = rec["judge"]
        print(f"  {rec['implementation']:<12} {rec['question_id']}: "
              + (f"correctness={verdict['correctness']} faithfulness={verdict['faithfulness']} — {verdict['reason']}"
                 if "error" not in verdict else f"ошибка судьи: {verdict['error']}"))
    return done


# ---------------------------------------------------------------- сводка
def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 2) if values else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for impl in IMPLEMENTATIONS:
        rows = [r for r in records if r["implementation"] == impl]
        if not rows:
            continue
        judged = [r["judge"] for r in rows if r.get("judge") and "error" not in r["judge"]]
        citable = [r for r in rows if r["kind"] != "out_of_base"]
        out[impl] = {
            "questions": len(rows), "errors": sum(bool(r.get("error")) for r in rows),
            "tokens_mean": _mean([r["total_tokens"] for r in rows]),
            "input_tokens_mean": _mean([r["input_tokens"] for r in rows]),
            "output_tokens_mean": _mean([r["output_tokens"] for r in rows]),
            "llm_calls_mean": _mean([r["llm_calls"] for r in rows]),
            "llm_calls_without_usage": sum(r.get("llm_calls_without_usage", 0) for r in rows),
            "latency_p50_ms": round(statistics.median(r["latency_ms"] for r in rows), 1),
            "latency_mean_ms": _mean([r["latency_ms"] for r in rows]),
            "handoffs_mean": _mean([r["handoff_count"] for r in rows]),
            "searches_mean": _mean([r["searches"] for r in rows]),
            "judged": len(judged),
            **{f"{name}_mean": _mean([v[name] for v in judged]) for name in SCORES},
            "facts_hit": sum(r["checks"]["facts_hit"] for r in rows),
            "facts_total": sum(r["checks"]["facts_total"] for r in rows),
            "expected": sum(bool(r["checks"]["expected"]) for r in rows),
            "with_citations": sum(bool(r["checks"]["citations"]) for r in citable), "citable": len(citable),
            "models": sorted({str(r.get("model")) for r in rows}),
        }
    if all(impl in out for impl in IMPLEMENTATIONS):
        single, multi = out["single_agent"], out["multi_agent"]

        def ratio(key: str) -> float | None:
            return round(multi[key] / single[key], 2) if single[key] and multi[key] is not None else None

        def diff(key: str) -> float | None:
            both = single[key] is not None and multi[key] is not None
            return round(multi[key] - single[key], 2) if both else None

        out["delta"] = {"tokens_x": ratio("tokens_mean"), "llm_calls_x": ratio("llm_calls_mean"),
                        "latency_p50_x": ratio("latency_p50_ms"), "handoffs_diff": diff("handoffs_mean"),
                        **{f"{name}_diff": diff(f"{name}_mean") for name in SCORES},
                        "facts_hit_diff": multi["facts_hit"] - single["facts_hit"],
                        "expected_diff": multi["expected"] - single["expected"]}
    return out


# ---------------------------------------------------------------- таблицы
def _num(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}".replace(",", " ")


def _times(value: float | None) -> str:
    return "—" if value is None else f"×{value:.2f} ({(value - 1) * 100:+.0f} %)"


def _plus(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:+.{digits}f}".replace("-", "−")


def main_table(summary: dict[str, Any]) -> str:
    single, multi = summary.get("single_agent", {}), summary.get("multi_agent", {})
    delta = summary.get("delta", {})
    n_single, n_multi = single.get("questions", 0), multi.get("questions", 0)
    rows = [
        ("Токены на запрос", _num(single.get("tokens_mean"), 0), _num(multi.get("tokens_mean"), 0),
         _times(delta.get("tokens_x"))),
        ("LLM-вызовов на запрос", _num(single.get("llm_calls_mean")), _num(multi.get("llm_calls_mean")),
         _times(delta.get("llm_calls_x"))),
        ("Latency p50 (мс)", _num(single.get("latency_p50_ms"), 0), _num(multi.get("latency_p50_ms"), 0),
         _times(delta.get("latency_p50_x"))),
        ("Передач управления на запрос", _num(single.get("handoffs_mean")), _num(multi.get("handoffs_mean")),
         _plus(delta.get("handoffs_diff"))),
        ("Качество: LLM-судья, correctness (1–5)", _num(single.get("correctness_mean"), 2),
         _num(multi.get("correctness_mean"), 2), _plus(delta.get("correctness_diff"), 2)),
        ("Качество: LLM-судья, faithfulness (1–5)", _num(single.get("faithfulness_mean"), 2),
         _num(multi.get("faithfulness_mean"), 2), _plus(delta.get("faithfulness_diff"), 2)),
        ("Ключевые факты эталона в ответе", f"{single.get('facts_hit', '—')}/{single.get('facts_total', '—')}",
         f"{multi.get('facts_hit', '—')}/{multi.get('facts_total', '—')}", _plus(delta.get("facts_hit_diff"), 0)),
        ("Ожидаемое поведение (все факты / честный отказ)", f"{single.get('expected', '—')}/{n_single}",
         f"{multi.get('expected', '—')}/{n_multi}", _plus(delta.get("expected_diff"), 0)),
        ("Ответы с цитатами [n] (вопросы по базе)", f"{single.get('with_citations', '—')}/{single.get('citable', '—')}",
         f"{multi.get('with_citations', '—')}/{multi.get('citable', '—')}", ""),
    ]
    head = (f"| Метрика | Single-agent (среднее по {n_single}) | Multi-agent (среднее по {n_multi}) | Δ |\n"
            "|---|---:|---:|---:|")
    return "\n".join([head, *(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows)])


def per_question_table(records: list[dict[str, Any]]) -> str:
    lines = [("| Вопрос | Реализация | Токены | LLM-вызовы (по агентам) | Latency, мс | Передачи | Поиски | Факты | "
              "Цитаты | Судья c/f |"), "|---|---|---:|---|---:|---:|---:|---:|---|---|"]
    for question in QUESTIONS:
        for impl in IMPLEMENTATIONS:
            rec = next((r for r in records if r["implementation"] == impl and r["question_id"] == question.id), None)
            if rec is None:
                continue
            calls = ", ".join(f"{name} {s['calls']}" for name, s in rec["by_agent"].items())
            verdict = rec.get("judge") or {}
            scores = f"{verdict['correctness']}/{verdict['faithfulness']}" if "correctness" in verdict else "—"
            checks = rec["checks"]
            facts = ("отказ" if checks["refused"] else "нет отказа") if question.kind == "out_of_base" else \
                f"{checks['facts_hit']}/{checks['facts_total']}"
            lines.append(f"| {question.id} ({question.kind}) | {TITLES[impl]} | {_num(rec['total_tokens'], 0)} | "
                         f"{rec['llm_calls']} ({calls}) | {_num(rec['latency_ms'], 0)} | {rec['handoff_count']} | "
                         f"{rec['searches']} | {facts} | {checks['citations'] or '—'} | {scores} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM-судья и таблицы 6.5 по experiments/results.json")
    parser.add_argument("--rejudge", action="store_true", help="оценить заново и уже оценённые ответы")
    parser.add_argument("--no-judge", action="store_true", help="без судьи: только таблицы")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args(argv)
    from experiments.common import console

    console()
    records = load_records(args.results)
    if not records:
        print(f"Нет записей в {args.results}: сначала experiments/single_agent_baseline.py и "
              "experiments/multi_agent_langgraph.py")
        return 1
    if not args.no_judge:
        from urllib.parse import urlsplit

        from app.core.config import get_settings

        cfg = get_settings()
        model = build_judge(cfg)
        host = urlsplit(cfg.ragas.judge_base_url or cfg.llm.base_url or "").hostname
        info = {"judge_model": cfg.ragas.judge_model, "judge_host": host}
        print(f"Судья: {info['judge_model']} @ {host}")
        if judge_records(records, model, info=info, rejudge=args.rejudge):
            write_json(args.results, records)
    summary = summarize(records)
    write_json(args.summary, summary)
    print("\n" + main_table(summary) + "\n\n" + per_question_table(records))
    print(f"\nСводка: {args.summary.relative_to(ROOT).as_posix() if args.summary.is_relative_to(ROOT) else args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
