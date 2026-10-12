"""
Таблицы для docs/rag_evaluation.md из результатов scripts/run_eval.py (блок 5.6): числа копируются из
CSV и JSON прогонов, а не переписываются руками.

    python scripts/eval_report.py                                   # все прогоны tests/eval/results/: таблица A/B
    python scripts/eval_report.py --labels baseline chunk_128       # только эти метки (последний прогон каждой)
    python scripts/eval_report.py --worst tests/eval/results/2026-10-13_1530_rerank.csv    # failure analysis
    python scripts/eval_report.py --check tests/eval/results/2026-10-13_1530_rerank.json   # пороги приёмки
    python scripts/eval_report.py --out var/eval_report.md          # то же в файл (UTF-8)

Таблица A/B: метка, что поменяли относительно первой строки (baseline), пять метрик со сдвигом от
baseline, средняя и p95 задержка, отказы. Прогоны с другим судьёй или другим golden dataset
сравнивать нельзя — такие строки помечены.

Failure analysis (--worst): N худших ответов по faithfulness (по возрастанию; отказы без фрагментов —
у них faithfulness нет — не входят, их число печатается отдельно), для каждого — вопрос, файлы и начало
фрагментов, ответ, метрики и диагноз по матрице 2×2: низкий faithfulness и низкий context_recall —
проблема retriever (нужного фрагмента в контексте не было), низкий faithfulness при высоком
context_recall — проблема генерации (контекст был, модель от него отошла). Граница «низкий» — 0.5.
Диагноз — подсказка: в отчёт он идёт после того, как ответ и фрагменты прочитаны глазами.

Пороги (--check): faithfulness > 0.7, answer_relevancy > 0.7, has_citation > 0.95 (цели чекпоинта 5).
Код выхода: 0 — готово (и пороги пройдены), 1 — порог не пройден, 2 — нечего читать.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "tests" / "eval" / "results"
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "has_citation")
THRESHOLDS = {"faithfulness": 0.7, "answer_relevancy": 0.7, "has_citation": 0.95}
LOW = 0.5
CONFIG_KEYS = ("chunking", "chunk_size", "chunk_overlap", "candidates_top_k", "similarity_top_k", "score_threshold",
               "llm_model", "temperature", "embeddings_model")


def load_runs(results: Path) -> list[dict[str, Any]]:
    """JSON прогонов (не .partial), по времени прогона."""
    runs = []
    for path in sorted(results.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and "metrics" in data and "label" in data:
            runs.append({**data, "_path": path})
    return sorted(runs, key=lambda r: (r.get("timestamp", ""), r["_path"].name))


def pick(runs: list[dict[str, Any]], labels: list[str] | None) -> list[dict[str, Any]]:
    """Последний прогон каждой метки — в порядке labels (первая — baseline)."""
    if not labels:
        return runs
    latest = {run["label"]: run for run in runs}
    missing = [label for label in labels if label not in latest]
    if missing:
        raise ValueError(f"нет прогонов с метками: {', '.join(missing)}")
    return [latest[label] for label in labels]


def _rerank(config: dict[str, Any]) -> str:
    rerank = config.get("rerank") or {}
    return f"on ({rerank.get('model')})" if rerank.get("enabled") else "off"


def changes(base: dict[str, Any], run: dict[str, Any]) -> str:
    """Чем конфигурация прогона отличается от baseline: «chunk_size 512→128»."""
    a, b = base.get("config", {}), run.get("config", {})
    diff = [f"{key} {a.get(key)}→{b.get(key)}" for key in CONFIG_KEYS if a.get(key) != b.get(key)]
    if _rerank(a) != _rerank(b):
        diff.append(f"re-ranker {_rerank(a).split(' ')[0]}→{_rerank(b).split(' ')[0]}")
    return ", ".join(diff) or "—"


def _mean(run: dict[str, Any], metric: str) -> float | None:
    return (run.get("metrics", {}).get(metric) or {}).get("mean")


def cell(run: dict[str, Any], base: dict[str, Any] | None, metric: str) -> str:
    value = _mean(run, metric)
    if value is None:
        return "—"
    text = f"{value:.3f}"
    reference = _mean(base, metric) if base is not None else None
    if base is not None and base is not run and reference is not None:
        text += f" ({value - reference:+.3f})"
    return text


def comparable(base: dict[str, Any], run: dict[str, Any]) -> list[str]:
    problems = []
    if base.get("judge", {}).get("judge_model") != run.get("judge", {}).get("judge_model"):
        problems.append("другой судья")
    if base.get("golden", {}).get("sha256") != run.get("golden", {}).get("sha256"):
        problems.append("другой golden")
    if base.get("judge", {}).get("citation_prompt") != run.get("judge", {}).get("citation_prompt"):
        problems.append("другой промпт has_citation")
    return problems


def ab_table(runs: list[dict[str, Any]]) -> str:
    """Таблица A/B: первая строка — baseline, у остальных — сдвиг от неё."""
    base = runs[0]
    lines = [("| Вариант | Изменение | faithfulness | answer_relevancy | context_precision | context_recall | "
              "has_citation | latency, мс (ср. / p95) | отказы (порог / модель) | файл |"),
             "|---|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        latency = run.get("latency_ms", {})
        refusals = run.get("refusals", {})
        warn = comparable(base, run)
        label = run["label"] + (f" ⚠️ {', '.join(warn)}" if warn else "")
        lines.append(f"| {label} | {changes(base, run) if run is not base else 'baseline'} | "
                     + " | ".join(cell(run, base, m) for m in METRICS)
                     + f" | {latency.get('mean', '—')} / {latency.get('p95', '—')} | "
                       f"{refusals.get('score_guard', '—')} / {refusals.get('model', '—')} | "
                       f"`{run['_path'].with_suffix('.csv').name}` |")
    judge, golden = base.get("judge", {}), base.get("golden", {})
    lines.append("")
    lines.append(f"Судья — `{judge.get('judge_model')}` ({judge.get('judge_host')}), эмбеддинги судьи — "
                 f"`{judge.get('judge_embeddings')}`; golden: вопросов {golden.get('items')} "
                 f"(`{golden.get('path')}`, sha256 {golden.get('sha256')}). В скобках — сдвиг от первой строки.")
    return "\n".join(lines)


def _float(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else math.nan
    except ValueError:
        return math.nan


def diagnosis(row: dict[str, Any]) -> str:
    faithfulness, recall = _float(row.get("faithfulness")), _float(row.get("context_recall"))
    if math.isnan(faithfulness):
        return "faithfulness нет: отказ без фрагментов"
    if faithfulness >= LOW:
        return "не провал по faithfulness"
    if math.isnan(recall):
        return "низкий faithfulness, context_recall нет (нет эталона) — смотреть фрагменты"
    if recall < LOW:
        return "retriever: низкий faithfulness и низкий context_recall — нужного фрагмента в контексте не было"
    return "generation: низкий faithfulness при высоком context_recall — контекст был, модель от него отошла"


def worst(csv_path: Path, n: int) -> str:
    """Failure analysis: n худших ответов по faithfulness (по возрастанию)."""
    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    scored = [r for r in rows if not math.isnan(_float(r.get("faithfulness")))]
    skipped = len(rows) - len(scored)
    scored.sort(key=lambda r: (_float(r["faithfulness"]), r["id"]))
    lines = [(f"Прогон `{csv_path.name}`: ответов {len(rows)}, faithfulness есть у {len(scored)} "
              f"(без него — {skipped}: отказ без фрагментов или сбой). Худшие по faithfulness, "
              f"{min(n, len(scored))}:"), ""]
    for number, row in enumerate(scored[:n], start=1):
        contexts = json.loads(row.get("retrieved_contexts") or "[]")
        files = json.loads(row.get("retrieved_files") or "[]")
        expected = json.loads(row.get("reference_sources") or "[]")
        shown = "; ".join(f"«{' '.join(c.split())[:120]}…»" for c in contexts[:3]) or "—"
        metrics = ", ".join(f"{m}={_float(row.get(m)):.2f}" for m in METRICS)
        lines += [f"**{number}. {row['id']}** — {row['user_input']}", "",
                  f"- retrieved: {', '.join(f'`{f}`' for f in files) or '—'}"
                  + (f" (эталон — {', '.join(f'`{f}`' for f in expected)})" if expected else ""),
                  f"- фрагменты (начало): {shown}",
                  f"- ответ: {' '.join((row.get('response') or '').split())}",
                  f"- эталон: {row.get('reference') or '—'}",
                  f"- метрики: {metrics}",
                  f"- диагноз (2×2): {diagnosis(row)}", ""]
    return "\n".join(lines)


def check(run: dict[str, Any]) -> tuple[bool, str]:
    lines, ok = [], True
    for metric, threshold in THRESHOLDS.items():
        value = _mean(run, metric)
        passed = value is not None and value > threshold
        ok &= passed
        shown = "—" if value is None else f"{value:.3f}"
        lines.append(f"{'OK  ' if passed else 'FAIL'} {metric} = {shown} (нужно > {threshold})")
    return ok, f"Пороги прогона {run['label']} ({run['_path'].name}):\n" + "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Таблицы docs/rag_evaluation.md из прогонов run_eval.py (блок 5.6)")
    parser.add_argument("--results", type=Path, default=RESULTS, help="папка прогонов")
    parser.add_argument("--labels", nargs="+", help="метки по порядку, первая — baseline")
    parser.add_argument("--worst", type=Path, help="CSV прогона для failure analysis")
    parser.add_argument("-n", type=int, default=5, help="сколько худших ответов (3–5 для отчёта)")
    parser.add_argument("--check", type=Path, help="JSON прогона: проверить пороги приёмки")
    parser.add_argument("--out", type=Path, help="записать в файл (UTF-8)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    code, parts = 0, []
    try:
        if args.worst:
            parts.append(worst(args.worst, args.n))
        if args.check:
            run = {**json.loads(args.check.read_text(encoding="utf-8")), "_path": args.check}
            ok, text = check(run)
            parts.append(text)
            code = 0 if ok else 1
        if not args.worst and not args.check:
            runs = pick(load_runs(args.results), args.labels)
            if not runs:
                raise ValueError(f"в {args.results} нет прогонов: сначала python scripts/run_eval.py")
            parts.append(ab_table(runs))
    except (OSError, ValueError, KeyError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    text = "\n\n".join(parts)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
