"""
Таблицы для docs/rag_evaluation.md из результатов scripts/run_eval.py (блок 5.6): числа копируются из
CSV и JSON прогонов, а не переписываются руками.

    python scripts/eval_report.py                                   # все прогоны tests/eval/results/: таблица A/B
    python scripts/eval_report.py --labels baseline chunk_128       # только эти метки (последний прогон каждой)
    python scripts/eval_report.py --worst tests/eval/results/2026-10-13_1530_rerank.csv    # failure analysis
    python scripts/eval_report.py --check tests/eval/results/2026-10-13_1530_rerank.json   # пороги приёмки
    python scripts/eval_report.py --out var/eval_report.md          # то же в файл (UTF-8)

Таблица A/B: метка, что поменяли относительно первой строки (baseline), пять метрик со сдвигом от
baseline, средняя и p95 задержка, отказы. Прогоны с другим судьёй, другим golden dataset или другим
промптом has_citation сравнивать нельзя — такие строки помечены.

Под ней — проверки без судьи, по CSV прогона (судья на 4B шумит, эти числа — нет):
- hit@5 и MRR@5 — есть ли среди пяти фрагментов, которые видела модель, фрагмент файла-источника
  эталона и на каком месте (метрики блока 5.4, по файлу), и сколько в пятёрке разных файлов: при мелкой
  нарезке несколько фрагментов одного файла вытесняют другие документы, а попадание по файлу ещё не
  значит, что найден фрагмент с ответом;
- «[N] в ответе» — доля ответов с маркером ссылки по регулярному выражению и совпадение с вердиктом
  has_citation: сверка судьи;
- «ссылка на эталон» — из ответов с маркерами доля тех, где хотя бы один [N] указывает на фрагмент
  файла-источника эталона. Нижняя граница точности ссылок: тот же факт бывает и в другом файле (FAQ).

Failure analysis (--worst): N худших ответов по faithfulness (по возрастанию; отказы без фрагментов —
у них faithfulness нет — не входят, их число печатается отдельно), для каждого — вопрос, файлы и начало
фрагментов, ответ, куда указывают его ссылки, метрики и диагноз по матрице 2×2: низкий faithfulness и
фрагмента эталона нет среди пяти (или низкий context_recall) — проблема retriever, низкий faithfulness
при найденном фрагменте и высоком context_recall — проблема генерации (контекст был, модель от него
отошла). Граница «низкий» — 0.5. Промах по файлу проверяется первым: на baseline context_recall = 1.0
даже там, где фрагмента эталона в пятёрке нет.
Диагноз — подсказка: в отчёт он идёт после того, как ответ и фрагменты прочитаны глазами.

Пороги (--check): faithfulness > 0.7, answer_relevancy > 0.7, has_citation > 0.95 (цели чекпоинта 5).
Код выхода: 0 — готово (и пороги пройдены), 1 — порог не пройден, 2 — нечего читать.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "tests" / "eval" / "results"
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "has_citation")
THRESHOLDS = {"faithfulness": 0.7, "answer_relevancy": 0.7, "has_citation": 0.95}
LOW = 0.5
MARKER = re.compile(r"\[(\d+)\]")
CONFIG_KEYS = ("chunking", "chunk_size", "chunk_overlap", "candidates_top_k", "similarity_top_k", "score_threshold",
               "llm_model", "temperature", "embeddings_model", "sources_fallback")
# Ключа нет в JSON прогонов до его появления — значит, тогда было так:
BEFORE_KEY = {"sources_fallback": False}


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
    old = {key: a.get(key, BEFORE_KEY.get(key)) for key in CONFIG_KEYS}
    new = {key: b.get(key, BEFORE_KEY.get(key)) for key in CONFIG_KEYS}
    diff = [f"{key} {old[key]}→{new[key]}" for key in CONFIG_KEYS if old[key] != new[key]]
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


def read_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def citations(row: dict[str, Any]) -> tuple[list[int], list[str], list[int]]:
    """Номера [N] из ответа, файлы фрагментов (в порядке промпта: [1] — первый) и места фрагментов
    файлов-источников эталона среди них."""
    numbers = [int(n) for n in MARKER.findall(row.get("response") or "")]
    files = json.loads(row.get("retrieved_files") or "[]")
    expected = set(json.loads(row.get("reference_sources") or "[]"))
    return numbers, files, [i for i, name in enumerate(files, start=1) if name in expected]


def judge_free(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Проверки без судьи по строкам CSV: hit@5, MRR@5, маркеры [N] и куда они указывают."""
    with_reference = [r for r in rows if json.loads(r.get("reference_sources") or "[]")]
    ranks = [citations(r)[2] for r in with_reference]
    marked = [r for r in rows if citations(r)[0]]
    agree = sum((bool(citations(r)[0]) == (_float(r.get("has_citation")) == 1.0))
                for r in rows if not math.isnan(_float(r.get("has_citation"))))
    on_reference = sum(any(n in citations(r)[2] for n in citations(r)[0]) for r in marked)
    shown = [citations(r)[1] for r in rows if citations(r)[1]]
    return {"rows": len(rows), "with_reference": len(with_reference),
            "hit": sum(bool(found) for found in ranks), "mrr": sum(1 / found[0] for found in ranks if found),
            "files": sum(len(set(files)) for files in shown) / (len(shown) or 1),
            "marked": len(marked), "agree": agree,
            "judged": sum(not math.isnan(_float(r.get("has_citation"))) for r in rows),
            "on_reference": on_reference}


def judge_free_table(runs: list[dict[str, Any]]) -> str:
    lines = ["Проверки без судьи (по CSV прогона):", "",
             "| Вариант | hit@5 | MRR@5 | файлов в пятёрке | [N] в ответе | совпадение с has_citation | ссылка на эталон |",
             "|---|---|---|---|---|---|---|"]
    for run in runs:
        csv_path = run["_path"].with_suffix(".csv")
        if not csv_path.is_file():
            lines.append(f"| {run['label']} | — | — | — | — | — | — |")
            continue
        s = judge_free(read_rows(csv_path))
        base = s["with_reference"] or 1
        lines.append(f"| {run['label']} | {s['hit']}/{s['with_reference']} ({s['hit'] / base:.3f}) | "
                     f"{s['mrr'] / base:.3f} | {s['files']:.1f} | {s['marked']}/{s['rows']} | {s['agree']}/{s['judged']} | "
                     f"{s['on_reference']}/{s['marked']} ({s['on_reference'] / (s['marked'] or 1):.3f}) |")
    return "\n".join(lines)


def diagnosis(row: dict[str, Any]) -> str:
    faithfulness, recall = _float(row.get("faithfulness")), _float(row.get("context_recall"))
    if math.isnan(faithfulness):
        return "faithfulness нет: отказ без фрагментов"
    if faithfulness >= LOW:
        return "не провал по faithfulness"
    missed = bool(json.loads(row.get("reference_sources") or "[]")) and not citations(row)[2]
    if missed:                                # судья на 4B бывает щедр к context_recall — промах по файлу важнее
        return ("retriever: фрагмента файла-источника эталона нет среди пяти"
                + (f" (context_recall={recall:.2f}: судья вывел эталон из соседних фрагментов)"
                   if not math.isnan(recall) and recall >= LOW else ""))
    if math.isnan(recall):
        return "низкий faithfulness, context_recall нет (нет эталона) — смотреть фрагменты"
    if recall < LOW:
        return "retriever: низкий faithfulness и низкий context_recall — нужного фрагмента в контексте не было"
    return "generation: низкий faithfulness при высоком context_recall — контекст был, модель от него отошла"


def worst(csv_path: Path, n: int) -> str:
    """Failure analysis: n худших ответов по faithfulness (по возрастанию)."""
    rows = read_rows(csv_path)
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
        numbers, _, reference_at = citations(row)
        cited = ", ".join(f"[{n}] `{files[n - 1]}`" if 1 <= n <= len(files) else f"[{n}] —" for n in numbers)
        at = ", ".join(f"[{i}]" for i in reference_at) or "нет среди пяти"
        lines += [f"**{number}. {row['id']}** — {row['user_input']}", "",
                  f"- retrieved: {', '.join(f'`{f}`' for f in files) or '—'}"
                  + (f" (эталон — {', '.join(f'`{f}`' for f in expected)})" if expected else ""),
                  f"- фрагменты (начало): {shown}",
                  f"- ответ: {' '.join((row.get('response') or '').split())}",
                  f"- эталон: {row.get('reference') or '—'}",
                  f"- метрики: {metrics}",
                  f"- ссылки ответа: {cited or 'нет'}; фрагмент эталона — {at}",
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
            parts.append(judge_free_table(runs))
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
