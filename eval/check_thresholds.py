"""
Можно ли релизить: агрегаты последнего прогона eval против порогов (блок 3.7).

    python eval/check_thresholds.py
    python eval/check_thresholds.py --run eval/runs/2026-10-07.json --thresholds eval/thresholds.yaml

Последний прогон — файл в eval/runs/ с самым поздним полем timestamp. Пороги — в
eval/thresholds.yaml: секция min (не меньше) и max (не больше). Каждый нарушенный
порог печатается отдельной строкой, а скрипт завершается с кодом 1 — так его можно
поставить последним шагом перед сменой промпта или модели. Код 2 — нечего проверять
(нет прогонов, битый файл, в прогоне нет нужного агрегата).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

EVAL_DIR = Path(__file__).resolve().parent


class CheckError(Exception):
    """Проверку нельзя выполнить: нет прогонов или порогов."""


def latest_run(runs_dir: Path) -> Path:
    candidates = []
    for path in runs_dir.glob("*.json"):
        try:
            timestamp = json.loads(path.read_text(encoding="utf-8")).get("timestamp", "")
        except (OSError, ValueError):
            continue   # битый или чужой файл — пропускаем
        candidates.append((timestamp, path.stat().st_mtime, path))
    if not candidates:
        raise CheckError(f"в {runs_dir} нет прогонов: сначала python eval/run_evaluation.py")
    return max(candidates)[2]


def load_thresholds(path: Path) -> dict[str, dict[str, float]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(data) - {"min", "max"}
    if unknown:
        raise CheckError(f"{path}: неизвестные секции {sorted(unknown)}; допустимы min и max")
    return {"min": data.get("min") or {}, "max": data.get("max") or {}}


def check(aggregates: dict[str, Any], thresholds: dict[str, dict[str, float]]) -> tuple[list[str], list[str]]:
    """-> (строки отчёта, нарушения)."""
    lines, failures = [], []
    for kind, op, ok in (("min", ">=", lambda v, t: v >= t), ("max", "<=", lambda v, t: v <= t)):
        for name, limit in thresholds[kind].items():
            value = aggregates.get(name)
            if value is None:
                message = f"{name}: нет в прогоне (ожидалось {op} {limit})"
                failures.append(message)
                lines.append(f"[FAIL] {message}")
            elif ok(value, limit):
                lines.append(f"[OK]   {name} = {value} ({op} {limit})")
            else:
                message = f"{name} = {value}, порог {op} {limit}"
                failures.append(message)
                lines.append(f"[FAIL] {message}")
    return lines, failures


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Проверка агрегатов eval-прогона по порогам")
    parser.add_argument("--runs-dir", default=str(EVAL_DIR / "runs"))
    parser.add_argument("--run", default=None, help="конкретный файл прогона вместо последнего")
    parser.add_argument("--thresholds", default=str(EVAL_DIR / "thresholds.yaml"))
    args = parser.parse_args(argv)

    try:
        run_path = Path(args.run) if args.run else latest_run(Path(args.runs_dir))
        run = json.loads(run_path.read_text(encoding="utf-8"))
        thresholds = load_thresholds(Path(args.thresholds))
    except (CheckError, OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Проверка не выполнена: {exc}")
        return 2

    print(f"Прогон: {run_path} (run_id {run.get('run_id')}, модель {run.get('model_under_test')}, "
          f"промпт {run.get('prompt_version')}, судья {run.get('judge_model')}, golden v{run.get('golden_version')})")
    lines, failures = check(run.get("aggregates", {}), thresholds)
    print("\n".join(lines))

    if not failures:
        print("Все пороги выполнены — можно релизить.")
        return 0

    print(f"\nНельзя релизить: нарушено порогов — {len(failures)}.")
    for failure in failures:
        print(f"  - {failure}")
    weak = [i for i in run.get("items", []) if i.get("scores") and i["scores"]["correctness"] < 3]
    for item in sorted(weak, key=lambda i: i["scores"]["correctness"]):
        print(f"  кейс {item['id']}: correctness={item['scores']['correctness']} — {item.get('explanation') or ''}")
    for item in run.get("items", []):
        if item.get("must_not_contain_hits"):
            print(f"  кейс {item['id']}: запрещённые слова {item['must_not_contain_hits']}")
        if item.get("error"):
            print(f"  кейс {item['id']}: ошибка {item['error']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
