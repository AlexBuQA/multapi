"""
Golden dataset после ручной вычитки (блок 5.6): tests/eval/golden_dataset_raw.csv + решения из
tests/eval/golden_review.yaml → tests/eval/golden_dataset.json.

    python scripts/build_golden.py                   # проверить вычитку и записать golden_dataset.json
    python scripts/build_golden.py --check           # только проверить: всё ли вычитано, хватает ли пар

Вычитка — по одной записи на каждую строку сырого CSV, ключ — её id:

    items:
      raw_001: {keep: true}
      raw_002: {keep: false, reason: "дубль raw_001"}
      raw_003:
        keep: true
        user_input: "Сколько дней действует ссылка для сброса пароля?"    # исправленный вопрос
        reference: "Ссылка действует 30 минут."                        # доразмеченный эталон
        note: "модель написала 30 дней — в password-reset.html 30 минут"

Правка вопроса или эталона — прямо в записи: сырой CSV остаётся как есть, и в репозитории видно, что
сгенерировала модель и что поменял человек. Строка без решения, неизвестный id, повтор вопроса или
меньше --min оставленных пар — ошибка: недочитанный golden не попадёт в оценку.

Формат golden_dataset.json — список записей: id, user_input, reference, reference_contexts,
reference_sources, synthesizer, edited (какие поля поправлены), note. Его читает scripts/run_eval.py.
Код выхода: 0 — готово, 1 — вычитка неполная или пар мало, 2 — ошибка чтения.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "tests" / "eval" / "golden_dataset_raw.csv"
REVIEW = ROOT / "tests" / "eval" / "golden_review.yaml"
OUT = ROOT / "tests" / "eval" / "golden_dataset.json"
EDITABLE = ("user_input", "reference", "reference_contexts", "reference_sources")


def read_raw(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    result = {}
    for row in rows:
        row["reference_contexts"] = json.loads(row.get("reference_contexts") or "[]")
        row["reference_sources"] = json.loads(row.get("reference_sources") or "[]")
        result[row["id"]] = row
    return result


def read_review(path: Path) -> dict[str, dict[str, Any]]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = data.get("items") or {}
    if not isinstance(items, dict):
        raise ValueError(f"{path.name}: items — словарь «id: решение»")
    return {str(key): (value or {}) for key, value in items.items()}


def build(raw: dict[str, dict[str, Any]], review: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Записи golden и список проблем вычитки (пустой — всё в порядке)."""
    problems = [f"{key}: нет решения keep" for key in raw if "keep" not in review.get(key, {})]
    problems += [f"{key}: такого id нет в сыром CSV" for key in review if key not in raw]
    golden, seen = [], {}
    for key, row in raw.items():
        decision = review.get(key, {})
        if decision.get("keep") is not True:
            if decision.get("keep") is False and not decision.get("reason"):
                problems.append(f"{key}: выброшена без reason")
            continue
        item = {"id": key, "synthesizer": row.get("synthesizer_name", "")}
        edited = []
        for field in EDITABLE:
            value = decision.get(field, row.get(field))
            if field in decision and decision[field] != row.get(field):
                edited.append(field)
            item[field] = value.strip() if isinstance(value, str) else value
        if not item["user_input"] or not item["reference"]:
            problems.append(f"{key}: пустой user_input или reference")
        question = " ".join(item["user_input"].lower().split())
        if question in seen:
            problems.append(f"{key}: тот же вопрос, что {seen[question]}")
        seen[question] = key
        item["edited"] = edited
        item["note"] = decision.get("note", "")
        golden.append(item)
    return golden, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="golden_dataset.json из сырого CSV и ручной вычитки (блок 5.6)")
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--review", type=Path, default=REVIEW)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--min", type=int, default=30, help="сколько пар должно остаться после вычитки")
    parser.add_argument("--check", action="store_true", help="только проверить, не записывать")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):     # вывод в файл на Windows (cp1251): «₽» и др. — не падать
        sys.stdout.reconfigure(errors="replace")
    try:
        raw, review = read_raw(args.raw), read_review(args.review)
    except (OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    golden, problems = build(raw, review)
    dropped = sum(1 for d in review.values() if d.get("keep") is False)
    edited = sum(1 for item in golden if item["edited"])
    print(f"Сырых пар {len(raw)}: оставлено {len(golden)} (из них поправлено {edited}), выброшено {dropped}")
    if len(golden) < args.min:
        problems.append(f"оставлено {len(golden)} пар — меньше {args.min}")
    if problems:
        print("Вычитка не закончена:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    if not args.check:
        args.out.write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"→ {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
