"""
Стоимость и объём индексации базы знаний эмбеддингами (блок 5.1).

    python scripts/indexing_cost.py
    python scripts/indexing_cost.py --corpus data/help_center.jsonl --benchmark-json docs/embeddings_benchmark.json

Считает токены документов data/help_center.jsonl словарём cl100k_base — токенизатором
text-embedding-3-small/-large и ada-002 (у них одинаковый) — и по ним:
1. цену индексации базы для моделей OpenAI: обычный запрос и Batch API (−50 %, ответ до 24 ч),
   для 56 документов базы и для 1 000 / 10 000 / 100 000 документов такой же длины;
2. объём векторов float32 по размерности модели (столько займёт индекс без накладных расходов БД);
3. цену запросов пользователей: вопрос тоже кодируется моделью (средняя длина — по
   tests/eval/mini_benchmark.json);
4. для локальных моделей цена — 0 $, а время индексации — по замеру
   scripts/embeddings_benchmark.py (docs/embeddings_benchmark.json), если он есть.

Цены — за 1 млн входных токенов, со страниц моделей OpenAI на 10.10.2026; проверяйте перед
расчётом бюджета. Без сети и словаря cl100k_base токены оцениваются как байты UTF-8 / 4 —
в выводе это написано.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "data" / "help_center.jsonl"
BENCHMARK = ROOT / "tests" / "eval" / "mini_benchmark.json"
BENCHMARK_JSON = ROOT / "docs" / "embeddings_benchmark.json"
PRICE_DATE = "10.10.2026"
# $ за 1 млн входных токенов (обычный запрос); Batch API — половина.
PRICES = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}
BATCH_DISCOUNT = 0.5
# Размерность векторов: облачные — по умолчанию; text-embedding-3 укорачивается параметром dimensions.
DIMENSIONS = {
    "bge-m3": 1024,
    "multilingual-e5-small": 384,
    "multilingual-e5-base": 768,
    "multilingual-e5-large": 1024,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-3-large, dimensions=1024": 1024,
}
SCALES = (1_000, 10_000, 100_000)
FLOAT32 = 4


def load_corpus(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def token_counter() -> tuple[Any, bool]:
    """(функция подсчёта, точный ли подсчёт)."""
    from app.services.embeddings import _cl100k, count_tokens

    return count_tokens, _cl100k() is not None


def cost(tokens: int, price_per_million: float) -> float:
    return tokens * price_per_million / 1_000_000


def money(value: float) -> str:
    """Доллары: три значащие цифры (видны и доли цента), не меньше двух знаков после точки."""
    if value <= 0:
        return "0 $"
    decimals = max(2, 2 - math.floor(math.log10(value)))
    return f"{value:,.{decimals}f}".replace(",", " ") + " $"


def duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f} с"
    if seconds < 2 * 3600:
        return f"{seconds / 60:.0f} мин"
    return f"{seconds / 3600:.1f} ч"


def size(num_bytes: float) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if num_bytes < 1024 or unit == "ГБ":
            return f"{num_bytes:.0f} {unit}" if unit == "Б" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} ГБ"


def is_local_result(result: dict[str, Any]) -> bool:
    """Модель на этом компьютере: Ollama (local:11434) или sentence-transformers (hf). Место —
    поле namespace; в JSON первых прогонов его нет — тогда из подписи openai@local:11434/…"""
    where = result.get("namespace")
    if where is None:
        label = str(result.get("model", ""))
        where = label.partition("@")[2] if "@" in label else label.partition(":")[2]
    return str(where).startswith(("local", "hf"))


def report(corpus: list[dict[str, str]], queries: list[str], count, exact: bool,
           measured: list[dict[str, Any]]) -> dict[str, Any]:
    doc_tokens = [count(doc["text"]) for doc in corpus]
    query_tokens = [count(query) for query in queries]
    total = sum(doc_tokens)
    avg = total / len(doc_tokens)
    data: dict[str, Any] = {
        "docs": len(corpus),
        "chars": sum(len(doc["text"]) for doc in corpus),
        "tokens": total,
        "tokens_avg": round(avg, 1),
        "tokens_max": max(doc_tokens),
        "query_tokens_avg": round(statistics.fmean(query_tokens), 1),
        "exact": exact,
        "prices": {},
        "storage": {},
        "local": [],
    }
    for model, price in PRICES.items():
        row = {"price": price, "corpus": cost(total, price), "corpus_batch": cost(total, price) * BATCH_DISCOUNT}
        for scale in SCALES:
            row[str(scale)] = cost(round(avg * scale), price)
            row[f"{scale}_batch"] = row[str(scale)] * BATCH_DISCOUNT
        row["queries_100k"] = cost(round(data["query_tokens_avg"] * 100_000), price)
        data["prices"][model] = row
    for model, dim in DIMENSIONS.items():
        data["storage"][model] = {"dim": dim, "vector_bytes": dim * FLOAT32,
                                  "corpus_bytes": dim * FLOAT32 * len(corpus),
                                  "100000_bytes": dim * FLOAT32 * 100_000}
    for result in measured:
        per_s = result.get("docs_per_s")
        if not per_s:
            continue
        data["local"].append({"model": result.get("model"), "docs_per_s": per_s,
                              "100000_s": round(100_000 / per_s), "query_ms": result.get("query_ms_median")})
    return data


def render(data: dict[str, Any]) -> str:
    how = "tiktoken cl100k_base" if data["exact"] else "оценка байты UTF-8 / 4 (словаря cl100k_base нет)"
    lines = [
        f"База: {data['docs']} документов, {data['chars']} символов, {data['tokens']} токенов ({how}); "
        f"в среднем {data['tokens_avg']} токена на документ, самый длинный — {data['tokens_max']}.",
        f"Вопрос пользователя в среднем {data['query_tokens_avg']} токена.",
        "",
        f"Цена индексации (цены OpenAI за 1 млн токенов на {PRICE_DATE}; Batch API — −50 %):",
        "",
        "| Модель | $ / 1M | Эта база | 1 000 док. | 10 000 док. | 100 000 док. | 100 000 док., Batch | "
        "100 000 вопросов |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, row in data["prices"].items():
        lines.append(f"| {model} | {row['price']} | {money(row['corpus'])} | {money(row['1000'])} | "
                     f"{money(row['10000'])} | {money(row['100000'])} | {money(row['100000_batch'])} | "
                     f"{money(row['queries_100k'])} |")
    lines += ["| bge-m3, multilingual-e5 (локально) | 0 | 0 $ | 0 $ | 0 $ | 0 $ | — | 0 $ |",
              "", "Объём векторов float32 (без индекса и метаданных БД):", "",
              "| Модель | Размерность | Один вектор | Эта база | 100 000 док. |", "|---|---|---|---|---|"]
    for model, row in data["storage"].items():
        lines.append(f"| {model} | {row['dim']} | {size(row['vector_bytes'])} | {size(row['corpus_bytes'])} | "
                     f"{size(row['100000_bytes'])} |")
    if data["local"]:
        lines += ["", "Локальные модели — время вместо денег (замер scripts/embeddings_benchmark.py):", "",
                  "| Модель | Документов/с | 100 000 док. | Один вопрос |", "|---|---|---|---|"]
        for row in data["local"]:
            query = f"{row['query_ms']:.0f} мс" if row["query_ms"] is not None else "—"
            lines.append(f"| {row['model']} | {row['docs_per_s']} | ≈ {duration(row['100000_s'])} | {query} |")
    else:
        lines += ["", "Время индексации локальными моделями — после python scripts/embeddings_benchmark.py "
                      "(docs/embeddings_benchmark.json)."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Цена и объём индексации базы знаний эмбеддингами (блок 5.1)")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--benchmark-json", type=Path, default=BENCHMARK_JSON)
    parser.add_argument("--json", action="store_true", help="вывести расчёт в JSON")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from app.observability.logging import setup_logging

    setup_logging("WARNING", stream=sys.stdout)
    corpus = load_corpus(args.corpus)
    queries = [item["query"] for item in json.loads(BENCHMARK.read_text(encoding="utf-8"))]
    measured = []
    if args.benchmark_json.exists():
        measured = [r for r in json.loads(args.benchmark_json.read_text(encoding="utf-8")).get("results", [])
                    if is_local_result(r)]
    count, exact = token_counter()
    data = report(corpus, queries, count, exact, measured)
    print(json.dumps(data, ensure_ascii=False, indent=2) if args.json else render(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
