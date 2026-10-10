"""
Фильтрация по metadata в Qdrant — три примера (блок 5.2, задача 6).

    python scripts/qdrant_filters_demo.py                    # «сегодня» — текущая дата
    python scripts/qdrant_filters_demo.py --today 2026-10-10 # воспроизвести прогон из docs/vector_store.md

Для каждого примера — вопрос пользователя, Python-код фильтра и top-3 из боевой коллекции
(QDRANT_COLLECTION) без фильтра и с ним: так видно, что меняет фильтр.

1. Match по строке: source = "release_notes.md" — ответ только из заметок о выпусках.
2. Range по дате: created_at >= сегодня − 30 дней (DatetimeRange) — без фильтра сверху
   старые статьи, с фильтром — свежие заметки.
3. must + must_not: только раздел billing, без архивных редакций (status = archived) —
   аналог «только тенант X, без архивных документов»: самый частый фильтр в production-RAG.
   Вопрос — о тарифе по старому названию: без фильтра сверху архив с ценами 2025 года, с
   фильтром — действующие тарифы и заметка о переименовании.

Вывод — Markdown для docs/vector_store.md. Код выхода: 0 — готово, 2 — ошибка.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOP_K = 3
DAYS = 30


def examples(today: date):
    """(название, вопрос, код фильтра для документации, Filter)."""
    from qdrant_client.models import DatetimeRange, FieldCondition, Filter, MatchValue

    since = (datetime(today.year, today.month, today.day, tzinfo=UTC) - timedelta(days=DAYS)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    return [
        ("Match по строке: только заметки о выпусках",
         "Что нового в мобильном приложении?",
         'Filter(must=[FieldCondition(key="source", match=MatchValue(value="release_notes.md"))])',
         Filter(must=[FieldCondition(key="source", match=MatchValue(value="release_notes.md"))])),
        (f"Range по дате: created_at не старше {DAYS} дней (с {since[:10]})",
         "Какие тарифы есть и сколько они стоят?",
         f'since = (datetime.now(UTC) - timedelta(days={DAYS})).strftime("%Y-%m-%dT%H:%M:%SZ")  # {since}\n'
         'Filter(must=[FieldCondition(key="created_at", range=DatetimeRange(gte=since))])',
         Filter(must=[FieldCondition(key="created_at", range=DatetimeRange(gte=since))])),
        ("must + must_not: раздел billing без архивных редакций",
         "Сколько стоит тариф «Профессиональный»?",
         'Filter(\n'
         '    must=[FieldCondition(key="category", match=MatchValue(value="billing"))],\n'
         '    must_not=[FieldCondition(key="status", match=MatchValue(value="archived"))],\n'
         ')',
         Filter(must=[FieldCondition(key="category", match=MatchValue(value="billing"))],
                must_not=[FieldCondition(key="status", match=MatchValue(value="archived"))])),
    ]


def rows(points) -> list[str]:
    lines = ["| # | doc_id | Заголовок | source | category | created_at | status | score |",
             "|---|---|---|---|---|---|---|---|"]
    for number, point in enumerate(points, start=1):
        p = point.payload or {}
        lines.append(f"| {number} | {p.get('doc_id')} | {p.get('title')} | {p.get('source')} | {p.get('category')} | "
                     f"{str(p.get('created_at', ''))[:10]} | {p.get('status')} | {point.score:.3f} |")
    if not points:
        lines.append("| — | ничего не найдено | | | | | | |")
    return lines


async def demo(args: argparse.Namespace) -> int:
    from app.core.config import get_settings
    from app.services import embeddings
    from app.services.vector_store import VectorStore, VectorStoreError

    store = VectorStore.from_settings(get_settings())
    try:
        if await store.points_count() == 0:
            raise VectorStoreError(f"Коллекция {store.collection} пуста: сначала python scripts/load_to_qdrant.py")
        print(f"Коллекция {store.collection}, сегодня {args.today.isoformat()}, top-{TOP_K}\n")
        for number, (title, query, code, query_filter) in enumerate(examples(args.today), start=1):
            vector = await asyncio.to_thread(embeddings.embed_query, query)
            plain = await store.search(vector, top_k=TOP_K)
            filtered = await store.search(vector, top_k=TOP_K, query_filter=query_filter)
            print(f"### {number}. {title}\n")
            print(f"Вопрос: «{query}»\n")
            print("```python\n" + code + "\n```\n")
            print("Без фильтра:\n")
            print("\n".join(rows(plain)) + "\n")
            print("С фильтром:\n")
            print("\n".join(rows(filtered)) + "\n")
        return 0
    finally:
        await store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Три примера фильтров Qdrant на базе знаний (блок 5.2)")
    parser.add_argument("--today", type=date.fromisoformat, default=date.today(),
                        help="дата «сегодня» для фильтра по дате, например 2026-10-10")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from pydantic import ValidationError

    from app.observability.logging import setup_logging
    from app.services.embeddings import EmbeddingError
    from app.services.vector_store import VectorStoreError

    setup_logging("WARNING", stream=sys.stdout)
    try:
        return asyncio.run(demo(args))
    except (VectorStoreError, EmbeddingError, ValidationError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
