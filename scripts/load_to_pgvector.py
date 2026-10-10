"""
Та же база знаний — в Postgres с pgvector (блок 5.2, задача 7).

    python scripts/load_to_pgvector.py               # расширение, таблица kb.documents, индексы + все фрагменты data/
    python scripts/load_to_pgvector.py --recreate    # удалить таблицу и залить заново (сменили модель)
    python scripts/load_to_pgvector.py --drop        # только удалить таблицу: сравнение закончено

Postgres — из DATABASE_URL (по умолчанию сервис postgres из compose на 127.0.0.1:5433; образ
pgvector/pgvector: docker compose up -d postgres), размерность — EMBEDDING_DIM, модель —
EMBEDDINGS__* блока 5.1. Шаги — как у scripts/load_to_qdrant.py:

1. Фрагменты из data/ — те же 110, с тем же payload.
2. CREATE EXTENSION IF NOT EXISTS vector (нужен 0.8+), схема kb, таблица
   documents (id uuid, embedding vector(N), payload jsonb); есть — сверяется размерность.
   Индексы: HNSW vector_cosine_ops (m=16, ef_construction=100 — как у Qdrant), HNSW по
   halfvec-выражению, GIN по payload.
3. Векторы — embed_documents (кеш блока 5.1: модель не вызывается, если Qdrant уже заливали);
   длина каждого сверяется с EMBEDDING_DIM до записи.
4. INSERT … ON CONFLICT (id) DO UPDATE одной транзакцией; id — тот же uuid5, что у точек Qdrant.
   Повторный запуск не копирует строки и не переписывает неизменённые: «новых 0, изменено 0».
5. Строки, которых больше нет в data/, удаляются; ANALYZE — статистика для планировщика.
6. Итог — count(*) и размеры таблицы и индексов.

Код выхода: 0 — готово, 2 — ошибка настроек, данных, модели или Postgres (текст — что сделать).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def kb(size: int) -> str:
    return f"{size / 1024:.0f} КБ"


async def load(args: argparse.Namespace) -> int:
    from load_to_qdrant import describe, embed

    from app.core.config import get_settings
    from app.services.documents import load_chunks
    from app.services.pgvector_store import PgVectorStore

    store = PgVectorStore.from_settings(get_settings())
    if args.drop:
        try:
            await store.drop()
            print(f"Таблица {store.qualified} удалена ({store.address}). Расширение vector и схема "
                  f"{store.schema} остаются; сравнение можно повторить, загрузив таблицу заново.")
            return 0
        finally:
            await store.close()
    chunks = load_chunks(args.data_dir)
    print(describe(chunks), flush=True)
    try:
        if args.recreate:
            await store.drop()
            print(f"Таблица {store.qualified} удалена (--recreate)", flush=True)
        await store.ensure_schema()
        print(f"Postgres {store.server_version} ({store.address}), pgvector {store.extension_version}, "
              f"таблица {store.qualified}, vector({store.dim})", flush=True)
        existing = await store.ids()
        vectors = await asyncio.to_thread(embed, chunks, store.dim)
        rows = [(chunk.point_id, vector, chunk.payload) for chunk, vector in zip(chunks, vectors, strict=True)]
        started = time.perf_counter()
        inserted, updated = await store.upsert(rows)
        stale = sorted(existing - {point_id for point_id, _, _ in rows})
        await store.delete(stale)
        await store.analyze()
        seconds = time.perf_counter() - started
        print(f"Строк: {len(rows)} — новых {inserted}, изменено {updated}, без изменений "
              f"{len(rows) - inserted - updated}; удалено устаревших: {len(stale)}; {seconds:.2f} с")
        sizes = await store.sizes()
        indexes = ", ".join(f"{name} {kb(size)}" for name, size in sizes.items() if name != "table")
        print(f"count(*): {await store.count()}; таблица {kb(sizes['table'])}; индексы: {indexes}")
        return 0
    finally:
        await store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Загрузка базы знаний из data/ в Postgres + pgvector (блок 5.2)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--recreate", action="store_true", help="удалить таблицу и создать заново")
    group.add_argument("--drop", action="store_true", help="только удалить таблицу")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("-v", "--verbose", action="store_true", help="лог INFO: запросы к модели")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from pydantic import ValidationError

    from app.observability.logging import setup_logging
    from app.services.documents import CorpusError
    from app.services.embeddings import EmbeddingError
    from app.services.pgvector_store import PgVectorError

    setup_logging("INFO" if args.verbose else "WARNING", stream=sys.stdout)
    try:
        return asyncio.run(load(args))
    except (PgVectorError, EmbeddingError, CorpusError, ValidationError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
