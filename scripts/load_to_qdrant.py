"""
Загрузка базы знаний в Qdrant (блок 5.2).

    python scripts/load_to_qdrant.py                # коллекция (если нет) + все фрагменты data/
    python scripts/load_to_qdrant.py --recreate     # удалить коллекцию и залить заново (сменили модель)
    python scripts/load_to_qdrant.py --dry-run      # только прочитать data/: сколько фрагментов и откуда
    python scripts/load_to_qdrant.py --no-prune     # не удалять точки, которых больше нет в data/

Адрес, ключ, коллекция и размерность — из .env (QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
EMBEDDING_DIM), модель эмбеддингов — EMBEDDINGS__* блока 5.1. Шаги:

1. Фрагменты из data/ (app/services/documents.py): справочный центр, заметки о выпусках,
   частые вопросы, архив прежних редакций — 110 фрагментов с payload source, text,
   created_at, category, product, status, doc_id, title.
2. ensure_collection: нет коллекции — создаётся (размерность EMBEDDING_DIM, COSINE, HNSW
   m=16, ef_construct=100) с payload-индексами source, created_at, category, product, status;
   есть — сверяется размерность: коллекция под другую модель — ошибка, а не тихая заливка.
3. Векторы — embed_documents (кеш блока 5.1: повторный запуск модель не вызывает). Длина
   каждого вектора сверяется с EMBEDDING_DIM до отправки в Qdrant.
4. upsert пачками по --batch-size (128), wait=True на последней — после неё можно искать.
   Id точки — uuid5 от файла и ключа фрагмента: повторный запуск перезаписывает те же точки.
5. Точки, которых больше нет в data/ (фрагмент удалили или переименовали), удаляются —
   коллекция совпадает с data/.
6. Итог — points_count из get_collection: после второго запуска тот же, что после первого.

Код выхода: 0 — готово, 2 — ошибка настроек, данных, модели или Qdrant (текст — что сделать).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BATCH = 128


def describe(chunks) -> str:
    sources = Counter(chunk.source.split("/")[0] if "/" in chunk.source else chunk.source for chunk in chunks)
    statuses = Counter(chunk.status for chunk in chunks)
    parts = ", ".join(f"{name} — {count}" for name, count in sources.items())
    return f"Фрагментов: {len(chunks)} ({parts}); актуальных {statuses['actual']}, архивных {statuses['archived']}"


def embed(chunks, dim: int):
    """Векторы фрагментов моделью блока 5.1 и проверка размерности до записи в Qdrant."""
    from app.services import embeddings

    service = embeddings.default_service()
    before = (service.stats.requests, service.stats.cache_hits)
    started = time.perf_counter()
    vectors = service.embed_documents([chunk.text for chunk in chunks])
    seconds = time.perf_counter() - started
    requests, hits = service.stats.requests - before[0], service.stats.cache_hits - before[1]
    print(f"Эмбеддинги: {service.label}, из кеша {hits}, запросов к модели {requests}, {seconds:.1f} с", flush=True)
    sizes = {len(vector) for vector in vectors}
    if sizes != {dim}:
        raise ValueError(f"Модель {service.label} возвращает векторы из {', '.join(map(str, sorted(sizes)))} чисел, "
                         f"а EMBEDDING_DIM={dim} — размерность коллекции. Либо в .env другая модель "
                         f"(EMBEDDINGS__MODEL), "
                         f"либо неверный EMBEDDING_DIM. Ничего не загружено.")
    return vectors


async def load(args: argparse.Namespace) -> int:
    from qdrant_client.models import PointStruct
    from tqdm import tqdm

    from app.core.config import get_settings
    from app.services.documents import load_chunks
    from app.services.vector_store import VectorStore

    chunks = load_chunks(args.data_dir)
    print(describe(chunks), flush=True)
    if args.dry_run:
        return 0
    cfg = get_settings()
    store = VectorStore.from_settings(cfg)
    try:
        print(f"Qdrant: {store.url}, коллекция {store.collection}, размерность {store.dim}", flush=True)
        if args.recreate:
            await store.drop()
            print(f"Коллекция {store.collection} удалена (--recreate)", flush=True)
        await store.ensure_collection()
        existing = await store.point_ids()
        vectors = await asyncio.to_thread(embed, chunks, store.dim)
        points = [PointStruct(id=chunk.point_id, vector=vector, payload=chunk.payload)
                  for chunk, vector in zip(chunks, vectors, strict=True)]
        with tqdm(total=len(points), unit=" точек", desc="upsert", file=sys.stdout, ascii=True) as bar:
            await store.upsert(points, batch_size=args.batch_size, on_batch=bar.update)
        current = {point.id for point in points}
        stale = sorted(existing - current)
        if stale and not args.no_prune:
            await store.delete(stale)
        info = await store.collection_info()
        new = len(current - existing)
        removed = len(stale) if not args.no_prune else 0
        print(f"Записано точек: {len(points)} (новых {new}, перезаписано {len(points) - new}); "
              f"удалено устаревших: {removed}" + (f", оставлено устаревших: {len(stale)}" if args.no_prune else ""))
        print(f"points_count: {info.points_count}; размерность {info.config.params.vectors.size}, "
              f"метрика {info.config.params.vectors.distance.value}, индексы payload: "
              f"{', '.join(sorted(info.payload_schema))}")
        return 0
    finally:
        await store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Загрузка базы знаний из data/ в Qdrant (блок 5.2)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="точек в одном upsert (128–256)")
    parser.add_argument("--recreate", action="store_true", help="удалить коллекцию и создать заново")
    parser.add_argument("--no-prune", action="store_true", help="не удалять точки, которых нет в data/")
    parser.add_argument("--dry-run", action="store_true", help="только прочитать data/")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("-v", "--verbose", action="store_true", help="лог INFO: запросы к модели и Qdrant")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if args.batch_size < 1:
        parser.error("--batch-size должен быть больше 0")

    from pydantic import ValidationError

    from app.observability.logging import setup_logging
    from app.services.documents import CorpusError
    from app.services.embeddings import EmbeddingError
    from app.services.vector_store import VectorStoreError

    setup_logging("INFO" if args.verbose else "WARNING", stream=sys.stdout)
    try:
        return asyncio.run(load(args))
    except (VectorStoreError, EmbeddingError, CorpusError, ValidationError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
