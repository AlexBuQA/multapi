"""
Cosine против Dot product на одних и тех же векторах (блок 5.2, задача 5).

    python scripts/compare_metrics.py           # сравнить и удалить временные коллекции
    python scripts/compare_metrics.py --keep    # оставить documents_cosine и documents_dot (посмотреть в дашборде)

1. Фрагменты — из боевой коллекции (QDRANT_COLLECTION, по умолчанию documents; её заполняет
   scripts/load_to_qdrant.py), векторы — заново от модуля эмбеддингов блока 5.1 (из его
   кеша, модель не вызывается). Не из Qdrant: коллекция COSINE нормирует векторы сама при
   записи, и по ней не видно, нормирует ли их наш модуль.
2. Длины векторов: у нормированных все равны 1 — тогда скалярное произведение и косинус
   совпадают, и ранжирование COSINE и DOT обязано совпасть (sanity check).
3. Временные коллекции documents_cosine (COSINE) и documents_dot (DOT) с этими векторами;
   пять вопросов пользователей (embed_query, блок 5.1) — top-5 в каждой; таблица Markdown
   «вопрос | top-5 cosine | top-5 dot | совпало ли ранжирование» — для docs/vector_store.md.
4. Контрольный опыт без Qdrant: те же векторы, умноженные на разные числа (так выглядели бы
   ненормированные векторы), — ранжирование по скалярному произведению на них меняется, по
   косинусу — нет. Это объясняет, зачем модуль эмбеддингов нормирует векторы.
5. Временные коллекции удаляются (кроме --keep): в проде живёт одна боевая коллекция.

Код выхода: 0 — ранжирование совпало на всех вопросах, 1 — нет (векторы не нормированы),
2 — ошибка (Qdrant, модель, пустая коллекция).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Реальные вопросы пользователей: golden dataset блока 3.7 и вопросы боту в Telegram.
QUERIES = (
    "Как сменить пароль?",
    "Письмо для подтверждения не пришло, что делать?",
    "API возвращает ошибку 429. Что это значит?",
    "Можно ли вернуть деньги за тариф?",
    "Как включить вход по отпечатку пальца в приложении?",
)
TOP_K = 5
TEMP = ("documents_cosine", "documents_dot")


def norm(vector) -> float:
    return math.sqrt(math.fsum(x * x for x in vector))


def scale_for(point_id: str) -> float:
    """Детерминированный «масштаб» 0.5–2.0 для контрольного опыта с ненормированными векторами."""
    return 0.5 + int(hashlib.md5(point_id.encode()).hexdigest()[:4], 16) / 0xFFFF * 1.5


def rank(query, docs, metric: str, scales=None) -> list[str]:
    """top-k doc_id перебором в Python: dot или cosine; docs — [(id, doc_id, вектор)], векторы
    опционально домножены на scales[id]."""
    scored = []
    for point_id, doc_id, vector in docs:
        scaled = [x * scales[point_id] for x in vector] if scales else vector
        dot = math.fsum(q * v for q, v in zip(query, scaled, strict=True))
        score = dot if metric == "dot" else dot / (norm(query) * norm(scaled))
        scored.append((score, doc_id))
    return [doc_id for _, doc_id in sorted(scored, key=lambda item: -item[0])[:TOP_K]]


def table(rows) -> str:
    lines = ["| Вопрос | top-5 cosine | top-5 dot | Совпало |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['query']} | {', '.join(row['cosine'])} | {', '.join(row['dot'])} | "
                     f"{'да' if row['same'] else '**нет**'} |")
    return "\n".join(lines)


async def compare(args: argparse.Namespace) -> int:
    from qdrant_client.models import Distance, PointStruct

    from app.core.config import get_settings
    from app.services import embeddings
    from app.services.vector_store import VectorStore, VectorStoreError

    cfg = get_settings()
    source = VectorStore.from_settings(cfg)
    stores = {"cosine": VectorStore.from_settings(cfg, collection=TEMP[0], distance=Distance.COSINE),
              "dot": VectorStore.from_settings(cfg, collection=TEMP[1], distance=Distance.DOT)}
    try:
        records = await source.scroll()
        if not records:
            raise VectorStoreError(f"Коллекция {source.collection} пуста: сначала python scripts/load_to_qdrant.py")
        vectors = await asyncio.to_thread(embeddings.embed_documents, [record.payload["text"] for record in records])
        norms = [norm(vector) for vector in vectors]
        print(f"Фрагментов в {source.collection}: {len(records)}; длина векторов модуля эмбеддингов "
              f"от {min(norms):.6f} "
              f"до {max(norms):.6f}" + (" — нормированы" if max(abs(n - 1) for n in norms) < 1e-3
                                         else " — НЕ нормированы"), flush=True)
        points = [PointStruct(id=record.id, vector=vector, payload=record.payload)
                  for record, vector in zip(records, vectors, strict=True)]
        docs = [(str(record.id), record.payload["doc_id"], vector)
                for record, vector in zip(records, vectors, strict=True)]
        for store in stores.values():
            await store.drop()                      # остатки прошлого запуска с --keep
            await store.ensure_collection()
            await store.upsert(points)
        rows = []
        for query in QUERIES:
            vector = await asyncio.to_thread(embeddings.embed_query, query)
            found = {name: await store.search(vector, top_k=TOP_K) for name, store in stores.items()}
            ids = {name: [point.payload["doc_id"] for point in hits] for name, hits in found.items()}
            top = {name: round(hits[0].score, 4) for name, hits in found.items()}
            rows.append({"query": query, **ids, "same": ids["cosine"] == ids["dot"], "top_scores": top})
        print()
        print(table(rows))
        print()
        for row in rows:
            print(f"  {row['query']}: score top-1 cosine {row['top_scores']['cosine']}, dot {row['top_scores']['dot']}")
        same = all(row["same"] for row in rows)
        print(f"\nРанжирование COSINE и DOT {'совпало на всех' if same else 'НЕ совпало на части'} "
              f"{len(rows)} вопросах.")

        scales = {point_id: scale_for(point_id) for point_id, _, _ in docs}
        cos_same = dot_same = 0
        for query in QUERIES:
            vector = await asyncio.to_thread(embeddings.embed_query, query)
            baseline = rank(vector, docs, "cosine")
            cos_same += rank(vector, docs, "cosine", scales) == baseline      # косинус от длины не зависит
            dot_same += rank(vector, docs, "dot", scales) == baseline
        print(f"Контрольный опыт (векторы домножены на 0.5–2.0, как ненормированные; расчёт в Python): "
              f"ранжирование как у исходных — cosine на {cos_same} из {len(QUERIES)}, dot на {dot_same} из "
              f"{len(QUERIES)} вопросов.")
        if not args.keep:
            for store in stores.values():
                await store.drop()
            print(f"Временные коллекции удалены: {', '.join(TEMP)}. Боевая — {source.collection} (COSINE).")
        return 0 if same else 1
    finally:
        for store in (source, *stores.values()):
            await store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cosine против Dot product на векторах боевой коллекции (блок 5.2)")
    parser.add_argument("--keep", action="store_true", help="не удалять documents_cosine и documents_dot")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from pydantic import ValidationError

    from app.observability.logging import setup_logging
    from app.services.embeddings import EmbeddingError
    from app.services.vector_store import VectorStoreError

    setup_logging("WARNING", stream=sys.stdout)
    try:
        return asyncio.run(compare(args))
    except (VectorStoreError, EmbeddingError, ValidationError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
