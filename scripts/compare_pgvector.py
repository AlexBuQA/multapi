"""
Qdrant против pgvector на одной базе знаний (блок 5.2, задача 7).

    python scripts/compare_pgvector.py                      # «сегодня» для фильтра по дате — текущая дата
    python scripts/compare_pgvector.py --today 2026-10-10   # как в docs/vector_store.md
    python scripts/compare_pgvector.py --runs 100           # замеров на вопрос (по умолчанию 50)

Сначала обе базы заполняются одними векторами: scripts/load_to_qdrant.py и scripts/load_to_pgvector.py.

1. Планы: как Postgres выполняет поиск top-5 (EXPLAIN) — по умолчанию и с enable_seqscan=off.
2. Задержка на 5 одинаковых вопросах (те же, что в compare_metrics.py). Вектор вопроса считается
   один раз (кеш модуля эмбеддингов), замеряется только поиск top-5 с payload — от вызова в
   Python до ответа: Qdrant (REST, как в сервисе — новое соединение на каждый запрос, см.
   VectorStore), pgvector с планом по умолчанию (одно соединение asyncpg), pgvector по HNSW и по
   HNSW над halfvec. На каждый вопрос и способ — 3 прогрева,
   затем --runs замеров; медиана и p95, мс. Отдельно — время самого поиска на сервере (у Qdrant —
   поле time ответа, у Postgres — EXPLAIN ANALYZE): разница — сеть, формат ответа и разбор в Python.
3. Совпадение top-5: pgvector против Qdrant и halfvec против полной точности.
4. Фильтры: те же три примера, что в qdrant_filters_demo.py, — Filter Qdrant против WHERE в SQL,
   top-3 обоих.
5. HNSW + WHERE: редкое условие (архивные редакции — 8 из 110) при поиске по HNSW. Postgres
   фильтрует уже найденных индексом кандидатов (hnsw.ef_search = 40), и строк может вернуться
   меньше top-k; итеративное сканирование pgvector 0.8 (hnsw.iterative_scan) добирает остальные.
6. Размеры таблицы и индексов в Postgres.

Вывод — Markdown для docs/vector_store.md. Код выхода: 0 — top-5 pgvector (план по умолчанию)
совпал с Qdrant на всех вопросах, 1 — нет, 2 — ошибка.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

TOP_K = 5
FILTER_TOP_K = 3
WARMUP = 3
DEFAULT_RUNS = 50
SERVER_RUNS = 20
FORCE_INDEX = {"enable_seqscan": "off", "enable_bitmapscan": "off"}   # только HNSW: без перебора и GIN
RARE_QUERY = "Как включить вход по отпечатку пальца в приложении?"
WAYS = {
    "qdrant": "Qdrant",
    "pg": "pgvector, план по умолчанию",
    "pg_hnsw": "pgvector, HNSW",
    "pg_half": "pgvector, HNSW halfvec",
}


# ---------------------------------------------------------------- вывод
def ms(values: list[float]) -> str:
    return f"{statistics.median(values):.2f} (p95 {percentile(values, 95):.2f})"


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def doc_ids(hits) -> list[str]:
    return [str((hit.payload or {}).get("doc_id")) for hit in hits]


def score_delta(reference, hits) -> float:
    """Наибольшая разница score одного и того же фрагмента в двух выдачах."""
    scores = {str(hit.payload.get("doc_id")): hit.score for hit in reference}
    return max((abs(scores[doc] - hit.score) for hit in hits if (doc := str(hit.payload.get("doc_id"))) in scores),
               default=0.0)


def literal(value) -> str:
    """Параметр запроса так, как его написали бы в SQL руками (только для показа в документации)."""
    if isinstance(value, dict):
        return "'" + json.dumps(value, ensure_ascii=False) + "'"
    if isinstance(value, datetime):
        return "'" + value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") + "'"
    return repr(value)


def shown_sql(store, where, top_k: int) -> str:
    """SELECT для документации: $1 — вектор вопроса, остальные параметры подставлены."""
    condition = where.sql
    for value in where.params:
        condition = condition.replace("{}", literal(value), 1)
    return (f"SELECT id, payload, 1 - (embedding <=> $1) AS score\n"
            f"FROM {store.qualified}\n"
            f"WHERE {condition}\n"
            f"ORDER BY embedding <=> $1\n"
            f"LIMIT {top_k};")


def sql_examples(today: date):
    """Те же три фильтра, что examples() в qdrant_filters_demo.py, — условиями WHERE."""
    from app.services.pgvector_store import Where

    since = datetime(today.year, today.month, today.day, tzinfo=UTC) - timedelta(days=30)
    return [
        Where("payload @> {}", ({"source": "release_notes.md"},)),
        Where("(payload->>'created_at')::timestamptz >= {}", (since,)),
        Where("payload @> {}\n  AND NOT payload @> {}", ({"category": "billing"}, {"status": "archived"})),
    ]


# ---------------------------------------------------------------- замеры
async def timed(call) -> tuple[float, list]:
    started = time.perf_counter()
    hits = await call()
    return (time.perf_counter() - started) * 1000, hits


async def measure(way: str, qdrants, pg, vector, runs: int) -> tuple[list[float], list]:
    """Прогрев и runs замеров одного способа на одном векторе; hits — из последнего замера.
    qdrants — {"qdrant": VectorStore}."""
    async def call():
        if way in qdrants:
            return await qdrants[way].search(vector, top_k=TOP_K)
        return await pg.search(vector, TOP_K, half=way == "pg_half")

    async def series():
        for _ in range(WARMUP):
            await call()
        times, hits = [], []
        for _ in range(runs):
            elapsed, hits = await timed(call)
            times.append(elapsed)
        return times, hits

    if way in ("pg_hnsw", "pg_half"):
        async with pg.settings(**FORCE_INDEX):
            return await series()
    return await series()


async def server_ms(way: str, qdrants, pg, vector, runs: int) -> list[float] | None:
    """Время самого поиска на сервере, мс: у Qdrant — поле time ответа REST, у Postgres —
    Execution Time из EXPLAIN ANALYZE. None — Qdrant в памяти процесса (тесты), сервера нет."""
    if way in qdrants:
        from qdrant_client.models import QueryRequest

        qdrant = qdrants[way]
        try:
            api = qdrant.client.http
        except NotImplementedError:
            return None
        request = QueryRequest(query=list(vector), limit=TOP_K, with_payload=True)
        result = []
        for _ in range(WARMUP + runs):
            response = await api.search_api.query_points(collection_name=qdrant.collection, query_request=request)
            result.append(float(response.time) * 1000)
        return result[WARMUP:]

    async def series():
        return [await pg.execution_ms(vector, TOP_K, half=way == "pg_half") for _ in range(WARMUP + runs)][WARMUP:]

    if way in ("pg_hnsw", "pg_half"):
        async with pg.settings(**FORCE_INDEX):
            return await series()
    return await series()


def median_cell(values: list[float] | None) -> str:
    return "—" if not values else f"{statistics.median(values):.2f}"


async def compare(args: argparse.Namespace) -> int:
    from compare_metrics import QUERIES
    from qdrant_filters_demo import examples

    from app.core.config import get_settings
    from app.services import embeddings
    from app.services.pgvector_store import PgVectorError, PgVectorStore, Where
    from app.services.vector_store import VectorStore, VectorStoreError

    cfg = get_settings()
    qdrant = VectorStore.from_settings(cfg)
    qdrants = {"qdrant": qdrant}
    pg = PgVectorStore.from_settings(cfg)
    try:
        await qdrant.ensure_collection()
        await pg.ensure_schema()
        points, rows = await qdrant.points_count(), await pg.count()
        if not points:
            raise VectorStoreError(f"Коллекция {qdrant.collection} пуста: сначала python scripts/load_to_qdrant.py")
        if rows != points:
            raise PgVectorError(f"В Qdrant {points} точек, в {pg.qualified} — {rows} строк: сначала "
                                "python scripts/load_to_pgvector.py (те же фрагменты и векторы)")
        print(f"Qdrant {qdrant.server_version}: {qdrant.collection}, {points} точек. Postgres {pg.server_version} "
              f"+ pgvector {pg.extension_version}: {pg.qualified}, {rows} строк. vector({pg.dim}), косинус, "
              f"HNSW m=16 / ef_construction=100 у обоих. Замеров на вопрос: {args.runs} (+{WARMUP} прогрева).\n")
        if urlsplit(qdrant.url).hostname == "localhost":
            print("Внимание: QDRANT_URL с localhost. В Windows localhost сначала пробует IPv6 (::1), где Qdrant "
                  "не слушает, и каждое новое соединение ждёт ~250 мс; задайте http://127.0.0.1:6333.\n")
        vectors = {query: await asyncio.to_thread(embeddings.embed_query, query) for query in QUERIES}

        # 1. Планы
        sample = vectors[QUERIES[0]]
        print("### Планы Postgres (top-5, 110 строк)\n")
        print(f"- по умолчанию: `{await pg.explain(sample, TOP_K)}`")
        async with pg.settings(**FORCE_INDEX):
            print(f"- с enable_seqscan=off: `{await pg.explain(sample, TOP_K)}`")
            print(f"- halfvec с enable_seqscan=off: `{await pg.explain(sample, TOP_K, half=True)}`\n")

        # 2–3. Задержка и совпадение
        times: dict[str, dict[str, list[float]]] = {way: {} for way in WAYS}
        hits: dict[str, dict[str, list]] = {way: {} for way in WAYS}
        for query in QUERIES:
            for way in WAYS:
                times[way][query], hits[way][query] = await measure(way, qdrants, pg, vectors[query], args.runs)
        print("### Задержка поиска top-5, мс: медиана (p95)\n")
        print("| Вопрос | " + " | ".join(WAYS.values()) + " |")
        print("|---|" + "---|" * len(WAYS))
        for query in QUERIES:
            print(f"| {query} | " + " | ".join(ms(times[way][query]) for way in WAYS) + " |")
        print("| **Все вопросы** | " + " | ".join(
            f"**{ms([t for query in QUERIES for t in times[way][query]])}**" for way in WAYS) + " |\n")

        server: dict[str, list[float] | None] = {}
        for way in WAYS:
            parts = [await server_ms(way, qdrants, pg, vectors[query], SERVER_RUNS) for query in QUERIES]
            server[way] = None if any(part is None for part in parts) else [t for part in parts for t in part]
        total = {way: [t for query in QUERIES for t in times[way][query]] for way in WAYS}
        print(f"Из чего складывается задержка, мс (медиана по всем вопросам; на сервере — {SERVER_RUNS} замеров "
              "на вопрос):\n")
        print("| | " + " | ".join(WAYS.values()) + " |")
        print("|---|" + "---|" * len(WAYS))
        print("| От вызова в Python до ответа | " + " | ".join(median_cell(total[way]) for way in WAYS) + " |")
        print("| Поиск на сервере (Qdrant: `time` ответа; Postgres: EXPLAIN ANALYZE) | "
              + " | ".join(median_cell(server[way]) for way in WAYS) + " |")
        print("| Остальное: соединение, формат ответа, разбор в Python | " + " | ".join(
            "—" if not server[way] else f"{statistics.median(total[way]) - statistics.median(server[way]):.2f}"
            for way in WAYS) + " |\n")

        print("### Совпадение top-5\n")
        print("| Вопрос | top-5 Qdrant | pgvector, по умолчанию | pgvector, HNSW | HNSW halfvec | "
              "max Δ score: pgvector / halfvec |")
        print("|---|---|---|---|---|---|")
        same_all = True
        for query in QUERIES:
            base = doc_ids(hits["qdrant"][query])
            cells = []
            for way in ("pg", "pg_hnsw", "pg_half"):
                found = doc_ids(hits[way][query])
                cells.append("= Qdrant" if found == base else "**" + ", ".join(found) + "**")
            same_all &= doc_ids(hits["pg"][query]) == base
            deltas = " / ".join(f"{score_delta(hits['qdrant'][query], hits[way][query]):.5f}"
                                for way in ("pg", "pg_half"))
            print(f"| {query} | {', '.join(base)} | " + " | ".join(cells) + f" | {deltas} |")
        print()

        # 4. Фильтры
        print(f"### Фильтры: Filter Qdrant и WHERE в SQL (сегодня {args.today.isoformat()}, top-{FILTER_TOP_K})\n")
        filters_same = 0
        for number, ((title, query, code, query_filter), where) in enumerate(
                zip(examples(args.today), sql_examples(args.today), strict=True), start=1):
            vector = await asyncio.to_thread(embeddings.embed_query, query)
            q_hits = await qdrant.search(vector, top_k=FILTER_TOP_K, query_filter=query_filter)
            p_hits = await pg.search(vector, FILTER_TOP_K, where)
            same = doc_ids(q_hits) == doc_ids(p_hits)
            filters_same += same
            print(f"#### {number}. {title}\n\nВопрос: «{query}»\n")
            print("```python\n" + code + "\n```\n")
            print("```sql\n" + shown_sql(pg, where, FILTER_TOP_K) + "\n```\n")
            print("| # | Qdrant | score | pgvector | score |\n|---|---|---|---|---|")
            for i in range(max(len(q_hits), len(p_hits))):
                q = q_hits[i] if i < len(q_hits) else None
                p = p_hits[i] if i < len(p_hits) else None
                print(f"| {i + 1} | {q.payload.get('doc_id') if q else '—'} | {f'{q.score:.3f}' if q else ''} | "
                      f"{p.payload.get('doc_id') if p else '—'} | {f'{p.score:.3f}' if p else ''} |")
            print(f"\nСовпало: {'да' if same else '**нет**'}\n")

        # 5. HNSW + WHERE
        rare = Where("payload @> {}", ({"status": "archived"},))
        vector = await asyncio.to_thread(embeddings.embed_query, RARE_QUERY)
        print(f"### HNSW + WHERE: только архивные редакции (8 из {rows}), top-{TOP_K}\n")
        print(f"Вопрос: «{RARE_QUERY}»\n")
        print("| Способ | План | Строк | doc_id |\n|---|---|---|---|")
        variants = [
            ("Qdrant, Filter status = archived", None, None),
            ("pgvector, план по умолчанию", {}, rare),
            ("pgvector, только HNSW, hnsw.ef_search = 40", FORCE_INDEX, rare),
            ("pgvector, только HNSW + hnsw.iterative_scan = strict_order",
             {**FORCE_INDEX, "hnsw.iterative_scan": "strict_order"}, rare),
        ]
        rare_counts = []
        for label, settings, where in variants:
            if settings is None:
                found = await qdrant.search(vector, top_k=TOP_K, query_filter=_archived_only())
                plan = "фильтр внутри поиска"
            else:
                async with pg.settings(**settings):
                    plan = f"`{await pg.explain(vector, TOP_K, where)}`"
                    found = await pg.search(vector, TOP_K, where)
            rare_counts.append(len(found))
            print(f"| {label} | {plan} | {len(found)} | {', '.join(doc_ids(found))} |")
        print()

        # 6. Размеры
        sizes = await pg.sizes()
        print("### Размеры в Postgres\n\n| Объект | КБ |\n|---|---|")
        for name, size in sizes.items():
            print(f"| {'таблица (с TOAST)' if name == 'table' else name} | {size / 1024:.0f} |")
        print()

        print(f"Итог: top-5 pgvector (план по умолчанию) {'совпал' if same_all else 'НЕ совпал'} с Qdrant на "
              f"{'всех' if same_all else 'части'} {len(QUERIES)} вопросах; фильтры совпали в {filters_same} из 3; "
              f"HNSW + WHERE без итеративного сканирования вернул {rare_counts[2]} из {TOP_K} строк, с ним — "
              f"{rare_counts[3]}.")
        return 0 if same_all else 1
    finally:
        await qdrant.close()
        await pg.close()


def _archived_only():
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    return Filter(must=[FieldCondition(key="status", match=MatchValue(value="archived"))])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qdrant против pgvector: задержка, совпадение, фильтры (блок 5.2)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="замеров на вопрос и способ")
    parser.add_argument("--today", type=date.fromisoformat, default=datetime.now().astimezone().date(),
                        help="дата «сегодня» для фильтра по дате, например 2026-10-10")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs должен быть больше 0")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from pydantic import ValidationError

    from app.observability.logging import setup_logging
    from app.services.embeddings import EmbeddingError
    from app.services.pgvector_store import PgVectorError
    from app.services.vector_store import VectorStoreError

    setup_logging("WARNING", stream=sys.stdout)
    try:
        return asyncio.run(compare(args))
    except (VectorStoreError, PgVectorError, EmbeddingError, ValidationError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
