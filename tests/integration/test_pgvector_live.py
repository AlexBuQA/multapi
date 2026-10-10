"""
pgvector на настоящем Postgres (блок 5.2, задача 7): таблица и индексы, upsert без дублей, поиск
и фильтры, HNSW + WHERE с итеративным сканированием, скрипты загрузки и сравнения.

    docker compose up -d postgres                # образ pgvector/pgvector из compose.yaml
    pytest tests/integration/test_pgvector_live.py -v

База — тестовая, как у тестов истории чатов: TEST_DATABASE_URL или DATABASE_URL с базой
<имя>_test. Каждый тест работает в своей схеме pytest_kb_<суффикс> и удаляет её в конце: данные
сервиса и таблицу kb.documents не трогает. Postgres не отвечает или в образе нет pgvector —
тесты пропускаются с подсказкой.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import math
import os
import sys
import uuid
from pathlib import Path

import asyncpg
import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.services.pgvector_store import PgVectorError, PgVectorStore, Where, sql_filter

DIM = 8
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")   # Qdrant :memory: в скриптах


def test_dsn() -> str:
    from sqlalchemy.engine import make_url

    from app.core.config import DatabaseSettings

    explicit = os.environ.get("TEST_DATABASE_URL")
    url = make_url(explicit or DatabaseSettings().database_url.get_secret_value())
    if not explicit:
        url = url.set(database=f"{url.database}_test")
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


test_dsn.__test__ = False   # не тест, а адрес тестовой базы
DSN = test_dsn()


async def probe() -> str | None:
    """None — можно тестировать; иначе причина пропуска."""
    from sqlalchemy.engine import make_url

    try:
        try:
            conn = await asyncpg.connect(DSN, timeout=3)
        except asyncpg.InvalidCatalogNameError:            # тестовой базы ещё нет — создаём, как тесты чата
            admin = await asyncpg.connect(make_url(DSN).set(database="postgres").render_as_string(
                hide_password=False), timeout=3)
            try:
                await admin.execute(f'CREATE DATABASE "{make_url(DSN).database}"')
            finally:
                await admin.close()
            conn = await asyncpg.connect(DSN, timeout=3)
    except Exception as exc:  # noqa: BLE001 — нет сервера, неверный пароль, нет прав
        return (f"Postgres недоступен ({type(exc).__name__}: {exc}). Запустите docker compose up -d postgres "
                "или задайте TEST_DATABASE_URL")
    try:
        if not await conn.fetchval("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"):
            return "в Postgres нет расширения pgvector: docker compose up -d postgres (образ pgvector/pgvector)"
    finally:
        await conn.close()
    return None


SKIP = asyncio.run(probe())
pytestmark = [pytestmark, pytest.mark.skipif(SKIP is not None, reason=SKIP or "")]


@pytest.fixture
def schema() -> str:
    return f"pytest_kb_{uuid.uuid4().hex[:8]}"


def drop_schema(schema: str) -> None:
    async def run():
        conn = await asyncpg.connect(DSN, timeout=3)
        try:
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        finally:
            await conn.close()

    asyncio.run(run())


@pytest.fixture
async def store(schema):
    s = PgVectorStore(DSN, DIM, schema=schema)
    yield s
    conn = await s.connection()
    await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    await s.close()


def vec(n: int) -> list[float]:
    """Векторы без равных расстояний: ось n % 8 плюс сдвиг, свой у каждого n."""
    v = [0.1] * DIM
    v[n % DIM] = 1.0
    v[(n + 1) % DIM] += 0.013 * n
    return v


def row(n: int, **payload):
    base = {"doc_id": f"D-{n}", "status": "archived" if n % 10 == 0 else "actual",
            "category": "billing" if n % 3 == 0 else "api", "created_at": f"2026-0{1 + n % 9}-01T00:00:00Z"}
    return str(uuid.UUID(int=n + 1)), vec(n), {**base, **payload}


def cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


# ---------------------------------------------------------------- таблица и строки
async def test_schema_indexes_and_idempotent_upsert(store, schema):
    await store.ensure_schema()
    await store.ensure_schema()                                 # повторный вызов ничего не меняет
    assert tuple(int(x) for x in store.extension_version.split(".")[:2]) >= (0, 8)
    conn = await store.connection()
    types = dict(await conn.fetch(
        "SELECT attname, format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = $1::regclass AND attnum > 0 AND NOT attisdropped", f"{schema}.documents"))
    assert types == {"id": "uuid", "embedding": f"vector({DIM})", "payload": "jsonb"}
    indexes = dict(await conn.fetch("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = $1", schema))
    assert "USING hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='100')" in \
        indexes["documents_embedding_hnsw"]
    assert f"((embedding)::halfvec({DIM})) halfvec_cosine_ops" in indexes["documents_embedding_half_hnsw"]
    assert "USING gin (payload jsonb_path_ops)" in indexes["documents_payload_gin"]

    rows = [row(n) for n in range(30)]
    assert await store.upsert(rows, batch_size=7) == (30, 0)
    assert await store.upsert(rows) == (0, 0)                   # то же самое — ни одной записи
    rows[5] = row(5, title="новый заголовок")
    assert await store.upsert(rows) == (0, 1)                   # изменился один payload
    assert await store.count() == 30
    assert await store.ids() == {r[0] for r in rows}
    await store.delete([rows[0][0], rows[1][0]])
    assert await store.count() == 28
    sizes = await store.sizes()
    assert set(sizes) == {"table", "documents_pkey", "documents_embedding_hnsw", "documents_embedding_half_hnsw",
                          "documents_payload_gin"}


async def test_search_matches_exact_cosine_and_filters(store):
    rows = [row(n) for n in range(40)]
    await store.upsert(rows)
    await store.analyze()
    query = vec(3)
    expected = sorted(rows, key=lambda r: -cosine(query, r[1]))[:5]
    hits = await store.search(query, 5)
    assert [h.id for h in hits] == [r[0] for r in expected]
    assert [round(h.score, 5) for h in hits] == [round(cosine(query, r[1]), 5) for r in expected]
    half = await store.search(query, 5, half=True)
    assert [h.id for h in half] == [h.id for h in hits]
    assert max(abs(a.score - b.score) for a, b in zip(hits, half, strict=True)) < 2e-3   # halfvec — ~3 знака

    no_archive = await store.search(query, 40, sql_filter())
    assert len(no_archive) == 36 and all(h.payload["status"] == "actual" for h in no_archive)
    billing = await store.search(query, 40, sql_filter(category="billing", include_archived=True))
    assert len(billing) == 14 and {h.payload["category"] for h in billing} == {"billing"}
    fresh = await store.search(query, 40, sql_filter(created_after="2026-08-01", include_archived=True))
    assert {h.payload["created_at"][:7] for h in fresh} == {"2026-08", "2026-09"}
    # must_not, как в Qdrant, пропускает строки без поля status
    await store.upsert([(str(uuid.UUID(int=999)), vec(99), {"doc_id": "NO-STATUS"})])
    assert "NO-STATUS" in {h.payload["doc_id"] for h in await store.search(query, 50, sql_filter())}


async def test_other_dimension_is_an_error(store, schema):
    await store.ensure_schema()
    other = PgVectorStore(DSN, 4, schema=schema)
    try:
        expected = r"создана для векторов из 8 чисел \(vector\(8\)\), а EMBEDDING_DIM=4.*--recreate"
        with pytest.raises(PgVectorError, match=expected):
            await other.ensure_schema()
    finally:
        await other.close()


async def test_hnsw_with_where_needs_iterative_scan(store):
    """Postgres фильтрует уже найденных HNSW кандидатов: при редком условии строк меньше
    top-k. Итеративное сканирование pgvector 0.8 добирает остальные."""
    rows = []
    for n in range(300):
        v = [0.05 * ((n * 7 + i) % 5) for i in range(DIM)]
        rare = n < 3
        v[DIM - 1 if rare else 0] += 3.0                       # редкие строки — у другой оси
        rows.append((str(uuid.UUID(int=n + 1)), v, {"doc_id": f"R-{n}", "status": "archived" if rare else "actual"}))
    await store.upsert(rows)
    await store.analyze()
    query = [1.0] + [0.0] * (DIM - 1)
    archived = Where("payload @> {}", ({"status": "archived"},))
    exact = await store.search(query, 3, archived)
    force = {"enable_seqscan": "off", "enable_bitmapscan": "off", "hnsw.ef_search": "10"}
    async with store.settings(**force):
        assert "documents_embedding_hnsw" in await store.explain(query, 3, archived)
        approximate = await store.search(query, 3, archived)
    async with store.settings(**force, **{"hnsw.iterative_scan": "strict_order"}):
        iterative = await store.search(query, 3, archived)
    assert len(exact) == 3 and len(approximate) < 3
    assert [h.id for h in iterative] == [h.id for h in exact]


# ---------------------------------------------------------------- скрипты
class TrigramBackend:
    """Модель-заглушка для скриптов: триграммы текста в 64 числах (как подставной сервер песочницы)."""

    provider, namespace, model, dimensions, local = "openai", "local:11434", "bge-m3", None, True

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * self.dim
            padded = f"  {text.lower()} "
            for i in range(len(padded) - 2):
                v[int(hashlib.md5(padded[i:i + 3].encode()).hexdigest()[:8], 16) % self.dim] += 1.0
            out.append(v)
        return out

    def close(self) -> None:
        pass


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_load_and_compare_scripts(monkeypatch, capsys, schema):
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Distance

    from app.core.config import Settings
    from app.services import embeddings
    from app.services.documents import load_chunks
    from app.services.embeddings import EmbeddingService
    from app.services.vector_store import VectorStore

    dim = 64
    client = AsyncQdrantClient(":memory:")
    service = EmbeddingService(TrigramBackend(dim), cache=None)

    def qdrant_from_settings(cls, cfg, *, collection=None, distance=Distance.COSINE, **client_options):
        return VectorStore(None, None, collection or "documents", dim, distance=distance, client=client)

    async def keep_open(self):                       # клиент :memory: общий на тест — как один процесс
        pass

    monkeypatch.setattr(VectorStore, "from_settings", classmethod(qdrant_from_settings))
    monkeypatch.setattr(VectorStore, "close", keep_open)
    monkeypatch.setattr(PgVectorStore, "from_settings", classmethod(lambda cls, cfg: cls(DSN, dim, schema=schema)))
    monkeypatch.setattr(embeddings, "default_service", lambda: service)
    monkeypatch.setenv("LLM__OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("app.core.config.get_settings", lambda: Settings(_env_file=None))
    total = len(load_chunks())
    try:
        assert load_script("load_to_qdrant").main([]) == 0
        loader = load_script("load_to_pgvector")
        assert loader.main([]) == 0
        first = capsys.readouterr().out
        assert f"новых {total}, изменено 0" in first and f"count(*): {total}" in first
        assert f"таблица {schema}.documents, vector({dim})" in first

        async def add_stale():
            store = PgVectorStore(DSN, dim, schema=schema)
            await store.upsert([(str(uuid.uuid4()), [0.1] * dim, {"doc_id": "OLD"})])
            await store.close()

        asyncio.run(add_stale())
        assert loader.main([]) == 0
        second = capsys.readouterr().out
        assert f"новых 0, изменено 0, без изменений {total}; удалено устаревших: 1" in second

        assert load_script("compare_pgvector").main(["--runs", "2", "--today", "2026-10-10"]) in (0, 1)
        out = capsys.readouterr().out
        for section in ("### Планы Postgres", "### Задержка поиска top-5", "### Совпадение top-5",
                        "### Фильтры: Filter Qdrant и WHERE в SQL", "### HNSW + WHERE", "### Размеры в Postgres"):
            assert section in out
        assert out.count("```sql") == 3 and "AND NOT payload @> '{\"status\": \"archived\"}'" in out
        server_row = next(line for line in out.splitlines() if line.startswith("| Поиск на сервере"))
        cells = server_row.rstrip(" |").split(" | ")
        assert cells[1] == "—"                                   # Qdrant в памяти: сервера нет
        assert all(float(cell) >= 0 for cell in cells[2:])       # Postgres: EXPLAIN ANALYZE
        assert "| pgvector, только HNSW + hnsw.iterative_scan = strict_order |" in out
        assert "| Вопрос | Qdrant | pgvector, план по умолчанию | pgvector, HNSW | pgvector, HNSW halfvec |" in out

        assert loader.main(["--drop"]) == 0                     # сравнение закончено — копия не нужна
        assert f"Таблица {schema}.documents удалена" in capsys.readouterr().out

        async def table_exists():
            conn = await asyncpg.connect(DSN, timeout=3)
            try:
                return await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", f"{schema}.documents")
            finally:
                await conn.close()

        assert asyncio.run(table_exists()) is False
    finally:
        from log_capture import quiet_logs

        quiet_logs()
        drop_schema(schema)


def test_compare_refuses_empty_table(monkeypatch, capsys, schema):
    """В Qdrant точки есть, а таблицу pgvector не заполняли — понятная ошибка, код 2."""
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Distance, PointStruct

    from app.core.config import Settings
    from app.services.vector_store import VectorStore

    client = AsyncQdrantClient(":memory:")

    def qdrant_from_settings(cls, cfg, *, collection=None, distance=Distance.COSINE, **client_options):
        return VectorStore(None, None, collection or "documents", DIM, distance=distance, client=client)

    async def fill():
        store = qdrant_from_settings(VectorStore, None)
        await store.ensure_collection()
        await store.upsert([PointStruct(id=str(uuid.UUID(int=1)), vector=vec(1), payload={"doc_id": "D-1"})])

    asyncio.run(fill())
    monkeypatch.setattr(VectorStore, "from_settings", classmethod(qdrant_from_settings))
    monkeypatch.setattr(PgVectorStore, "from_settings", classmethod(lambda cls, cfg: cls(DSN, DIM, schema=schema)))
    monkeypatch.setenv("LLM__OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("app.core.config.get_settings", lambda: Settings(_env_file=None))
    try:
        assert load_script("compare_pgvector").main(["--runs", "1"]) == 2
        err = capsys.readouterr().err
        assert f"В Qdrant 1 точек, в {schema}.documents — 0 строк" in err and "load_to_pgvector.py" in err
    finally:
        from log_capture import quiet_logs

        quiet_logs()
        drop_schema(schema)
