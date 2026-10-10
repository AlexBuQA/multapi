"""
pgvector (блок 5.2, задача 7) без базы: SQL поиска и фильтров, текстовый вид векторов, адрес
базы, ошибки подключения, параметры сеанса, вспомогательные функции скрипта сравнения.

Поиск на настоящем Postgres с pgvector — tests/integration/test_pgvector_live.py.
"""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import asyncpg
import pytest

from app.core.config import Settings
from app.services import pgvector_store as pgv
from app.services.pgvector_store import (
    PgVectorError,
    PgVectorStore,
    PgVectorUnavailable,
    Where,
    and_,
    plan_summary,
    sql_filter,
)

ROOT = Path(__file__).resolve().parents[2]
DSN = "postgresql://multapi:secret-pass@127.0.0.1:5433/multapi"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- фильтры и SQL
def test_where_numbers_parameters_after_query_ones():
    where = Where("payload @> {} AND NOT payload @> {}", ({"category": "billing"}, {"status": "archived"}))
    assert where.render(2) == "payload @> $2 AND NOT payload @> $3"
    with pytest.raises(ValueError, match="не столько, сколько параметров"):
        Where("payload @> {}", ()).render(2)
    assert and_(None, None) is None
    joined = and_(Where("a = {}", (1,)), None, Where("b = {}", (2,)))
    assert joined.sql == "a = {} AND b = {}" and joined.params == (1, 2)


def test_sql_filter_matches_qdrant_build_filter():
    # По умолчанию, как в /kb/search, — без архивных редакций: NOT payload @> …, а не <>,
    # чтобы, как must_not в Qdrant, проходили и строки без поля status.
    default = sql_filter()
    assert default.sql == "NOT payload @> {}" and default.params == ({"status": "archived"},)
    assert sql_filter(include_archived=True) is None
    full = sql_filter(category="billing", source="faq.md", created_after="2026-09-10")
    assert full.sql == ("payload @> {} AND payload @> {} AND (payload->>'created_at')::timestamptz >= {} "
                        "AND NOT payload @> {}")
    assert full.params == ({"category": "billing"}, {"source": "faq.md"}, datetime(2026, 9, 10, tzinfo=UTC),
                           {"status": "archived"})


@pytest.mark.parametrize(("value", "expected"), [
    ("2026-09-10", datetime(2026, 9, 10, tzinfo=UTC)),
    ("2026-09-10T00:00:00Z", datetime(2026, 9, 10, tzinfo=UTC)),
    (date(2026, 9, 10), datetime(2026, 9, 10, tzinfo=UTC)),
    (datetime(2026, 9, 10, 3), datetime(2026, 9, 10, 3, tzinfo=UTC)),   # noqa: DTZ001 — наивное время -> UTC
])
def test_dates_become_aware_datetimes(value, expected):
    assert pgv.to_datetime(value) == expected


def test_search_sql_uses_index_friendly_order_by():
    store = PgVectorStore(DSN, 1024)
    sql = store.search_sql(5)
    assert sql == ("SELECT id::text AS id, payload, embedding <=> $1::vector(1024) AS distance FROM kb.documents "
                   "ORDER BY embedding <=> $1::vector(1024) LIMIT 5")
    half = store.search_sql(3, sql_filter(), half=True)
    # Для halfvec сортировка — по тому же выражению, что в индексе, иначе индекс не подходит.
    assert "ORDER BY embedding::halfvec(1024) <=> $1::halfvec(1024) LIMIT 3" in half
    assert "WHERE NOT payload @> $2" in half
    for bad in (0, 1001, "5"):
        with pytest.raises(ValueError):
            store.search_sql(bad)


def test_indexes_repeat_qdrant_hnsw():
    from app.services.vector_store import HNSW

    store = PgVectorStore(DSN, 1024)
    indexes = store.indexes()
    assert set(indexes) == {"documents_embedding_hnsw", "documents_embedding_half_hnsw", "documents_payload_gin"}
    assert indexes["documents_embedding_hnsw"].startswith("hnsw (embedding vector_cosine_ops)")
    assert indexes["documents_embedding_half_hnsw"].startswith("hnsw ((embedding::halfvec(1024)) halfvec_cosine_ops)")
    assert indexes["documents_payload_gin"] == "gin (payload jsonb_path_ops)"
    assert f"m = {HNSW.m}, ef_construction = {HNSW.ef_construct}" in indexes["documents_embedding_hnsw"]


@pytest.mark.parametrize("name", ["kb; DROP TABLE x", "KB", "kb.documents", "", "1kb"])
def test_schema_and_table_names_are_checked(name):
    with pytest.raises(ValueError, match="только латиница"):
        PgVectorStore(DSN, 8, schema=name)


def test_plan_summary():
    seq = [{"Plan": {"Node Type": "Limit", "Plans": [{"Node Type": "Sort", "Plans": [{"Node Type": "Seq Scan"}]}]}}]
    hnsw = json.dumps([{"Plan": {"Node Type": "Limit", "Plans": [
        {"Node Type": "Index Scan", "Index Name": "documents_embedding_hnsw"}]}}])
    assert plan_summary(seq) == "Limit → Sort → Seq Scan"
    assert plan_summary(hnsw) == "Limit → Index Scan (documents_embedding_hnsw)"


# ---------------------------------------------------------------- векторы и адрес
def test_vector_literal_keeps_float32_exactly():
    values = [struct.unpack("f", struct.pack("f", x))[0] for x in (0.1, -0.333333, 1e-7, 0.123456789, 1.0, 0.0)]
    text = pgv.vector_literal(values)
    assert text.startswith("[") and text.endswith("]") and " " not in text
    parsed = pgv.parse_vector(text)
    assert [struct.pack("f", x) for x in parsed] == [struct.pack("f", x) for x in values]
    assert pgv.parse_vector("[]") == []


def test_dsn_for_asyncpg_and_messages_without_password():
    assert pgv.asyncpg_dsn("postgresql+asyncpg://u:p@h:5433/db") == "postgresql://u:p@h:5433/db"
    store = PgVectorStore(DSN, 8)
    assert store.address == "127.0.0.1:5433/multapi"
    assert "secret-pass" not in store.address


def test_from_settings():
    cfg = Settings(_env_file=None, database_url="postgresql+asyncpg://multapi:pw@127.0.0.1:5433/multapi",
                   embedding_dim=1024)
    store = PgVectorStore.from_settings(cfg)
    assert store.dsn == "postgresql://multapi:pw@127.0.0.1:5433/multapi"
    assert (store.dim, store.qualified) == (1024, "kb.documents")
    with pytest.raises(PgVectorError, match="EMBEDDING_DIM не задан"):
        PgVectorStore.from_settings(cfg.model_copy(update={"embedding_dim": None}))


@pytest.mark.parametrize(("exc", "cls", "text"), [
    (ConnectionRefusedError(111, "Connect call failed"), PgVectorUnavailable, "docker compose up -d postgres"),
    (TimeoutError(), PgVectorUnavailable,
     r"Нет связи с Postgres \(127\.0\.0\.1:5433/multapi\): TimeoutError: нет ответа за 10 с"),
    (asyncpg.InvalidPasswordError("password authentication failed"), PgVectorError, "не принял пароль"),
    (asyncpg.InvalidCatalogNameError('database "multapi" does not exist'), PgVectorError, "нет такой базы"),
])
async def test_connection_errors_are_explained(monkeypatch, exc, cls, text):
    async def fail(*args, **kwargs):
        raise exc

    monkeypatch.setattr(asyncpg, "connect", fail)
    with pytest.raises(cls, match=text) as info:
        await PgVectorStore(DSN, 8).connection()
    assert "secret-pass" not in str(info.value)


class FakeConn:
    def __init__(self) -> None:
        self.sql: list[str] = []

    def is_closed(self) -> bool:
        return False

    async def execute(self, sql: str, *args) -> str:
        self.sql.append(sql)
        return "OK"


def ready_store(conn: FakeConn) -> PgVectorStore:
    store = PgVectorStore(DSN, 8)
    store._conn, store.extension_version = conn, "0.8.7"
    return store


async def test_session_settings_are_reset_even_after_error():
    conn = FakeConn()
    store = ready_store(conn)
    with pytest.raises(RuntimeError):
        async with store.settings(enable_seqscan="off", **{"hnsw.iterative_scan": "strict_order"}):
            raise RuntimeError("поиск упал")
    assert conn.sql == ["SET enable_seqscan = off", "SET hnsw.iterative_scan = strict_order",
                        "RESET enable_seqscan", "RESET hnsw.iterative_scan"]


@pytest.mark.parametrize("values", [{"work_mem": "1GB"}, {"enable_seqscan": "off; DROP TABLE kb.documents"}])
async def test_session_settings_are_whitelisted(values):
    conn = FakeConn()
    with pytest.raises(ValueError, match="не из списка"):
        async with ready_store(conn).settings(**values):
            pass
    assert conn.sql == []


async def test_wrong_dimension_is_rejected_before_sql():
    conn = FakeConn()
    store = ready_store(conn)
    with pytest.raises(PgVectorError, match="вектор из 4 чисел, а таблица kb.documents — для 8"):
        await store.upsert([("00000000-0000-0000-0000-000000000001", [0.1] * 4, {})])
    with pytest.raises(PgVectorError, match="вектор запроса: вектор из 3 чисел"):
        await store.search([0.1] * 3)
    with pytest.raises(ValueError, match="batch_size"):
        await store.upsert([], batch_size=0)
    assert conn.sql == []


def test_version_tuple():
    assert pgv.version_tuple("0.8.7") == (0, 8, 7) >= pgv.MIN_VERSION
    assert pgv.version_tuple("0.7.4") < pgv.MIN_VERSION
    assert pgv.version_tuple(None) == ()


# ---------------------------------------------------------------- скрипт сравнения
def test_compare_helpers():
    compare = load_script("compare_pgvector")
    assert compare.percentile([5.0, 1.0, 3.0, 2.0, 4.0], 95) == 5.0
    assert compare.percentile(list(map(float, range(1, 101))), 95) == 95.0
    assert compare.ms([1.0, 2.0, 3.0]) == "2.00 (p95 3.00)"
    store = PgVectorStore(DSN, 8)
    where = compare.sql_examples(date(2026, 10, 10))[1]
    shown = compare.shown_sql(store, where, 3)
    assert "WHERE (payload->>'created_at')::timestamptz >= '2026-09-10T00:00:00Z'" in shown
    assert shown.endswith("LIMIT 3;") and "FROM kb.documents" in shown


def test_compare_filters_mirror_qdrant_demo():
    """Три условия WHERE — те же фильтры, что в qdrant_filters_demo.py, в том же порядке."""
    compare = load_script("compare_pgvector")
    demo = load_script("qdrant_filters_demo")
    today = date(2026, 10, 10)
    qdrant, sql = demo.examples(today), compare.sql_examples(today)
    assert len(qdrant) == len(sql) == 3
    match, since, composite = sql
    assert qdrant[0][3].must[0].match.value == match.params[0]["source"]
    assert qdrant[1][3].must[0].range.gte == since.params[0]
    assert qdrant[2][3].must[0].match.value == composite.params[0]["category"]
    assert qdrant[2][3].must_not[0].match.value == composite.params[1]["status"]
    assert "AND NOT payload @>" in composite.sql
