"""
Векторная база (блок 5.2): VectorStore, база знаний, загрузка, cosine/dot, фильтры, /kb/search.

Qdrant — в режиме :memory: (AsyncQdrantClient(location=":memory:")): тот же клиент и те же
модели запросов, но в процессе, без сети и сервера. Payload-индексов в нём нет (Qdrant
предупреждает об этом) — их создание проверяет живой тест tests/integration/test_vector_store_live.py
на настоящем Qdrant. Эмбеддинги — FakeBackend из тестов блока 5.1 (нормированные векторы без модели).
"""
from __future__ import annotations

import importlib.util
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, ScoredPoint

from app.core.config import Settings
from app.services import documents, embeddings
from app.services import vector_store as vs
from app.services.documents import CorpusError, load_chunks, point_id
from app.services.embeddings import EmbeddingService
from app.services.vector_store import (
    HNSW,
    PAYLOAD_INDEXES,
    VectorStore,
    VectorStoreError,
    VectorStoreUnavailable,
    build_filter,
)

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

ROOT = Path(__file__).resolve().parents[2]
DIM = 8


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def memory_store(dim: int = DIM, collection: str = "documents", distance: Distance = Distance.COSINE,
                 client: AsyncQdrantClient | None = None) -> VectorStore:
    return VectorStore(None, None, collection, dim, distance=distance, client=client or AsyncQdrantClient(":memory:"))


def vector(seed: int, dim: int = DIM) -> list[float]:
    return [1.0 if i == seed % dim else 0.1 for i in range(dim)]


def point(n: int, dim: int = DIM, **payload) -> PointStruct:
    return PointStruct(id=str(uuid.UUID(int=n + 1)), vector=vector(n, dim),
                       payload={"doc_id": f"D-{n}", "status": "actual", **payload})


class HashBackend:
    """Модель-заглушка для скриптов: вектор из букв текста, у одинаковых текстов — одинаковый."""

    provider, namespace, model, dimensions, local = "openai", "local:11434", "bge-m3", None, True

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        out = []
        for text in texts:
            v = [0.01] * self.dim
            for ch in text.lower():
                v[ord(ch) % self.dim] += 1.0
            out.append(v)
        return out

    def close(self) -> None:
        pass


# ---------------------------------------------------------------- коллекция
async def test_ensure_collection_creates_with_dim_cosine_and_hnsw():
    store = memory_store()
    await store.ensure_collection()
    await store.ensure_collection()                       # повторно — ничего не меняет
    info = await store.collection_info()
    assert (info.config.params.vectors.size, info.config.params.vectors.distance) == (DIM, Distance.COSINE)
    assert (info.config.hnsw_config.m, info.config.hnsw_config.ef_construct) == (16, 100)
    assert (HNSW.m, HNSW.ef_construct) == (16, 100)
    assert store.server_version and await store.points_count() == 0
    await store.close()


async def test_payload_indexes_cover_filter_fields():
    assert PAYLOAD_INDEXES == {"source": "keyword", "created_at": "datetime", "category": "keyword",
                               "product": "keyword", "status": "keyword"}
    client = AsyncQdrantClient(":memory:")
    created: list[tuple[str, str]] = []
    original = client.create_payload_index

    async def spy(collection_name, field_name, field_schema=None, **kwargs):
        created.append((field_name, field_schema))
        return await original(collection_name, field_name, field_schema, **kwargs)

    client.create_payload_index = spy
    store = memory_store(client=client)
    await store.ensure_collection()
    assert dict(created) == PAYLOAD_INDEXES
    await store.close()


async def test_existing_collection_with_other_dim_is_an_error():
    client = AsyncQdrantClient(":memory:")
    await memory_store(dim=DIM, client=client).ensure_collection()
    with pytest.raises(VectorStoreError, match=r"создана для векторов из 8 чисел, а EMBEDDING_DIM=1024.*--recreate"):
        await memory_store(dim=1024, client=client).ensure_collection()
    with pytest.raises(VectorStoreError, match="метрикой Cosine, а нужна Dot"):
        await memory_store(client=client, distance=Distance.DOT).ensure_collection()
    await client.close()


async def test_version_mismatch_is_logged():
    from types import SimpleNamespace

    from log_capture import captured_logs, events

    store = memory_store()

    async def old_server():
        return SimpleNamespace(version="1.12.0")

    store.client.info = old_server
    with captured_logs() as logs:
        await store.ensure_collection()
    [warning] = events(logs, "qdrant_version_mismatch")
    assert warning["server"] == "1.12.0" and warning["client"].startswith("1.")
    await store.close()


# ---------------------------------------------------------------- точки
async def test_upsert_batches_wait_only_on_last():
    client = AsyncQdrantClient(":memory:")
    calls: list[tuple[int, bool]] = []
    original = client.upsert

    async def spy(collection_name, points, wait=True, **kwargs):
        calls.append((len(points), wait))
        return await original(collection_name, points=points, wait=wait, **kwargs)

    client.upsert = spy
    store = memory_store(client=client)
    await store.ensure_collection()
    progress: list[int] = []
    await store.upsert([point(n) for n in range(7)], batch_size=3, on_batch=progress.append)
    assert calls == [(3, False), (3, False), (1, True)] and progress == [3, 3, 1]
    await store.upsert([point(n) for n in range(7)])        # те же id — перезапись, не копии
    assert calls[-1] == (7, True) and await store.points_count() == 7
    await store.close()


async def test_wrong_dim_is_rejected_before_sending():
    client = AsyncQdrantClient(":memory:")
    store = memory_store(client=client)
    await store.ensure_collection()
    bad = point(1, dim=DIM + 1)
    with pytest.raises(VectorStoreError, match=rf"точка {bad.id}: вектор из 9 чисел, а коллекция documents — для 8"):
        await store.upsert([point(0), bad])
    assert await store.points_count() == 0                  # первая точка тоже не ушла
    with pytest.raises(VectorStoreError, match="вектор запроса: вектор из 3 чисел"):
        await store.search([1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        await store.upsert([point(0)], batch_size=0)
    await store.close()


async def test_search_returns_scored_points_with_payload_and_filters():
    store = memory_store()
    await store.ensure_collection()
    await store.upsert([point(0, category="billing", source="faq.md", created_at="2026-09-29T00:00:00Z"),
                        point(1, category="billing", source="faq.md", created_at="2025-01-01T00:00:00Z",
                              status="archived"),
                        point(2, category="api", source="release_notes.md", created_at="2026-10-01T00:00:00Z")])
    hits = await store.search(vector(1), top_k=3)
    assert all(isinstance(hit, ScoredPoint) for hit in hits) and hits[0].payload["doc_id"] == "D-1"
    billing = await store.search(vector(1), query_filter=build_filter(category="billing"))
    assert [hit.payload["doc_id"] for hit in billing] == ["D-0"]                  # архивная исключена
    with_archive = await store.search(vector(1), query_filter=build_filter(category="billing", include_archived=True))
    assert {hit.payload["doc_id"] for hit in with_archive} == {"D-0", "D-1"}
    fresh = await store.search(vector(1), query_filter=build_filter(created_after="2026-09-10", include_archived=True))
    assert {hit.payload["doc_id"] for hit in fresh} == {"D-0", "D-2"}
    by_source = await store.search(vector(0), query_filter=build_filter(source="release_notes.md"))
    assert [hit.payload["doc_id"] for hit in by_source] == ["D-2"]
    await store.close()


def test_build_filter_shapes():
    assert build_filter(include_archived=True) is None
    only_archive_rule = build_filter()
    assert only_archive_rule.must is None and only_archive_rule.must_not[0].key == "status"
    full = build_filter(category="billing", source="faq.md", created_after="2026-09-10")
    assert [c.key for c in full.must] == ["category", "source", "created_at"]


async def test_scroll_ids_and_delete():
    store = memory_store()
    await store.ensure_collection()
    await store.upsert([point(n) for n in range(5)])
    ids = await store.point_ids()
    assert ids == {str(uuid.UUID(int=n + 1)) for n in range(5)}
    await store.delete(sorted(ids)[:2])
    await store.delete([])
    assert await store.points_count() == 3
    await store.drop()
    assert not await store.client.collection_exists("documents")
    await store.close()


# ---------------------------------------------------------------- ошибки Qdrant
class Failing:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def collection_exists(self, name):
        raise self.exc

    async def info(self):
        raise self.exc

    async def get_collection(self, name):
        raise self.exc

    async def close(self):
        pass


@pytest.mark.parametrize("exc, cls, text", [
    (ResponseHandlingException(httpx.ConnectError("refused")), VectorStoreUnavailable,
     r"Нет связи с Qdrant \(http://qdrant\.invalid:6333\): ConnectError: refused\. .*docker compose up -d qdrant"),
    # Тайм-аут httpx без текста: в сообщении — его тип, а не пустое место перед точкой.
    (ResponseHandlingException(httpx.ReadTimeout("")), VectorStoreUnavailable, r"\): ReadTimeout — нет ответа\. "),
    (UnexpectedResponse(401, "Unauthorized", b"Invalid API key or JWT", httpx.Headers()), VectorStoreError,
     r"не принял ключ \(401\): QDRANT_API_KEY"),
    (UnexpectedResponse(500, "Internal", b'{"status": {"error": "disk full"}}', httpx.Headers()), VectorStoreError,
     "Qdrant ответил 500: .*disk full"),
])
async def test_errors_are_explained(exc, cls, text):
    store = VectorStore("http://qdrant.invalid:6333", None, "documents", DIM, client=Failing(exc))
    with pytest.raises(cls, match=text) as caught:
        await store.ensure_collection()
    assert caught.value.__cause__ is exc


def test_client_is_created_once_and_without_insecure_warning_for_local(recwarn):
    store = VectorStore("http://localhost:6333", "key-1234567890", "documents", DIM)
    assert isinstance(store.client, AsyncQdrantClient)
    assert not [w for w in recwarn if "insecure" in str(w.message)]
    with pytest.warns(UserWarning, match="insecure connection"):
        VectorStore("http://qdrant.example.com:6333", "key-1234567890", "documents", DIM)   # внешний — нужен https
    with pytest.raises(VectorStoreError, match="QDRANT_URL не задан"):
        VectorStore(None, None, "documents", DIM)


@pytest.mark.parametrize(("url", "keepalive"), [
    ("http://qdrant:6333", 0),                   # app в compose
    ("http://127.0.0.1:6333", 0),
    ("http://192.168.1.20:6333", 0),
    ("https://xyz.cloud.qdrant.io:6333", None),  # внешний по https — пул qdrant-client по умолчанию
])
def test_local_qdrant_gets_new_connection_per_request(monkeypatch, url, keepalive):
    """Keep-alive к Qdrant по HTTP добавлял ~40 мс к каждому ответу (delayed ACK + Нейгл):
    для локальных адресов — новое соединение на каждый запрос, как qdrant-client делает для localhost."""
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(vs, "AsyncQdrantClient", Recorder)
    VectorStore(url, "key-1234567890", "documents", DIM)
    limits = seen.get("limits")
    assert (limits.max_keepalive_connections if limits else None) == keepalive
    assert seen["check_compatibility"] is False
    explicit = httpx.Limits(max_keepalive_connections=4)
    VectorStore(url, "key-1234567890", "documents", DIM, limits=explicit)
    assert seen["limits"] is explicit                            # свои limits не перезаписываются


def test_from_settings(clean_env):
    cfg = Settings(_env_file=None, qdrant_url="http://localhost:6333/", qdrant_api_key="k" * 20, embedding_dim=1024)
    store = VectorStore.from_settings(cfg)
    assert (store.url, store.collection, store.dim) == ("http://localhost:6333", "documents", 1024)
    other = VectorStore.from_settings(cfg, collection="documents_dot", distance=Distance.DOT)
    assert (other.collection, other.distance) == ("documents_dot", Distance.DOT)
    with pytest.raises(VectorStoreError, match="QDRANT_URL и EMBEDDING_DIM"):
        VectorStore.from_settings(Settings(_env_file=None))


# ---------------------------------------------------------------- настройки
@pytest.fixture
def clean_env(monkeypatch):
    """src/config.py загружает .env разработчика в окружение — его значения тестам не нужны."""
    for name in ("EMBEDDING_DIM", "QDRANT_COLLECTION", "QDRANT_API_KEY", "EMBEDDINGS__DIMENSIONS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QDRANT_URL", "none")


def test_settings_defaults_and_validation(clean_env):
    cfg = Settings(_env_file=None)
    assert (cfg.qdrant_url, cfg.qdrant_collection, cfg.embedding_dim) == (None, "documents", None)
    assert Settings(_env_file=None, qdrant_url="none").qdrant_url is None
    with pytest.raises(ValueError, match="EMBEDDING_DIM — нет"):
        Settings(_env_file=None, qdrant_url="http://localhost:6333")
    with pytest.raises(ValueError, match="EMBEDDINGS__DIMENSIONS=512, а EMBEDDING_DIM=1536"):
        Settings(_env_file=None, qdrant_url="http://q:6333", embedding_dim=1536, embeddings={"dimensions": 512})
    with pytest.raises(ValueError, match="QDRANT_URL: ожидается"):
        Settings(_env_file=None, qdrant_url="localhost:6333", embedding_dim=8)
    with pytest.raises(ValueError, match="QDRANT_COLLECTION"):
        Settings(_env_file=None, qdrant_collection="docs.v2")


def test_settings_from_environment(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "secret-key-123456")
    monkeypatch.setenv("QDRANT_COLLECTION", "kb")
    monkeypatch.setenv("EMBEDDING_DIM", "1024")
    cfg = Settings(_env_file=None)
    assert (cfg.qdrant_url, cfg.qdrant_collection, cfg.embedding_dim) == ("http://qdrant:6333", "kb", 1024)
    assert "secret-key" not in repr(cfg)


# ---------------------------------------------------------------- lifespan и DI
async def test_open_vector_store(monkeypatch, clean_env):
    from log_capture import captured_logs, events

    assert await vs.open_vector_store(Settings(_env_file=None)) is None
    cfg = Settings(_env_file=None, qdrant_url="http://localhost:6333", embedding_dim=DIM)
    monkeypatch.setattr(VectorStore, "from_settings", classmethod(lambda cls, cfg: memory_store()))
    store = await vs.open_vector_store(cfg)
    assert await store.client.collection_exists("documents")
    await store.close()

    down = VectorStore("http://localhost:6333", None, "documents", DIM,
                       client=Failing(ResponseHandlingException(httpx.ConnectError("refused"))))
    monkeypatch.setattr(VectorStore, "from_settings", classmethod(lambda cls, cfg: down))
    with captured_logs() as logs:
        assert await vs.open_vector_store(cfg) is down              # сервис стартует без Qdrant
    assert events(logs, "vector_store_unavailable")

    client = AsyncQdrantClient(":memory:")
    await memory_store(dim=4, client=client).ensure_collection()
    monkeypatch.setattr(VectorStore, "from_settings", classmethod(lambda cls, cfg: memory_store(client=client)))
    with pytest.raises(VectorStoreError, match="создана для векторов из 4 чисел"):
        await vs.open_vector_store(cfg)                            # коллекция под другую модель — ошибка старта


async def test_lifespan_creates_one_store_and_closes_it(mocker):
    from app.main import app

    store = memory_store()
    closed = mocker.spy(store, "close")
    mocker.patch("app.main.open_vector_store", mocker.AsyncMock(return_value=store))
    mocker.patch("app.main.setup_tracing", return_value=None)
    async with app.router.lifespan_context(app):
        assert app.state.vector_store is store
    closed.assert_awaited_once()


# ---------------------------------------------------------------- /kb/search
@pytest.fixture
async def kb_app(monkeypatch):
    from app.deps.providers import get_vector_store
    from app.main import app

    store = memory_store()
    await store.ensure_collection()
    await store.upsert([point(0, title="Возврат оплаты", category="billing", source="help_center.jsonl",
                              created_at="2026-05-19T00:00:00Z", text="Вернуть оплату можно в течение 14 дней."),
                        point(1, title="Возврат (2025)", category="billing", source="archive/refunds_2025.md",
                              created_at="2025-03-12T00:00:00Z", status="archived", text="7 дней."),
                        point(2, title="Лимиты API", category="api", source="help_center.jsonl",
                              created_at="2025-12-16T00:00:00Z", text="429")])
    seen: list[str] = []

    def fake_embed_query(text: str) -> list[float]:
        seen.append(text)
        query = vector(1)
        query[0] = 0.6                           # ближе к D-0, чем к D-2: порядок без ничьих
        return query

    monkeypatch.setattr(embeddings, "embed_query", fake_embed_query)
    app.dependency_overrides[get_vector_store] = lambda: store
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http, seen
    app.dependency_overrides.pop(get_vector_store, None)
    await store.close()


async def test_kb_search_excludes_archive_by_default(kb_app):
    http, seen = kb_app
    response = await http.get("/kb/search", params={"q": "Как вернуть деньги?", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert seen == ["Как вернуть деньги?"] and body["collection"] == "documents"
    assert [hit["doc_id"] for hit in body["hits"]] == ["D-0", "D-2"]
    assert body["hits"][0]["text"] == "Вернуть оплату можно в течение 14 дней." and body["hits"][0]["score"] > 0
    archived = (await http.get("/kb/search", params={"q": "x", "include_archived": "true"})).json()
    assert archived["hits"][0]["doc_id"] == "D-1" and archived["hits"][0]["status"] == "archived"
    billing = (await http.get("/kb/search", params={"q": "x", "category": "billing"})).json()
    assert [hit["doc_id"] for hit in billing["hits"]] == ["D-0"]
    recent = (await http.get("/kb/search", params={"q": "x", "created_after": "2026-01-01"})).json()
    assert [hit["doc_id"] for hit in recent["hits"]] == ["D-0"]


async def test_kb_search_validation_and_errors(kb_app, monkeypatch):
    http, _ = kb_app
    assert (await http.get("/kb/search", params={"q": ""})).status_code == 422
    assert (await http.get("/kb/search", params={"q": "x", "top_k": 50})).status_code == 422
    assert (await http.get("/kb/search", params={"q": "   "})).json()["error"]["code"] == "empty_query"

    def broken(text):
        raise embeddings.EmbeddingError("Модель bge-m3 не найдена (404): скачайте её: ollama pull bge-m3.")

    monkeypatch.setattr(embeddings, "embed_query", broken)
    response = await http.get("/kb/search", params={"q": "x"})
    assert response.status_code == 503 and response.json()["error"]["code"] == "embeddings_unavailable"


async def test_kb_search_without_vector_store_is_503():
    from app.main import app

    app.state.vector_store = None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        response = await http.get("/kb/search", params={"q": "x"})
    assert response.status_code == 503 and response.json()["error"]["code"] == "vector_store_not_configured"


# ---------------------------------------------------------------- база знаний
def test_corpus_has_100_plus_chunks_with_payload():
    chunks = load_chunks()
    assert len(chunks) >= 100
    assert {chunk.source for chunk in chunks} >= {"help_center.jsonl", "release_notes.md", "faq.md",
                                                  "archive/refunds_2025.md"}
    for chunk in chunks:
        payload = chunk.payload
        assert {"source", "text", "created_at", "category"} <= set(payload) and len(payload) >= 4
        assert payload["text"].strip() and payload["status"] in ("actual", "archived")
        assert datetime.strptime(payload["created_at"], "%Y-%m-%dT%H:%M:%SZ")
    assert len({chunk.point_id for chunk in chunks}) == len(chunks)


def test_corpus_supports_filter_examples():
    """Для примеров фильтров: свежие заметки за 30 дней до 10.10.2026, старые статьи, архив."""
    chunks = load_chunks()
    fresh = [c.doc_id for c in chunks if c.created_at >= "2026-09-10"]
    assert fresh and all(doc_id.startswith("RN-") for doc_id in fresh)
    assert sum(c.created_at < "2026-01-01" for c in chunks) > 30
    archived = {c.doc_id: c for c in chunks if c.status == "archived"}
    assert "ARC-TARIFFS-2025" in archived and archived["ARC-TARIFFS-2025"].category == "billing"
    assert "«Профессиональный»" in archived["ARC-TARIFFS-2025"].text


def test_point_ids_are_deterministic_uuid5():
    chunk = next(c for c in load_chunks() if c.doc_id == "KB-030")
    assert chunk.point_id == point_id("help_center.jsonl", "KB-030") == str(
        uuid.uuid5(documents.NAMESPACE, "help_center.jsonl#KB-030"))
    assert uuid.UUID(chunk.point_id).version == 5
    assert [c.point_id for c in load_chunks()] == [c.point_id for c in load_chunks()]


def test_markdown_parsing(tmp_path):
    (tmp_path / "notes.md").write_text(
        "---\ncategory: api\nproduct: api\nstatus: actual\ncreated_at: 2026-01-02\n---\n"
        "# Заголовок файла\n\nВступление.\n\n"
        "## Первый раздел\n<!-- id: N-1; created_at: 2026-09-29; category: billing -->\nТекст первого.\n\n"
        "## Второй раздел\nТекст второго.\n", encoding="utf-8")
    first, second = documents.read_markdown(tmp_path / "notes.md", "notes.md")
    assert (first.doc_id, first.key, first.category) == ("N-1", "N-1", "billing")
    assert first.created_at == "2026-09-29T00:00:00Z"
    assert first.text == "Первый раздел\nТекст первого." and first.title == "Первый раздел"
    assert (second.doc_id, second.key, second.category, second.created_at) == ("notes-1", "1", "api",
                                                                               "2026-01-02T00:00:00Z")


@pytest.mark.parametrize("body, error", [
    ("## Раздел\n<!-- id N-1 -->\nТекст", "нет двоеточия"),
    ("## Раздел\n<!-- created_at: вчера -->\nТекст", "не дата"),
    ("## Раздел\n<!-- status: draft -->\nТекст", "status «draft»"),
    ("## Раздел\n<!-- id: X -->\n", "не хватает полей text"),
    ("Без разделов", "нет ни одного раздела"),
    ("## A\n<!-- id: X -->\nТекст\n## B\n<!-- id: X -->\nТекст", "Повторяется ключ «X»"),
])
def test_markdown_errors_name_the_place(tmp_path, body, error):
    (tmp_path / "bad.md").write_text("---\ncategory: api\nproduct: api\nstatus: actual\ncreated_at: 2026-01-02\n---\n"
                                     + body, encoding="utf-8")
    with pytest.raises(CorpusError, match=error):
        load_chunks(tmp_path, ["bad.md"])


def test_timezone_is_normalized_to_utc():
    assert documents.to_rfc3339("2026-09-29T03:00:00+03:00", "x") == "2026-09-29T00:00:00Z"
    assert documents.to_rfc3339(datetime(2026, 9, 29, tzinfo=UTC), "x") == "2026-09-29T00:00:00Z"


# ---------------------------------------------------------------- скрипты
@pytest.fixture
def script_env(monkeypatch, tmp_path):
    """Скрипты с Qdrant :memory: и моделью-заглушкой; один клиент на тест — как один процесс."""
    client = AsyncQdrantClient(":memory:")
    backend = HashBackend()
    service = EmbeddingService(backend, cache=None)

    def from_settings(cls, cfg, *, collection=None, distance=Distance.COSINE):
        return VectorStore(None, None, collection or "documents", DIM, distance=distance, client=client)

    async def keep_open(self):                        # скрипт закрывает store, а клиент общий на тест
        pass

    monkeypatch.setattr(VectorStore, "from_settings", classmethod(from_settings))
    monkeypatch.setattr(VectorStore, "close", keep_open)
    monkeypatch.setattr(embeddings, "default_service", lambda: service)
    monkeypatch.setattr("app.core.config.get_settings", lambda: Settings(_env_file=None))
    yield client, backend
    from log_capture import quiet_logs

    quiet_logs()


async def count(client: AsyncQdrantClient, name: str = "documents") -> int:
    return (await client.get_collection(name)).points_count


def test_loader_is_idempotent_and_prunes(script_env, capsys):
    import asyncio

    client, backend = script_env
    loader = load_script("load_to_qdrant")
    assert loader.main([]) == 0
    first = capsys.readouterr().out
    total = len(load_chunks())
    assert asyncio.run(count(client)) == total
    assert f"points_count: {total}" in first and f"новых {total}" in first

    stale = PointStruct(id=point_id("removed.md", "0"), vector=vector(0), payload={"doc_id": "OLD"})
    asyncio.run(client.upsert("documents", [stale]))
    assert loader.main(["--batch-size", "50"]) == 0
    second = capsys.readouterr().out
    assert asyncio.run(count(client)) == total                                  # без дублей
    assert "новых 0" in second and "удалено устаревших: 1" in second


def test_loader_refuses_wrong_dimension(script_env, capsys, monkeypatch):
    import asyncio

    client, _ = script_env
    monkeypatch.setattr(embeddings, "default_service", lambda: EmbeddingService(HashBackend(dim=DIM + 4), cache=None))
    assert load_script("load_to_qdrant").main([]) == 2
    err = capsys.readouterr().err
    assert "возвращает векторы из 12 чисел, а EMBEDDING_DIM=8" in err and "Ничего не загружено" in err
    assert asyncio.run(count(client)) == 0


def test_loader_dry_run(capsys):
    assert load_script("load_to_qdrant").main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"Фрагментов: {len(load_chunks())} (help_center.jsonl — 56")


def test_cosine_and_dot_agree_on_normalized_vectors(script_env, capsys):
    client, _ = script_env
    assert load_script("load_to_qdrant").main([]) == 0
    capsys.readouterr()
    compare = load_script("compare_metrics")
    assert compare.main([]) == 0
    out = capsys.readouterr().out
    assert "— нормированы" in out and out.count("| да |") == len(compare.QUERIES)
    assert "cosine на 5 из 5" in out and "Временные коллекции удалены" in out
    import asyncio

    names = {c.name for c in asyncio.run(client.get_collections()).collections}
    assert names == {"documents"}


def test_cosine_and_dot_differ_on_raw_vectors(script_env, capsys, monkeypatch):
    """Если бы модуль эмбеддингов не нормировал векторы, DOT ранжировал бы иначе — скрипт это видит."""
    _, backend = script_env
    assert load_script("load_to_qdrant").main([]) == 0
    capsys.readouterr()
    monkeypatch.setattr(embeddings, "embed_documents", lambda texts: backend.embed(texts))
    monkeypatch.setattr(embeddings, "embed_query", lambda text: backend.embed([text])[0])
    assert load_script("compare_metrics").main([]) == 1
    out = capsys.readouterr().out
    assert "НЕ нормированы" in out and "| **нет** |" in out


def test_filters_demo(script_env, capsys):
    assert load_script("load_to_qdrant").main([]) == 0
    capsys.readouterr()
    assert load_script("qdrant_filters_demo").main(["--today", "2026-10-10"]) == 0
    out = capsys.readouterr().out
    assert out.count("Без фильтра:") == 3 and out.count("С фильтром:") == 3
    assert 'MatchValue(value="release_notes.md")' in out and "2026-09-10T00:00:00Z" in out
    assert "must_not=[FieldCondition(key=\"status\"" in out
    sections = out.split("### ")[1:]
    match_rows = sections[0].split("С фильтром:")[1]
    assert "release_notes.md" in match_rows and "faq.md" not in match_rows and "help_center" not in match_rows
    date_rows = sections[1].split("С фильтром:")[1]
    assert all(line.split(" | ")[5] >= "2026-09-10" for line in date_rows.splitlines() if line.startswith("| 1"))
    composite = sections[2].split("С фильтром:")[1]
    assert "archived" not in composite and "billing" in composite


def test_demo_queries_are_domain_questions():
    compare = load_script("compare_metrics")
    assert len(compare.QUERIES) == 5 and len(set(compare.QUERIES)) == 5
    assert all(q.endswith("?") and len(q.split()) >= 3 for q in compare.QUERIES)
