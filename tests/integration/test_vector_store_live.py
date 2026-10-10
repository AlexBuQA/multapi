"""
Smoke-тест VectorStore на настоящем Qdrant (блок 5.2): ensure_collection, upsert, search.

    docker compose up -d qdrant
    pytest tests/integration/test_vector_store_live.py -v

Адрес и ключ — QDRANT_URL и QDRANT_API_KEY из .env (или QDRANT_TEST_URL / QDRANT_TEST_API_KEY
в окружении). Qdrant не отвечает — тест пропускается. Работает во временной коллекции
pytest_smoke_<случайный суффикс> и удаляет её в конце: боевую коллекцию не трогает.

В отличие от unit-тестов (Qdrant :memory:), здесь видно то, чего нет в локальном режиме:
payload-индексы с типами, HNSW-конфигурацию сервера, ответ на неверный ключ.
"""
from __future__ import annotations

import os
import socket
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse

import pytest
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENV = dotenv_values(ROOT / ".env") if (ROOT / ".env").exists() else {}
URL = os.environ.get("QDRANT_TEST_URL") or ENV.get("QDRANT_URL") or ""
API_KEY = os.environ.get("QDRANT_TEST_API_KEY") or ENV.get("QDRANT_API_KEY") or None
DIM = 8


def reachable(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.hostname:
        return False
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not reachable(URL), reason=f"Qdrant недоступен: QDRANT_URL={URL or '(не задан)'}")


def vector(seed: int) -> list[float]:
    return [1.0 if i == seed % DIM else 0.05 * (i + 1) for i in range(DIM)]


async def test_vector_store_on_real_qdrant():
    from qdrant_client.models import PayloadSchemaType, PointStruct

    from app.services.vector_store import PAYLOAD_INDEXES, VectorStore, VectorStoreError, build_filter

    store = VectorStore(URL, API_KEY, f"pytest_smoke_{uuid.uuid4().hex[:8]}", DIM)
    try:
        await store.ensure_collection()
        await store.ensure_collection()
        info = await store.collection_info()
        assert info.config.params.vectors.size == DIM
        assert (info.config.hnsw_config.m, info.config.hnsw_config.ef_construct) == (16, 100)
        schema = {name: index.data_type for name, index in info.payload_schema.items()}
        assert schema == PAYLOAD_INDEXES and schema["created_at"] == PayloadSchemaType.DATETIME

        points = [PointStruct(id=str(uuid.UUID(int=n + 1)), vector=vector(n),
                              payload={"doc_id": f"D-{n}", "category": "billing" if n % 2 else "api",
                                       "status": "archived" if n % 10 == 1 else "actual",
                                       "created_at": f"2026-{1 + n % 9:02d}-15T00:00:00Z", "source": "smoke.md",
                                       "text": f"фрагмент {n}"})
                  for n in range(300)]
        await store.upsert(points, batch_size=128)
        assert await store.points_count() == 300
        await store.upsert(points, batch_size=128)                  # повтор — те же id, без дублей
        assert await store.points_count() == 300

        hits = await store.search(vector(1), top_k=5)
        # у точек n, n+8, n+16… одинаковые векторы: ближайшие пять — из этой группы, близость 1
        assert len(hits) == 5 and all(int(hit.payload["doc_id"][2:]) % DIM == 1 for hit in hits)
        assert hits[0].score > 0.99
        query_filter = build_filter(category="billing", created_after="2026-06-01")
        filtered = await store.search(vector(1), top_k=10, query_filter=query_filter)
        assert filtered and all(hit.payload["category"] == "billing" and hit.payload["status"] == "actual"
                                and hit.payload["created_at"] >= "2026-06-01" for hit in filtered)

        with pytest.raises(VectorStoreError, match="вектор из 3 чисел"):
            await store.search([1.0, 0.0, 0.0])
        wrong_key = VectorStore(URL, "wrong-key-for-smoke-test", store.collection, DIM)
        try:
            if API_KEY:
                with pytest.raises(VectorStoreError, match="не принял ключ"):
                    await wrong_key.points_count()
        finally:
            await wrong_key.close()
    finally:
        await store.drop()
        await store.close()
