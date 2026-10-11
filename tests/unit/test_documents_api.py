"""
POST /documents/upload, GET /documents/status, POST /documents/reindex (блок 5.5): файл ложится в
data/uploads/, BackgroundTasks индексирует его тем же IngestionPipeline (UPSERTS), ответ — 202.
Qdrant :memory:, эмбеддинги — триграммы; корпус — копия во временной папке.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import httpx
import pytest
from qdrant_client import QdrantClient
from rag_fakes import DIM, TrigramEmbeddings

from app.core.config import Settings, get_settings
from app.deps.providers import get_ingestion
from app.main import app
from app.routers.documents import safe_name
from app.services.ingestion import IngestionService, ProjectEmbedding

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "admin-token-for-tests-0123"
ADMIN = {"X-Admin-Token": TOKEN}
PDF = ROOT / "data" / "support" / "support-regulation_v2.3.pdf"


@pytest.fixture
def documents_app(tmp_path):
    root = tmp_path / "data"
    (root / "billing").mkdir(parents=True)
    shutil.copy2(ROOT / "data" / "billing" / "refunds_v2.docx", root / "billing")
    cfg = Settings(_env_file=None, qdrant_url="http://127.0.0.1:6333", embedding_dim=DIM, admin_token=TOKEN,
                   llm={"openai_api_key": "k"},
                   rag={"data_dir": root, "docstore_dir": tmp_path / "ingest", "collection": "kb_test",
                        "upload_max_mb": 1})
    client = QdrantClient(":memory:")
    ingest = IngestionService(cfg, client=client, embed_model=ProjectEmbedding(TrigramEmbeddings()))
    app.dependency_overrides[get_settings] = lambda: cfg
    app.dependency_overrides[get_ingestion] = lambda: ingest
    app.state.rag = None
    yield root, ingest, client
    app.dependency_overrides.pop(get_settings, None)
    app.dependency_overrides.pop(get_ingestion, None)


async def http() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_upload_returns_202_and_indexes_in_background(documents_app):
    root, _, client = documents_app
    async with await http() as api:
        response = await api.post("/documents/upload", headers=ADMIN,
                                  files={"file": ("tariffs.pdf", PDF.read_bytes(), "application/pdf")})
        assert response.status_code == 202
        body = response.json()
        assert body == {"status": "queued", "file": "uploads/tariffs.pdf", "size": PDF.stat().st_size,
                        "replaced": False, "status_url": "/documents/status?file=uploads/tariffs.pdf"}
        status = (await api.get(body["status_url"])).json()      # ASGITransport ждёт и фоновые задачи
        assert status == {"file": "uploads/tariffs.pdf", "state": "new", "documents": 4, "error": None,
                          "renamed_to": None, "indexed": True}
        assert (root / "uploads" / "tariffs.pdf").is_file()
        points = client.scroll("kb_test", limit=50, with_payload=["source", "category", "page"])[0]
        assert {(p.payload["source"], p.payload["category"]) for p in points} == {("uploads/tariffs.pdf", "uploads")}
        again = await api.post("/documents/upload", headers=ADMIN,
                               files={"file": ("tariffs.pdf", PDF.read_bytes(), "application/pdf")})
        assert again.json()["replaced"] is True
        assert (await api.get("/documents/status", params={"file": "uploads/tariffs.pdf"})).json()["state"] == "unchanged"
        assert client.count("kb_test").count == 4                 # UPSERTS: копий нет


async def test_broken_upload_is_failed_and_renamed(documents_app):
    root, _, _ = documents_app
    async with await http() as api:
        await api.post("/documents/upload", headers=ADMIN, files={"file": ("broken.pdf", b"not a pdf", "application/pdf")})
        status = (await api.get("/documents/status", params={"file": "uploads/broken.pdf"})).json()
    assert status["state"] == "failed" and status["renamed_to"] == "uploads/broken.pdf.failed"
    assert status["indexed"] is False and str(root) not in status["error"]
    assert (root / "uploads" / "broken.pdf.failed").is_file()


@pytest.mark.parametrize(("headers", "filename", "content", "status", "code"), [
    ({}, "a.pdf", b"x", 401, "unauthorized"),
    ({"X-Admin-Token": "wrong-token-0123456789"}, "a.pdf", b"x", 401, "unauthorized"),
    (ADMIN, "main.py", b"x", 415, "unsupported_format"),
    (ADMIN, ".env.md", b"x", 422, "bad_file_name"),
    (ADMIN, "a.pdf", b"", 422, "empty_file"),
])
async def test_upload_rejects(documents_app, headers, filename, content, status, code):
    async with await http() as api:
        response = await api.post("/documents/upload", headers=headers, files={"file": (filename, content, "text/plain")})
    assert response.status_code == status and response.json()["error"]["code"] == code


async def test_upload_too_large_is_413_before_parsing(documents_app):
    async with await http() as api:
        response = await api.post("/documents/upload", headers=ADMIN,
                                  files={"file": ("big.pdf", b"x" * (3 * 1024 * 1024), "application/pdf")})
    assert response.status_code == 413 and response.json()["error"]["code"] == "request_too_large"
    assert not (documents_app[0] / "uploads" / "big.pdf").exists()


@pytest.mark.parametrize(("name", "expected"), [("../../app/main.md", "main.md"), ("C:\\Users\\x\\отчёт.pdf", "отчёт.pdf"),
                                                ("  notes.MD ", "notes.MD")])
def test_safe_name_keeps_only_file_name(name, expected):
    assert safe_name(name) == expected


async def test_status_unknown_and_reindex(documents_app):
    _, ingest, client = documents_app
    async with await http() as api:
        assert (await api.get("/documents/status", params={"file": "nope.pdf"})).status_code == 404
        assert (await api.post("/documents/reindex")).status_code == 401
        response = await api.post("/documents/reindex", headers=ADMIN)
    assert response.status_code == 202 and response.json() == {"status": "queued", "collection": "kb_test"}
    assert ingest.last_report is not None and ingest.last_report.count("new") == 1
    assert client.count("kb_test").count == 1


async def test_documents_without_qdrant_is_503():
    app.state.ingestion = None
    async with await http() as api:
        response = await api.get("/documents/status", params={"file": "x.pdf"})
    assert response.status_code == 503 and response.json()["error"]["code"] == "rag_not_configured"
