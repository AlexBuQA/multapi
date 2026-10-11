"""
Корпус и индексация RAG (блок 5.5) без сети: настоящий корпус data/<категория>/… (66 документов
HTML, DOCX, Markdown, PDF), ридеры и метаданные, IngestionPipeline с SimpleDocumentStore и
DocstoreStrategy.UPSERTS поверх Qdrant :memory:, эмбеддинги — триграммы текста.

Главные свойства: повторный запуск — «0 changed, N unchanged» и ни одного вызова модели
эмбеддингов; изменённый файл заменяет свои фрагменты, а не добавляет копии; нечитаемый файл —
.failed и строка ingest_file_failed в логе; полный путь (с именем пользователя Windows) никуда не
попадает.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient
from rag_fakes import DIM, TrigramEmbeddings, threaded

from app.core.config import Settings
from app.services import ingestion
from app.services.ingestion import (
    IngestError,
    IngestionService,
    ProjectEmbedding,
    category_of,
    file_metadata,
    is_ignored,
    list_files,
    read_file,
    split_front_matter,
    version_of,
)

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CATEGORIES = {"account", "api", "billing", "data-management", "integrations", "mobile", "notifications", "releases",
              "start", "support", "team"}


def load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- корпус
def test_corpus_has_66_documents_of_four_formats_in_categories():
    files = list_files(DATA)
    relative = [p.relative_to(DATA).as_posix() for p in files if not p.relative_to(DATA).as_posix().startswith("uploads/")]
    assert len(relative) == 66
    assert Counter(Path(p).suffix for p in relative) == {".html": 26, ".md": 17, ".docx": 14, ".pdf": 9}
    assert {p.split("/")[0] for p in relative} == CATEGORIES                  # всё — в data/<категория>/
    assert not any(p.startswith(("archive/", "rag-block-03/")) or "/" not in p for p in relative)  # .ragignore


def test_every_document_has_author_date_and_text():
    documents = 0
    for path in list_files(DATA):
        if path.relative_to(DATA).parts[0] == "uploads":
            continue
        docs = read_file(path, DATA)
        documents += len(docs)
        for doc in docs:
            assert doc.text.strip() and doc.metadata["author"] and re.fullmatch(r"\d{4}-\d{2}-\d{2}",
                                                                              doc.metadata["created_at"])
            assert set(doc.excluded_embed_metadata_keys) == set(doc.metadata)      # в вектор — только текст
    assert documents == 102                                                      # страницы PDF, разделы MD


def test_inventory_doc_matches_corpus():
    text = (ROOT / "docs" / "data_inventory.md").read_text(encoding="utf-8")
    assert "Файлов: **66** в 11 категориях" in text
    assert "HTML — 26, Markdown — 17, DOCX — 14, PDF — 9" in text
    assert "Документов LlamaIndex: **102**" in text
    for fmt, count in (("HTML", 26), ("Markdown", 17), ("DOCX", 14), ("PDF", 9)):
        assert re.search(rf"^\| {fmt} \| [^|]+ \| {count} \|", text, re.MULTILINE), fmt
    assert len(re.findall(r"^\| `[a-z-]+/[^`]+` \|", text, re.MULTILINE)) == 66            # опись по файлам


def test_build_corpus_reproduces_committed_documents(tmp_path):
    """Генератор детерминирован: те же файлы и тот же текст (HTML и Markdown — побайтно)."""
    build_corpus = load_script("build_corpus")
    written = build_corpus.build(tmp_path)
    assert len(written) == 66
    for path in written:
        relative = path.relative_to(tmp_path)
        committed = DATA / relative
        assert committed.is_file(), relative
        if path.suffix in (".html", ".md"):
            assert path.read_bytes() == committed.read_bytes().replace(b"\r\n", b"\n"), relative
        else:
            assert [d.text for d in read_file(path, tmp_path)] == [d.text for d in read_file(committed, DATA)]


# ---------------------------------------------------------------- ридеры и метаданные
def test_html_reader_takes_article_without_navigation():
    [doc] = read_file(DATA / "billing" / "refunds_v2.docx", DATA)
    assert doc.metadata["version"] == "2" and doc.metadata["author"] == "Финансовый отдел"
    [page] = read_file(DATA / "account" / "password-reset.html", DATA)
    assert page.text.startswith("Восстановление (сброс) пароля")
    assert "Справка «Личный кабинет» ›" not in page.text and "©" not in page.text   # шапка и подвал — мимо
    assert page.metadata["title"] == "Восстановление (сброс) пароля"


def test_html_without_article_falls_back_to_body(tmp_path):
    (tmp_path / "misc").mkdir()
    path = tmp_path / "misc" / "page.html"
    path.write_text("<html><head><title>Заметка</title></head><body><p>Текст страницы.</p></body></html>",
                    encoding="utf-8")
    [doc] = read_file(path, tmp_path)
    assert doc.text == "Текст страницы." and doc.metadata["category"] == "misc"


def test_markdown_front_matter_goes_to_metadata_not_text():
    docs = read_file(DATA / "api" / "faq-api.md", DATA)
    assert [d.id_ for d in docs] == ["api/faq-api.md#1", "api/faq-api.md#2", "api/faq-api.md#3"]   # раздел — документ
    assert all("---" not in d.text and "author:" not in d.text for d in docs)
    assert docs[0].metadata["title"] == "Частые вопросы: API" and docs[0].metadata["created_at"] == "2026-06-15"
    assert split_front_matter("---\ntitle: a: b\n---\nтекст") == ({}, "текст")        # не YAML — но и не в тексте


def test_pdf_is_read_by_pages_with_page_numbers():
    docs = read_file(DATA / "support" / "support-regulation_v2.3.pdf", DATA)
    assert [d.metadata["page"] for d in docs] == [1, 2, 3, 4]
    assert [d.id_ for d in docs] == [f"support/support-regulation_v2.3.pdf#p{n}" for n in range(1, 5)]
    assert "15 минут" in docs[1].text and docs[0].metadata["version"] == "2.3"
    assert docs[0].metadata["author"] == "Служба поддержки" and docs[0].metadata["created_at"] == "2026-01-20"


def test_metadata_has_no_absolute_paths_and_stable_key_order():
    meta = file_metadata(DATA / "releases" / "release-notes_2025-h1.pdf", DATA)
    assert list(meta) == ["file_name", "source", "category", "format", "title", "author", "created_at",
                          "last_modified"]
    assert meta["source"] == "releases/release-notes_2025-h1.pdf" and meta["category"] == "releases"
    assert not any(str(ROOT) in str(value) for value in meta.values())


@pytest.mark.parametrize(("name", "version"), [("refunds_v2.docx", "2"), ("support-regulation_v2.3.pdf", "2.3"),
                                               ("policy-V3.md", "3"), ("release-notes_2025-h1.pdf", None),
                                               ("api-v2-keys.md", None), ("x_v2.pdf.failed", "2")])
def test_version_from_file_name(name, version):
    assert version_of(name) == version


def test_category_and_ignore_rules():
    from pathlib import PurePosixPath

    assert category_of(PurePosixPath("billing/x.docx")) == "billing"
    assert category_of(PurePosixPath("uploads/x.pdf")) == "uploads"
    assert category_of(PurePosixPath("x.md")) == "general"
    patterns = ["archive/", "faq.md", "*.jsonl", "drafts/*.md"]
    assert is_ignored("archive/api_v1.md", patterns) and is_ignored("billing/archive/x.md", patterns)
    assert is_ignored("faq.md", patterns) and is_ignored("billing/faq.md", patterns)   # без «/» — имя на любой глубине
    assert is_ignored("help_center.jsonl", patterns) and is_ignored("drafts/a.md", patterns)
    assert not is_ignored("billing/faq-billing.md", patterns) and not is_ignored("team/drafts/a.md", patterns)


def test_list_files_skips_hidden_failed_and_unsupported(tmp_path):
    (tmp_path / "billing").mkdir()
    for name in ("a.md", "b.pdf.failed", ".hidden.md", "c.json", "d.txt", ".x.pdf.part"):
        (tmp_path / "billing" / name).write_text("текст", encoding="utf-8")
    (tmp_path / ".ragignore").write_text("# комментарий\nd.txt\n", encoding="utf-8")
    assert [p.name for p in list_files(tmp_path)] == ["a.md"]
    with pytest.raises(IngestError, match="Нет папки корпуса"):
        list_files(tmp_path / "missing")


# ---------------------------------------------------------------- IngestionPipeline + UPSERTS
def corpus_copy(tmp_path: Path, files: tuple[str, ...] = ("billing/refunds_v2.docx", "api/faq-api.md",
                                                           "support/support-regulation_v2.3.pdf",
                                                           "account/password-reset.html")) -> Path:
    root = tmp_path / "data"
    for relative in files:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DATA / relative, target)
    return root


def service(root: Path, client: QdrantClient, embedder: TrigramEmbeddings | None = None,
            **rag: Any) -> IngestionService:
    cfg = Settings(_env_file=None, qdrant_url="http://127.0.0.1:6333", embedding_dim=DIM,
                   llm={"openai_api_key": "k"},
                   rag={"data_dir": root, "docstore_dir": root.parent / "ingest", "collection": "kb_test", **rag})
    return IngestionService(cfg, client=client, embed_model=ProjectEmbedding(embedder or TrigramEmbeddings()))


def test_second_run_is_all_unchanged_without_embeddings(tmp_path):
    root, client, embedder = corpus_copy(tmp_path), QdrantClient(":memory:"), TrigramEmbeddings()
    first = threaded(service(root, client, embedder).run)
    assert (first.count("new"), first.total, first.chunks) == (4, 4, 9)        # 1 + 3 раздела + 4 страницы + 1
    assert client.count("kb_test").count == 9 and (root.parent / "ingest" / "kb_test_docstore.json").is_file()
    calls = embedder.document_calls
    again = threaded(service(root, client, embedder).run)                      # новый процесс: docstore с диска
    assert (again.count("changed"), again.count("unchanged"), again.chunks) == (0, 4, 0)
    assert "0 changed, 4 unchanged" in again.summary()
    assert embedder.document_calls == calls and client.count("kb_test").count == 9


def test_changed_file_replaces_its_chunks_and_removed_file_is_deleted(tmp_path):
    root, client = corpus_copy(tmp_path), QdrantClient(":memory:")
    threaded(service(root, client).run)
    faq = root / "api" / "faq-api.md"
    faq.write_text(faq.read_text(encoding="utf-8").split("\n## Чем тестовая")[0] + "\n", encoding="utf-8")
    (root / "account" / "password-reset.html").unlink()
    report = threaded(service(root, client).run)
    states = {item.path: item.state for item in report.files}
    assert states == {"api/faq-api.md": "changed", "billing/refunds_v2.docx": "unchanged",
                      "support/support-regulation_v2.3.pdf": "unchanged", "account/password-reset.html": "removed"}
    sources = [p.payload["source"] for p in client.scroll("kb_test", limit=100, with_payload=["source"])[0]]
    assert Counter(sources) == {"api/faq-api.md": 2, "billing/refunds_v2.docx": 1,
                                "support/support-regulation_v2.3.pdf": 4}      # раздел исчез — и его фрагмент


def test_broken_and_empty_files_are_renamed_and_logged(tmp_path):
    from log_capture import captured_logs, events

    root, client = corpus_copy(tmp_path, ("billing/refunds_v2.docx",)), QdrantClient(":memory:")
    (root / "uploads").mkdir()
    (root / "uploads" / "broken.pdf").write_bytes(b"not a pdf")
    (root / "uploads" / "broken.docx").write_bytes(b"not a zip")
    (root / "uploads" / "empty.md").write_text("---\ntitle: пусто\n---\n", encoding="utf-8")
    with captured_logs("WARNING") as records:
        report = threaded(service(root, client).run)
    assert (report.count("new"), report.count("failed")) == (1, 3)
    assert sorted(p.name for p in (root / "uploads").iterdir()) == ["broken.docx.failed", "broken.pdf.failed",
                                                                    "empty.md.failed"]
    failed = events(records, "ingest_file_failed")
    assert {e["file"] for e in failed} == {"uploads/broken.pdf", "uploads/broken.docx", "uploads/empty.md"}
    assert all(str(tmp_path) not in e["error"] for e in failed)                 # без полного пути
    again = threaded(service(root, client).run)                                # .failed больше не читается
    assert (again.count("failed"), again.count("unchanged")) == (0, 1)


def held_open(path: Path) -> bool:
    for fd in Path("/proc/self/fd").iterdir():
        try:
            if os.readlink(fd) == str(path):
                return True
        except OSError:
            continue
    return False


WINDOWS = sys.platform == "win32"


@pytest.mark.skipif(not WINDOWS and not Path("/proc/self/fd").is_dir(), reason="нужна Windows или /proc (Linux)")
def test_broken_files_are_closed_before_rename_as_windows_requires(tmp_path, monkeypatch):
    """Windows не переименовывает открытый файл, Linux — переименовывает. На Windows работает
    настоящий os.replace, на Linux он ведёт себя как в Windows: PermissionError, если процесс ещё
    держит файл открытым (видно по /proc/self/fd). Так pymupdf.open(path) на битом PDF держал
    файл, пока было живо исключение FileDataError, и broken.pdf не становился .failed."""
    real_replace = os.replace

    def windows_replace(src, dst):
        if held_open(Path(src)):
            raise PermissionError(13, "Процесс не может получить доступ к файлу, так как он занят", str(src))
        real_replace(src, dst)

    root, client = corpus_copy(tmp_path, ("billing/refunds_v2.docx",)), QdrantClient(":memory:")
    (root / "uploads").mkdir()
    for name, content in (("broken.pdf", b"not a pdf"), ("half.pdf", b"%PDF-1.7\n1 0 obj\n<<"),
                          ("broken.docx", b"not a zip")):
        (root / "uploads" / name).write_bytes(content)
    if not WINDOWS:
        monkeypatch.setattr(ingestion.os, "replace", windows_replace)
    report = threaded(service(root, client).run)
    assert {f.path: f.renamed_to for f in report.files if f.state == "failed"} == {
        "uploads/broken.pdf": "uploads/broken.pdf.failed", "uploads/half.pdf": "uploads/half.pdf.failed",
        "uploads/broken.docx": "uploads/broken.docx.failed"}


def test_lost_docstore_does_not_duplicate_points(tmp_path):
    """id фрагмента — uuid5 от документа и номера: без docstore те же точки перезаписываются."""
    root, client = corpus_copy(tmp_path), QdrantClient(":memory:")
    threaded(service(root, client).run)
    shutil.rmtree(root.parent / "ingest")
    report = threaded(service(root, client).run)
    assert report.count("new") == 4 and client.count("kb_test").count == 9


def test_empty_collection_resets_docstore_and_rebuild(tmp_path):
    root, client = corpus_copy(tmp_path), QdrantClient(":memory:")
    threaded(service(root, client).run)
    client.delete_collection("kb_test")                                        # том Qdrant удалили
    report = threaded(service(root, client).run)
    assert report.count("new") == 4 and client.count("kb_test").count == 9
    rebuilt = threaded(service(root, client).run, rebuild=True)
    assert rebuilt.count("new") == 4 and client.count("kb_test").count == 9


def test_new_embedding_model_or_chunking_reindexes_documents(tmp_path):
    root, client = corpus_copy(tmp_path), QdrantClient(":memory:")
    threaded(service(root, client).run)
    report = threaded(service(root, client, chunking="fixed", chunk_size=256, chunk_overlap=32).run)
    assert report.count("changed") == 4                                        # метка нарезки — в hash документа
    labels = {p.payload["chunking"] for p in client.scroll("kb_test", limit=100, with_payload=["chunking"])[0]}
    assert labels == {"fixed:256/32"}


def test_single_file_run_and_status(tmp_path):
    root, client = corpus_copy(tmp_path, ("billing/refunds_v2.docx",)), QdrantClient(":memory:")
    ingest = service(root, client)
    threaded(ingest.run)
    (root / "uploads").mkdir()
    upload = root / "uploads" / "faq-api.md"
    shutil.copy2(DATA / "api" / "faq-api.md", upload)
    ingest.queue("uploads/faq-api.md")
    assert ingest.status["uploads/faq-api.md"].state == "queued"
    report = threaded(ingest.run, [upload])
    assert [(f.path, f.state) for f in report.files] == [("uploads/faq-api.md", "new")]   # прочие — не тронуты
    assert ingest.status["uploads/faq-api.md"].state == "new" and client.count("kb_test").count == 4
    with pytest.raises(IngestError, match="вне корпуса"):
        threaded(ingest.run, [DATA / "billing" / "refunds_v2.docx"])


def test_wrong_vector_size_asks_for_rebuild(tmp_path):
    from qdrant_client.models import Distance, VectorParams

    from app.services.rag_common import RAGError

    root, client = corpus_copy(tmp_path), QdrantClient(":memory:")
    client.create_collection("kb_test", vectors_config=VectorParams(size=DIM * 2, distance=Distance.COSINE))
    with pytest.raises(RAGError, match="--rebuild"):
        threaded(service(root, client).run)


# ---------------------------------------------------------------- scripts/ingest.py, data_inventory.py
def test_ingest_cli_reports_and_rejects_foreign_corpus(tmp_path, monkeypatch, capsys):
    from log_capture import quiet_logs

    root = corpus_copy(tmp_path)
    (root / "uploads").mkdir()
    (root / "uploads" / "broken.pdf").write_bytes(b"x")
    cli = load_script("ingest")
    client = QdrantClient(":memory:")
    monkeypatch.setattr(client, "close", lambda: None)
    cfg = Settings(_env_file=None, qdrant_url="http://127.0.0.1:6333", embedding_dim=DIM, llm={"openai_api_key": "k"},
                   rag={"data_dir": root, "docstore_dir": tmp_path / "ingest", "collection": "kb_test"})
    monkeypatch.setattr("app.core.config.get_settings", lambda: cfg)
    original = IngestionService.__init__

    def init(self, cfg, **kwargs):
        original(self, cfg, client=client, embed_model=ProjectEmbedding(TrigramEmbeddings()), **kwargs)

    monkeypatch.setattr(IngestionService, "__init__", init)
    try:
        assert threaded(cli.main, [str(root)]) == 1                           # один файл не прочитался
        out = capsys.readouterr().out
        assert "4 new, 0 changed, 0 unchanged, 1 failed" in out and "uploads/broken.pdf → uploads/broken.pdf.failed" in out
        assert threaded(cli.main, [str(root)]) == 0
        assert "0 new, 0 changed, 4 unchanged, 0 failed, 0 removed" in capsys.readouterr().out
        assert threaded(cli.main, [str(DATA / "rag-block-03")]) == 2          # чужой корпус без --collection
        assert "смешались бы два корпуса" in capsys.readouterr().err
        assert threaded(cli.main, [str(root), "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "4 файлов, 9 документов" in out and "billing/" not in out.split("Метаданные")[0]
    finally:
        quiet_logs()


def test_inventory_render_counts(tmp_path):
    inventory = load_script("data_inventory")
    root = corpus_copy(tmp_path)
    text = inventory.render(inventory.collect(root, 512, 64, "recursive"), "data", "recursive:512/64")
    assert "Файлов: **4** в 4 категориях" in text and "Документов LlamaIndex: **9**" in text
    assert "| `support/support-regulation_v2.3.pdf` | PDF |" in text and "| 4 | 4 | Служба поддержки | 2026-01-20 | 2.3 |" in text


def test_ingestion_module_has_no_absolute_path_metadata_keys():
    assert "file_path" not in ingestion.LLM_EXCLUDED
    docs = read_file(DATA / "account" / "password-reset.html", DATA)
    assert "file_path" not in docs[0].metadata and "tag_id" not in docs[0].metadata


def test_failed_embeddings_do_not_leave_documents_marked_unchanged(tmp_path):
    """Модель эмбеддингов упала посреди записи изменённого файла: следующий запуск пишет его заново."""
    from app.services.embeddings import EmbeddingError

    root, client = corpus_copy(tmp_path, ("api/faq-api.md", "billing/refunds_v2.docx")), QdrantClient(":memory:")
    threaded(service(root, client).run)
    faq = root / "api" / "faq-api.md"
    faq.write_text(faq.read_text(encoding="utf-8") + "\n## Новый вопрос\n\nНовый ответ.\n", encoding="utf-8")

    class Down(TrigramEmbeddings):
        def embed_documents(self, texts):
            raise EmbeddingError("модель эмбеддингов не отвечает")

    with pytest.raises(EmbeddingError):
        threaded(service(root, client, Down()).run)
    report = threaded(service(root, client).run)
    assert {item.path: item.state for item in report.files} == {"api/faq-api.md": "new",
                                                                "billing/refunds_v2.docx": "unchanged"}
    sources = Counter(p.payload["source"] for p in client.scroll("kb_test", limit=100, with_payload=["source"])[0])
    assert sources == {"api/faq-api.md": 4, "billing/refunds_v2.docx": 1}
