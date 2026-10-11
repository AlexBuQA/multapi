"""
Индексация корпуса RAG (блок 5.5): data/<категория>/… → Qdrant, инкрементально.

    python scripts/ingest.py data/                    # новые и изменённые файлы; повторно — «0 changed, N unchanged»
    python scripts/ingest.py data/ --rebuild          # удалить коллекцию и docstore, проиндексировать заново
    python scripts/ingest.py data/ --dry-run          # только прочитать: файлы, документы, фрагменты, метаданные
    python scripts/ingest.py data/uploads/new.pdf     # один файл (внутри корпуса)

Форматы — PDF, DOCX, HTML, Markdown (и .txt), что не входит в корпус — data/.ragignore.
Конвейер, метаданные и docstore — app/services/ingestion.py, docs/rag.md («Ingestion»).
Коллекция — RAG__COLLECTION (kb_docs), Qdrant — QDRANT_URL, модель эмбеддингов — EMBEDDINGS__*.

Файл, который не читается, переименовывается в <имя>.failed и попадает в лог (ingest_file_failed)
и в таблицу ниже; остальные индексируются. Исправили файл — уберите .failed из имени и
запустите снова.

Код выхода: 0 — готово; 1 — часть файлов не прочиталась (.failed); 2 — ошибка настроек, Qdrant
или модели эмбеддингов (текст — что сделать).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MARKS = {"new": "+", "changed": "~", "failed": "!", "removed": "-"}


def dry_run(root: Path, cfg: object) -> int:
    """Прочитать корпус без Qdrant и эмбеддингов: сколько файлов, документов и фрагментов."""
    from app.services import chunking
    from app.services.ingestion import list_files, read_file

    settings = cfg.rag                                   # type: ignore[attr-defined]
    splitter = chunking.build_splitter("recursive" if settings.chunking == "semantic" else settings.chunking,
                                       chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    files, documents, chunks, failed = Counter(), Counter(), Counter(), []
    sample = None
    for path in list_files(root):
        relative = path.relative_to(root).as_posix()
        try:
            docs = read_file(path, root)
        except Exception as exc:  # noqa: BLE001 — dry-run файлы не переименовывает, только показывает
            failed.append((relative, f"{type(exc).__name__}: {exc.__cause__ or exc}"))
            continue
        fmt = docs[0].metadata["format"]
        files[fmt] += 1
        documents[fmt] += len(docs)
        chunks[fmt] += len(splitter.get_nodes_from_documents(docs))
        sample = sample or docs[0]
    print(f"Корпус {root.name}/: {sum(files.values())} файлов, {sum(documents.values())} документов, "
          f"{sum(chunks.values())} фрагментов ({settings.chunk_size}/{settings.chunk_overlap})")
    for fmt in sorted(files):
        print(f"  {fmt:5} файлов {files[fmt]:3}  документов {documents[fmt]:3}  фрагментов {chunks[fmt]:3}")
    for relative, reason in failed:
        print(f"  ! {relative}: {reason}")
    if sample is not None:
        print(f"Метаданные первого документа ({sample.id_}): {sample.metadata}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Инкрементальная индексация корпуса RAG в Qdrant (блок 5.5)")
    parser.add_argument("path", nargs="?", type=Path, help="папка корпуса (data/) или файл в ней; "
                        "по умолчанию — RAG__DATA_DIR")
    parser.add_argument("--rebuild", action="store_true", help="удалить коллекцию и docstore и проиндексировать заново")
    parser.add_argument("--dry-run", action="store_true", help="только прочитать файлы, без Qdrant и эмбеддингов")
    parser.add_argument("--collection", help="другая коллекция (по умолчанию RAG__COLLECTION) — для другого корпуса")
    parser.add_argument("--verbose", action="store_true", help="показать и неизменённые файлы")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from pydantic import ValidationError

    from app.core.config import get_settings
    from app.observability.logging import setup_logging
    from app.services import embeddings
    from app.services.ingestion import IngestError, IngestionService
    from app.services.rag import no_implicit_event_loop
    from app.services.rag_common import RAGError

    # В консоль — предупреждения (ingest_file_failed и т. п.); итог печатается таблицей ниже.
    setup_logging("WARNING", stream=sys.stderr)
    no_implicit_event_loop()
    try:
        cfg = get_settings()
        configured = cfg.rag.data_dir.resolve()
        target = (args.path or configured).resolve()
        root, files = (target, None) if target.is_dir() or not target.exists() else (configured, [target])
        if args.dry_run:
            return dry_run(root, cfg)
        if root != configured and not args.collection:
            raise IngestError(f"Папка {args.path} — не корпус RAG__DATA_DIR ({cfg.rag.data_dir.name}/): в коллекции "
                              f"{cfg.rag.collection} смешались бы два корпуса. Укажите --collection или RAG__DATA_DIR")
        service = IngestionService(cfg, data_dir=root, collection=args.collection)
        try:
            print(f"Корпус {root.name}/ → коллекция {service.collection} (Qdrant {cfg.qdrant_url}), "
                  f"модель {service.embed_model.model_name}, нарезка {service.chunking_label}", flush=True)
            report = service.run(files, rebuild=args.rebuild)
            points = service.points()
        finally:
            service.close()
    except (RAGError, embeddings.EmbeddingError, ValidationError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    for item in report.files:
        if item.state != "unchanged" or args.verbose:
            note = f" — {item.error}" if item.error else (f", документов {item.documents}" if item.documents else "")
            renamed = f" → {item.renamed_to}" if item.renamed_to else ""
            print(f"  {MARKS.get(item.state, ' ')} {item.state:9} {item.path}{renamed}{note}")
    print(report.summary())
    print(f"Точек в коллекции {report.collection}: {points}")
    return 1 if report.count("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
