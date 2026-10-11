"""
Документы корпуса RAG (блок 5.5): загрузка нового файла и переиндексация без перезапуска сервиса.

    POST /documents/upload   (X-Admin-Token, multipart: file)  → 202 {"status": "queued", "file": "uploads/x.pdf", …}
    GET  /documents/status?file=uploads/x.pdf                   → {"state": "new", "indexed": true, …}
    POST /documents/reindex  (X-Admin-Token)                    → 202: весь корпус, новые и изменённые файлы

Загрузка сохраняет файл в RAG__UPLOAD_DIR (data/uploads/, категория uploads) и ставит в
BackgroundTasks индексацию этого файла: IngestionPipeline с DocstoreStrategy.UPSERTS — тот же
конвейер, что у scripts/ingest.py. Ответ 202 уходит сразу; через несколько секунд (bge-m3 на CPU —
до минуты) фрагменты в коллекции, и /rag/query и чаты находят их. Файл с тем же именем — новая версия:
UPSERTS заменит его фрагменты, а не добавит копии. Не прочитался — переименован в .failed, причина —
в GET /documents/status и в логе ingest_file_failed.

Пополнять базу знаний — право администратора, поэтому загрузка и переиндексация — по
X-Admin-Token (ADMIN_TOKEN), как /chats/admin/*. Имя файла — только последняя часть пути
(«../../app/main.py» → main.py, а .py не принимается); форматы — PDF, DOCX, HTML, Markdown, TXT;
размер — до RAG__UPLOAD_MAX_MB (больше — 413 ещё до разбора формы, app/core/body_limit.py).
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Request, UploadFile

from app.admin.deps import require_admin
from app.chat.domain import RequestError
from app.deps.providers import IngestionDep, SettingsDep
from app.observability.logging import get_logger
from app.schemas.rag import DocumentStatus, UploadAccepted
from app.services.ingestion import FORMATS, FileResult, IngestionService

router = APIRouter(prefix="/documents", tags=["documents"])
log = get_logger()

NAME_MAX = 150
CHUNK = 1024 * 1024
_UNSAFE = re.compile(r"[\x00-\x1f<>:\"|?*]")


def safe_name(filename: str | None) -> str:
    """Имя файла без пути (и Windows-, и POSIX-разделители), без управляющих и запрещённых в Windows
    символов, не скрытое, с поддерживаемым расширением."""
    name = re.split(r"[\\/]", (filename or "").strip())[-1].strip()
    if not name or name.startswith(".") or len(name) > NAME_MAX or _UNSAFE.search(name):
        raise RequestError(422, "bad_file_name", f"Недопустимое имя файла: латиница, кириллица, цифры, «-», «_», "
                                                 f"«.», до {NAME_MAX} символов.")
    if Path(name).suffix.lower() not in FORMATS:
        raise RequestError(415, "unsupported_format",
                           f"Формат не поддерживается: нужны {', '.join(sorted({s.lstrip('.') for s in FORMATS}))}.")
    return name


async def read_limited(file: UploadFile, limit: int) -> bytes:
    data = bytearray()
    while chunk := await file.read(CHUNK):
        data.extend(chunk)
        if len(data) > limit:
            raise RequestError(413, "file_too_large", f"Файл больше {limit // CHUNK} МБ (RAG__UPLOAD_MAX_MB).")
    return bytes(data)


def _write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part")      # скрытый: ingest его не возьмёт
    temporary.write_bytes(data)
    os.replace(temporary, target)


def index_files(ingestion: IngestionService, rag: Any, files: list[Path] | None) -> None:
    """Фоновая задача (BackgroundTasks — в пуле потоков): индексация и, если RAG ещё не был
    подключён к коллекции (её не было на старте), подключение."""
    try:
        report = ingestion.run(files)
    except Exception as exc:  # noqa: BLE001 — фоновая задача: ответ уже ушёл, итог — в лог и статус
        log.error("ingest_background_failed", files=[p.name for p in files or []], error=str(exc)[:300])
        for path in files or []:
            relative = path.relative_to(ingestion.root).as_posix()
            ingestion.status[relative] = FileResult(relative, "failed", error=str(exc)[:300])
        return
    if rag is not None and report.total:
        try:
            rag.ensure_ready()
        except Exception as exc:  # noqa: BLE001 — подключится при следующем запросе (ensure_ready)
            log.warning("rag_not_ready", error=str(exc)[:300])


@router.post(
    "/upload",
    status_code=202,
    response_model=UploadAccepted,
    dependencies=[Depends(require_admin)],
    summary="Загрузить документ в базу знаний (RAG)",
    description="Сохраняет файл в data/uploads/ и индексирует его в фоне (BackgroundTasks, IngestionPipeline, "
                "UPSERTS). Ответ 202 — сразу; результат — GET /documents/status. Нужен заголовок X-Admin-Token.",
    responses={401: {"description": "Нет или неверный X-Admin-Token"}, 413: {"description": "Файл слишком большой"},
               415: {"description": "Формат не поддерживается"}, 503: {"description": "RAG выключен"}},
)
async def upload(file: UploadFile, request: Request, background: BackgroundTasks, ingestion: IngestionDep,
                 settings: SettingsDep) -> UploadAccepted:
    name = safe_name(file.filename)
    data = await read_limited(file, settings.rag.upload_max_mb * CHUNK)
    if not data:
        raise RequestError(422, "empty_file", "Файл пустой.")
    target = settings.rag.uploads / name
    replaced = target.exists()
    await asyncio.to_thread(_write, target, data)
    relative = target.resolve().relative_to(ingestion.root).as_posix()
    ingestion.queue(relative)
    background.add_task(index_files, ingestion, getattr(request.app.state, "rag", None), [target])
    log.info("document_uploaded", file=relative, size=len(data), replaced=replaced)
    return UploadAccepted(file=relative, size=len(data), replaced=replaced,
                          status_url=f"/documents/status?file={relative}")


@router.get(
    "/status",
    response_model=DocumentStatus,
    summary="Результат индексации файла",
    description="queued — ждёт очереди; new / changed / unchanged — в индексе; failed — не прочитался (.failed) или "
                "Qdrant и модель эмбеддингов были недоступны. Статусы — в памяти процесса с его старта.",
)
async def status(file: str, ingestion: IngestionDep) -> DocumentStatus:
    item = ingestion.status.get(file.strip().lstrip("/"))
    if item is None:
        raise RequestError(404, "document_not_found", "Этот файл не индексировался с запуска сервиса.")
    return DocumentStatus(file=item.path, state=item.state, documents=item.documents, error=item.error,
                          renamed_to=item.renamed_to, indexed=item.state in {"new", "changed", "unchanged"})


@router.post(
    "/reindex",
    status_code=202,
    dependencies=[Depends(require_admin)],
    summary="Переиндексировать корпус",
    description="Весь RAG__DATA_DIR в фоне: новые и изменённые файлы, удаление фрагментов исчезнувших. "
                "Неизменённые пропускаются (UPSERTS). Итог — в логе ingest_done.",
)
async def reindex(request: Request, background: BackgroundTasks, ingestion: IngestionDep) -> dict[str, str]:
    background.add_task(index_files, ingestion, getattr(request.app.state, "rag", None), None)
    return {"status": "queued", "collection": ingestion.collection}
