"""
База знаний для векторного поиска (блок 5.2): фрагменты документов из data/ с метаданными.

Источники (CORPUS) — файлы предметной области, а не тестовые строки:
- data/help_center.jsonl — статьи справочного центра (блок 5.1): одна строка — один фрагмент;
- data/release_notes.md — заметки о выпусках «Что нового», у каждой своя дата;
- data/faq.md — частые вопросы поддержки;
- data/archive/*.md — прежние редакции статей (status: archived): тарифы 2025 года, старые
  сроки возврата и требования к паролю. В выдаче они нужны только для истории — фильтр
  must_not status=archived убирает их.

Markdown: YAML front matter задаёт значения по умолчанию для файла (title, category, product,
status, created_at), каждый раздел «## …» — один фрагмент. Комментарий сразу под заголовком
уточняет фрагмент: «<!-- id: RN-2026-09-29; created_at: 2026-09-29; category: billing -->».
Текст фрагмента — заголовок и абзацы раздела: заголовок несёт смысл («Можно ли войти без
пароля?»), и модель эмбеддингов должна его видеть.

Payload точки (Chunk.payload): source, text, created_at, category — обязательные по заданию —
и doc_id, title, product, status, chunk_index. category — предметное поле: раздел справки
(account, billing, api, …), по нему фильтруется поиск.

Id точки — uuid5(NAMESPACE, «source#key»): одинаковый при каждом запуске, поэтому повторная
загрузка перезаписывает те же точки, а не добавляет копии. key — id статьи (KB-030, RN-…), а
если его нет — номер раздела в файле. Номер раздела сдвигается, когда в начало файла
дописывают новый раздел, поэтому у заметок и вопросов id задан явно.
"""
from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
# Пространство имён uuid5 для id точек. Сменить — значит получить другие id и дубли при
# следующей загрузке, поэтому это константа, а не настройка.
NAMESPACE = uuid.UUID("6f1d3c2e-5b7a-4e0f-9a8d-2c4b6e8f0a15")
CORPUS = ("help_center.jsonl", "release_notes.md", "faq.md", "archive/*.md")
STATUSES = ("actual", "archived")
REQUIRED = ("title", "category", "product", "status", "created_at")

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_META = re.compile(r"^<!--\s*(.*?)\s*-->\s*$")


class CorpusError(ValueError):
    """Ошибка в файле базы знаний: где и что исправить."""


@dataclass(frozen=True)
class Chunk:
    source: str          # файл относительно data/, например release_notes.md
    key: str             # стабильный ключ внутри файла: id статьи или номер раздела
    chunk_index: int     # номер фрагмента в файле, с 0
    doc_id: str
    title: str
    text: str
    created_at: str      # RFC 3339, UTC: 2026-09-29T00:00:00Z
    category: str
    product: str
    status: str          # actual | archived

    @property
    def point_id(self) -> str:
        return point_id(self.source, self.key)

    @property
    def payload(self) -> dict[str, Any]:
        return {"source": self.source, "text": self.text, "created_at": self.created_at,
                "category": self.category, "product": self.product, "status": self.status,
                "doc_id": self.doc_id, "title": self.title, "chunk_index": self.chunk_index}


def point_id(source: str, key: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{source}#{key}"))


def to_rfc3339(value: Any, where: str) -> str:
    """Дата из файла (2026-09-29 или с временем) — в RFC 3339 UTC, как ждёт индекс DATETIME."""
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime(value.year, value.month, value.day)
    else:
        try:
            moment = datetime.fromisoformat(str(value).strip())
        except ValueError as exc:
            raise CorpusError(f"{where}: created_at «{value}» — не дата, нужен вид 2026-09-29") from exc
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC).replace(tzinfo=None)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _chunk(source: str, key: str, index: int, fields: dict[str, Any], where: str) -> Chunk:
    missing = [name for name in (*REQUIRED, "text", "doc_id") if not str(fields.get(name) or "").strip()]
    if missing:
        raise CorpusError(f"{where}: не хватает полей {', '.join(missing)}")
    status = str(fields["status"]).strip()
    if status not in STATUSES:
        raise CorpusError(f"{where}: status «{status}» — допустимо {', '.join(STATUSES)}")
    return Chunk(source=source, key=key, chunk_index=index, doc_id=str(fields["doc_id"]).strip(),
                 title=str(fields["title"]).strip(), text=str(fields["text"]).strip(),
                 created_at=to_rfc3339(fields["created_at"], where), category=str(fields["category"]).strip(),
                 product=str(fields["product"]).strip(), status=status)


def read_jsonl(path: Path, source: str) -> list[Chunk]:
    chunks = []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, line in enumerate(lines):
        where = f"{source}, строка {index + 1}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"{where}: не JSON ({exc.msg})") from exc
        doc_id = str(row.get("id") or "").strip()
        if not doc_id:
            raise CorpusError(f"{where}: нет id")
        chunks.append(_chunk(source, doc_id, index, {**row, "doc_id": doc_id}, where))
    return chunks


def _parse_meta(line: str, where: str) -> dict[str, str]:
    match = _META.match(line.strip())
    if not match:
        return {}
    meta = {}
    for part in filter(None, (item.strip() for item in match.group(1).split(";"))):
        name, sep, value = part.partition(":")
        if not sep:
            raise CorpusError(f"{where}: в комментарии «{part}» нет двоеточия, нужен вид «ключ: значение»")
        meta[name.strip()] = value.strip()
    return meta


def read_markdown(path: Path, source: str) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    defaults: dict[str, Any] = {}
    match = _FRONT_MATTER.match(raw)
    if match:
        defaults = yaml.safe_load(match.group(1)) or {}
        raw = raw[match.end():]
    sections: list[tuple[str, list[str], int]] = []
    for number, line in enumerate(raw.split("\n"), start=1):
        if line.startswith("## "):
            sections.append((line[3:].strip(), [], number))
        elif sections:
            sections[-1][1].append(line)
    chunks = []
    stem = source.rsplit("/", 1)[-1].removesuffix(".md")
    for index, (heading, body, line_no) in enumerate(sections):
        where = f"{source}, раздел «{heading}» (строка {line_no})"
        meta: dict[str, str] = {}
        if body and body[0].strip().startswith("<!--"):
            meta = _parse_meta(body[0], where)
            body = body[1:]
        text = "\n".join(line.rstrip() for line in body).strip()
        key = meta.get("id") or str(index)
        fields = {**defaults, "title": heading, **meta, "text": f"{heading}\n{text}" if text else "",
                  "doc_id": meta.get("id") or f"{stem}-{index}"}
        chunks.append(_chunk(source, key, index, fields, where))
    if not chunks:
        raise CorpusError(f"{source}: нет ни одного раздела «## …»")
    return chunks


def corpus_files(data_dir: Path = DATA_DIR, patterns: Iterable[str] = CORPUS) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        found = sorted(data_dir.glob(pattern))
        if not found:
            raise CorpusError(f"В {data_dir} нет файлов {pattern}")
        files.extend(found)
    return files


def load_chunks(data_dir: Path = DATA_DIR, patterns: Iterable[str] = CORPUS) -> list[Chunk]:
    """Все фрагменты базы знаний. Одинаковый id у двух фрагментов — ошибка, а не тихая перезапись."""
    chunks: list[Chunk] = []
    for path in corpus_files(data_dir, patterns):
        source = path.relative_to(data_dir).as_posix()
        reader = read_jsonl if path.suffix == ".jsonl" else read_markdown
        chunks.extend(reader(path, source))
    seen: dict[str, Chunk] = {}
    for chunk in chunks:
        other = seen.setdefault(chunk.point_id, chunk)
        if other is not chunk:
            raise CorpusError(f"Повторяется ключ «{chunk.key}» в {chunk.source}: id точек совпадут")
    doc_ids: dict[str, Chunk] = {}
    for chunk in chunks:
        other = doc_ids.setdefault(chunk.doc_id, chunk)
        if other is not chunk:
            raise CorpusError(f"doc_id «{chunk.doc_id}» встречается в {other.source} и {chunk.source}")
    return chunks
