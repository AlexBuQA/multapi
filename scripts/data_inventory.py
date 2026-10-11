"""
Опись корпуса RAG (блок 5.5): число файлов, форматы, размер, категории — в docs/data_inventory.md.

    python scripts/data_inventory.py                              # напечатать опись
    python scripts/data_inventory.py --out docs/data_inventory.md # перезаписать документ

Читает те же файлы и теми же ридерами, что scripts/ingest.py (app/services/ingestion.py), но без
Qdrant и эмбеддингов: документы — страницы PDF и разделы Markdown, фрагменты — нарезка RAG__CHUNKING
(semantic считается как recursive: ей нужна модель эмбеддингов). Тест tests/unit/test_ingestion.py
сверяет числа в docs/data_inventory.md с папкой data/ — опись не отстаёт от корпуса.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FORMAT_NAMES = {"html": "HTML", "docx": "DOCX", "md": "Markdown", "pdf": "PDF", "txt": "TXT"}
READERS = {"html": "HTMLTagReader: `<article>`", "docx": "DocxReader (docx2txt)",
           "md": "MarkdownReader: раздел — документ", "pdf": "PyMuPDFReader: страница — документ",
           "txt": "текст как есть"}


@dataclass
class Row:
    path: str
    category: str
    fmt: str
    size: int
    documents: int
    chunks: int
    author: str
    created_at: str
    version: str


def collect(root: Path, chunk_size: int, chunk_overlap: int, strategy: str) -> list[Row]:
    from app.services import chunking
    from app.services.ingestion import list_files, read_file

    splitter = chunking.build_splitter("recursive" if strategy == "semantic" else strategy,
                                       chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    rows = []
    for path in list_files(root):
        docs = read_file(path, root)
        meta = docs[0].metadata
        rows.append(Row(meta["source"], meta["category"], meta["format"], path.stat().st_size, len(docs),
                        len(splitter.get_nodes_from_documents(docs)), meta.get("author", "—"),
                        meta.get("created_at", "—"), meta.get("version", "—")))
    return rows


def kb(size: int) -> str:
    return f"{size / 1024:.1f}".replace(".", ",")


def render(rows: list[Row], root_name: str, chunking_label: str) -> str:
    total_size = sum(r.size for r in rows)
    by_format: dict[str, list[Row]] = defaultdict(list)
    by_category: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_format[row.fmt].append(row)
        by_category[row.category].append(row)
    lines = [
        "# Опись корпуса RAG (блок 5.5)",
        "",
        ("Сгенерировано `python scripts/data_inventory.py --out docs/data_inventory.md` — числа сверяет тест "
         "`tests/unit/test_ingestion.py`. Корпус собран из базы знаний проекта (`data/help_center.jsonl`, "
         "`data/faq.md`, `data/release_notes.md`) скриптом `scripts/build_corpus.py`; что в `data/` не входит "
         "в корпус — `data/.ragignore`. Размеры — файлов с переводами строк LF, как в архиве и в git; после "
         "checkout в Windows с core.autocrlf HTML и Markdown на диске на несколько байт больше."),
        "",
        "## Итог",
        "",
        f"- Файлов: **{len(rows)}** в {len(by_category)} категориях (`{root_name}/<категория>/…`).",
        "- Форматы: " + ", ".join(f"{FORMAT_NAMES[f]} — {len(by_format[f])}" for f in sorted(by_format,
                                  key=lambda f: -len(by_format[f]))) + ".",
        f"- Общий размер: **{kb(total_size)} КБ** ({total_size} байт).",
        (f"- Документов LlamaIndex: **{sum(r.documents for r in rows)}** (страница PDF и раздел Markdown — "
         f"отдельный документ), фрагментов при нарезке `{chunking_label}`: **{sum(r.chunks for r in rows)}**."),
        (f"- С автором в самом документе: {sum(1 for r in rows if r.author != '—')}, с редакцией в имени "
         f"файла: {sum(1 for r in rows if r.version != '—')}."),
        "",
        "## По форматам",
        "",
        "| Формат | Ридер | Файлов | Документов | Фрагментов | Размер, КБ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for fmt in sorted(by_format, key=lambda f: -len(by_format[f])):
        group = by_format[fmt]
        lines.append(f"| {FORMAT_NAMES[fmt]} | {READERS[fmt]} | {len(group)} | {sum(r.documents for r in group)} | "
                     f"{sum(r.chunks for r in group)} | {kb(sum(r.size for r in group))} |")
    lines += ["", "## По категориям", "", "Категория — первая папка пути (`data/billing/…` → `billing`).", "",
              "| Категория | Файлов | Форматы | Авторы | Размер, КБ |", "|---|---:|---|---|---:|"]
    for category in sorted(by_category):
        group = by_category[category]
        formats = Counter(r.fmt for r in group)
        authors = sorted({r.author for r in group if r.author != "—"})
        lines.append(f"| `{category}` | {len(group)} | "
                     + ", ".join(f"{FORMAT_NAMES[f]} {n}" for f, n in sorted(formats.items()))
                     + f" | {'; '.join(authors) or '—'} | {kb(sum(r.size for r in group))} |")
    lines += ["", "## Файлы", "", "| Файл | Формат | КБ | Документов | Фрагментов | Автор | Создан | Редакция |",
              "|---|---|---:|---:|---:|---|---|---|"]
    for row in rows:
        lines.append(f"| `{row.path}` | {FORMAT_NAMES[row.fmt]} | {kb(row.size)} | {row.documents} | {row.chunks} | "
                     f"{row.author} | {row.created_at} | {row.version} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Опись корпуса RAG: файлы, форматы, размер (блок 5.5)")
    parser.add_argument("path", nargs="?", type=Path, help="папка корпуса; по умолчанию RAG__DATA_DIR")
    parser.add_argument("--out", type=Path, help="записать Markdown в файл, например docs/data_inventory.md")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from app.core.config import get_settings
    from app.services import chunking

    cfg = get_settings().rag
    root = (args.path or cfg.data_dir).resolve()
    label = chunking.label("recursive" if cfg.chunking == "semantic" else cfg.chunking,
                           chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    text = render(collect(root, cfg.chunk_size, cfg.chunk_overlap, cfg.chunking), root.name, label)
    if args.out:
        args.out.write_text(text, encoding="utf-8", newline="\n")
        print(f"Опись записана: {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
