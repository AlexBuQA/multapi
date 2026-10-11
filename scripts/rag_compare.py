"""
Две версии RAG на пяти вопросах (блок 5.3): LlamaIndex (app/services/rag.py) против версии без
фреймворка (app/services/rag_baremetal.py).

    python scripts/rag_compare.py                  # обе версии; коллекций нет — проиндексирует
    python scripts/rag_compare.py --rebuild        # пересобрать обе коллекции
    python scripts/rag_compare.py --only llamaindex
    python scripts/rag_compare.py --out var/rag_compare.md      # Markdown ещё и в файл (UTF-8)
    python scripts/rag_compare.py --json var/rag_compare.json   # ещё и полные ответы в JSON

Перед запуском — Qdrant (docker compose up -d qdrant) и Ollama с моделями эмбеддингов и ответа
из .env. На CPU прогон занимает минуты: десять ответов модели.

1. Пять вопросов: три «хороших» (ответ — в одном файле), один средний (ответ — в таблице PDF, а
   слова вопроса тянут и соседний файл про оплату картой) и один вне базы (честный отказ).
2. По каждому вопросу и версии — короткий ответ, top-1 файл и близость, релевантен ли top-1
   (да — файл с ответом, частично — соседний по теме, нет), есть ли в ответе ожидаемые факты (да —
   все части ответа, частично — не все; у вопроса об API это лимиты и код 429) и сработал ли отказ
   «В базе знаний ответа нет.» (порог RAG__MIN_SCORE или сама модель).
3. Строки кода конвейера (индексация и запрос) — по PIPELINE в обоих модулях, без импортов,
   docstring-ов, комментариев и пустых строк.

Вывод — Markdown для docs/rag.md. Код выхода: 0 — прогон прошёл, 2 — ошибка.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# kind: хороший — ответ в одном файле; средний — нужный факт в таблице PDF, рядом отвлекающий файл;
# вне базы — ответа нет. answer — файлы с ответом, related — соседние по теме. expect — части ответа:
# каждая часть — варианты написания; ответ полный, если в нём есть все части.
QUESTIONS: list[dict[str, Any]] = [
    {"kind": "хороший", "question": "Сколько стоит тариф «Команда» в месяц?",
     "answer": {"02_tariffs_and_payment.md"}, "related": set(), "expect": [["4 900", "4900"]]},
    {"kind": "хороший", "question": "Сколько запросов в минуту можно делать к API и что будет при превышении лимита?",
     "answer": {"04_api.md"}, "related": set(), "expect": [["60", "300", "1 200", "1200"], ["429"]]},
    {"kind": "хороший", "question": "Как часто счета и акты выгружаются в 1С?",
     "answer": {"08_integrations_1c.docx"}, "related": set(), "expect": [["30 минут"]]},
    {"kind": "средний", "question": "Ночью не проходит оплата картой — как быстро ответит поддержка?",
     "answer": {"09_support_rules.pdf"}, "related": {"03_refunds_and_card_payments.txt"},
     "expect": [["1 час"]]},
    {"kind": "вне базы", "question": "Есть ли интеграция с Битрикс24?",
     "answer": set(), "related": {"08_integrations_1c.docx"}, "expect": []},
]
VERSIONS = {"llamaindex": "LlamaIndex", "baremetal": "bare-metal"}
MODULES = {"llamaindex": "app/services/rag.py", "baremetal": "app/services/rag_baremetal.py"}
ANSWER_CHARS = 160


# ---------------------------------------------------------------- строки кода
def _definition(tree: ast.Module, dotted: str) -> ast.AST:
    node: Any = tree
    for part in dotted.split("."):
        node = next(child for child in node.body
                    if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and child.name == part)
    return node


def code_lines(path: Path, names: tuple[str, ...]) -> int:
    """Строки определений names без импортов, docstring-ов, комментариев и пустых строк."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    counted: set[int] = set()
    for name in names:
        node = _definition(tree, name)
        skip: set[int] = set()
        for inner in ast.walk(node):
            if isinstance(inner, ast.Import | ast.ImportFrom):
                skip.update(range(inner.lineno, inner.end_lineno + 1))
            if (isinstance(inner, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and inner.body
                    and isinstance(inner.body[0], ast.Expr) and isinstance(inner.body[0].value, ast.Constant)
                    and isinstance(inner.body[0].value.value, str)):
                skip.update(range(inner.body[0].lineno, inner.body[0].end_lineno + 1))
        start = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
        for number in range(start, node.end_lineno + 1):
            text = lines[number - 1].strip()
            if number not in skip and text and not text.startswith("#"):
                counted.add(number)
    return len(counted)


def loc_table() -> str:
    from app.services import rag, rag_baremetal

    pipelines = {"llamaindex": rag.PIPELINE, "baremetal": rag_baremetal.PIPELINE}
    rows = ["| Версия | Индексация | Запрос | Всего | Что считалось (строк) |", "|---|---:|---:|---:|---|"]
    for version, pipeline in pipelines.items():
        path = ROOT / MODULES[version]
        ingestion = code_lines(path, pipeline["ingestion"])
        query = code_lines(path, pipeline["query"])
        total = code_lines(path, pipeline["ingestion"] + pipeline["query"])
        names = ", ".join(f"`{name}` {code_lines(path, (name,))}" for name in pipeline["ingestion"] + pipeline["query"])
        rows.append(f"| {VERSIONS[version]} | {ingestion} | {query} | {total} | {names} |")
    return "\n".join(rows)


# ---------------------------------------------------------------- прогон
def relevance(item: dict[str, Any], top: str | None) -> str:
    if not item["answer"]:
        return "— (в базе нет)"
    if top in item["answer"]:
        return "да"
    return "частично" if top in item["related"] else "нет"


def has_fact(item: dict[str, Any], answer: str) -> str:
    from app.services.rag_common import is_no_answer

    if not item["expect"]:
        return "отказ" if is_no_answer(answer) else "**ответил без базы**"
    if is_no_answer(answer):
        return "отказ"
    text = answer.replace("\u00a0", " ").replace("\u202f", " ")      # неразрывные пробелы в «4 900»
    found = sum(any(variant in text for variant in part) for part in item["expect"])
    return "да" if found == len(item["expect"]) else "частично" if found else "нет"


def short(text: str) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= ANSWER_CHARS else text[:ANSWER_CHARS - 1] + "…"


def open_services(cfg: Any, only: str | None, rebuild: bool) -> dict[str, Any]:
    from app.services.rag import RAGService, no_implicit_event_loop
    from app.services.rag_baremetal import BareMetalRAG

    no_implicit_event_loop()
    services: dict[str, Any] = {}
    for version, factory in (("llamaindex", RAGService), ("baremetal", BareMetalRAG)):
        if only and version != only:
            continue
        started = time.perf_counter()
        service = factory(cfg).build(rebuild=rebuild)
        state = (f"проиндексировано фрагментов — {service.indexed}" if service.indexed
                 else "подключились к готовой")
        print(f"{VERSIONS[version]}: коллекция {service.collection}, {state} "
              f"({time.perf_counter() - started:.1f} с)", file=sys.stderr)
        services[version] = service
    return services


def run(cfg: Any, services: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for number, item in enumerate(QUESTIONS, 1):
        for version, service in services.items():
            started = time.perf_counter()
            answer = service.answer(item["question"])
            seconds = time.perf_counter() - started
            top = answer["sources"][0] if answer["sources"] else {"source": None, "score": 0.0}
            results.append({"n": number, "kind": item["kind"], "question": item["question"], "version": version,
                            "seconds": round(seconds, 1), "top_source": top["source"], "top_score": answer["top_score"],
                            "relevant": relevance(item, top["source"]), "fact": has_fact(item, answer["answer"]),
                            "below_threshold": answer["top_score"] < cfg.rag.min_score, **answer})
            print(f"{number}/{len(QUESTIONS)} {VERSIONS[version]}: {seconds:.1f} с", file=sys.stderr)
    return results


def report(cfg: Any, services: dict[str, Any], results: list[dict[str, Any]]) -> str:
    first = next(iter(services.values()))
    embed = first.embed_model.model_name if hasattr(first, "embed_model") else first.embedder.label
    lines = [
        f"## Прогон 5 вопросов — {datetime.now().astimezone():%Y-%m-%d %H:%M}",
        "",
        (f"Эмбеддинги `{embed}` (EMBEDDING_DIM={cfg.embedding_dim}), модель ответа "
         f"`{cfg.rag.llm_model or cfg.llm.default_model}`, top-k={cfg.rag.similarity_top_k}, "
         f"RAG__MIN_SCORE={cfg.rag.min_score}, фрагменты {cfg.rag.chunk_size}/{cfg.rag.chunk_overlap} токенов."),
        "",
        "| # | Тип | Вопрос | Версия | Ответ (кратко) | top-1 (близость) | Релевантен | Факт в ответе | Время, с |",
        "|---|---|---|---|---|---|---|---|---:|",
    ]
    for row in results:
        threshold = " (ниже порога — модель не вызывалась)" if row["below_threshold"] else ""
        lines.append(f"| {row['n']} | {row['kind']} | {row['question']} | {VERSIONS[row['version']]} | "
                     f"{short(row['answer'])}{threshold} | {row['top_source']} ({row['top_score']:.3f}) | "
                     f"{row['relevant']} | {row['fact']} | {row['seconds']} |")
    lines += ["", "Top-3 источника:", ""]
    for row in results:
        sources = ", ".join(f"{s['source']} {s['score']:.3f}" for s in row["sources"])
        lines.append(f"- {row['n']}, {VERSIONS[row['version']]}: {sources}")
    lines += ["", "## Строки кода конвейера", "", loc_table(), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG: LlamaIndex против bare-metal на пяти вопросах (блок 5.3)")
    parser.add_argument("--rebuild", action="store_true", help="пересобрать коллекции обеих версий")
    parser.add_argument("--only", choices=sorted(VERSIONS), help="только одна версия")
    parser.add_argument("--out", type=Path, help="сохранить Markdown в файл (UTF-8; перенаправление > в Windows "
                                                 "PowerShell 5.1 записало бы UTF-16)")
    parser.add_argument("--json", type=Path, help="сохранить полные ответы и источники в JSON")
    parser.add_argument("--loc", action="store_true", help="только таблица строк кода, без моделей и Qdrant")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if args.loc:
        print(loc_table())
        return 0

    from pydantic import ValidationError

    from app.core.config import get_settings
    from app.observability.logging import setup_logging
    from app.services.embeddings import EmbeddingError
    from app.services.rag_common import RAGError

    setup_logging("WARNING", stream=sys.stderr)
    services: dict[str, Any] = {}
    try:
        cfg = get_settings()
        services = open_services(cfg, args.only, args.rebuild)
        results = run(cfg, services)
        markdown = report(cfg, services, results)
        print(markdown)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(markdown, encoding="utf-8")
            print(f"Markdown — {args.out}", file=sys.stderr)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Ответы целиком — {args.json}", file=sys.stderr)
        return 0
    except (RAGError, EmbeddingError, ValidationError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    finally:
        for service in services.values():
            service.close()


if __name__ == "__main__":
    sys.exit(main())
