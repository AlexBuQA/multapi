"""
Цитаты [N] у модели ответа (блок 5.5): какой вариант промпта заставляет llama3.2 ставить номера
фрагментов и не ломает отказ на вопрос, ответа на который в базе нет.

    python scripts/citation_experiment.py                         # все варианты, таблица в консоль
    python scripts/citation_experiment.py --out var/citations.md  # ещё и в файл (UTF-8)
    python scripts/citation_experiment.py --variants rules,example

Зачем: на Windows llama3.2 (3B) отвечала «Срок вернуть оплату составляет 14 дней после списания.» без
[1] — и с правилом в системном промпте, и с напоминанием последней строкой перед ответом. Варианты
(тексты — из app/services/rag_common.py):

- rules — номер после каждого предложения: правило RULES и напоминание INLINE_FORMAT («номер
  фрагмента — в конце каждого предложения»);
- example — тот же промпт и перед вопросом пример диалога: контекст из двух фрагментов на
  постороннюю тему (библиотека — чтобы модель не перенесла факты примера в ответ) и ответ с [1], [2];
- sources — RULES, SOURCES_RULE и напоминание ANSWER_FORMAT: последняя строка «Источники: [1], [3]».
  Выиграл замер на Windows и стал промптом сервиса (chat_messages). Первый замер шёл с образцом фраз
  «Первое утверждение [1]…» в RULES; модель вставила его в ответ в боте, образец убран, sources
  перемерен (docs/rag.md, «Цитаты (Windows)»);
- example_sources — пример диалога, ответ в котором заканчивается строкой «Источники: …».

Вопросы: 5 по базе (ответ есть) и 2 о продукте без ответа в базе — их близость выше порога, отказ
на них даёт промпт; вариант, на котором модель начинает на них отвечать, не годится. Поиск — один
раз на вопрос, фрагменты одни и те же для всех вариантов. Нужны Qdrant с проиндексированным корпусом
(python scripts/ingest.py data/), модель эмбеддингов и модель ответа из .env. Вызовов модели —
вариантов × 7; на CPU это 5–15 минут. Код выхода: 0 — готово, 2 — ошибка.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IN_BASE = [
    "Каков срок возврата оплаты?",
    "Сколько стоит тариф «Бизнес» в месяц?",
    "Сколько действует ссылка для сброса пароля?",
    "Сколько раз сервис повторяет доставку вебхука, если мой сервер не ответил 200?",
    "Как подключить уведомления в Telegram?",
]
NOT_IN_BASE = ["Можно ли оплатить тариф криптовалютой?", "Работает ли приложение на Apple Watch?"]

DEMO_FRAGMENTS = [("Книгу выдают на 21 день. Продлить срок можно один раз в разделе «Мои книги».", "library-rules.md", None),
                  ("За каждый день просрочки начисляется штраф 10 рублей.", "library-fines.md", None)]
DEMO_QUESTION = "На сколько дней выдают книгу и что будет, если вернуть её позже?"
DEMO_INLINE = ("Книгу выдают на 21 день, продлить срок можно один раз [1]. "
               "За каждый день просрочки начисляется штраф 10 рублей [2].")
DEMO_LINE = ("Книгу выдают на 21 день, продлить срок можно один раз. "
             "За каждый день просрочки начисляется штраф 10 рублей.\nИсточники: [1], [2]")


def variants(product: str) -> dict[str, Callable[[str, str], list[Any]]]:
    """Вариант → функция (вопрос, нумерованный контекст) → сообщения для модели."""
    from llama_index.core.llms import ChatMessage, MessageRole

    from app.services.rag_common import (
        ANSWER_FORMAT,
        INLINE_FORMAT,
        answer_template,
        numbered_context,
        system_prompt,
    )

    inline_template, line_template = answer_template(INLINE_FORMAT), answer_template(ANSWER_FORMAT)
    inline_system, line_system = system_prompt(product, sources_line=False), system_prompt(product)
    demo_context = numbered_context(DEMO_FRAGMENTS)

    def system(text: str) -> Any:
        return ChatMessage(role=MessageRole.SYSTEM, content=text)

    def user(template: str, question: str, context: str) -> Any:
        return ChatMessage(role=MessageRole.USER, content=template.format(context_str=context, query_str=question))

    def assistant(text: str) -> Any:
        return ChatMessage(role=MessageRole.ASSISTANT, content=text)

    def rules(question: str, context: str) -> list[Any]:
        return [system(inline_system), user(inline_template, question, context)]

    def example(question: str, context: str) -> list[Any]:
        return [system(inline_system), user(inline_template, DEMO_QUESTION, demo_context),
                assistant(DEMO_INLINE), user(inline_template, question, context)]

    def sources(question: str, context: str) -> list[Any]:
        return [system(line_system), user(line_template, question, context)]

    def example_sources(question: str, context: str) -> list[Any]:
        return [system(line_system), user(line_template, DEMO_QUESTION, demo_context),
                assistant(DEMO_LINE), user(line_template, question, context)]

    return {"rules": rules, "example": example, "sources": sources, "example_sources": example_sources}


def run(service: Any, product: str, names: list[str], questions: dict[str, list[str]]) -> list[dict[str, Any]]:
    from app.services.rag_common import cited_numbers, is_no_answer, llm_error

    builders = variants(product)
    found = {q: service.retrieve(q) for group in questions.values() for q in group}
    rows = []
    for name in names:
        for group, items in questions.items():
            for question in items:
                retrieval = found[question]
                if not retrieval.confident:
                    rows.append({"variant": name, "group": group, "question": question, "answer": None,
                                 "refused": True, "cited": [], "seconds": 0.0, "guard": True})
                    continue
                started = time.perf_counter()
                try:
                    reply = service.llm.chat(builders[name](question, retrieval.context()))
                except Exception as exc:
                    explained = llm_error(exc, service.cfg)
                    if explained is exc:
                        raise
                    raise explained from exc
                answer = (reply.message.content or "").strip()
                row = {"variant": name, "group": group, "question": question, "answer": answer,
                       "refused": is_no_answer(answer), "cited": cited_numbers(answer, len(retrieval.nodes)),
                       "seconds": round(time.perf_counter() - started, 1), "guard": False}
                rows.append(row)
                print(f"{name} · {question[:40]} → {'отказ' if row['refused'] else row['cited'] or 'без [N]'} "
                      f"({row['seconds']} с)", file=sys.stderr)
    return rows


def summarize(rows: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    out = []
    for name in names:
        mine = [r for r in rows if r["variant"] == name]
        inside = [r for r in mine if r["group"] == "in"]
        answered = [r for r in inside if not r["refused"]]
        outside = [r for r in mine if r["group"] == "near"]
        timed = [r["seconds"] for r in mine if not r["guard"]]
        out.append({"variant": name, "answered": len(answered), "in": len(inside),
                    "cited": sum(1 for r in answered if r["cited"]),
                    "refused_near": sum(1 for r in outside if r["refused"]), "near": len(outside),
                    "seconds": round(sum(timed) / len(timed), 1) if timed else 0.0})
    return out


def report(rows: list[dict[str, Any]], names: list[str], model: str, when: datetime | None = None) -> str:
    lines = [f"## Цитаты в ответах — {(when or datetime.now().astimezone()):%Y-%m-%d %H:%M}", "",
             f"Модель ответа `{model}`; вопросов по базе {len(IN_BASE)}, без ответа в базе {len(NOT_IN_BASE)}.", "",
             "| Вариант | Ответил по базе | С цитатами [N] | Отказ без ответа в базе | Среднее время, с |",
             "|---|---:|---:|---:|---:|"]
    for s in summarize(rows, names):
        lines.append(f"| {s['variant']} | {s['answered']} из {s['in']} | {s['cited']} из {s['answered']} | "
                     f"{s['refused_near']} из {s['near']} | {s['seconds']} |")
    lines += ["", "| Вариант | Вопрос | Ответ | [N] |", "|---|---|---|---|"]
    for r in rows:
        text = "отказ порогом" if r["guard"] else " ".join((r["answer"] or "").split()).replace("|", "/")
        lines.append(f"| {r['variant']} | {r['question']} | {text[:220]} | {', '.join(map(str, r['cited'])) or '—'} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Какой промпт заставляет модель ставить цитаты [N] (блок 5.5)")
    parser.add_argument("--variants", default="rules,example,sources,example_sources",
                        help="варианты через запятую: rules, example, sources, example_sources")
    parser.add_argument("--out", type=Path, help="сохранить Markdown в файл (UTF-8)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from pydantic import ValidationError

    from app.core.config import get_settings
    from app.observability.logging import setup_logging
    from app.services.embeddings import EmbeddingError
    from app.services.rag import RAGService, no_implicit_event_loop
    from app.services.rag_common import RAGError

    setup_logging("WARNING", stream=sys.stderr)
    no_implicit_event_loop()
    try:
        cfg = get_settings()
        product = cfg.support.product_name
        names = [n.strip() for n in args.variants.split(",") if n.strip()]
        unknown = set(names) - set(variants(product))
        if unknown:
            raise ValueError(f"Неизвестные варианты: {', '.join(sorted(unknown))}")
        service = RAGService(cfg).build(create=False)
        try:
            rows = run(service, product, names, {"in": IN_BASE, "near": NOT_IN_BASE})
            markdown = report(rows, names, cfg.rag.llm_model or cfg.llm.default_model)
        finally:
            service.close()
    except (RAGError, EmbeddingError, ValidationError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    print(markdown)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown, encoding="utf-8")
        print(f"Markdown — {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
