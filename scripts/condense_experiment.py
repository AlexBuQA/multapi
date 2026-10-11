"""
Condense в чате (блок 5.5): как переписывается follow-up и что находит по нему поиск.

    python scripts/condense_experiment.py                         # таблица в консоль
    python scripts/condense_experiment.py --out var/condense.md   # ещё и в файл (UTF-8)

Зачем: на Windows llama3.2 переписала «А при оплате по счёту?» после вопроса о возврате в «Какова
сумма, которую я могу заплатить по счёту?» — тема возврата пропала, поиск нашёл не тот документ, и
чат ответил «По базе не нашёл». Варианты запроса к поиску:

- as_is — follow-up как есть, без condense: так его видит поиск без переписывания;
- plain — condense до замера: системное правило и диалог (condense_messages(..., examples=False));
- examples — condense сервиса: перед диалогом образцы CONDENSE_EXAMPLES (app/chat/knowledge.py);
- concat — без модели: прошлый вопрос пользователя и follow-up одной строкой.

Пары — первый вопрос, ответ на него (как в истории чата), follow-up и файл, который поиск должен
найти в top-5. Последняя пара — смена темы: condense не должен тянуть в новый вопрос старый. Нужны
Qdrant с проиндексированным корпусом, модель эмбеддингов и модель ответа из .env (condense в сервисе
— та же модель, LLM__DEFAULT_MODEL). Вызовов модели — 2 × пар, короткие; на CPU 1–3 минуты.
Код выхода: 0 — готово, 2 — ошибка.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAIRS = [
    ("Каков срок возврата оплаты?", "Вернуть оплату можно в течение 14 дней после списания.",
     "А при оплате по счёту?", ("billing/refunds_v2.docx",)),
    ("Сколько стоит тариф «Команда»?", "Тариф «Команда» стоит 4 900 ₽ в месяц.",
     "А «Бизнес»?", ("billing/faq-billing.md", "releases/release-notes_2025-h2.pdf")),
    ("Как подключить уведомления в Telegram?",
     "Откройте «Уведомления» → «Telegram», нажмите «Подключить», затем «Старт» в боте.",
     "А как их отключить?", ("notifications/telegram-notifications.html",)),
    ("Каков срок возврата оплаты?", "Вернуть оплату можно в течение 14 дней после списания.",
     "Как подключить уведомления в Telegram?", ("notifications/telegram-notifications.html",)),
]
VARIANTS = ("as_is", "plain", "examples", "concat")


def history(first: str, answer: str) -> list[Any]:
    from app.chat.domain import ChatMessage

    chat = uuid4()
    return [ChatMessage(chat_id=chat, role="user", content=first, tokens=0),
            ChatMessage(chat_id=chat, role="assistant", content=answer, tokens=0)]


def rewrite(llm: Any, messages: list[dict[str, Any]]) -> str | None:
    """Тот же разбор ответа, что в сервисе (clean_query), и та же температура 0. Предел ответа — как у
    модели /rag/query (RAG__MAX_TOKENS): берётся только первая строка, длина на неё не влияет."""
    from llama_index.core.llms import ChatMessage

    from app.chat.knowledge import clean_query

    reply = llm.chat([ChatMessage(role=m["role"], content=m["content"]) for m in messages], temperature=0.0)
    return clean_query(reply.message.content or "")


def query_for(variant: str, llm: Any, first: str, answer: str, follow_up: str) -> str:
    from app.chat.knowledge import condense_messages

    if variant == "as_is":
        return follow_up
    if variant == "concat":
        return f"{first} {follow_up}"
    messages = condense_messages(history(first, answer), follow_up, examples=variant == "examples")
    assert messages is not None
    return rewrite(llm, messages) or follow_up            # модель не ответила — как в сервисе: вопрос как есть


def run(service: Any, pairs: list[tuple[str, str, str, tuple[str, ...]]] | None = None) -> list[dict[str, Any]]:
    from app.services.rag_common import llm_error

    rows = []
    for first, answer, follow_up, expected in pairs or PAIRS:
        for variant in VARIANTS:
            started = time.perf_counter()
            try:
                query = query_for(variant, service.llm, first, answer, follow_up)
            except Exception as exc:
                explained = llm_error(exc, service.cfg)
                if explained is exc:
                    raise
                raise explained from exc
            seconds = round(time.perf_counter() - started, 1)
            found = service.retrieve(query)
            files = [node.metadata.get("source") for node in found.nodes]
            rank = next((i for i, f in enumerate(files, start=1) if f in expected), None)
            rows.append({"variant": variant, "first": first, "follow_up": follow_up, "query": query,
                         "top_file": files[0] if files else None, "top_score": round(found.top_score, 3),
                         "rank": rank, "confident": found.confident, "seconds": seconds})
            print(f"{variant} · {follow_up} → {query[:60]} · {'top-' + str(rank) if rank else 'мимо'}", file=sys.stderr)
    return rows


def report(rows: list[dict[str, Any]], model: str, when: datetime | None = None) -> str:
    pairs = len({(r["first"], r["follow_up"]) for r in rows})
    lines = [f"## Condense — {(when or datetime.now().astimezone()):%Y-%m-%d %H:%M}", "",
             f"Модель `{model}`; пар «вопрос → follow-up»: {pairs}. Попадание — нужный файл в top-5 поиска.", "",
             "| Вариант | Попаданий | Из них первым | Среднее время, с |", "|---|---:|---:|---:|"]
    for variant in VARIANTS:
        mine = [r for r in rows if r["variant"] == variant]
        hits = [r for r in mine if r["rank"]]
        timed = [r["seconds"] for r in mine]
        lines.append(f"| {variant} | {len(hits)} из {len(mine)} | {sum(1 for r in hits if r['rank'] == 1)} | "
                     f"{round(sum(timed) / len(timed), 1) if timed else 0.0} |")
    lines += ["", "| Вариант | Первый вопрос → follow-up | Запрос к поиску | top_score | Лучший файл | Нужный файл |",
              "|---|---|---|---:|---|---|"]
    for r in rows:
        query = r["query"].replace("|", "/")
        lines.append(f"| {r['variant']} | {r['first']} → {r['follow_up']} | {query} | {r['top_score']:.3f} | "
                     f"{r['top_file'] or '—'} | {'top-' + str(r['rank']) if r['rank'] else 'нет в top-5'} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Condense follow-up в чате: варианты запроса к поиску (блок 5.5)")
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
        service = RAGService(cfg).build(create=False)
        try:
            rows = run(service)
            markdown = report(rows, cfg.rag.llm_model or cfg.llm.default_model)
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
