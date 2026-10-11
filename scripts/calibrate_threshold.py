"""
Калибровка порога отказа RAG__SCORE_THRESHOLD (блок 5.5): распределение близости лучшего фрагмента
на корпусе для вопросов по базе и вопросов не по ней.

    python scripts/calibrate_threshold.py                          # таблицы в консоль
    python scripts/calibrate_threshold.py --out var/threshold.md   # ещё и в файл (UTF-8) — для docs/rag.md

Score-guard сравнивает с порогом top_score — косинусную близость лучшего из 10 кандидатов. Шкала
своя у каждой модели эмбеддингов: у OpenAI text-embedding-3-small близкие тексты — 0.3–0.6, у bge-m3 —
0.5–0.8, поэтому порог берётся не «как в курсе», а по распределению на своём корпусе:

- «по базе» — 28 вопросов golden dataset блока 5.4 (tests/eval/retrieval_dataset.json): корпус
  блока 5.5 собран из той же базы знаний, ответы на них в нём есть;
- «посторонние» — OFF_TOPIC: темы, к продукту не относящиеся (борщ, погода, курс доллара). Их и
  должен отсекать score-guard;
- «о продукте, ответа нет» — NOT_IN_BASE: интеграция с Битрикс24, оплата криптовалютой, тёмная тема.
  Слова у них те же, что в базе («тариф», «оплата», «приложение»), поэтому близость — как у вопросов
  по базе; порог их не отличит, отказ на них — второй слой, промпт («По базе не нашёл…»). В подбор
  порога они не входят, скрипт только показывает, сколько из них дойдёт до модели;
- «follow-up» — короткие уточнения без истории («А при оплате по счёту?»): так их видит поиск без
  condense. Видно, почему в чате вопрос сначала переписывается.

Рекомендуемый порог — середина зазора между посторонними и вопросами по базе, если они разделяются
(максимум посторонних < минимум по базе), иначе — порог с наименьшим числом ошибок (вопрос по базе
отклонён + посторонний пропущен к модели); при равенстве ошибок — меньший порог: лишний вызов модели
дешевле отказа на вопрос, ответ на который есть (второй слой — промпт — всё равно ответит «по базе не
нашёл»).

Нужны Qdrant с проиндексированным корпусом (python scripts/ingest.py data/) и модель эмбеддингов из
.env. Модель ответа не вызывается. Код выхода: 0 — готово, 2 — ошибка.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "tests" / "eval" / "retrieval_dataset.json"
OFF_TOPIC = [
    "Как приготовить борщ?",
    "Какая погода завтра в Москве?",
    "Кто написал роман «Война и мир»?",
    "Как часто поливать фикус зимой?",
    "Какой курс доллара на сегодня?",
    "Как настроить Wi-Fi роутер дома?",
    "Посоветуй фильм на вечер",
    "Сколько планет в Солнечной системе?",
]
NOT_IN_BASE = [
    "Есть ли интеграция с Битрикс24?",
    "Можно ли оплатить тариф криптовалютой?",
    "Есть ли в личном кабинете тёмная тема?",
    "Работает ли приложение на Apple Watch?",
]
FOLLOW_UPS = ["А при оплате по счёту?", "А для них?", "А на тарифе «Бизнес»?", "А сколько это стоит?"]
GROUPS = {"in": "по базе", "off": "посторонние", "near": "о продукте, ответа нет", "follow": "follow-up без истории"}


def questions() -> dict[str, list[str]]:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    return {"in": [item["question"] for item in data["questions"]], "off": OFF_TOPIC, "near": NOT_IN_BASE,
            "follow": FOLLOW_UPS}


def measure(service: Any, groups: dict[str, list[str]]) -> list[dict[str, Any]]:
    rows = []
    for group, items in groups.items():
        for question in items:
            found = service.retrieve(question)
            top = found.nodes[0].metadata if found.nodes else {}
            rows.append({"group": group, "question": question, "top_score": round(found.top_score, 3),
                         "top_file": top.get("source") or top.get("file_name")})
            print(f"{GROUPS[group]}: {found.top_score:.3f} — {question}", file=sys.stderr)
    return rows


def summary(scores: list[float]) -> dict[str, float]:
    ordered = sorted(scores)

    def pct(p: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))]

    return {"n": len(ordered), "min": ordered[0], "p10": pct(0.1), "median": statistics.median(ordered),
            "p90": pct(0.9), "max": ordered[-1]}


def recommend(inside: list[float], outside: list[float]) -> tuple[float, int, str]:
    """Порог, число ошибок на нём и как выбран."""
    if max(outside) < min(inside):
        return round((max(outside) + min(inside)) / 2, 2), 0, "середина зазора между группами"
    candidates = sorted({round(x, 2) for x in inside + outside} | {round(x + 0.01, 2) for x in outside})
    best = min(candidates, key=lambda t: (sum(s < t for s in inside) + sum(s >= t for s in outside), t))
    errors = sum(s < best for s in inside) + sum(s >= best for s in outside)
    return best, errors, "наименьшее число ошибок (группы пересекаются)"


def report(rows: list[dict[str, Any]], current: float, model: str, collection: str,
           when: datetime | None = None) -> str:
    scores = {group: [r["top_score"] for r in rows if r["group"] == group] for group in GROUPS}
    threshold, errors, how = recommend(scores["in"], scores["off"])
    lines = [f"## Калибровка порога — {(when or datetime.now().astimezone()):%Y-%m-%d %H:%M}", "",
             f"Эмбеддинги `{model}`, коллекция `{collection}`; top_score — близость лучшего из кандидатов.", "",
             "| Группа | Вопросов | min | p10 | медиана | p90 | max |", "|---|---:|---:|---:|---:|---:|---:|"]
    for group, title in GROUPS.items():
        if not scores[group]:
            continue
        s = summary(scores[group])
        lines.append(f"| {title} | {s['n']} | {s['min']:.3f} | {s['p10']:.3f} | {s['median']:.3f} | "
                     f"{s['p90']:.3f} | {s['max']:.3f} |")
    if errors == 0:
        recommended = (f"- Зазор между посторонними и вопросами по базе: {max(scores['off']):.3f}–"
                       f"{min(scores['in']):.3f}; рекомендуемый порог — его середина, **{threshold}**.")
    else:
        recommended = f"- Рекомендуемый порог: **{threshold}** ({how}; ошибок: {errors})."
    refused_in = sum(x < current for x in scores["in"])
    passed_off = sum(x >= current for x in scores["off"])
    lines += ["", recommended,
              (f"- Текущий RAG__SCORE_THRESHOLD={current}: вопросов по базе отклонено {refused_in} из "
               f"{len(scores['in'])}, посторонних пропущено к модели {passed_off} из {len(scores['off'])}.")]
    if scores["near"]:
        passed_near = sum(x >= current for x in scores["near"])
        lines.append(f"- Вопросы о продукте без ответа в базе: до модели доходит {passed_near} из "
                     f"{len(scores['near'])} — по близости их от вопросов по базе не отличить, отказ на них даёт промпт.")
    lines += ["", "| Группа | Вопрос | top_score | Лучший фрагмент |", "|---|---|---:|---|"]
    for row in sorted(rows, key=lambda r: (list(GROUPS).index(r["group"]), -r["top_score"])):
        wrong = {"in": row["top_score"] < current, "off": row["top_score"] >= current}.get(row["group"], False)
        lines.append(f"| {GROUPS[row['group']]} | {row['question']} | {row['top_score']:.3f}{' ⚠️' if wrong else ''} | "
                     f"{row['top_file'] or '—'} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Калибровка RAG__SCORE_THRESHOLD по распределению близости (блок 5.5)")
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
        current = cfg.rag.score_threshold
        # Порог 0 и без re-ranker: нужна только близость лучшего кандидата.
        probe = cfg.model_copy(update={"rag": cfg.rag.model_copy(update={"score_threshold": 0.0}),
                                       "rerank": cfg.rerank.model_copy(update={"enabled": False})})
        service = RAGService(probe).build(create=False)
        try:
            rows = measure(service, questions())
            markdown = report(rows, current, service.embed_model.model_name, service.collection)
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
