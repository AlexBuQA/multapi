"""
Пост-фактум оценка трейсов Phoenix (блок 5.6, шаг 9, опционально): phoenix.evals.HallucinationEvaluator
на ответах, которые RAG дал в scripts/trace_queries.py, — доля галлюцинаций на свободных вопросах, то
есть на трафике вне golden dataset.

    python scripts/trace_queries.py                 # сначала трафик: 20+ вопросов, трейсы в Phoenix
    python scripts/phoenix_hallucinations.py        # оценка этих трейсов, метки — обратно в Phoenix
    python scripts/phoenix_hallucinations.py --since 2026-10-12T09:00 --no-annotate

Спаны — через phoenix.client.Client().spans.get_spans_dataframe(): в arize-phoenix 20 так называется
то, что в материалах курса — px.Client().get_spans_dataframe() (старого px.Client в пакете больше нет).
Из каждого трейса RAGService.answer берётся LLM-span модели ответа: input — промпт целиком (системные
правила, пронумерованные фрагменты базы знаний и вопрос), output — то, что ответила модель. Evaluator
проверяет, всё ли в ответе подтверждается этим входом: grounded или hallucinated, с объяснением.
Отказ score-guard (top_score ниже порога) модель не вызывает — LLM-span нет, оценивать нечего: такие
трейсы считаются отдельно.

Какой трейс — какой вопрос (golden или свободный), берётся из var/trace_queries.jsonl: оттуда же время
начала прогона, с которого читаются спаны. Судья — тот же, что у RAGAS (RAGAS__JUDGE_*: по умолчанию
qwen3:4b-instruct в Ollama), через phoenix.evals.LLM. Метки пишутся аннотацией hallucination на корневой
span трейса (POST /rag/query) — в списке трейсов Phoenix появляется колонка с меткой; её снимок — в
docs/screenshots/. Таблица — tests/eval/results/phoenix/{YYYY-MM-DD_HHMM}_hallucination.csv.

Нужны Phoenix с трейсами, судья и группа eval (requirements-eval.txt). Код выхода: 0 — готово, 1 — нет
трейсов для оценки, 2 — ошибка.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERIES = ROOT / "var" / "trace_queries.jsonl"
OUT_DIR = ROOT / "tests" / "eval" / "results" / "phoenix"
ANSWER_SPAN = "RAGService.answer"
FREE_KINDS = ("base", "product", "offtopic", "unknown")
COLUMNS = ("trace_id", "root_span_id", "kind", "question", "label", "score", "response", "explanation", "error")


def read_queries(path: Path) -> tuple[datetime | None, dict[str, str]]:
    """Начало прогона trace_queries.py и вид каждого вопроса (golden / base / product / offtopic)."""
    if not path.is_file():
        return None, {}
    started, kinds = None, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("meta"):
            started = datetime.fromisoformat(record["started"])
        else:
            kinds[record["question"]] = record["kind"]
    return started, kinds


def _messages(value: Any) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else []


def transcript(messages: list[dict[str, Any]]) -> str:
    """Сообщения LLM-span'а одним текстом: «system: …», «user: …» — так их видит evaluator."""
    return "\n\n".join(f"{m.get('message.role', '?')}: {m.get('message.content') or ''}" for m in messages)


def _question(span: Any) -> str:
    try:
        return str(json.loads(span["attributes.input.value"]).get("question") or "")
    except (TypeError, ValueError, AttributeError):
        return ""


def rag_answers(spans: Any, kinds: dict[str, str]) -> list[dict[str, Any]]:
    """Ответы RAG из спанов: по одной строке на трейс с RAGService.answer. У строки с LLM-span'ом есть
    input/output для evaluator; без него (отказ score-guard) — guard=True."""
    rows = []
    for trace_id, trace in spans.groupby("context.trace_id", sort=False):
        answers = trace[trace["name"] == ANSWER_SPAN]
        if answers.empty:
            continue                                   # не RAG: /chat, /health, загрузка документов
        answer = answers.iloc[0]
        parents = dict(zip(trace["context.span_id"], trace["parent_id"], strict=False))

        def under_answer(span_id: str, answer_id: str = answer["context.span_id"],
                         parents: dict[str, str | None] = parents) -> bool:
            seen = set()
            while span_id and span_id not in seen:
                seen.add(span_id)
                span_id = parents.get(span_id)
                if span_id == answer_id:
                    return True
            return False

        llm = trace[(trace["span_kind"] == "LLM")
                    & trace["attributes.llm.input_messages"].apply(lambda v: bool(_messages(v)))
                    & trace["context.span_id"].apply(under_answer)]
        roots = trace[trace["parent_id"].isna()]
        question = _question(answer)
        row = {"trace_id": trace_id, "root_span_id": (roots.iloc[0] if len(roots) else answer)["context.span_id"],
               "question": question, "kind": kinds.get(question, "unknown"), "guard": llm.empty,
               "start_time": answer["start_time"]}
        if not llm.empty:
            span = llm.sort_values("start_time").iloc[-1]   # внутренний OpenAILike.chat: у него вход и выход
            output = _messages(span.get("attributes.llm.output_messages"))
            row["input"] = transcript(_messages(span["attributes.llm.input_messages"]))
            row["output"] = (output[0].get("message.content") if output else None) or ""
        rows.append(row)
    return sorted(rows, key=lambda r: r["start_time"])


def build_llm(cfg: Any) -> tuple[Any, dict[str, Any]]:
    """Судья RAGAS__JUDGE_* как phoenix.evals.LLM (провайдер openai: Ollama и OpenRouter) и параметры
    вызова. Прокси и сертификаты ОС — как у судьи RAGAS (app/eval/metrics.py)."""
    import httpx
    from phoenix.evals import LLM

    from app.core.config import (
        http_client_options,
        is_local_url,
        is_openrouter,
        provider_headers,
        proxy_for,
    )
    from app.eval.metrics import JUDGE_TITLE, judge_endpoint

    base_url, key = judge_endpoint(cfg)
    options = http_client_options(proxy_for(base_url, cfg.llm.proxy_url),
                                  cfg.llm.use_system_certs and not is_local_url(base_url))
    # max_retries клиента задаёт сам phoenix.evals (0); повторы — у evaluate_dataframe(max_retries=…).
    shared = {"api_key": key, "base_url": base_url, "timeout": cfg.ragas.judge_timeout,
              "default_headers": provider_headers(base_url, JUDGE_TITLE)}
    llm = LLM(provider="openai", model=cfg.ragas.judge_model, **shared,
              sync_client_kwargs={"http_client": httpx.Client(**options)} if options else None,
              async_client_kwargs={"http_client": httpx.AsyncClient(**options)} if options else None)
    params: dict[str, Any] = {"temperature": 0.0, "max_tokens": cfg.ragas.judge_max_tokens}
    if is_openrouter(base_url):
        params["extra_body"] = {"provider": {"require_parameters": True}}
    return llm, params


def evaluate(rows: list[dict[str, Any]], llm: Any, params: dict[str, Any]) -> None:
    """HallucinationEvaluator по строкам с LLM-span'ом: label, score (1.0 — галлюцинация), explanation."""
    import pandas as pd
    from phoenix.evals import evaluate_dataframe
    from phoenix.evals.metrics.hallucination import HallucinationEvaluator

    judged = [r for r in rows if not r["guard"]]
    if not judged:
        return
    frame = pd.DataFrame({"input": [r["input"] for r in judged], "output": [r["output"] for r in judged]})
    result = evaluate_dataframe(dataframe=frame, evaluators=[HallucinationEvaluator(llm=llm, **params)],
                                hide_tqdm_bar=True, max_retries=3)
    for row, (_, scored) in zip(judged, result.iterrows(), strict=True):
        score = scored.get("hallucination_score")
        score = json.loads(score) if isinstance(score, str) else score
        if isinstance(score, dict) and score.get("label"):
            row.update(label=score["label"], score=score.get("score"), explanation=score.get("explanation") or "")
        else:
            details = scored.get("hallucination_execution_details")
            row.update(label="error", score=None, explanation="",
                       error=" ".join(str(details).split())[:300] if details is not None else "нет оценки")


def summary(rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for title, kinds in (("свободные вопросы", FREE_KINDS), ("golden", ("golden",)), ("все", None)):
        group = [r for r in rows if kinds is None or r["kind"] in kinds]
        if not group:
            continue
        judged = [r for r in group if r.get("label") in ("grounded", "hallucinated")]
        bad = [r for r in judged if r["label"] == "hallucinated"]
        guard = sum(1 for r in group if r["guard"])
        errors = sum(1 for r in group if r.get("label") == "error")
        share = f"{len(bad) / len(judged):.0%}" if judged else "—"
        lines.append(f"{title}: трейсов {len(group)}, отказ score-guard {guard}, оценено {len(judged)}, "
                     f"hallucinated {len(bad)} ({share})" + (f", ошибок судьи {errors}" if errors else ""))
    return lines


def annotate(client: Any, rows: list[dict[str, Any]]) -> int:
    """Метки на корневой span трейса: аннотация hallucination (LLM) — видна в списке трейсов Phoenix."""
    import pandas as pd

    marked = [r for r in rows if r.get("label") in ("grounded", "hallucinated")]
    if not marked:
        return 0
    frame = pd.DataFrame({"span_id": [r["root_span_id"] for r in marked], "label": [r["label"] for r in marked],
                          "score": [float(r["score"] or 0.0) for r in marked],
                          "explanation": [r["explanation"][:2000] for r in marked]})
    client.spans.log_span_annotations_dataframe(dataframe=frame, annotation_name="hallucination",
                                                annotator_kind="LLM", sync=True)
    return len(marked)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "label": row.get("label") or ("guard" if row["guard"] else ""),
                             "response": row.get("output", ""), "explanation": row.get("explanation", ""),
                             "error": row.get("error", "")})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HallucinationEvaluator Phoenix на трейсах RAG (блок 5.6, шаг 9)")
    parser.add_argument("--phoenix", default="http://127.0.0.1:6006", help="адрес Phoenix")
    parser.add_argument("--project", default=None, help="проект Phoenix; пусто => PHOENIX_PROJECT_NAME из .env")
    parser.add_argument("--queries", type=Path, default=QUERIES, help="JSONL прогона trace_queries.py")
    parser.add_argument("--since", default=None, help="спаны с этого момента (ISO, UTC); пусто => начало "
                                                      "прогона trace_queries.py или последние сутки")
    parser.add_argument("--limit", type=int, default=20000, help="сколько спанов читать")
    parser.add_argument("--out", type=Path, default=None, help="CSV; пусто => tests/eval/results/phoenix/…")
    parser.add_argument("--no-annotate", action="store_true", help="не записывать метки в Phoenix")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):     # вывод в файл на Windows (cp1251): «₽» и др. — не падать
        sys.stdout.reconfigure(errors="replace")

    from phoenix.client import Client

    from app.core.config import get_settings

    cfg = get_settings()
    project = args.project or cfg.phoenix_project_name
    started, kinds = read_queries(args.queries)
    if args.since:
        since = datetime.fromisoformat(args.since)
        since = since if since.tzinfo else since.replace(tzinfo=UTC)
    else:
        since = started or datetime.now(UTC) - timedelta(days=1)
    client = Client(base_url=args.phoenix.rstrip("/"))
    try:
        spans = client.spans.get_spans_dataframe(project_name=project, start_time=since, limit=args.limit, timeout=60)
    except Exception as exc:  # noqa: BLE001 — Phoenix недоступен, проекта нет
        print(f"Phoenix {args.phoenix}, проект {project}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    rows = rag_answers(spans, kinds) if not spans.empty else []
    if not rows:
        print(f"В проекте {project} с {since:%Y-%m-%d %H:%M} UTC нет трейсов {ANSWER_SPAN}: сначала "
              f"python scripts/trace_queries.py", file=sys.stderr)
        return 1
    judged = sum(1 for r in rows if not r["guard"])
    print(f"Трейсов RAG: {len(rows)} (с ответом модели {judged}); судья {cfg.ragas.judge_model} …")
    llm, params = build_llm(cfg)
    evaluate(rows, llm, params)

    print(f"\n{'вид':<8} {'метка':<12} вопрос → объяснение судьи")
    for row in rows:
        label = row.get("label") or "guard"
        why = row.get("explanation") or row.get("error") or ("отказ score-guard, модель не вызывалась"
                                                             if row["guard"] else "")
        print(f"{row['kind']:<8} {label:<12} {row['question'][:50]} → {' '.join(why.split())[:90]}")
    print("\n" + "\n".join(summary(rows)))

    out = args.out or OUT_DIR / f"{datetime.now():%Y-%m-%d_%H%M}_hallucination.csv"
    write_csv(rows, out)
    print(f"→ {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    if not args.no_annotate:
        count = annotate(client, rows)
        print(f"Метки hallucination записаны на {count} трейсов: {args.phoenix.rstrip('/')} → проект {project}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
