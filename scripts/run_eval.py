"""
Оценка RAG по golden dataset (блок 5.6): RAG блока 5.5 отвечает на вопросы, судья RAGAS 0.4 ставит
оценки. Результат — tests/eval/results/{YYYY-MM-DD_HHMM}_{label}.csv (строка на вопрос) и .json
(средние, задержка, конфигурация RAG и судьи) — audit log, по которому строится ряд прогонов.

    python scripts/run_eval.py --label baseline                        # текущая конфигурация сервиса
    python scripts/run_eval.py --label chunk_128 --chunk-size 128      # эксперимент A: нарезка
    python scripts/run_eval.py --label rerank --rerank on              # эксперимент B: re-ranker блока 5.4
    python scripts/run_eval.py --label smoke --limit 3                 # быстрая проверка на трёх вопросах
    python scripts/run_eval.py --resume tests/eval/results/2026-10-12_1530_baseline.partial.jsonl

Метрики (app/eval/metrics.py): faithfulness, answer_relevancy, context_precision, context_recall из
ragas.metrics.collections и has_citation (@discrete_metric). Плюс задержка ответа RAG, мс.

Два прохода, потому что на CPU модель ответа (llama3.2) и судья (qwen3:4b-instruct) не помещаются в
память вместе и вытесняли бы друг друга на каждом вопросе:
1. ответы — RAGService.evaluate_inputs, по одному вопросу: тот же путь, что POST /rag/query (поиск,
   score-guard, промпт с цитатами), плюс полный текст фрагментов для метрик. Задержка — этого прохода:
   судья в это время не работает и её не искажает;
2. оценки — asyncio.gather по всем вопросам, одновременно не больше RAGAS__CONCURRENCY строк (Ollama на
   CPU всё равно ставит запросы в очередь).
Каждый готовый ответ и каждая готовая строка оценок сразу дописываются в {…}.partial.jsonl: прогон на
CPU — несколько часов (Windows, qwen3:4b-instruct на CPU: около 7,5 мин судьи на вопрос), и после сбоя или
Ctrl+C его продолжает --resume. После успешного прогона файл удаляется.

Параметры эксперимента — флагами, по одному (--chunk-size, --top-k, --rerank …); всё остальное — из
.env, как у сервиса. Корпус индексируется здесь же в коллекцию оценки
{RAG__COLLECTION}_{нарезка}_{размер}_{перекрытие} (kb_docs_recursive_512_64, kb_docs_recursive_128_64):
у каждой нарезки своя, прогоны с одной нарезкой (baseline и rerank) делят одну. Коллекцию сервиса
kb_docs оценка не читает и не пишет: в ней ещё и загрузки пользователей (POST /documents/upload,
data/uploads — например, тарифы 2027 года из блока 5.5), которых нет в golden dataset, а на тарифных
вопросах они спорили бы с корпусом. Конфигурация, судья (с отпечатком промпта has_citation) и хеш golden
записываются в JSON — сравнивать можно только прогоны с одним судьёй, одним промптом и одним golden.

Нужны Qdrant, Ollama с моделью ответа, эмбеддингов и судьёй, группа eval (requirements-eval.txt).
Код выхода: 0 — готово, 2 — ошибка.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "has_citation")
CSV_COLUMNS = ("id", "user_input", "reference", "response", *METRICS, "latency_ms", "top_score", "confident",
               "retrieved_files", "reference_sources", "retrieved_contexts", "judge_seconds", "errors")
LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,40}$")
SHORT = {"faithfulness": "faith", "answer_relevancy": "relev", "context_precision": "prec",
         "context_recall": "recall", "has_citation": "cite"}


# ---------------------------------------------------------------- golden и конфигурация
def load_golden(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    rows = []
    for number, item in enumerate(items, start=1):
        if not str(item.get("user_input") or "").strip():
            continue                          # строка без вопроса — нечего спрашивать у RAG
        rows.append({**item, "id": str(item.get("id") or f"q{number:03d}"), "reference": item.get("reference") or ""})
    return rows


def configure(cfg: Any, args: argparse.Namespace) -> tuple[Any, str]:
    """Настройки с параметрами эксперимента и коллекция оценки для них — по нарезке, и для нарезки
    сервиса тоже (не kb_docs: там загрузки пользователей). Валидаторы RagSettings проверяют сочетание
    (overlap < chunk_size, кандидатов не меньше top-K)."""
    from app.core.config import RagSettings, RerankSettings

    rag_update = {key: value for key, value in {
        "chunking": args.chunking, "chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap,
        "similarity_top_k": args.top_k, "candidates_top_k": args.candidates, "score_threshold": args.threshold,
        "llm_model": args.llm_model}.items() if value is not None}
    rag = RagSettings(**{**cfg.rag.model_dump(), **rag_update})
    rerank = RerankSettings(**{**cfg.rerank.model_dump(), **({"enabled": args.rerank == "on"} if args.rerank else {})})
    updated = cfg.model_copy(update={"rag": rag, "rerank": rerank})
    collection = args.collection or f"{cfg.rag.collection}_{rag.chunking}_{rag.chunk_size}_{rag.chunk_overlap}"
    return updated, collection


def describe(cfg: Any, collection: str) -> dict[str, Any]:
    rag = cfg.rag
    return {"collection": collection, "chunking": rag.chunking, "chunk_size": rag.chunk_size,
            "chunk_overlap": rag.chunk_overlap, "candidates_top_k": rag.candidates_top_k,
            "similarity_top_k": rag.similarity_top_k, "score_threshold": rag.score_threshold,
            "rerank": {"enabled": cfg.rerank.enabled, "provider": cfg.rerank.provider if cfg.rerank.enabled else None,
                       "model": cfg.rerank.model if cfg.rerank.enabled else None},
            "llm_model": rag.llm_model or cfg.llm.default_model, "temperature": rag.temperature,
            "embeddings_model": cfg.embeddings.model, "embedding_dim": cfg.embedding_dim}


# ---------------------------------------------------------------- журнал прогона
class Journal:
    """{stamp}_{label}.partial.jsonl: первая строка — meta, дальше — answer и score по мере готовности."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.meta: dict[str, Any] = {}
        self.answers: dict[str, dict[str, Any]] = {}
        self.scores: dict[str, dict[str, Any]] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                kind = record.pop("kind")
                if kind == "meta":
                    self.meta = record
                elif kind == "answer":
                    self.answers[record["id"]] = record
                elif kind == "score":
                    self.scores[record["id"]] = record

    def write(self, kind: str, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": kind, **record}, ensure_ascii=False) + "\n")
        if kind == "meta":
            self.meta = record
        elif kind == "answer":
            self.answers[record["id"]] = record
        else:
            self.scores[record["id"]] = record


def golden_hash(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps([[r["id"], r["user_input"], r["reference"]] for r in rows], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- проходы
async def answer_all(rag: Any, rows: list[dict[str, Any]], journal: Journal) -> None:
    """Проход 1: ответы RAG по одному вопросу (задержка без судьи)."""
    from app.services.rag_common import RAGError

    todo = [row for row in rows if row["id"] not in journal.answers]
    for done, row in enumerate(todo, start=1):
        try:
            result = await asyncio.to_thread(rag.evaluate_inputs, row["user_input"])
            error = None
        except RAGError:
            raise                             # Qdrant, эмбеддинги, модель недоступны — дальше смысла нет
        except Exception as exc:  # noqa: BLE001 — сбой на одном вопросе: пустой ответ, метрики — NaN
            result = {"answer": "", "retrieved_contexts": [], "retrieved_files": [], "top_score": 0.0,
                      "confident": False, "latency_ms": None}
            error = f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"
        record = {"id": row["id"], "response": result["answer"], "retrieved_contexts": result["retrieved_contexts"],
                  "retrieved_files": result.get("retrieved_files", []), "top_score": result["top_score"],
                  "confident": result["confident"], "latency_ms": result["latency_ms"], "error": error}
        journal.write("answer", record)
        mark = "отказ" if not result["confident"] else f"{len(result['retrieved_contexts'])} фрагм."
        print(f"[ответ {done}/{len(todo)}] {row['id']} · {mark} · {result['latency_ms']} мс · {row['user_input'][:60]}",
              file=sys.stderr)


async def judge_all(metrics: Any, rows: list[dict[str, Any]], journal: Journal, concurrency: int) -> None:
    """Проход 2: оценки судьи — asyncio.gather, одновременно не больше concurrency строк."""
    from app.eval.metrics import score_row

    todo = [row for row in rows if row["id"] not in journal.scores]
    semaphore = asyncio.Semaphore(concurrency)
    counter = {"done": 0}
    started = time.perf_counter()

    async def one(row: dict[str, Any]) -> None:
        answer = journal.answers[row["id"]]
        async with semaphore:
            t0 = time.perf_counter()
            scores, errors = await score_row(metrics, question=row["user_input"], reference=row["reference"],
                                             response=answer["response"], contexts=answer["retrieved_contexts"])
            seconds = round(time.perf_counter() - t0, 1)
        if answer.get("error"):
            errors = {"answer": answer["error"], **errors}
        journal.write("score", {"id": row["id"], "scores": scores, "errors": errors, "judge_seconds": seconds})
        counter["done"] += 1
        elapsed = time.perf_counter() - started
        eta = elapsed / counter["done"] * (len(todo) - counter["done"])
        shown = " ".join(f"{SHORT[k]}={v:.2f}" for k, v in scores.items() if not math.isnan(v))
        print(f"[оценка {counter['done']}/{len(todo)}] {row['id']} · {shown} · {seconds} с · осталось ~{eta / 60:.0f} мин",
              file=sys.stderr)

    await asyncio.gather(*(one(row) for row in todo))


# ---------------------------------------------------------------- итог
def frame(rows: list[dict[str, Any]], journal: Journal) -> Any:
    import pandas as pd

    records = []
    for row in rows:
        answer, score = journal.answers[row["id"]], journal.scores[row["id"]]
        records.append({"id": row["id"], "user_input": row["user_input"], "reference": row["reference"],
                        "response": answer["response"], **score["scores"], "latency_ms": answer["latency_ms"],
                        "top_score": answer["top_score"], "confident": answer["confident"],
                        "retrieved_files": json.dumps(answer["retrieved_files"], ensure_ascii=False),
                        "reference_sources": json.dumps(row.get("reference_sources") or [], ensure_ascii=False),
                        "retrieved_contexts": json.dumps(answer["retrieved_contexts"], ensure_ascii=False),
                        "judge_seconds": score["judge_seconds"],
                        "errors": json.dumps(score["errors"], ensure_ascii=False) if score["errors"] else ""})
    df = pd.DataFrame(records, columns=list(CSV_COLUMNS))
    df["latency_ms"] = df["latency_ms"].astype("Int64")     # целые мс и пусто у сбоя, а не 2400.0
    return df


def aggregates(df: Any, journal: Journal) -> dict[str, Any]:
    def stats(column: str) -> dict[str, Any]:
        values = df[column].astype(float)
        valid = values.dropna()
        return {"mean": round(float(valid.mean()), 4) if len(valid) else None, "n": len(valid),
                "nan": int(values.isna().sum())}

    latency = df["latency_ms"].dropna().astype(float)
    answered = df["response"].fillna("").astype(str) != ""        # пустой ответ — сбой RAG, а не отказ
    refused = ~df["confident"].astype(bool) & answered
    refusals_guard = int((refused & (df["retrieved_contexts"] == "[]")).sum())
    return {**journal.meta, "rows": len(df),
            "metrics": {name: stats(name) for name in METRICS},
            "latency_ms": {"mean": round(float(latency.mean())) if len(latency) else None,
                           "p50": round(float(latency.quantile(0.5))) if len(latency) else None,
                           "p95": round(float(latency.quantile(0.95))) if len(latency) else None},
            "refusals": {"score_guard": refusals_guard, "model": int(refused.sum()) - refusals_guard},
            "rag_errors": int((~answered).sum()),
            "rows_with_errors": int((df["errors"] != "").sum())}


def summary_text(df: Any, agg: dict[str, Any]) -> str:
    lines = [(f"\n{agg['label']}: {agg['rows']} вопросов, судья {agg['judge']['judge_model']}, "
              f"коллекция {agg['config']['collection']}")]
    for name in METRICS:
        m = agg["metrics"][name]
        mean = "—" if m["mean"] is None else f"{m['mean']:.3f}"
        lines.append(f"  {name:<18} {mean}   (n={m['n']}, NaN={m['nan']})")
    lines.append(f"  {'latency, мс':<18} {agg['latency_ms']['mean']} (p50 {agg['latency_ms']['p50']}, "
                 f"p95 {agg['latency_ms']['p95']})")
    # Отказ без фрагментов faithfulness не имеет — в худшие не попадает (failure analysis: eval_report.py --worst).
    worst = df.dropna(subset=["faithfulness"]).sort_values("faithfulness").head(5)
    lines.append("\nХудшие по faithfulness (подробно: python scripts/eval_report.py --worst <CSV>):")
    for _, row in worst.iterrows():
        lines.append(f"  {row['id']} · faithfulness={row['faithfulness']:.2f} · "
                     f"context_recall={row['context_recall']:.2f} · {row['user_input'][:70]}")
    return "\n".join(lines)


# ---------------------------------------------------------------- запуск
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGAS-оценка RAG по golden dataset (блок 5.6)")
    parser.add_argument("--golden", type=Path, help="golden dataset (по умолчанию RAGAS__GOLDEN_PATH)")
    parser.add_argument("--label", default="baseline", help="метка конфигурации: baseline, chunk_128, rerank …")
    parser.add_argument("--out-dir", type=Path, help="папка результатов (по умолчанию RAGAS__RESULTS_DIR)")
    parser.add_argument("--limit", type=int, help="только первые N вопросов")
    parser.add_argument("--ids", help="только эти id через запятую")
    parser.add_argument("--concurrency", type=int, help="строк у судьи одновременно (по умолчанию RAGAS__CONCURRENCY)")
    parser.add_argument("--resume", type=Path, help="продолжить прогон по его .partial.jsonl")
    parser.add_argument("--answers-only", action="store_true", help="только ответы RAG; оценки — потом через --resume")
    group = parser.add_argument_group("параметры эксперимента (по умолчанию — из .env, как у сервиса)")
    group.add_argument("--chunking", choices=["fixed", "recursive", "semantic"])
    group.add_argument("--chunk-size", type=int)
    group.add_argument("--chunk-overlap", type=int)
    group.add_argument("--top-k", type=int, help="фрагментов в контекст модели (RAG__SIMILARITY_TOP_K)")
    group.add_argument("--candidates", type=int, help="кандидатов из Qdrant (RAG__CANDIDATES_TOP_K)")
    group.add_argument("--rerank", choices=["on", "off"], help="re-ranker блока 5.4 (RERANK__*)")
    group.add_argument("--threshold", type=float, help="порог score-guard (RAG__SCORE_THRESHOLD)")
    group.add_argument("--llm-model", help="модель ответа (RAG__LLM_MODEL)")
    group.add_argument("--collection", help="коллекция Qdrant (по умолчанию — kb_docs_<нарезка>_<размер>_<перекрытие>)")
    args = parser.parse_args(argv)
    if not LABEL.match(args.label):
        parser.error("--label: латиница, цифры, _ . - (это часть имени файла)")
    return args


async def run(args: argparse.Namespace) -> int:
    from app.core.config import get_settings
    from app.eval.metrics import (
        build_embeddings,
        build_judge,
        build_metrics,
        judge_cache,
        judge_info,
    )
    from app.services.rag import RAGService

    base = get_settings()
    out_dir = args.out_dir or base.ragas.results_dir
    golden_path = args.golden or base.ragas.golden_path
    rows = load_golden(golden_path)
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        rows = [r for r in rows if r["id"] in wanted]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print(f"Ошибка: в {golden_path} нет вопросов для прогона", file=sys.stderr)
        return 2

    if args.resume:
        journal = Journal(args.resume)
        if not journal.meta:
            print(f"Ошибка: {args.resume} — не журнал прогона", file=sys.stderr)
            return 2
        meta = journal.meta
        print(f"Продолжаю {meta['label']} от {meta['timestamp']}: ответов {len(journal.answers)}, "
              f"оценок {len(journal.scores)} из {len(rows)}; конфигурация — из журнала", file=sys.stderr)
        replay = argparse.Namespace(**{k: None for k in ("chunking", "chunk_size", "chunk_overlap", "top_k",
                                                         "candidates", "threshold", "llm_model", "rerank")})
        c = meta["config"]
        replay.chunking, replay.chunk_size, replay.chunk_overlap = c["chunking"], c["chunk_size"], c["chunk_overlap"]
        replay.top_k, replay.candidates, replay.threshold = c["similarity_top_k"], c["candidates_top_k"], c["score_threshold"]
        replay.llm_model, replay.rerank, replay.collection = c["llm_model"], "on" if c["rerank"]["enabled"] else "off", c["collection"]
        cfg, collection = configure(base, replay)
        stamp, label = meta["timestamp"], meta["label"]
        out_dir = args.out_dir or args.resume.parent           # результат — рядом с журналом
        if meta["golden"]["sha256"] != golden_hash(rows):
            print("Внимание: golden dataset изменился после начала прогона — оценки разных версий смешаются",
                  file=sys.stderr)
    else:
        cfg, collection = configure(base, args)
        stamp, label = datetime.now().strftime("%Y-%m-%d_%H%M"), args.label
        journal = Journal(out_dir / f"{stamp}_{label}.partial.jsonl")
        journal.write("meta", {"label": label, "timestamp": stamp, "config": describe(cfg, collection),
                               "judge": judge_info(cfg), "golden": {"path": golden_path.name, "items": len(rows),
                                                                     "sha256": golden_hash(rows)}})

    started = time.perf_counter()
    rag = RAGService(cfg, collection=collection)
    try:
        report = await asyncio.to_thread(rag.ingestion().run)       # корпус — в коллекцию оценки
        print(f"Коллекция {collection}: {report.summary()}", file=sys.stderr)
        await asyncio.to_thread(rag.build, create=False)
        await answer_all(rag, rows, journal)
    finally:
        rag.close()
    answers_seconds = round(time.perf_counter() - started)
    if args.answers_only:
        print(f"Ответы готовы ({answers_seconds} с). Оценки: python scripts/run_eval.py --resume {journal.path}",
              file=sys.stderr)
        return 0

    started = time.perf_counter()
    judge = build_judge(cfg, cache=judge_cache(cfg))
    metrics = build_metrics(judge, build_embeddings(cfg))
    await judge_all(metrics, rows, journal, args.concurrency or cfg.ragas.concurrency)
    judge_seconds = round(time.perf_counter() - started)

    df = frame(rows, journal)
    agg = aggregates(df, journal)
    agg["seconds"] = {"answers": answers_seconds, "judge": judge_seconds}
    agg["versions"] = versions()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stamp}_{label}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    csv_path.with_suffix(".json").write_text(json.dumps(agg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    journal.path.unlink(missing_ok=True)
    print(summary_text(df, agg))
    print(f"\n→ {csv_path}\n→ {csv_path.with_suffix('.json')}", file=sys.stderr)
    return 0


def versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    result = {}
    for package in ("ragas", "llama-index-core", "openai"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "-"
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from pydantic import ValidationError

    from app.observability.logging import setup_logging
    from app.services.embeddings import EmbeddingError
    from app.services.rag import no_implicit_event_loop
    from app.services.rag_common import RAGError

    setup_logging("WARNING", stream=sys.stderr)
    no_implicit_event_loop()
    try:
        return asyncio.run(run(args))
    except (RAGError, EmbeddingError, ValidationError, ValueError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nПрервано. Готовое сохранено в .partial.jsonl — продолжить: --resume", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
