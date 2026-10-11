"""
Эксперимент с нарезкой (блок 5.4): три стратегии, re-ranker и подбор параметров — по метрикам на
golden dataset (tests/eval/retrieval_dataset.json), а не на глаз.

    python scripts/chunking_experiment.py --out var/chunking_experiment.md   # всё; Markdown ещё и в файл
    python scripts/chunking_experiment.py --no-rerank       # без re-ranker (нет sentence-transformers)
    python scripts/chunking_experiment.py --rebuild         # переиндексировать коллекции заново
    python scripts/chunking_experiment.py --json var/chunking_experiment.json   # и все строки с выдачей

Перед запуском — Qdrant (docker compose up -d qdrant) и Ollama с bge-m3. Re-ranker —
BAAI/bge-reranker-v2-m3 (pip install -r requirements-embeddings.txt; веса ~2,3 ГБ скачиваются при
первом запуске) или Cohere (RERANK__PROVIDER=cohere и RERANK__API_KEY в .env).

1. Стратегии fixed, recursive, semantic (app/services/chunking.py) на 512/64 — коллекции
   docs_fixed, docs_recursive, docs_semantic в Qdrant из .env. Повторный запуск подключается к
   готовым коллекциям, если в них та же нарезка и то же число фрагментов. Для каждой: фрагментов
   всего и на документ, средняя длина (токены cl100k_base и символы), Hit Rate@5, MRR@10,
   Recall@10 (app/services/retrieval_eval.py) и скорость поиска — медиана и p95 по вопросам, мс.
   Метрики двух уровней: по документу (фрагмент нужного файла, как в задании) и строгие — по
   фрагменту, где есть часть ответа дословно (evidence golden dataset). Плюс сколько токенов
   занимают первые 5 фрагментов — это цена контекста для модели.
   Вектор вопроса считается заранее (одинаков для всех стратегий), замеряется сам поиск:
   LlamaIndex-ретривер + Qdrant.
2. Re-ranker на лучшей стратегии: из Qdrant — 10 или 20 кандидатов, re-ranker пересортировывает их,
   метрики — по новому порядку. Время — поиск плюс переранжирование одного вопроса.
3. Подбор параметров лучшей стратегии: chunk_size 256/512 × overlap 32/64 (у semantic —
   buffer_size 1/2 × breakpoint 90/95), top-K 10 и 20 кандидатов; с re-ranker и без. Коллекции
   docs_<стратегия>_<параметры>.
4. Итог — конфигурация с лучшими метриками (при равенстве — без re-ranker и с меньшим top-K) и
   строки для .env.

Лучшая — по строгим Hit Rate@5 и MRR@10, затем документным Hit Rate@5 и MRR@10, затем Recall@10
(строгий, документный): документный уровень на 10 файлах почти всегда 1.0 и стратегии не различает.
Один вопрос из 28 — это 0.036 любой метрики: разница меньше — шум. Код выхода: 0 — прогон прошёл,
2 — ошибка.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COLLECTIONS = {"fixed": "docs_fixed", "recursive": "docs_recursive", "semantic": "docs_semantic"}
SIZE_GRID = [(512, 64), (512, 32), (256, 64), (256, 32)]          # chunk_size, overlap
SEMANTIC_GRID = [(1, 95), (1, 90), (2, 95), (2, 90)]              # buffer_size, breakpoint
TOP_KS = (10, 20)
BASE_TOP_K = 10
WARMUP, RUNS = 2, 3


def say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class Config:
    strategy: str
    chunk_size: int = 512
    chunk_overlap: int = 64
    buffer_size: int = 1
    breakpoint: int = 95

    @property
    def label(self) -> str:
        from app.services.chunking import label

        return label(self.strategy, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
                     buffer_size=self.buffer_size, breakpoint_percentile_threshold=self.breakpoint)

    @property
    def params(self) -> str:
        if self.strategy == "semantic":
            return f"buffer {self.buffer_size}, breakpoint {self.breakpoint}"
        return f"{self.chunk_size} / {self.chunk_overlap}"

    @property
    def collection(self) -> str:
        if self == Config(self.strategy):
            return COLLECTIONS[self.strategy]
        if self.strategy == "semantic":
            return f"docs_semantic_b{self.buffer_size}_p{self.breakpoint}"
        return f"docs_{self.strategy}_{self.chunk_size}_{self.chunk_overlap}"


class Experiment:
    """Индексы, поиск и метрики. client, embed_model и reranker можно подменить (тесты)."""

    def __init__(self, cfg: Any, *, client: Any, embed_model: Any, dataset: list[dict[str, Any]],
                 reranker: Any = None, rebuild: bool = False) -> None:
        self.cfg, self.client, self.embed_model = cfg, client, embed_model
        self.dataset, self.reranker, self.rebuild = dataset, reranker, rebuild
        self.vectors: dict[str, list[float]] = {}
        self.indexes: dict[Config, tuple[Any, dict[str, Any]]] = {}
        self.seconds_per_pair: float | None = None      # скорость re-ranker из последнего замера
        self.reranker_warm = False
        self.rows: dict[tuple[Config, int, bool], dict[str, Any]] = {}    # прогон не повторяется

    # ------------------------------------------------------------ индексы
    def index(self, config: Config) -> tuple[Any, dict[str, Any]]:
        """Коллекция конфигурации: готовая с той же нарезкой — подключиться, иначе собрать."""
        if config in self.indexes:
            return self.indexes[config]
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        from app.services import chunking
        from app.services.rag import load_corpus

        started = time.perf_counter()
        documents = load_corpus(self.cfg.rag.lab_dir, self.embed_model.model_name, config.label)
        splitter = chunking.build_splitter(config.strategy, chunk_size=config.chunk_size,
                                           chunk_overlap=config.chunk_overlap, embed_model=self.embed_model,
                                           buffer_size=config.buffer_size,
                                           breakpoint_percentile_threshold=config.breakpoint)
        nodes = splitter.get_nodes_from_documents(documents)
        stats = chunking.chunk_stats(nodes, len({d.metadata["file_name"] for d in documents}))
        store = QdrantVectorStore(client=self.client, collection_name=config.collection)
        if self._reusable(config, len(nodes)):
            index = VectorStoreIndex.from_vector_store(store, embed_model=self.embed_model)
            state = "готовая"
        else:
            if self.client.collection_exists(config.collection):
                self.client.delete_collection(config.collection)
            index = VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(vector_store=store),
                                     embed_model=self.embed_model)
            state = "проиндексирована"
        stats["index_s"] = round(time.perf_counter() - started, 1)
        say(f"{config.collection}: {state}, фрагментов {stats['chunks']}, средняя длина {stats['avg_tokens']} "
            f"токенов ({stats['index_s']} с)")
        self.indexes[config] = (index, stats)
        return index, stats

    def _reusable(self, config: Config, chunks: int) -> bool:
        if self.rebuild or not self.client.collection_exists(config.collection):
            return False
        if self.client.count(config.collection, exact=True).count != chunks:
            return False
        records, _ = self.client.scroll(config.collection, limit=1, with_payload=["chunking", "embed_model"])
        payload = (records[0].payload or {}) if records else {}
        return payload.get("chunking") == config.label and payload.get("embed_model") == self.embed_model.model_name

    # ------------------------------------------------------------ поиск
    def vector(self, question: str) -> list[float]:
        if question not in self.vectors:
            self.vectors[question] = self.embed_model.get_query_embedding(question)
        return self.vectors[question]

    def search(self, config: Config, top_k: int) -> tuple[dict[str, list[Any]], list[float]]:
        """Кандидаты по каждому вопросу и время поиска (мс) — RUNS замеров на вопрос после прогрева."""
        from llama_index.core.schema import QueryBundle

        index, _ = self.index(config)
        retriever = index.as_retriever(similarity_top_k=top_k)
        bundles = {item["question"]: QueryBundle(item["question"], embedding=self.vector(item["question"]))
                   for item in self.dataset}
        for bundle in list(bundles.values())[:WARMUP]:
            retriever.retrieve(bundle)
        found: dict[str, list[Any]] = {}
        timings: list[float] = []
        for question, bundle in bundles.items():
            for _ in range(RUNS):
                started = time.perf_counter()
                found[question] = retriever.retrieve(bundle)
                timings.append((time.perf_counter() - started) * 1000)
        return found, timings

    def rerank(self, found: dict[str, list[Any]]) -> tuple[dict[str, list[Any]], float | None, bool]:
        """Пересортировать кандидатов. Возвращает выдачу, мс re-ranker на вопрос и признак оценки.
        Время на вопрос = скорость одной пары × число кандидатов: оценки пар кешируются (top-10 после
        top-20, одинаковые фрагменты разных конфигураций), и прямой замер занизил бы время. Если в
        этом прогоне часть пар взята из кеша, значение — оценка (≈) по скорости посчитанных пар."""
        from app.services.reranker import Candidate

        if not self.reranker_warm:              # скачивание весов и загрузка модели — не в замер
            self.reranker.rerank("прогрев", [Candidate(text=f"прогрев re-ranker {time.time()}")])
            self.reranker_warm = True
        ranked: dict[str, list[Any]] = {}
        seconds = scored = 0.0
        for question, nodes in found.items():
            before = self.reranker.pairs_scored
            started = time.perf_counter()
            order = self.reranker.rerank(question, [Candidate(text=n.get_content(), doc_id=n.metadata.get("file_name"),
                                                              score=n.score or 0.0, payload=n) for n in nodes])
            if self.reranker.pairs_scored > before:
                seconds += time.perf_counter() - started
                scored += self.reranker.pairs_scored - before
            ranked[question] = [candidate.payload for candidate in order]
        if scored:
            self.seconds_per_pair = seconds / scored
        if self.seconds_per_pair is None:
            return ranked, None, False
        candidates = [len(nodes) for nodes in found.values()]
        return ranked, self.seconds_per_pair * statistics.mean(candidates) * 1000, scored < sum(candidates)

    def run(self, config: Config, top_k: int = BASE_TOP_K, *, rerank: bool = False) -> dict[str, Any]:
        """Строка таблицы: метрики, статистика нарезки, время. details — выдача по вопросам."""
        if (config, top_k, rerank) in self.rows:
            return self.rows[(config, top_k, rerank)]
        from app.services.embeddings import count_tokens
        from app.services.retrieval_eval import evaluate, evaluate_evidence, recall

        found, timings = self.search(config, top_k)
        rerank_ms, estimated = None, False
        if rerank:
            say(f"{config.collection}: re-ranker, {top_k} кандидатов × {len(found)} вопросов…")
            found, rerank_ms, estimated = self.rerank(found)
        ranking = {q: [n.metadata.get("file_name") for n in nodes] for q, nodes in found.items()}
        texts = {q: [n.get_content() for n in nodes] for q, nodes in found.items()}
        details: list[dict[str, Any]] = []
        metrics = evaluate(lambda question: ranking[question], self.dataset, details=details)
        evidence_details: list[dict[str, Any]] = []
        strict = evaluate_evidence(lambda question: texts[question], self.dataset, details=evidence_details)
        for row, ev in zip(details, evidence_details, strict=True):
            row["ev_first_rank"] = ev["first_rank"]
            row["ev_recall"] = ev["recall"]
        recall_k = statistics.mean(recall(ranking[i["question"]], i["relevant_doc_ids"], top_k) for i in self.dataset)
        context = statistics.mean(sum(count_tokens(text) for text in chunks[:5]) for chunks in texts.values())
        _, stats = self.index(config)
        retrieval = statistics.median(timings)
        row = {"strategy": config.strategy, "params": config.params, "collection": config.collection,
               "config": config, "top_k": top_k, "rerank": rerank, **metrics, "recall@k": round(recall_k, 4),
               **{f"ev_{key}": value for key, value in strict.items()}, "top5_tokens": round(context),
               **{k: v for k, v in stats.items() if k != "index_s"},
               "retrieval_ms": round(retrieval, 1), "retrieval_p95_ms": round(_p95(timings), 1),
               "rerank_ms": round(rerank_ms) if rerank_ms is not None else None, "rerank_estimated": estimated,
               "total_ms": round(retrieval + (rerank_ms or 0.0), 1), "details": details}
        self.rows[(config, top_k, rerank)] = row
        return row


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


def score_key(row: dict[str, Any]) -> tuple:
    """Лучше — выше строгие Hit Rate@5 и MRR@10, затем документные, затем Recall@10; при равенстве —
    без re-ranker, меньший top-K и меньше токенов в контексте."""
    return (row["ev_hit_rate@5"], row["ev_mrr@10"], row["hit_rate@5"], row["mrr@10"], row["ev_recall@10"],
            row["recall@10"], not row["rerank"], -row["top_k"], -row["top5_tokens"])


def grid(strategy: str) -> list[Config]:
    if strategy == "semantic":
        return [Config("semantic", buffer_size=b, breakpoint=p) for b, p in SEMANTIC_GRID]
    return [Config(strategy, chunk_size=size, chunk_overlap=overlap) for size, overlap in SIZE_GRID]


# ---------------------------------------------------------------- отчёт
def _metrics(row: dict[str, Any]) -> str:
    return f"{row['hit_rate@5']:.3f} | {row['mrr@10']:.3f} | {row['recall@10']:.3f}"


def _strict(row: dict[str, Any]) -> str:
    return f"{row['ev_hit_rate@5']:.3f} | {row['ev_mrr@10']:.3f} | {row['ev_recall@10']:.3f}"


def _speed(row: dict[str, Any]) -> str:
    if not row["rerank"]:
        return f"{row['retrieval_ms']} (p95 {row['retrieval_p95_ms']})"
    if row["rerank_ms"] is None:
        return f"{row['retrieval_ms']} + re-ranker (из кеша, не замерялся)"
    return f"{row['retrieval_ms']} + re-ranker {'≈' if row['rerank_estimated'] else ''}{row['rerank_ms']}"


def strategy_table(rows: list[dict[str, Any]]) -> str:
    lines = [("| Стратегия | Коллекция | Фрагментов | На документ | Средняя длина chunk, токенов (символов) | Мин–макс, "
              "токенов | Hit Rate@5 | MRR@10 | Recall@10 | Строгие: Hit Rate@5 | MRR@10 | Recall@10 | Токенов в top-5 | "
              "Скорость retrieval, мс (медиана, p95) |"),
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        name = row["strategy"] + (" + re-ranker" if row["rerank"] else "")
        lines.append(f"| {name} | `{row['collection']}` | {row['chunks']} | {row['chunks_per_doc']} | "
                     f"{row['avg_tokens']} ({row['avg_chars']}) | {row['min_tokens']}–{row['max_tokens']} | "
                     f"{_metrics(row)} | {_strict(row)} | {row['top5_tokens']} | {_speed(row)} |")
    return "\n".join(lines)


def rerank_table(rows: list[dict[str, Any]], reranker_name: str) -> str:
    lines = [("| Конфигурация | Кандидатов (top-K) | Hit Rate@5 | MRR@10 | Recall@10 | Recall@K | Строгие: Hit Rate@5 | "
              "MRR@10 | Recall@10 | Средняя длина chunk, токенов | Токенов в top-5 | Скорость, мс на вопрос |"),
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        name = f"{row['strategy']} {row['params']}" + (f" + {reranker_name}" if row["rerank"] else " без re-ranker")
        lines.append(f"| {name} | {row['top_k']} | {_metrics(row)} | {row['recall@k']:.3f} | {_strict(row)} | "
                     f"{row['avg_tokens']} | {row['top5_tokens']} | {_speed(row)} |")
    return "\n".join(lines)


def misses_table(rows: list[dict[str, Any]], dataset: list[dict[str, Any]]) -> str:
    """Вопросы, где хотя бы у одной конфигурации фрагмент с ответом — не первый: на маленьком
    корпусе метрики почти у потолка, и различия между конфигурациями — это именно такие вопросы."""
    names = [row["strategy"] + (f" {row['params']}" if row["config"] != Config(row["strategy"]) else "")
             + (" + rr" if row["rerank"] else "") for row in rows]
    ranks = [{d["id"]: (d["first_rank"], d["ev_first_rank"]) for d in row["details"]} for row in rows]

    def cell(pair: tuple[int | None, int | None]) -> str:
        return " / ".join("—" if rank is None else str(rank) for rank in pair)

    lines = ["| Вопрос | Тип | Нужные документы | " + " | ".join(names) + " |",
             "|---|---|---|" + "---:|" * len(rows)]
    count = 0
    for item in dataset:
        found = [r.get(item["id"], (None, None)) for r in ranks]
        if all(ev == 1 for _, ev in found):
            continue
        count += 1
        lines.append(f"| {item['id']}. {item['question']} | {item.get('type', '')} | "
                     f"{', '.join(item['relevant_doc_ids'])} | " + " | ".join(cell(pair) for pair in found) + " |")
    if not count:
        return "Фрагмент с ответом — первый в выдаче у всех конфигураций на всех вопросах."
    return ("Позиция первого фрагмента нужного документа / первого фрагмента с ответом (— — нет среди "
            "кандидатов) на вопросах, где хотя бы у одной конфигурации фрагмент с ответом не первый "
            "(rr — с re-ranker, top-K 10):\n\n" + "\n".join(lines))


def type_table(row: dict[str, Any]) -> str:
    from app.services.retrieval_eval import by_type

    strict = [{"type": d["type"], "hit": 1.0 if d["ev_first_rank"] and d["ev_first_rank"] <= 5 else 0.0,
               "rr": 1.0 / d["ev_first_rank"] if d["ev_first_rank"] and d["ev_first_rank"] <= 10 else 0.0,
               "recall": d["ev_recall"]} for d in row["details"]]
    lines = ["| Тип вопроса | Вопросов | Hit Rate@5 | MRR@10 | Recall@10 | Строгие: Hit Rate@5 | MRR@10 | Recall@10 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    strict_by_type = by_type(strict)
    for kind, m in by_type(row["details"]).items():
        e = strict_by_type[kind]
        lines.append(f"| {kind} | {m['n']} | {m['hit']:.3f} | {m['mrr']:.3f} | {m['recall']:.3f} | {e['hit']:.3f} | "
                     f"{e['mrr']:.3f} | {e['recall']:.3f} |")
    return "\n".join(lines)


def env_lines(row: dict[str, Any]) -> str:
    config: Config = row["config"]
    lines = [f"RAG__CHUNKING={config.strategy}"]
    if config.strategy == "semantic":
        lines += [f"RAG__SEMANTIC_BUFFER_SIZE={config.buffer_size}", f"RAG__SEMANTIC_BREAKPOINT={config.breakpoint}"]
    else:
        lines += [f"RAG__CHUNK_SIZE={config.chunk_size}", f"RAG__CHUNK_OVERLAP={config.chunk_overlap}"]
    lines += [f"RAG__CANDIDATES_TOP_K={row['top_k']}", f"RERANK__ENABLED={'true' if row['rerank'] else 'false'}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- прогон
def experiment(exp: Experiment, *, sweep_rerank: bool = True) -> tuple[str, dict[str, Any]]:
    cfg, dataset, reranker = exp.cfg, exp.dataset, exp.reranker
    rr_name = getattr(reranker, "model", "re-ranker") if reranker else ""
    say(f"Вопросов: {len(dataset)}. Стратегии: {', '.join(COLLECTIONS)}.")
    strategies = [exp.run(Config(name)) for name in COLLECTIONS]
    best = max(strategies, key=score_key)
    best_config: Config = best["config"]
    say(f"Лучшая стратегия: {best['strategy']} (документ {_metrics(best)}; строго {_strict(best)})")

    rerank_rows = [best]
    if reranker:
        rerank_rows += [exp.run(best_config, top_k, rerank=True) for top_k in sorted(TOP_KS, reverse=True)][::-1]

    sweep: list[dict[str, Any]] = []
    for config in grid(best_config.strategy):
        for top_k in TOP_KS:
            sweep.append(exp.run(config, top_k))
        if reranker and sweep_rerank:
            for top_k in sorted(TOP_KS, reverse=True):        # сначала 20: оценки первых 10 — из кеша
                sweep.append(exp.run(config, top_k, rerank=True))
    sweep.sort(key=lambda row: (-row["config"].chunk_size, -row["config"].chunk_overlap, row["config"].buffer_size,
                                -row["config"].breakpoint, row["rerank"], row["top_k"]))
    final = max(sweep + rerank_rows, key=score_key)

    first = exp.embed_model.model_name
    types = ", ".join(f"{kind} — {n}" for kind, n in _type_counts(dataset).items())
    report = [
        f"## Прогон — {datetime.now().astimezone():%Y-%m-%d %H:%M}",
        "",
        (f"Корпус `{cfg.rag.lab_dir.name}`, эмбеддинги `{first}` (EMBEDDING_DIM={cfg.embedding_dim}), golden dataset — "
         f"{len(dataset)} вопросов ({types}). Метрики — по документам: id фрагмента — имя его файла. Один вопрос — "
         f"{1 / len(dataset):.3f} любой метрики."),
        "",
        "### Стратегии (chunk_size 512, overlap 64, top-K 10)",
        "",
        strategy_table(strategies + [r for r in rerank_rows if r["rerank"] and r["top_k"] == BASE_TOP_K]),
        "",
        f"### Re-ranker на лучшей стратегии ({best['strategy']})",
        "",
        rerank_table(rerank_rows, rr_name) if reranker else "Re-ranker не запускался (--no-rerank).",
        "",
        f"### Подбор параметров ({best['strategy']})",
        "",
        rerank_table(sweep, rr_name),
        "",
        "### По типам вопросов (итоговая конфигурация)",
        "",
        type_table(final),
        "",
        "### Промахи",
        "",
        misses_table(strategies + [r for r in sweep if r["top_k"] == BASE_TOP_K and not any(r is s for s in strategies)],
                     dataset),
        "",
        (f"### Итог: {final['strategy']} {final['params']}, top-K {final['top_k']}, "
         f"{'с re-ranker' if final['rerank'] else 'без re-ranker'} — документ {_metrics(final)}, строго {_strict(final)}"),
        "",
        "```",
        env_lines(final),
        "```",
        "",
    ]
    data = {"strategies": strategies, "rerank": rerank_rows, "sweep": sweep, "final": final}
    return "\n".join(report), data


def _type_counts(dataset: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in dataset:
        counts[item.get("type") or "—"] = counts.get(item.get("type") or "—", 0) + 1
    return counts


def _jsonable(data: dict[str, Any]) -> Any:
    def row(r: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in r.items() if k != "config"}
        out["config"] = {"strategy": r["config"].strategy, "chunk_size": r["config"].chunk_size,
                         "chunk_overlap": r["config"].chunk_overlap, "buffer_size": r["config"].buffer_size,
                         "breakpoint": r["config"].breakpoint}
        return out
    return {key: [row(r) for r in value] if isinstance(value, list) else row(value) for key, value in data.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Нарезка, re-ranker и подбор параметров по golden dataset (блок 5.4)")
    parser.add_argument("--rebuild", action="store_true", help="переиндексировать все коллекции эксперимента")
    parser.add_argument("--no-rerank", action="store_true", help="без re-ranker")
    parser.add_argument("--no-sweep-rerank", action="store_true", help="re-ranker только на лучшей стратегии, "
                        "в подборе параметров — без него (быстрее на CPU)")
    parser.add_argument("--rerank-cache", action="store_true", help="оценки re-ranker из SQLite (RERANK__CACHE_PATH): "
                        "повторный прогон быстрее, но время re-ranker не замеряется")
    parser.add_argument("--out", type=Path, help="сохранить Markdown в файл (UTF-8)")
    parser.add_argument("--json", type=Path, help="сохранить строки таблиц и выдачу по вопросам в JSON")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from pydantic import ValidationError

    from app.core.config import get_settings
    from app.observability.logging import setup_logging
    from app.services import embeddings
    from app.services.rag import ProjectEmbedding, no_implicit_event_loop
    from app.services.rag_common import RAGError
    from app.services.reranker import RerankError
    from app.services.retrieval_eval import load_dataset
    from app.services.vector_store import make_qdrant_client

    setup_logging("WARNING", stream=sys.stderr)
    no_implicit_event_loop()
    client = reranker = None
    try:
        cfg = get_settings()
        if not cfg.qdrant_url:
            raise RAGError("QDRANT_URL не задан: эксперимент индексирует коллекции в Qdrant")
        corpus = [p.name for p in cfg.rag.lab_dir.iterdir() if p.is_file()]
        dataset = load_dataset(corpus=corpus)
        client = make_qdrant_client(cfg.qdrant_url, cfg.qdrant_api_key.get_secret_value() if cfg.qdrant_api_key else None,
                                    asynchronous=False)
        if not args.no_rerank:
            reranker = build_experiment_reranker(cfg, persistent_cache=args.rerank_cache)
        exp = Experiment(cfg, client=client, embed_model=ProjectEmbedding(embeddings.default_service()),
                         dataset=dataset, reranker=reranker, rebuild=args.rebuild)
        markdown, data = experiment(exp, sweep_rerank=not args.no_sweep_rerank)
        print(markdown)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(markdown, encoding="utf-8")
            say(f"Markdown — {args.out}")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")
            say(f"JSON — {args.json}")
        return 0
    except (RAGError, RerankError, embeddings.EmbeddingError, ValidationError, ValueError) as exc:
        say(f"Ошибка: {exc}")
        return 2
    finally:
        if reranker is not None:
            reranker.close()
        if client is not None:
            client.close()


def build_experiment_reranker(cfg: Any, *, persistent_cache: bool) -> Any:
    """Re-ranker из RERANK__* независимо от RERANK__ENABLED: эксперимент сравнивает с ним и без."""
    from app.services.reranker import build_reranker

    rerank = cfg.rerank.model_copy(update={"enabled": True, "cache_enabled": persistent_cache})
    return build_reranker(cfg.model_copy(update={"rerank": rerank}))


if __name__ == "__main__":
    sys.exit(main())
