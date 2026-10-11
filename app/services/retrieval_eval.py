"""
Метрики поиска по golden dataset (блок 5.4): Hit Rate@5, MRR@10, Recall@10.

    from app.services.retrieval_eval import evaluate, load_dataset
    evaluate(retrieve, load_dataset())   # {"hit_rate@5": 0.93, "mrr@10": 0.81, "recall@10": 0.95}

retrieve(question) -> список id документов в порядке выдачи — по одному на найденный фрагмент,
поэтому id повторяются: три фрагмента одного файла подряд — три позиции. Документ — файл корпуса
(data/rag-block-03), его id — имя файла; эталон — relevant_doc_ids вопроса (1–3 файла).

- Hit Rate@k — доля вопросов, у которых среди первых k фрагментов есть фрагмент нужного файла.
- MRR@k — среднее 1 / позиция первого фрагмента нужного файла среди первых k (нет — 0).
- Recall@k — доля нужных файлов вопроса, фрагменты которых попали в первые k, в среднем по
  вопросам. Для вопроса с одним нужным файлом совпадает с Hit Rate@k.

Определения те же, что у HitRate, MRR и Recall из llama_index.core.evaluation, если отдать им id
документов первых k фрагментов (тесты сверяют). Модуль не зависит от LlamaIndex: функцию поиска
можно отдать любую — версии без фреймворка тоже.

Строгий уровень — evaluate_evidence(): те же три метрики, но «нужный» фрагмент — тот, в тексте
которого есть часть ответа дословно (evidence вопроса; без учёта регистра и переносов строк). На
корпусе из 10 файлов документный уровень почти всегда 1.0 — фрагмент нужного файла найдётся
среди первых пяти почти при любой нарезке. А вот целым ли дошёл до модели сам ответ, зависит от
того, где прошли границы фрагментов. Recall — доля частей ответа (у вопроса их 1–3), найденных в
первых k фрагментах.
"""
from __future__ import annotations

import json
import re
import statistics
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "tests" / "eval" / "retrieval_dataset.json"
HIT_K, MRR_K, RECALL_K = 5, 10, 10
MAX_RELEVANT = 3


def load_dataset(path: Path = DATASET, *, corpus: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Вопросы golden dataset. Проверяет схему: question, relevant_doc_ids из 1–3 разных id; id
    вопросов не повторяются; если передан corpus (имена файлов) — все id есть в корпусе."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) else data
    known = set(corpus) if corpus is not None else None
    seen: set[str] = set()
    for number, item in enumerate(questions, 1):
        where = f"{Path(path).name}, вопрос {item.get('id', number)}"
        if not str(item.get("question", "")).strip():
            raise ValueError(f"{where}: пустой question")
        relevant = item.get("relevant_doc_ids")
        if not isinstance(relevant, list) or not 1 <= len(relevant) <= MAX_RELEVANT or len(set(relevant)) != len(relevant):
            raise ValueError(f"{where}: relevant_doc_ids — от 1 до {MAX_RELEVANT} разных id документов")
        if known is not None and (missing := [doc for doc in relevant if doc not in known]):
            raise ValueError(f"{where}: нет в корпусе {', '.join(missing)}")
        evidence = item.get("evidence", [])
        if not isinstance(evidence, list) or any(not _alternatives(piece) for piece in evidence):
            raise ValueError(f"{where}: evidence — список цитат (строка или список вариантов)")
        key = str(item.get("id", number))
        if key in seen:
            raise ValueError(f"{where}: id вопроса повторяется")
        seen.add(key)
    if not questions:
        raise ValueError(f"{Path(path).name}: нет вопросов")
    return questions


# ---------------------------------------------------------------- метрики одного вопроса
def first_relevant_rank(retrieved: Sequence[str], relevant: Iterable[str], k: int | None = None) -> int | None:
    """Позиция (с 1) первого фрагмента нужного документа среди первых k; нет — None."""
    wanted = set(relevant)
    for rank, doc_id in enumerate(retrieved[:k] if k else retrieved, 1):
        if doc_id in wanted:
            return rank
    return None


def hit_rate(retrieved: Sequence[str], relevant: Iterable[str], k: int = HIT_K) -> float:
    return 1.0 if first_relevant_rank(retrieved, relevant, k) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str], k: int = MRR_K) -> float:
    rank = first_relevant_rank(retrieved, relevant, k)
    return 1.0 / rank if rank else 0.0


def recall(retrieved: Sequence[str], relevant: Iterable[str], k: int = RECALL_K) -> float:
    wanted = set(relevant)
    return len(wanted & set(retrieved[:k])) / len(wanted) if wanted else 0.0


# ---------------------------------------------------------------- по набору вопросов
def evaluate(retrieve: Callable[[str], Sequence[str]], dataset: Sequence[dict[str, Any]], *,
             hit_k: int = HIT_K, mrr_k: int = MRR_K, recall_k: int = RECALL_K,
             details: list[dict[str, Any]] | None = None) -> dict[str, float]:
    """Hit Rate@hit_k, MRR@mrr_k и Recall@recall_k по всем вопросам — одним словарем:
    {"hit_rate@5": …, "mrr@10": …, "recall@10": …}. details — список, куда дописать строку на
    вопрос (id, тип, позиция первого нужного документа, метрики, выдача)."""
    hits, ranks, recalls = [], [], []
    for item in dataset:
        retrieved = list(retrieve(item["question"]))
        relevant = item["relevant_doc_ids"]
        row = {"id": item.get("id"), "type": item.get("type"), "relevant": relevant,
               "first_rank": first_relevant_rank(retrieved, relevant),
               "hit": hit_rate(retrieved, relevant, hit_k), "rr": reciprocal_rank(retrieved, relevant, mrr_k),
               "recall": recall(retrieved, relevant, recall_k), "retrieved": retrieved}
        hits.append(row["hit"])
        ranks.append(row["rr"])
        recalls.append(row["recall"])
        if details is not None:
            details.append(row)
    if not hits:
        raise ValueError("Пустой набор вопросов")
    return {f"hit_rate@{hit_k}": round(statistics.mean(hits), 4), f"mrr@{mrr_k}": round(statistics.mean(ranks), 4),
            f"recall@{recall_k}": round(statistics.mean(recalls), 4)}


# ---------------------------------------------------------------- строгий уровень: фрагмент с ответом
def normalize(text: str) -> str:
    """Для сравнения цитат: нижний регистр, неразрывные пробелы и переносы строк — обычный пробел."""
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ").replace("\u202f", " ")).strip().lower()


def _alternatives(piece: Any) -> list[str]:
    values = piece if isinstance(piece, list) else [piece]
    return [value for value in values if isinstance(value, str) and value.strip()]


def covered(text: str, evidence: Sequence[Any]) -> set[int]:
    """Номера частей ответа, которые целиком есть в тексте фрагмента."""
    flat = normalize(text)
    return {number for number, piece in enumerate(evidence)
            if any(normalize(variant) in flat for variant in _alternatives(piece))}


def evaluate_evidence(retrieve_texts: Callable[[str], Sequence[str]], dataset: Sequence[dict[str, Any]], *,
                      hit_k: int = HIT_K, mrr_k: int = MRR_K, recall_k: int = RECALL_K,
                      details: list[dict[str, Any]] | None = None) -> dict[str, float]:
    """Hit Rate, MRR и Recall по фрагментам, где есть часть ответа (evidence). retrieve_texts(question)
    -> тексты фрагментов в порядке выдачи. Вопросы без evidence пропускаются."""
    hits, ranks, recalls = [], [], []
    for item in dataset:
        evidence = item.get("evidence") or []
        if not evidence:
            continue
        found = [covered(text, evidence) for text in retrieve_texts(item["question"])]
        first = next((rank for rank, pieces in enumerate(found, 1) if pieces), None)
        in_top = set().union(*found[:recall_k]) if found else set()
        row = {"id": item.get("id"), "type": item.get("type"), "first_rank": first,
               "hit": 1.0 if first and first <= hit_k else 0.0, "rr": 1.0 / first if first and first <= mrr_k else 0.0,
               "recall": len(in_top) / len(evidence)}
        hits.append(row["hit"])
        ranks.append(row["rr"])
        recalls.append(row["recall"])
        if details is not None:
            details.append(row)
    if not hits:
        raise ValueError("Ни у одного вопроса нет evidence")
    return {f"hit_rate@{hit_k}": round(statistics.mean(hits), 4), f"mrr@{mrr_k}": round(statistics.mean(ranks), 4),
            f"recall@{recall_k}": round(statistics.mean(recalls), 4)}


def by_type(details: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Те же метрики по типам вопросов (fact, paraphrase, comparison, multi_step) — из details."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in details:
        groups.setdefault(row.get("type") or "—", []).append(row)
    return {kind: {"n": len(rows), "hit": round(statistics.mean(r["hit"] for r in rows), 4),
                   "mrr": round(statistics.mean(r["rr"] for r in rows), 4),
                   "recall": round(statistics.mean(r["recall"] for r in rows), 4)}
            for kind, rows in groups.items()}
