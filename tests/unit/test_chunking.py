"""
Блок 5.4 без сети: три стратегии нарезки на настоящем корпусе data/rag-block-03, метрики поиска
(в том числе сверка с HitRate / MRR / Recall из LlamaIndex), golden dataset, re-ranker (cross-encoder
с подменной моделью и Cohere через httpx.MockTransport), выбор нарезки и re-ranker в RAGService и
scripts/chunking_experiment.py на Qdrant :memory: с эмбеддингами из триграмм.

Настоящие числа (bge-m3, bge-reranker-v2-m3) — docs/chunking_experiment.md.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import Document, QueryBundle
from pydantic import ValidationError
from qdrant_client import QdrantClient
from rag_fakes import CORPUS, DIM, RecordingLLM, TrigramEmbeddings, threaded

from app.core.config import Settings
from app.services import chunking, retrieval_eval
from app.services.embeddings import count_tokens
from app.services.rag import ProjectEmbedding, RAGService, load_corpus
from app.services.rag_baremetal import read_file
from app.services.reranker import (
    Candidate,
    CohereReranker,
    CrossEncoderReranker,
    RerankError,
    ScoreCache,
    build_reranker,
)

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

ROOT = Path(__file__).resolve().parents[2]
FILES = sorted(p.name for p in CORPUS.iterdir())


def settings(rerank: dict[str, Any] | None = None, **rag: Any) -> Settings:
    return Settings(_env_file=None, qdrant_url="http://127.0.0.1:6333", embedding_dim=DIM,
                    llm={"openai_api_key": "test-key", "base_url": "http://127.0.0.1:11434/v1",
                         "default_model": "llama3.2"},
                    rag={"data_dir": CORPUS, "min_score": 0.0, **rag}, rerank=rerank or {})


def corpus(label: str = "test") -> list[Document]:
    return load_corpus(CORPUS, "trigram", label)


SENTENCE_END = re.compile(r"[.!?…»):]\s*$")


def ends_at_boundary(nodes, documents) -> float:
    """Доля фрагментов, которые кончаются концом предложения или строки исходного документа."""
    texts = {d.id_: d.text for d in documents}
    ok = sum(1 for n in nodes if SENTENCE_END.search(n.get_content())
             or texts[n.ref_doc_id][n.end_char_idx:n.end_char_idx + 1] in ("", "\n"))
    return ok / len(nodes)


# ---------------------------------------------------------------- нарезка
def test_russian_sentences_keep_text_and_abbreviations():
    text = "Цена — 4 900 ₽ в месяц, т. е. 58 800 ₽ за двенадцать месяцев. Обмен с 1С каждые 30 минут!\n\nЕщё? Да."
    parts = chunking.split_sentences_ru(text)
    assert "".join(parts) == text                     # пробелы и переносы не теряются
    assert [p.strip() for p in parts] == ["Цена — 4 900 ₽ в месяц, т. е. 58 800 ₽ за двенадцать месяцев.",
                                          "Обмен с 1С каждые 30 минут!", "Ещё?", "Да."]   # «т. е.» — не конец
    assert chunking.split_sentences_ru("") == []


def test_fixed_size_cuts_by_tokens_not_sentences():
    splitter = chunking.fixed_size()
    assert (splitter.chunk_size, splitter.chunk_overlap) == (512, 64)
    nodes = splitter.get_nodes_from_documents(corpus())
    assert all(count_tokens(n.get_content()) <= 512 for n in nodes)
    long_docs = [n for n in nodes if n.metadata["file_name"] == "07_team_and_data.md"]
    assert len(long_docs) >= 3
    assert any(not SENTENCE_END.search(n.get_content()) for n in nodes)       # режет посреди предложения
    first, second = long_docs[0].get_content(), long_docs[1].get_content()
    assert first.split()[-5:] == second.split()[:5] or second[:40] in first   # перекрытие


def test_recursive_splits_by_paragraphs_and_russian_sentences():
    splitter = chunking.recursive(256, 32)
    assert splitter.paragraph_separator == "\n\n"
    assert splitter._chunking_tokenizer_fn is chunking.split_sentences_ru
    documents = corpus()
    nodes = splitter.get_nodes_from_documents(documents)
    assert all(count_tokens(n.get_content()) <= 256 for n in nodes)
    assert ends_at_boundary(nodes, documents) == 1.0           # конец предложения или строки
    fixed = chunking.fixed_size(256, 32).get_nodes_from_documents(documents)
    assert ends_at_boundary(fixed, documents) < 0.5            # baseline режет посреди фразы
    # Особенность: заголовок раздела («## …» — отдельный абзац) может остаться в конце фрагмента, а
    # текст раздела уйти в следующий — разбор в docs/chunking_experiment.md.
    assert any(n.get_content().rstrip().splitlines()[-1].startswith("## ") for n in nodes)


class TopicEmbedding(BaseEmbedding):
    """Две темы: предложения про растения — один вектор, остальные — другой."""

    def _vector(self, text: str) -> list[float]:
        return [1.0, 0.0] if re.search(r"фикус|полив|растени", text.lower()) else [0.0, 1.0]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)


def test_semantic_splits_where_meaning_changes():
    text = ("Тариф «Старт» стоит 990 ₽. Тариф «Команда» стоит 4 900 ₽. Тариф «Бизнес» стоит 14 900 ₽. "
            "Оплатить можно картой. Счета лежат в биллинге. Фикус поливают раз в неделю. "
            "Растения пересаживают весной. Полив зимой реже. Растения любят свет. Фикусу нужен свет.")
    splitter = chunking.semantic(TopicEmbedding(model_name="topics"))
    assert (splitter.buffer_size, splitter.breakpoint_percentile_threshold) == (1, 95)
    nodes = splitter.get_nodes_from_documents([Document(text=text)])
    assert len(nodes) == 2                             # граница — на смене темы (±1 предложение из-за buffer_size)
    assert not re.search(r"Фикус|Растения", nodes[0].get_content()) and "Тариф" not in nodes[1].get_content()


def test_build_splitter_and_labels():
    assert isinstance(chunking.build_splitter("fixed", chunk_size=256, chunk_overlap=32), type(chunking.fixed_size()))
    with pytest.raises(ValueError, match="нужна модель эмбеддингов"):
        chunking.build_splitter("semantic")
    with pytest.raises(ValueError, match="Неизвестная стратегия"):
        chunking.build_splitter("hierarchical")
    assert chunking.label("recursive", chunk_size=256, chunk_overlap=32) == "recursive:256/32"
    assert chunking.label("semantic", chunk_size=256, buffer_size=2, breakpoint_percentile_threshold=90) == "semantic:2/90"


def test_chunk_stats():
    nodes = chunking.recursive().get_nodes_from_documents(corpus())
    stats = chunking.chunk_stats(nodes)
    assert stats["documents"] == 10 and stats["chunks"] == len(nodes)
    assert stats["chunks_per_doc"] == round(len(nodes) / 10, 1)
    assert stats["min_tokens"] <= stats["avg_tokens"] <= stats["max_tokens"] <= 512
    assert stats["avg_chars"] > stats["avg_tokens"]


# ---------------------------------------------------------------- метрики
def test_metrics_by_hand():
    retrieved = ["a.md", "b.md", "a.md", "c.md", "d.md", "e.md"]
    assert retrieval_eval.hit_rate(retrieved, ["c.md"], k=5) == 1.0
    assert retrieval_eval.hit_rate(retrieved, ["c.md"], k=3) == 0.0
    assert retrieval_eval.reciprocal_rank(retrieved, ["c.md"]) == 0.25
    assert retrieval_eval.reciprocal_rank(retrieved, ["c.md"], k=3) == 0.0
    assert retrieval_eval.recall(retrieved, ["c.md", "z.md"]) == 0.5
    assert retrieval_eval.recall(retrieved, ["a.md", "b.md", "e.md"], k=2) == pytest.approx(2 / 3)
    assert retrieval_eval.first_relevant_rank(retrieved, ["z.md"]) is None


@pytest.mark.parametrize(("retrieved", "relevant"), [
    (["a", "b", "a", "c", "d", "e", "f", "g", "h", "i", "j", "k"], ["c"]),
    (["a", "a", "a", "b", "b", "c", "a", "d", "e", "c", "z", "y"], ["d", "z"]),
    (["x", "y", "z"], ["a", "b", "c"]),
    (["b", "a", "c", "a", "b", "c", "d", "d", "e", "f", "g", "h"], ["a", "b", "c"]),
])
def test_metrics_match_llamaindex(retrieved, relevant):
    from llama_index.core.evaluation.retrieval.metrics import MRR, HitRate, Recall

    assert retrieval_eval.hit_rate(retrieved, relevant, 5) == HitRate().compute(
        expected_ids=relevant, retrieved_ids=retrieved[:5]).score
    assert retrieval_eval.reciprocal_rank(retrieved, relevant, 10) == MRR().compute(
        expected_ids=relevant, retrieved_ids=retrieved[:10]).score
    assert retrieval_eval.recall(retrieved, relevant, 10) == Recall().compute(
        expected_ids=relevant, retrieved_ids=retrieved[:10]).score


def test_evaluate_returns_three_numbers_and_details():
    dataset = [{"id": "q1", "type": "fact", "question": "раз", "relevant_doc_ids": ["a"]},
               {"id": "q2", "type": "comparison", "question": "два", "relevant_doc_ids": ["b", "c"]}]
    results = {"раз": ["x", "a"], "два": ["x", "x", "x", "x", "x", "b"]}
    details: list[dict[str, Any]] = []
    metrics = retrieval_eval.evaluate(results.__getitem__, dataset, details=details)
    assert metrics == {"hit_rate@5": 0.5, "mrr@10": round((1 / 2 + 1 / 6) / 2, 4), "recall@10": 0.75}
    assert [d["first_rank"] for d in details] == [2, 6]
    by_type = retrieval_eval.by_type(details)
    assert by_type["fact"] == {"n": 1, "hit": 1.0, "mrr": 0.5, "recall": 1.0}
    with pytest.raises(ValueError, match="Пустой набор"):
        retrieval_eval.evaluate(results.__getitem__, [])


def test_evaluate_evidence_needs_the_answer_inside_a_chunk():
    dataset = [{"id": "q1", "question": "SLA", "relevant_doc_ids": ["09.pdf"],
                "evidence": ["оплата картой 1 час", ["круглосуточно", "24/7"]]}]
    chunks = {"SLA": ["Регламент поддержки", "Высокий Не проходит \nОплата картой\n1 час 1 рабочий день",
                      "дежурный инженер (круглосуточно)"]}
    details: list[dict[str, Any]] = []
    metrics = retrieval_eval.evaluate_evidence(chunks.__getitem__, dataset, details=details)
    assert metrics == {"hit_rate@5": 1.0, "mrr@10": 0.5, "recall@10": 1.0}
    cut = {"SLA": ["Высокий Не проходит оплата", "картой 1 час 1 рабочий день"]}     # граница разрезала ответ
    assert retrieval_eval.evaluate_evidence(cut.__getitem__, dataset) == {"hit_rate@5": 0.0, "mrr@10": 0.0,
                                                                         "recall@10": 0.0}
    assert retrieval_eval.covered("…ДЕЖУРНЫЙ 24/7…", dataset[0]["evidence"]) == {1}


def test_golden_dataset():
    dataset = retrieval_eval.load_dataset(corpus=FILES)
    assert len(dataset) >= 20
    assert {item["type"] for item in dataset} == {"fact", "paraphrase", "comparison", "multi_step"}
    assert any(len(item["relevant_doc_ids"]) == 3 for item in dataset)
    assert not any("10_houseplants.txt" in item["relevant_doc_ids"] for item in dataset)   # отвлекающий документ
    assert len({item["question"] for item in dataset}) == len(dataset)
    texts = {name: retrieval_eval.normalize(read_file(CORPUS / name)) for name in FILES}
    for item in dataset:
        assert item["evidence"], item["id"]
        for piece in item["evidence"]:                 # каждая цитата есть дословно в одном из нужных файлов
            variants = piece if isinstance(piece, list) else [piece]
            found = {name for name, text in texts.items() for v in variants if retrieval_eval.normalize(v) in text}
            assert found and found <= set(item["relevant_doc_ids"]), (item["id"], piece, found)


@pytest.mark.parametrize(("item", "message"), [
    ({"id": "q1", "question": " ", "relevant_doc_ids": ["a.md"]}, "пустой question"),
    ({"id": "q1", "question": "?", "relevant_doc_ids": []}, "от 1 до 3"),
    ({"id": "q1", "question": "?", "relevant_doc_ids": ["a.md", "b.md", "c.md", "d.md"]}, "от 1 до 3"),
    ({"id": "q1", "question": "?", "relevant_doc_ids": ["a.md", "a.md"]}, "от 1 до 3"),
    ({"id": "q1", "question": "?", "relevant_doc_ids": ["нет.md"]}, "нет в корпусе нет.md"),
    ({"id": "q1", "question": "?", "relevant_doc_ids": ["a.md"], "evidence": [""]}, "evidence"),
])
def test_load_dataset_rejects_bad_items(tmp_path, item, message):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"questions": [item]}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        retrieval_eval.load_dataset(path, corpus=["a.md", "b.md", "c.md", "d.md"])
    duplicate = {"id": "q1", "question": "?", "relevant_doc_ids": ["a.md"]}
    path.write_text(json.dumps([duplicate, duplicate]), encoding="utf-8")
    with pytest.raises(ValueError, match="повторяется"):
        retrieval_eval.load_dataset(path)


# ---------------------------------------------------------------- re-ranker
class FakeCrossEncoder:
    """Подменная модель sentence-transformers: оценка — число общих слов вопроса и фрагмента."""

    instances: ClassVar[list[FakeCrossEncoder]] = []

    def __init__(self, model: str, *, device: str, max_length: int) -> None:
        self.args = (model, device, max_length)
        self.calls: list[list[tuple[str, str]]] = []
        FakeCrossEncoder.instances.append(self)

    def predict(self, pairs, batch_size, show_progress_bar, convert_to_numpy):
        self.calls.append(list(pairs))
        self.batch_size = batch_size
        return [float(len(set(q.lower().split()) & set(t.lower().split()))) for q, t in pairs]


CANDIDATES = [Candidate("Фикус поливают раз в неделю", "10.txt", 0.9),
              Candidate("Тариф «Команда» стоит 4 900 ₽ в месяц", "02.md", 0.7),
              Candidate("Сколько стоит тариф «Команда» в месяц? 4 900 ₽", "02.md", 0.6)]


def test_cross_encoder_reranks_and_caches(tmp_path):
    reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3", max_length=256, batch_size=4,
                                    cache=ScoreCache(tmp_path / "rerank.sqlite"), loader=FakeCrossEncoder)
    top = reranker.rerank("сколько стоит тариф «Команда» в месяц?", CANDIDATES, top_n=2)
    assert [c.doc_id for c in top] == ["02.md", "02.md"]
    assert top[0].text.startswith("Сколько стоит") and top[0].vector_score == 0.6 and top[0].score > top[1].score
    model = FakeCrossEncoder.instances[-1]
    assert model.args == ("BAAI/bge-reranker-v2-m3", "cpu", 256) and model.batch_size == 4
    assert reranker.pairs_scored == 3
    again = reranker.rerank("сколько стоит тариф «Команда» в месяц?", CANDIDATES)
    assert len(again) == 3 and len(model.calls) == 1 and reranker.pairs_scored == 3      # из кеша
    reranker.close()
    reopened = CrossEncoderReranker(cache=ScoreCache(tmp_path / "rerank.sqlite"), loader=FakeCrossEncoder)
    reopened.rerank("сколько стоит тариф «Команда» в месяц?", CANDIDATES)
    assert reopened.pairs_scored == 0 and reopened._model is None    # SQLite: модель даже не загружалась
    reopened.close()
    assert reranker.rerank("вопрос", []) == []


def test_cross_encoder_without_sentence_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RerankError, match="pip install -r requirements-embeddings.txt"):
        CrossEncoderReranker().rerank("вопрос", CANDIDATES)
    with pytest.raises(RerankError, match="sentence-transformers"):
        build_reranker(settings(rerank={"enabled": True, "cache_enabled": False}))


def cohere(handler) -> CohereReranker:
    return CohereReranker("test-key", base_url="https://api.cohere.test",
                          http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_cohere_rerank_request_and_order():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "x", "results": [{"index": 2, "relevance_score": 0.9},
                                                                {"index": 1, "relevance_score": 0.5},
                                                                {"index": 0, "relevance_score": 0.01}]})

    top = cohere(handler).rerank("сколько стоит «Команда»?", CANDIDATES, top_n=2)
    assert [c.score for c in top] == [0.9, 0.5] and top[0].text == CANDIDATES[2].text
    request = seen[0]
    assert str(request.url) == "https://api.cohere.test/v2/rerank"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {"model": "rerank-v3.5", "query": "сколько стоит «Команда»?",
                                           "documents": [c.text for c in CANDIDATES], "top_n": 3}


@pytest.mark.parametrize(("response", "message"), [
    (httpx.Response(401, json={"message": "invalid api token"}), "не принял ключ"),
    (httpx.Response(429, text="Too Many Requests"), "ответил 429"),
    (httpx.ConnectError("refused"), "не ответил: ConnectError"),
])
def test_cohere_errors_are_explained(response, message):
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(response, Exception):
            raise response
        return response

    with pytest.raises(RerankError, match=message):
        cohere(handler).rerank("вопрос", CANDIDATES)


def test_rerank_settings_and_factory():
    assert build_reranker(settings()) is None                         # по умолчанию выключен
    with pytest.raises(ValidationError, match="RERANK__API_KEY"):
        settings(rerank={"enabled": True, "provider": "cohere"})
    reranker = build_reranker(settings(rerank={"enabled": True, "provider": "cohere", "api_key": "test-key",
                                               "cache_enabled": False}))
    assert isinstance(reranker, CohereReranker) and reranker.model == "rerank-v3.5"
    reranker.close()
    cfg = Settings(_env_file=None)
    assert (cfg.rag.chunking, cfg.rag.candidates_top_k, cfg.rerank.enabled) == ("recursive", 10, False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, rag={"chunking": "hierarchical"})
    with pytest.raises(ValidationError, match="RAG__CANDIDATES_TOP_K=2 меньше RAG__SIMILARITY_TOP_K=3"):
        Settings(_env_file=None, rag={"candidates_top_k": 2})


def test_rerank_settings_from_environment(monkeypatch):
    monkeypatch.setenv("RAG__CHUNKING", "semantic")
    monkeypatch.setenv("RAG__SEMANTIC_BREAKPOINT", "90")
    monkeypatch.setenv("RAG__CANDIDATES_TOP_K", "20")
    monkeypatch.setenv("RERANK__ENABLED", "true")
    monkeypatch.setenv("RERANK__MAX_LENGTH", "384")
    cfg = Settings(_env_file=None)
    assert (cfg.rag.chunking, cfg.rag.semantic_breakpoint, cfg.rag.candidates_top_k) == ("semantic", 90, 20)
    assert (cfg.rerank.enabled, cfg.rerank.provider, cfg.rerank.max_length) == (True, "cross-encoder", 384)


# ---------------------------------------------------------------- RAGService: нарезка и re-ranker из настроек
class ReverseReranker:
    """Re-ranker, который переворачивает порядок кандидатов, — видно, что модель получила его порядок."""

    model = "reverse"

    def __init__(self) -> None:
        self.seen: list[int] = []

    def rerank(self, query: str, candidates, top_n=None):
        self.seen.append(len(candidates))
        result = list(reversed(candidates))
        return result[:top_n] if top_n else result

    def close(self) -> None:
        pass


def test_rag_service_uses_chunking_and_reranker():
    client = QdrantClient(":memory:")
    reranker = ReverseReranker()
    cfg = settings(chunking="fixed", chunk_size=256, chunk_overlap=32, candidates_top_k=6)
    service = RAGService(cfg, client=client, embed_model=ProjectEmbedding(TrigramEmbeddings()), llm=RecordingLLM(),
                         reranker=reranker).build()
    expected = chunking.fixed_size(256, 32).get_nodes_from_documents(corpus())
    assert service.indexed == len(expected)
    record = client.scroll("rag_block_03", limit=1, with_payload=["chunking"])[0][0]
    assert record.payload["chunking"] == "fixed:256/32"
    vector_only = threaded(service.query_engine.retrieve, QueryBundle("Сколько стоит тариф «Команда»?"))
    answer = threaded(service.answer, "Сколько стоит тариф «Команда»?")
    assert reranker.seen == [6] and len(answer["sources"]) == 3
    assert [s["score"] for s in answer["sources"]] == [round(n.score, 3) for n in vector_only][::-1][:3]
    assert answer["top_score"] == round(max(n.score for n in vector_only), 3)       # порог — по близости Qdrant
    assert len(service.llm.prompts) == 1                                            # модель вызвана


def test_rag_service_warns_when_collection_has_other_chunking():
    from log_capture import captured_logs, events

    client = QdrantClient(":memory:")
    embedder = TrigramEmbeddings()
    RAGService(settings(), client=client, embed_model=ProjectEmbedding(embedder), llm=RecordingLLM()).build()
    with captured_logs("WARNING") as records:
        other = RAGService(settings(chunking="fixed"), client=client, embed_model=ProjectEmbedding(embedder),
                           llm=RecordingLLM()).build()
    assert other.indexed == 0 and other.chunking_mismatch == "recursive:512/64"   # работает, но предупреждает
    assert events(records, "rag_chunking_mismatch")[0]["configured"] == "fixed:512/64"


# ---------------------------------------------------------------- scripts/chunking_experiment.py
def load_experiment():
    spec = importlib.util.spec_from_file_location("chunking_experiment", ROOT / "scripts" / "chunking_experiment.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["chunking_experiment"] = module          # dataclass Config ищет свой модуль в sys.modules
    spec.loader.exec_module(module)
    return module


def test_experiment_offline():
    script = load_experiment()
    client = QdrantClient(":memory:")
    embedder = TrigramEmbeddings()
    dataset = retrieval_eval.load_dataset(corpus=FILES)[:8]
    reranker = CrossEncoderReranker(loader=FakeCrossEncoder)
    exp = script.Experiment(settings(), client=client, embed_model=ProjectEmbedding(embedder), dataset=dataset,
                            reranker=reranker)
    markdown, data = threaded(script.experiment, exp)
    assert {c.name for c in client.get_collections().collections} >= {"docs_fixed", "docs_recursive", "docs_semantic"}
    assert [row["strategy"] for row in data["strategies"]] == ["fixed", "recursive", "semantic"]
    for row in data["strategies"] + data["rerank"] + data["sweep"]:
        assert {"hit_rate@5", "mrr@10", "recall@10", "ev_hit_rate@5", "avg_tokens", "retrieval_ms"} <= set(row)
        assert 0 <= row["hit_rate@5"] <= 1 and row["retrieval_ms"] > 0
    assert [(r["rerank"], r["top_k"]) for r in data["rerank"]] == [(False, 10), (True, 10), (True, 20)]
    assert data["rerank"][1]["rerank_estimated"] and not data["rerank"][2]["rerank_estimated"]   # top-10 — из кеша
    assert len(data["sweep"]) == 4 * 4                       # 4 нарезки × (top-K 10, 20) × (без, с re-ranker)
    best = data["strategies"][[r["strategy"] for r in data["strategies"]].index(data["rerank"][0]["strategy"])]
    assert best is max(data["strategies"], key=script.score_key)
    for header in ("Стратегия", "Hit Rate@5", "MRR@10", "Recall@10", "Средняя длина chunk", "Скорость retrieval, мс",
                   "### Re-ranker на лучшей стратегии", "### Подбор параметров", "### Итог"):
        assert header in markdown
    assert re.search(r"RAG__CHUNKING=(fixed|recursive|semantic)\n", markdown)
    assert "RERANK__ENABLED=" in markdown and "RAG__CANDIDATES_TOP_K=" in markdown
    json.dumps(script._jsonable(data), ensure_ascii=False)      # --json сериализуется

    calls = embedder.document_calls
    again = script.Experiment(settings(), client=client, embed_model=ProjectEmbedding(embedder), dataset=dataset)
    threaded(again.index, script.Config("recursive"))
    threaded(again.index, script.Config("fixed"))
    assert embedder.document_calls == calls                  # готовые коллекции: корпус заново не эмбеддится


def test_experiment_score_key_and_env_lines():
    script = load_experiment()
    base = {"ev_hit_rate@5": 1.0, "ev_mrr@10": 0.8, "hit_rate@5": 1.0, "mrr@10": 0.9, "ev_recall@10": 1.0,
            "recall@10": 1.0, "rerank": False, "top_k": 10, "top5_tokens": 1000}
    assert script.score_key({**base, "ev_mrr@10": 0.9}) > script.score_key(base)
    assert script.score_key(base) > script.score_key({**base, "rerank": True})     # при равенстве — без re-ranker
    assert script.score_key(base) > script.score_key({**base, "top_k": 20})
    row = {**base, "config": script.Config("recursive", chunk_size=256, chunk_overlap=32)}
    assert script.env_lines(row) == ("RAG__CHUNKING=recursive\nRAG__CHUNK_SIZE=256\nRAG__CHUNK_OVERLAP=32\n"
                                     "RAG__CANDIDATES_TOP_K=10\nRERANK__ENABLED=false")
    assert script.Config("semantic", buffer_size=2, breakpoint=90).collection == "docs_semantic_b2_p90"
    assert script.Config("fixed").collection == "docs_fixed"
    assert script.Config("fixed", chunk_size=256, chunk_overlap=32).collection == "docs_fixed_256_32"
