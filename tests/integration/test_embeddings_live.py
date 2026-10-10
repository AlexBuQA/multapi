"""
Живая проверка эмбеддингов (блок 5.1, маркер llm): настоящие модели на mini_benchmark.

    pytest -m llm tests/integration/test_embeddings_live.py -v

- bge-m3 в Ollama (модель по умолчанию): 1024 числа, длина 1, нужный фрагмент ближе ложного
  хотя бы в 7 вопросах из 10. Ollama не запущена или модель не скачана — тест пропускается
  (ollama pull bge-m3).
- multilingual-e5-small через sentence-transformers — smoke-тест префиксов: печатает
  близость нужной пары и отрыв от ложной для трёх вариантов — с префиксами, вопрос без
  «query: » при базе с «passage: », без префиксов везде. Проверяет, что префиксы доходят
  до модели (векторы другие) и что с ними нужный фрагмент ближе ложного хотя бы в 8 из 10.
  Гипотеза задания «без префиксов близость нужной пары падает» на наших данных для абсолютной
  близости не подтвердилась (Windows, 10.10.2026: 0.884 с префиксами, 0.886 без) — поэтому
  она не утверждение теста, а печатаемый замер; разбор — docs/embeddings.md.
  Нет пакета — пропускается (pip install -r requirements-embeddings.txt); первый запуск
  скачивает модель с Hugging Face (~470 МБ).

Кеш выключен: проверяется модель, а не файл кеша.
"""
from __future__ import annotations

import json
import socket
import statistics
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.llm

OLLAMA_URL = "http://localhost:11434/v1"
ITEMS = json.loads((ROOT / "tests" / "eval" / "mini_benchmark.json").read_text(encoding="utf-8"))


def reachable(url: str) -> bool:
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=1):
            return True
    except OSError:
        return False


def pair_scores(query_vectors, doc_vectors):
    from app.services.embeddings import cosine

    rel = [cosine(q, doc_vectors[2 * n]) for n, q in enumerate(query_vectors)]
    irr = [cosine(q, doc_vectors[2 * n + 1]) for n, q in enumerate(query_vectors)]
    return rel, irr


def documents() -> list[str]:
    return [text for item in ITEMS for text in (item["relevant"], item["irrelevant"])]


@pytest.mark.skipif(not reachable(OLLAMA_URL), reason=f"Ollama недоступна: {OLLAMA_URL}")
def test_bge_m3_in_ollama():
    from app.services.embeddings import EmbeddingError, EmbeddingService, OpenAICompatibleBackend

    service = EmbeddingService(OpenAICompatibleBackend(model="bge-m3", base_url=OLLAMA_URL, api_key="ollama",
                                                       timeout=300), cache=None)
    try:
        queries = [service.embed_query(item["query"]) for item in ITEMS]
        docs = service.embed_documents(documents())
    except EmbeddingError as exc:
        pytest.skip(str(exc))
    finally:
        service.close()
    assert service.dimension == 1024
    assert all(abs(sum(x * x for x in vector) - 1) < 1e-4 for vector in queries + docs)
    rel, irr = pair_scores(queries, docs)
    assert sum(r > i for r, i in zip(rel, irr, strict=True)) >= 7


def test_e5_prefixes_smoke():
    pytest.importorskip("sentence_transformers")
    from app.services.embeddings import EmbeddingService, SentenceTransformerBackend, cosine

    service = EmbeddingService(SentenceTransformerBackend("intfloat/multilingual-e5-small"), cache=None)
    try:
        queries = [service.embed_query(item["query"]) for item in ITEMS]
        raw_queries = service.embed_texts([item["query"] for item in ITEMS])
        docs = service.embed_documents(documents())
        raw_docs = service.embed_texts(documents())
    finally:
        service.close()
    variants = {"с префиксами (как надо)": pair_scores(queries, docs),
                "вопрос без query:, база с passage:": pair_scores(raw_queries, docs),
                "без префиксов везде": pair_scores(raw_queries, raw_docs)}
    print("\nE5 small: близость нужной пары / отрыв rel − irr / пары rel > irr")
    for title, (rel, irr) in variants.items():
        margin = statistics.fmean(r - i for r, i in zip(rel, irr, strict=True))
        right = sum(r > i for r, i in zip(rel, irr, strict=True))
        print(f"  {title:<36} {statistics.fmean(rel):.4f} / {margin:+.4f} / {right}/{len(rel)}")
    assert min(cosine(a, b) for a, b in zip(queries, raw_queries, strict=True)) < 0.9999   # префикс дошёл до модели
    rel, irr = variants["с префиксами (как надо)"]
    assert sum(r > i for r, i in zip(rel, irr, strict=True)) >= 8
