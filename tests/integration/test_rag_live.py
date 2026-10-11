"""
Живой RAG (блок 5.3, маркер llm): обе версии на настоящих Qdrant, модели эмбеддингов и модели
ответа из .env (bge-m3 и llama3.2 в Ollama).

    docker compose up -d qdrant
    pytest -m llm tests/integration/test_rag_live.py -v -s

Корпус индексируется во временные коллекции pytest_rag_<суффикс> (боевые rag_block_03 и
rag_block_03_baremetal не трогаются) и удаляется в конце. Qdrant или модель не отвечают — тест
пропускается. На CPU — несколько минут: два вопроса × две версии плюс эмбеддинги корпуса (при
повторе — из кеша).

Проверяется то, что от удачи модели почти не зависит: форма ответа, нужный файл первым для
вопроса о тарифе, отказ без вызова модели для вопроса не по базе (если близость ниже порога)
или честное «нет» от модели. Ответы печатаются (-s) — их разбор в docs/rag.md.
"""
from __future__ import annotations

import os
import socket
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse

import pytest
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENV = dotenv_values(ROOT / ".env") if (ROOT / ".env").exists() else {}
URL = os.environ.get("QDRANT_TEST_URL") or ENV.get("QDRANT_URL") or ""

GOOD = "Сколько стоит тариф «Команда» в месяц?"
OUT_OF_BASE = "Есть ли интеграция с Битрикс24?"


def reachable(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.hostname:
        return False
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 443), timeout=1):
            return True
    except OSError:
        return False


pytestmark = [pytest.mark.llm,
              pytest.mark.skipif(not reachable(URL), reason=f"Qdrant недоступен: QDRANT_URL={URL or '(не задан)'}")]


@pytest.fixture(scope="module")
def cfg():
    from app.core.config import Settings

    suffix = uuid.uuid4().hex[:8]
    settings = Settings(qdrant_url=URL, rag={"collection": f"pytest_rag_{suffix}",
                                             "baremetal_collection": f"pytest_rag_bm_{suffix}"})
    if settings.llm.base_url and not reachable(settings.llm.base_url):
        pytest.skip(f"Модель ответа недоступна: LLM__BASE_URL={settings.llm.base_url}")
    return settings


@pytest.fixture(scope="module", params=["llamaindex", "baremetal"])
def service(request, cfg):
    from app.services.rag import RAGService, no_implicit_event_loop
    from app.services.rag_baremetal import BareMetalRAG

    no_implicit_event_loop()
    rag = RAGService(cfg) if request.param == "llamaindex" else BareMetalRAG(cfg)
    try:
        rag.build()
        yield rag
    finally:
        rag.client.delete_collection(rag.collection)
        rag.close()


def show(name: str, answer: dict) -> None:
    top = answer["sources"][0]
    print(f"\n{name}: {answer['answer']!r}\n  top_score={answer['top_score']} top-1={top['source']} ({top['score']})")


def test_good_question(service):
    from app.services.rag_common import is_no_answer

    answer = service.answer(GOOD)
    show(type(service).__name__, answer)
    assert len(answer["sources"]) >= 3
    assert answer["sources"][0]["source"] == "02_tariffs_and_payment.md"
    assert not is_no_answer(answer["answer"])


def test_out_of_base_question(service):
    from app.services.rag_common import is_no_answer

    good = service.answer(GOOD)
    answer = service.answer(OUT_OF_BASE)
    show(type(service).__name__, answer)
    assert answer["top_score"] < good["top_score"]
    assert is_no_answer(answer["answer"]), "модель ответила на вопрос, которого нет в базе"
