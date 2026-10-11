"""
Чат по базе знаний (блок 5.5): POST /chats/{id}/messages с RAG — контекст и правило цитировать в
запросе к модели, score-guard без вызова модели, condense follow-up по истории чата, событие
sources в SSE перед done, оценка вместе с показанными источниками.

Поиск подменён FakeRag: настоящий Retrieval из app/services/rag.py с двумя фрагментами, модель —
FakeLLM. Сам поиск по Qdrant проверяют tests/unit/test_rag.py и test_ingestion.py.
"""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import httpx
import pytest
from chat_fakes import FakeLLM, make_settings
from llama_index.core.schema import NodeWithScore, TextNode

from app.chat.deps import get_chat_rag, get_llm_client
from app.chat.knowledge import (
    CONDENSE_EXAMPLES,
    CONDENSE_SYSTEM,
    clean_query,
    condense_messages,
    with_knowledge,
)
from app.chat.routes import sanitize, sse_event
from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.main import app
from app.schemas.chat import ChatRequest, content_text
from app.services.rag import Retrieval
from app.services.rag_common import ANSWER_FORMAT, CHAT_RULES, NO_ANSWER, RAGNotIndexed

REFUND = "Вернуть оплату можно в течение 14 дней после списания. Деньги приходят на карту за 5–10 рабочих дней, " \
         "при оплате по счёту — на расчётный счёт плательщика."
REGULATION = "Сроки считаются с момента регистрации заявки и зависят от приоритета."


class FakeRag:
    """Поиск: два фрагмента с близостью score; ниже порога — пустой контекст (score-guard)."""

    def __init__(self, score: float = 0.71, threshold: float = 0.5, error: Exception | None = None) -> None:
        self.score, self.threshold, self.error = score, threshold, error
        self.queries: list[str] = []

    def ensure_ready(self) -> None:
        if self.error is not None:
            raise self.error

    def retrieve(self, question: str) -> Retrieval:
        self.queries.append(question)
        nodes = [NodeWithScore(node=TextNode(text=REFUND, metadata={
                     "file_name": "refunds_v2.docx", "source": "billing/refunds_v2.docx", "category": "billing"}),
                     score=self.score),
                 NodeWithScore(node=TextNode(text=REGULATION, metadata={
                     "file_name": "support-regulation_v2.3.pdf", "source": "support/support-regulation_v2.3.pdf",
                     "category": "support", "page": 2}), score=self.score - 0.1)]
        return Retrieval(question, nodes if self.score >= self.threshold else [], self.score, 10, self.threshold)


def knowledge_reply(req: ChatRequest) -> str:
    """Модель: переписывает follow-up (condense) или отвечает по контексту с цитатой [1]."""
    system = "\n".join(content_text(m.content) for m in req.messages if m.role == "system")
    if CONDENSE_SYSTEM in system:
        return "Самостоятельный вопрос: «Куда вернут деньги при оплате по счёту?»"
    if "[1] (refunds_v2.docx)" in system:
        return "Вернуть оплату можно в течение 14 дней [1]."
    return "Без базы знаний."


@pytest.fixture
def rag_chat(tmp_path):
    settings, llm, rag = make_settings(tmp_path), FakeLLM(knowledge_reply), FakeRag()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[get_chat_rag] = lambda: rag
    yield settings, llm, rag
    app.dependency_overrides.clear()


async def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def new_chat(http: httpx.AsyncClient) -> str:
    response = await http.post("/chats", json={"owner_external_id": f"rag-{uuid4().hex[:10]}", "interface": "cli"})
    return response.json()["chat_id"]


def events(body: str) -> list[tuple[str, dict[str, Any]]]:
    """(имя события, JSON) — с учётом строки event:."""
    result = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        name, data = "message", []
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data.append(line[len("data: "):])
        assert len(data) == 1, block                                   # событие — одна строка data:
        result.append((name, json.loads(data[0])))
    return result


async def ask(http: httpx.AsyncClient, chat_id: str, text: str) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    response = await http.post(f"/chats/{chat_id}/messages", data={"content": text})
    assert response.status_code == 200, response.text
    parsed = events(response.text)
    return "".join(e["delta"] for _, e in parsed if e["type"] == "token"), parsed


# ---------------------------------------------------------------- ответ по базе
async def test_answer_from_knowledge_base_streams_sources_before_done(rag_chat):
    _, llm, rag = rag_chat
    async with await client() as http:
        chat_id = await new_chat(http)
        answer, parsed = await ask(http, chat_id, "Каков срок возврата?")
    assert answer == "Вернуть оплату можно в течение 14 дней [1]."
    assert [name for name, _ in parsed[-2:]] == ["sources", "message"]           # sources — перед done
    sources = parsed[-2][1]
    assert sources["type"] == "sources" and sources["confident"] is True and sources["top_score"] == 0.71
    first, second = sources["sources"]
    assert {k: first[k] for k in ("id", "file_name", "page", "score", "cited")} == {
        "id": 1, "file_name": "refunds_v2.docx", "page": None, "score": 0.71, "cited": True}
    assert (second["id"], second["page"], second["cited"]) == (2, 2, False) and second["snippet"].startswith("Сроки")
    assert parsed[-1][1]["type"] == "done" and parsed[-1][1]["message_id"]
    assert rag.queries == ["Каков срок возврата?"]
    [req] = llm.requests                                                          # condense не было: вопрос первый
    rules, context = req.messages[0], req.messages[1]
    assert rules.role == "system" and CHAT_RULES in rules.content
    assert context.role == "system" and context.content[0].media == "document"    # не режется пределом вопроса
    assert "[1] (refunds_v2.docx)" in context.content[0].text
    assert "[2] (support-regulation_v2.3.pdf, стр. 2)" in context.content[0].text
    assert context.content[0].text.endswith(f"В ответе: {ANSWER_FORMAT}")       # формат цитат — в конце контекста
    assert req.temperature == 0.1


async def test_score_guard_answers_without_model(rag_chat):
    from log_capture import captured_logs
    from log_capture import events as log_events

    _, llm, rag = rag_chat
    rag.score = 0.2
    async with await client() as http:
        chat_id = await new_chat(http)
        with captured_logs("INFO") as records:
            answer, parsed = await ask(http, chat_id, "Как приготовить борщ?")
        history = (await http.get(f"/chats/{chat_id}/messages")).json()
    assert answer == NO_ANSWER and llm.requests == []                            # модель не вызывалась
    assert parsed[-2] == ("sources", {"type": "sources", "confident": False, "top_score": 0.2, "sources": []})
    guard = log_events(records, "rag_score_guard")
    assert guard and guard[0]["where"] == "chat" and guard[0]["threshold"] == 0.5
    messages = history["messages"] if isinstance(history, dict) else history
    assert [(m["role"], m["content"]) for m in messages] == [("user", "Как приготовить борщ?"),
                                                             ("assistant", NO_ANSWER)]


async def test_follow_up_is_condensed_for_search_and_history_has_no_old_citations(rag_chat):
    _, llm, rag = rag_chat
    async with await client() as http:
        chat_id = await new_chat(http)
        await ask(http, chat_id, "Каков срок возврата?")
        _, parsed = await ask(http, chat_id, "А при оплате по счёту?")
    assert rag.queries == ["Каков срок возврата?", "Куда вернут деньги при оплате по счёту?"]
    condense, answer = llm.requests[1], llm.requests[2]
    assert condense.messages[0].content == CONDENSE_SYSTEM and condense.temperature == 0.0
    assert [m.role for m in condense.messages] == ["system", *["user", "assistant"] * len(CONDENSE_EXAMPLES), "user"]
    dialog = condense.messages[-1].content                                       # диалог чата — после образцов
    assert "Пользователь: Каков срок возврата?" in dialog and "Ассистент: Вернуть оплату можно в течение 14 дней." in dialog
    assert "Последний вопрос: А при оплате по счёту?" in dialog
    history = [m for m in answer.messages if m.role == "assistant"]
    assert history and all("[1]" not in m.content for m in history)              # старые цитаты вычищены
    assert answer.messages[-1].content == "А при оплате по счёту?"                # модель видит вопрос как есть
    assert parsed[-2][1]["confident"] is True


async def test_condense_failure_falls_back_to_raw_question(rag_chat, tmp_path):
    _, _, rag = rag_chat

    class CondenseDown(FakeLLM):
        async def stream(self, req):
            if req.messages[0].content == CONDENSE_SYSTEM:
                raise LLMError("провайдер недоступен")
            async for delta in super().stream(req):
                yield delta

    llm = CondenseDown(knowledge_reply)
    app.dependency_overrides[get_llm_client] = lambda: llm
    async with await client() as http:
        chat_id = await new_chat(http)
        await ask(http, chat_id, "Каков срок возврата?")
        answer, _ = await ask(http, chat_id, "А при оплате по счёту?")
    assert rag.queries[-1] == "А при оплате по счёту?" and answer.startswith("Вернуть оплату")


async def test_knowledge_base_down_is_503_and_question_not_saved(rag_chat):
    _, llm, rag = rag_chat
    rag.error = RAGNotIndexed("Коллекция kb_docs пуста")
    async with await client() as http:
        chat_id = await new_chat(http)
        response = await http.post(f"/chats/{chat_id}/messages", data={"content": "Каков срок возврата?"})
        history = (await http.get(f"/chats/{chat_id}/messages")).json()
    assert response.status_code == 503 and response.json()["error"]["code"] == "rag_not_indexed"
    assert "индексируется" in response.json()["error"]["message"] and llm.requests == []
    assert (history["messages"] if isinstance(history, dict) else history) == []


async def test_model_refusal_has_no_sources(rag_chat):
    _, llm, _ = rag_chat
    llm.reply = lambda req: "По базе не нашёл, могу эскалировать."
    async with await client() as http:
        chat_id = await new_chat(http)
        _, parsed = await ask(http, chat_id, "Есть ли интеграция с Битрикс24?")
    assert parsed[-2][1] == {"type": "sources", "confident": False, "top_score": 0.71, "sources": []}


async def test_chat_without_rag_has_no_sources_event(tmp_path):
    settings, llm = make_settings(tmp_path), FakeLLM()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.state.rag = FakeRag()
    try:
        settings.rag.chat = False                                                # RAG__CHAT=false
        async with await client() as http:
            chat_id = await new_chat(http)
            _, parsed = await ask(http, chat_id, "Привет")
        assert [e["type"] for _, e in parsed if e["type"] != "token"] == ["done"]
    finally:
        app.state.rag = None
        app.dependency_overrides.clear()


# ---------------------------------------------------------------- оценка с источниками
async def test_feedback_stores_shown_sources(rag_chat, tmp_path):
    settings, _, _ = rag_chat
    async with await client() as http:
        chat_id = await new_chat(http)
        _, parsed = await ask(http, chat_id, "Каков срок возврата?")
        message_id = parsed[-1][1]["message_id"]
        shown = [{"id": 1, "file_name": "refunds_v2.docx", "source": "billing/refunds_v2.docx", "page": None,
                  "score": 0.71, "cited": True}]
        response = await http.post(f"/chats/{chat_id}/messages/{message_id}/feedback",
                                   json={"value": "down", "sources": shown})
        bad = await http.post(f"/chats/{chat_id}/messages/{message_id}/feedback",
                              json={"value": "up", "sources": [{"id": 0}]})
    assert response.status_code == 200 and response.json()["saved"] is True
    assert bad.status_code == 422
    rows = [json.loads(line) for line in (settings.chat_storage_dir / "feedback.jsonl").read_text("utf-8").splitlines()]
    assert rows[-1]["value"] == "down" and rows[-1]["sources"] == [
        {"id": 1, "file_name": "refunds_v2.docx", "source": "billing/refunds_v2.docx", "score": 0.71, "cited": True}]


async def test_repositories_store_feedback_sources(repo):
    from app.chat.domain import ChatMessage

    chat = await repo.create_chat(f"rag-{uuid4().hex[:8]}", "cli", None)
    answer = ChatMessage(chat_id=chat.id, role="assistant", content="Ответ [1]", tokens=3)
    await repo.append_message(chat.id, answer)
    result = await repo.add_feedback(chat.id, answer.id, chat.owner_external_id, "down",
                                     sources=[{"id": 1, "file_name": "refunds_v2.docx", "score": 0.71}])
    assert result.saved is True and result.value == "down"


# ---------------------------------------------------------------- помощники
def test_sse_sanitize_keeps_event_on_one_line():
    assert sanitize("шаги:\n1. раз\r\n2. два\r3.") == "шаги:\\n1. раз\\n2. два\\n3."
    event = sse_event({"type": "token", "delta": "a\nb c"})
    assert event.count("\n") == 2 and event.startswith("data: ") and event.endswith("\n\n")
    named = sse_event({"type": "sources", "sources": []}, event="sources")
    assert named.split("\n")[:2] == ["event: sources", 'data: {"type": "sources", "sources": []}']


def test_condense_helpers():
    from app.chat.domain import ChatMessage

    chat = uuid4()
    assert condense_messages([], "А для них?") is None                          # вопросов ещё не было
    prior = [ChatMessage(chat_id=chat, role="user", content="Каков срок возврата?", tokens=5),
             ChatMessage(chat_id=chat, role="assistant", content="14 дней [1].", tokens=5)]
    messages = condense_messages(prior, "А для них?")
    assert messages[-1]["content"].startswith("Диалог:\nПользователь: Каков срок возврата?\nАссистент: 14 дней.")
    assert messages[2] == {"role": "assistant", "content": "На сколько дней выдают учебник?"}   # первый образец
    assert "книг" not in messages[-1]["content"]                            # образцы — не в диалоге чата
    plain = condense_messages(prior, "А для них?", examples=False)          # без образцов — как до замера
    assert [m["role"] for m in plain] == ["system", "user"] and plain[1] == messages[-1]
    assert clean_query("Самостоятельный вопрос: «Куда вернут деньги?»\nпояснение") == "Куда вернут деньги?"
    assert clean_query("  \n ") is None


def test_with_knowledge_inserts_context_after_system_prompt():
    rag = FakeRag()
    messages = [{"role": "system", "content": "правила"}, {"role": "user", "content": "вопрос"},
                {"role": "assistant", "content": "ответ [1][2]."}, {"role": "user", "content": "ещё"}]
    result = with_knowledge(messages, rag.retrieve("вопрос"))
    assert [m["role"] for m in result] == ["system", "system", "user", "assistant", "user"]
    assert result[2]["content"] == "вопрос" and result[3]["content"] == "ответ."
