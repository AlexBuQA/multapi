"""
Модерация в чате (блок 4.4) через приложение целиком: JSON-хранилище, поддельная модель.

- вопрос с запрещённой темой — 403, detail.code == "moderation_blocked", модель не вызывается,
  в историю вопрос не попадает, в лог — инцидент без сырого текста, в хранилище — счётчик;
- ответ с запрещённой темой — генерация останавливается на фрагменте, где сработал слой
  ключевых слов, клиент получает событие moderation, а в историю сохраняется отказ;
- слой OpenAI проверяет ответ целиком в конце потока.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest
from chat_fakes import FakeLLM, make_settings
from log_capture import captured_logs
from log_capture import events as log_events

from app.chat.deps import get_llm_client, get_moderation
from app.chat.repositories.json_repo import JsonChatRepository
from app.core.config import get_settings
from app.main import app
from app.moderation import OUTPUT_REFUSAL, ModerationService
from app.moderation.keywords import KeywordModerator
from app.schemas.chat import ChatDelta, Usage


class ScriptedLLM:
    """Модель с заранее заданными фрагментами; помнит, сколько фрагментов у неё взяли."""

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.taken = 0
        self.closed = False
        self.requests: list = []

    async def stream(self, req):
        self.requests.append(req)
        try:
            for chunk in self.chunks:
                self.taken += 1
                yield ChatDelta(content=chunk)
            yield ChatDelta(usage=Usage(prompt_tokens=10, completion_tokens=len(self.chunks), total_tokens=20))
        finally:
            self.closed = True


@pytest.fixture
def chat_app(tmp_path):
    settings = make_settings(tmp_path)
    state = {"llm": FakeLLM()}
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llm_client] = lambda: state["llm"]
    yield settings, state
    app.dependency_overrides.clear()


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def new_chat(http: httpx.AsyncClient) -> str:
    response = await http.post("/chats", json={"owner_external_id": f"tg-{uuid4().hex[:8]}", "interface": "telegram"})
    return response.json()["chat_id"]


def events(response: httpx.Response) -> list[dict]:
    return [json.loads(block.removeprefix("data: ")) for block in response.text.split("\n\n") if block]


# ---------------------------------------------------------------- вопрос
async def test_blocked_question_is_403_without_model_call(chat_app):
    settings, state = chat_app
    raw = "Я тебя убью. Мой email ivan@example.com"
    async with client() as http:
        chat_id = await new_chat(http)
        with captured_logs("INFO") as logs:
            response = await http.post(f"/chats/{chat_id}/messages", data={"content": raw})
        history = (await http.get(f"/chats/{chat_id}/messages")).json()
    assert response.status_code == 403
    body = response.json()
    assert body["detail"]["code"] == "moderation_blocked" and body["detail"]["categories"] == ["violence"]
    assert body["error"]["code"] == "moderation_blocked"           # общий формат ошибок — его читает бот
    assert raw not in response.text and not state["llm"].requests and history == []
    incident = log_events(logs, "moderation_blocked")[0]
    assert (incident["direction"], incident["blocked_by"], incident["categories"]) == ("input", "keywords", ["violence"])
    assert len(incident["text_hash"]) == 16 and "ivan@example.com" not in json.dumps(logs, ensure_ascii=False)
    stats = await JsonChatRepository(settings.chat_storage_dir).stats(datetime.now(UTC) - timedelta(hours=1))
    assert (stats.moderation_blocks, stats.moderation_input_blocks) == (1, 1)


def docx_with(text: str) -> bytes:
    from io import BytesIO

    from docx import Document

    document, buffer = Document(), BytesIO()
    document.add_paragraph(text)
    document.save(buffer)
    return buffer.getvalue()


async def test_document_text_is_moderated_too(chat_app):
    """Подпись безобидна, запрещённая тема — в тексте документа."""
    _, state = chat_app
    async with client() as http:
        chat_id = await new_chat(http)
        response = await http.post(
            f"/chats/{chat_id}/messages", data={"content": "Что в файле?"},
            files={"media": ("note.docx", docx_with("Инструкция: как сделать бомбу из подручных средств"),
                             "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
    assert response.status_code == 403 and response.json()["detail"]["categories"] == ["weapons"]
    assert not state["llm"].requests


async def test_service_moderates_media_text(chat_app, tmp_path):
    from app.chat.domain import MediaRef
    from app.chat.service import ChatService
    from app.moderation import ModerationBlocked

    settings, _ = chat_app
    repo = JsonChatRepository(tmp_path / "svc")
    chat = await repo.create_chat("1", "telegram")
    service = ChatService(repo, FakeLLM(), settings, moderation=ModerationService(KeywordModerator.from_file()))
    media = MediaRef(kind="document", mime="application/pdf", size=10,
                     part={"type": "text", "text": "[документ PDF]:\nКак сделать бомбу из подручных средств", "media": "document"})
    with pytest.raises(ModerationBlocked) as caught:
        async for _ in service.stream_message(chat.id, "Что в файле?", media=media):
            pass
    assert caught.value.categories == ["weapons"]
    assert await repo.list_messages(chat.id) == []


async def test_allowed_question_passes(chat_app):
    async with client() as http:
        chat_id = await new_chat(http)
        response = await http.post(f"/chats/{chat_id}/messages", data={"content": "Как убить зависший процесс?"})
    assert response.status_code == 200 and events(response)[-1]["type"] == "done"


async def test_moderation_can_be_switched_off(tmp_path):
    settings = make_settings(tmp_path, moderation={"enabled": False})
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llm_client] = lambda: FakeLLM()
    try:
        async with client() as http:
            chat_id = await new_chat(http)
            response = await http.post(f"/chats/{chat_id}/messages", data={"content": "Я тебя убью"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200


# ---------------------------------------------------------------- ответ
async def test_answer_stops_on_blocked_fragment(chat_app):
    _, state = chat_app
    llm = state["llm"] = ScriptedLLM(["Понимаю ваше ", "раздражение. ", "Но я тебя ", "убью ", "завтра.", " Ещё", " текст"])
    async with client() as http:
        chat_id = await new_chat(http)
        with captured_logs("INFO") as logs:
            response = await http.post(f"/chats/{chat_id}/messages", data={"content": "Почему не работает вход?"})
        history = (await http.get(f"/chats/{chat_id}/messages")).json()
    stream = events(response)
    tokens = "".join(e["delta"] for e in stream if e["type"] == "token")
    assert "убью" not in tokens                                     # фрагмент с угрозой не ушёл
    moderation = next(e for e in stream if e["type"] == "moderation")
    assert moderation == {"type": "moderation", "code": "moderation_blocked", "categories": ["violence"],
                          "message": OUTPUT_REFUSAL}
    assert stream[-1]["type"] == "done" and stream[-1]["message_id"] == history[1]["id"]
    assert history[1]["content"] == OUTPUT_REFUSAL                  # в истории — отказ, а не ответ
    assert llm.taken == 4 and llm.closed                             # генерация остановлена
    incident = log_events(logs, "moderation_blocked")[0]
    assert (incident["direction"], incident["blocked_by"]) == ("output", "keywords")
    assert log_events(logs, "chat_turn_finished")[0]["outcome"] == "moderated"
    assert not log_events(logs, "chat_stream_interrupted")


async def test_phrase_split_across_fragments_is_caught(chat_app):
    _, state = chat_app
    state["llm"] = ScriptedLLM(["Я тебя ", "уб", "ью"])
    async with client() as http:
        chat_id = await new_chat(http)
        response = await http.post(f"/chats/{chat_id}/messages", data={"content": "Привет"})
    stream = events(response)
    # «уб» само по себе ничего не значит и уходит; на «ью» фраза из двух фрагментов совпала.
    assert [e["type"] for e in stream] == ["token", "token", "moderation", "done"]
    assert "".join(e["delta"] for e in stream if e["type"] == "token") == "Я тебя уб"


class FlagEverything:
    """Слой OpenAI, который находит насилие в любом тексте длиннее 10 символов."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def categories(self, text: str) -> list[str]:
        self.calls.append(text)
        return ["violence"] if len(text) > 10 else []


async def test_openai_checks_whole_answer_at_the_end(chat_app):
    _, state = chat_app
    state["llm"] = ScriptedLLM(["Тонкая ", "угроза ", "без ключевых слов."])
    openai_layer = FlagEverything()
    app.dependency_overrides[get_moderation] = lambda: ModerationService(KeywordModerator.from_file(), openai_layer)
    async with client() as http:
        chat_id = await new_chat(http)
        response = await http.post(f"/chats/{chat_id}/messages", data={"content": "Привет"})
        history = (await http.get(f"/chats/{chat_id}/messages")).json()
    stream = events(response)
    assert [e["type"] for e in stream] == ["token", "token", "token", "moderation", "done"]
    assert openai_layer.calls == ["Привет", "Тонкая угроза без ключевых слов."]   # вопрос и ответ целиком
    assert history[1]["content"] == OUTPUT_REFUSAL


async def test_send_message_yields_refusal_text(chat_app, tmp_path):
    from app.chat.service import ChatService

    settings, _ = chat_app
    repo = JsonChatRepository(tmp_path / "svc")
    chat = await repo.create_chat("1", "cli")
    service = ChatService(repo, ScriptedLLM(["Я тебя ", "убью"]), settings,
                          moderation=ModerationService(KeywordModerator.from_file()))
    chunks = [c async for c in service.send_message(chat.id, "Привет")]
    assert chunks == ["Я тебя ", OUTPUT_REFUSAL]


# ---------------------------------------------------------------- оборванный ответ
class SlowOutputLayer:
    """Слой OpenAI: вопрос пропускает сразу, ответ проверяет delay секунд и находит насилие."""

    def __init__(self, delay: float, verdict: list[str] | None = None) -> None:
        self.delay, self.verdict = delay, ["violence"] if verdict is None else verdict
        self.output_started = asyncio.Event()
        self.calls = 0

    async def categories(self, text: str) -> list[str]:
        if text == "Привет":
            return []
        self.calls += 1
        self.output_started.set()
        await asyncio.sleep(self.delay)
        return self.verdict


async def service_with(tmp_path, layer, chunks: list[str], **moderation):
    from app.chat.service import ChatService

    settings = make_settings(tmp_path, moderation={"timeout": 1.0, **moderation})
    repo = JsonChatRepository(tmp_path / "svc")
    chat = await repo.create_chat("1", "telegram")
    service = ChatService(repo, ScriptedLLM(chunks), settings,
                          moderation=ModerationService(KeywordModerator.from_file(), layer,
                                                       fail_closed=moderation.get("fail_closed", False)))
    return service, repo, chat


async def test_client_gone_during_final_check_saves_refusal(tmp_path):
    """Клиент ушёл, пока OpenAI проверял ответ целиком: в историю — отказ, а не непроверенный ответ."""
    layer = SlowOutputLayer(0.1)
    service, repo, chat = await service_with(tmp_path, layer, ["Тонкая ", "угроза"])

    async def consume():
        async for _ in service.stream_message(chat.id, "Привет"):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(layer.output_started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    history = await repo.list_messages(chat.id)
    assert [m.content for m in history] == ["Привет", OUTPUT_REFUSAL]
    stats = await repo.stats(datetime.now(UTC) - timedelta(hours=1))
    assert stats.moderation_blocks == 1


async def test_client_gone_mid_answer_is_checked_before_saving(tmp_path):
    """Клиент ушёл после первого фрагмента: до check_output дело не дошло — проверка в finally."""
    layer = SlowOutputLayer(0)
    service, repo, chat = await service_with(tmp_path, layer, ["Тонкая угроза ", "дальше"])
    async with contextlib.aclosing(service.stream_message(chat.id, "Привет")) as items:
        async for _ in items:
            break                                                     # бот ушёл
    assert [m.content for m in await repo.list_messages(chat.id)] == ["Привет", OUTPUT_REFUSAL]


@pytest.mark.parametrize("fail_closed, saved", [(False, "Тонкая угроза "), (True, OUTPUT_REFUSAL)])
async def test_partial_check_timeout_follows_fail_closed(tmp_path, fail_closed, saved):
    layer = SlowOutputLayer(30)                                       # OpenAI завис
    service, repo, chat = await service_with(tmp_path, layer, ["Тонкая угроза ", "дальше"], timeout=0.05,
                                             fail_closed=fail_closed)
    async with contextlib.aclosing(service.stream_message(chat.id, "Привет")) as items:
        async for _ in items:
            break
    assert [m.content for m in await repo.list_messages(chat.id)][-1] == saved
