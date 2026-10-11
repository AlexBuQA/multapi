"""
Бот и RAG (блок 5.5): debounce правок по чату, событие sources, подпись с источниками и оценка
👍/👎 вместе с показанными источниками.
"""
from __future__ import annotations

import json
from uuid import UUID, uuid4

import httpx
import pytest
from aiogram.methods import AnswerCallbackQuery, EditMessageText
from bot_fakes import FakeBackend, callback_update, message_update, new_chat_id

from bot import texts
from bot.config import BotSettings
from bot.services import streaming
from bot.services.backend_client import BackendClient, make_http
from bot.services.sources import (
    SHOWN_SOURCES,
    ShownSources,
    legend,
    shown,
    without_sources_line,
)
from bot.services.streaming import EditDebounce, StreamRenderer, answer_with_stream

SOURCES = [
    {"id": 1, "file_name": "refunds_v2.docx", "source": "billing/refunds_v2.docx", "category": "billing",
     "page": None, "score": 0.71, "snippet": "Вернуть оплату можно в течение 14 дней", "cited": True},
    {"id": 2, "file_name": "support-regulation_v2.3.pdf", "source": "support/support-regulation_v2.3.pdf",
     "category": "support", "page": 2, "score": 0.62, "snippet": "Сроки ответа", "cited": False},
    {"id": 3, "file_name": "faq-billing.md", "source": "billing/faq-billing.md", "category": "billing",
     "page": None, "score": 0.55, "snippet": "Частые вопросы", "cited": True},
]
ANSWER = "Вернуть оплату можно в течение 14 дней [1], заявка — в разделе «Помощь» [3]."


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def source(bot, chat: int):
    return message_update("вопрос", chat).message.as_(bot)


@pytest.fixture(autouse=True)
def clean_sources():
    SHOWN_SOURCES.clear()
    yield
    SHOWN_SOURCES.clear()


# ---------------------------------------------------------------- debounce по чату
def test_edit_debounce_is_per_chat():
    clock = Clock()
    debounce = EditDebounce(0.7, clock)
    assert debounce.ready(1) and debounce.wait_time(1) == 0
    debounce.mark(1)
    assert not debounce.ready(1) and debounce.ready(2)               # другой чат — свой счётчик
    clock.now += 0.5
    assert debounce.wait_time(1) == pytest.approx(0.2)
    clock.now += 0.2
    assert debounce.ready(1)


async def test_edits_respect_700_ms_and_final_waits_its_turn(bot, session, monkeypatch):
    chat, clock = new_chat_id(), Clock()
    debounce = EditDebounce(0.7, clock)
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(streaming.asyncio, "sleep", fake_sleep)
    renderer = StreamRenderer(source(bot, chat), debounce=debounce, clock=clock)
    for at, chunk in ((0.0, "Раз "), (0.3, "два "), (0.69, "три "), (0.71, "четыре "), (1.0, "пять")):
        clock.now = 100.0 + at
        await renderer.feed(chunk)
    await renderer.finish()
    edits = [m.text for m in session.of(EditMessageText)]
    assert session.sent_texts(chat) == ["Раз "]
    assert edits == ["Раз два три четыре ", "Раз два три четыре пять"]   # правка не раньше 0,7 с
    assert slept == [pytest.approx(0.41)]                                # финальная — дождалась очереди
    assert debounce.last_edit_at[chat] == clock.now


async def test_two_answers_in_one_chat_share_debounce(bot, session):
    chat, clock = new_chat_id(), Clock()
    debounce = EditDebounce(0.7, clock)
    first = StreamRenderer(source(bot, chat), debounce=debounce, clock=clock)
    second = StreamRenderer(source(bot, chat), debounce=debounce, clock=clock)
    await first.feed("Первый ответ ")
    clock.now += 0.8
    await first.feed("растёт")                                          # правка: прошло 0,8 с
    await second.feed("Второй ")
    await second.feed("тоже")                                           # сразу после правки первого — нет
    assert [m.text for m in session.of(EditMessageText)] == ["Первый ответ растёт"]


def test_edit_mode_is_default_and_interval_not_below_700_ms():
    settings = BotSettings(bot_token="42:TEST-TOKEN", _env_file=None)
    assert (settings.bot_streaming, settings.bot_edit_interval) == ("edit", 0.7)
    with pytest.raises(ValueError):
        BotSettings(bot_token="42:TEST-TOKEN", bot_edit_interval=0.3, _env_file=None)
    assert streaming.EDIT_INTERVAL == 0.7


# ---------------------------------------------------------------- источники под ответом
def test_shown_prefers_cited_and_skips_refusals():
    assert [s["id"] for s in shown(SOURCES, True)] == [1, 3]
    uncited = [{**s, "cited": False} for s in SOURCES] + [{**SOURCES[0], "id": 4, "cited": False}]
    assert [s["id"] for s in shown(uncited, True)] == [1, 2, 3]          # не сослалась — первые три
    assert shown(SOURCES, False) == [] and shown([], True) == []
    assert legend(shown(SOURCES, True)) == (f"\n\n{texts.SOURCES_TITLE}\n[1] refunds_v2.docx\n[3] faq-billing.md")
    assert legend([SOURCES[1]]).endswith("[2] support-regulation_v2.3.pdf, стр. 2")
    assert legend([]) == ""


def test_shown_sources_memory_is_bounded():
    memory = ShownSources(limit=2)
    ids = [uuid4() for _ in range(3)]
    for message_id in ids:
        memory.remember(message_id, SOURCES[:1])
    assert memory.get(ids[0]) is None and memory.get(ids[2]) == SOURCES[:1]
    memory.remember(uuid4(), [])                                         # пустое не запоминается
    assert len(memory._items) == 2


async def test_answer_gets_legend_and_sources_are_remembered(bot, session):
    chat = new_chat_id()
    backend = FakeBackend(ANSWER, sources=SOURCES, confident=True)
    await answer_with_stream(source(bot, chat), backend, uuid4(), "Каков срок возврата?")
    [text] = session.on_screen(chat)
    assert text == ANSWER + f"\n\n{texts.SOURCES_TITLE}\n[1] refunds_v2.docx\n[3] faq-billing.md"
    assert SHOWN_SOURCES.get(backend.answers[-1].message_id) == [SOURCES[0], SOURCES[2]]


@pytest.mark.parametrize(("answer", "expected"), [
    ("Срок — 14 дней после списания. Источники: [2][3].", "Срок — 14 дней после списания."),   # ответы llama3.2
    ("Срок — 14 дней.\nИсточники: [2], [3]", "Срок — 14 дней."),
    ("Срок — 14 дней. Источники: [1], [2][3]", "Срок — 14 дней."),
    ("Срок — 14 дней [1].", "Срок — 14 дней [1]."),                                          # строки нет
    ("Источники платежей: карта и СБП [2].", "Источники платежей: карта и СБП [2]."),         # не та строка
])
def test_without_sources_line(answer, expected):
    assert without_sources_line(answer) == expected


@pytest.mark.parametrize("mode", ["edit", "draft"])
async def test_model_sources_line_is_replaced_by_legend(bot, session, mode):
    """Модель заканчивает ответ строкой «Источники: [1], [3]» — в Telegram вместо неё подпись с теми же
    номерами и именами файлов, а не два списка источников подряд."""
    chat = new_chat_id()
    backend = FakeBackend("Вернуть оплату можно в течение 14 дней.\nИсточники: [1], [3]", sources=SOURCES,
                          confident=True)
    await answer_with_stream(source(bot, chat), backend, uuid4(), "Каков срок возврата?", mode=mode)
    assert session.on_screen(chat) == [
        f"Вернуть оплату можно в течение 14 дней.\n\n{texts.SOURCES_TITLE}\n[1] refunds_v2.docx\n[3] faq-billing.md"]


@pytest.mark.parametrize(("answer", "shown_text"), [
    ("По базе не нашёл, могу эскалировать.", "По базе не нашёл, могу эскалировать."),
    # так отказывала llama3.2 в замере на Windows: строка «Источники» без подписи ни на что не указывает
    ("По базе не нашёл. Могу эскалировать. Источники: [5]", "По базе не нашёл. Могу эскалировать."),
])
async def test_refusal_has_no_legend(bot, session, answer, shown_text):
    chat = new_chat_id()
    backend = FakeBackend(answer, sources=[], confident=False)
    await answer_with_stream(source(bot, chat), backend, uuid4(), "Как приготовить борщ?")
    assert session.on_screen(chat) == [shown_text]
    assert SHOWN_SOURCES.get(backend.answers[-1].message_id) is None


async def test_feedback_sends_shown_sources(dp, bot, session):
    backend = dp["backend"] = FakeBackend(ANSWER, sources=SOURCES, confident=True)
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("Каков срок возврата?", chat))
    [(key, markup)] = [(k, m) for k, m in session.markups.items() if k[0] == chat]
    down = markup.inline_keyboard[0][1].callback_data
    await dp.feed_update(bot, callback_update(down, chat, message_id=key[1], text=ANSWER))
    chat_id = backend.chats[(str(chat), "telegram")]
    answer_id = backend.answers[-1].message_id
    assert backend.feedback == {(chat_id, answer_id): "down"}
    assert backend.feedback_sources[(chat_id, answer_id)] == [SOURCES[0], SOURCES[2]]
    assert session.of(AnswerCallbackQuery)[-1].text == texts.FEEDBACK_THANKS


async def test_feedback_without_remembered_sources_still_works(dp, bot, session):
    backend = dp["backend"] = FakeBackend()                              # ответ без базы знаний
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("Привет", chat))
    [(key, markup)] = [(k, m) for k, m in session.markups.items() if k[0] == chat]
    await dp.feed_update(bot, callback_update(markup.inline_keyboard[0][0].callback_data, chat, message_id=key[1]))
    chat_id = backend.chats[(str(chat), "telegram")]
    assert backend.feedback_sources[(chat_id, backend.answers[-1].message_id)] is None


# ---------------------------------------------------------------- BackendClient
CHAT = UUID("9f76d71a-fe47-4a61-8c2f-757a5a300737")
ANSWER_ID = "0b6f4f0e-5d77-4c5e-9d8e-3b1f6f2a9c10"


def client(handler) -> BackendClient:
    http = make_http("http://backend.test/", timeout=15, transport=httpx.MockTransport(handler))
    return BackendClient(http, retry_delays=(0, 0))


async def test_client_reads_sources_event():
    payload = {"type": "sources", "confident": True, "top_score": 0.71, "sources": SOURCES}
    body = (f'data: {json.dumps({"type": "token", "delta": "Ответ [1]"}, ensure_ascii=False)}\n\n'
            f"event: sources\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            f'data: {{"type": "done", "message_id": "{ANSWER_ID}"}}\n\n')
    stream = client(lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"},
                                                   content=body.encode())).send_message(CHAT, "вопрос")
    assert [chunk async for chunk in stream] == ["Ответ [1]"]
    assert stream.sources == SOURCES and stream.confident is True and str(stream.message_id) == ANSWER_ID


async def test_client_sends_feedback_with_sources():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"message_id": ANSWER_ID, "value": "down", "saved": True})

    await client(handler).send_feedback(CHAT, UUID(ANSWER_ID), "down", sources=[SOURCES[1]])
    assert json.loads(seen[0].content) == {"value": "down", "sources": [
        {"id": 2, "file_name": "support-regulation_v2.3.pdf", "source": "support/support-regulation_v2.3.pdf",
         "page": 2, "score": 0.62, "cited": False}]}                       # без snippet и category


def test_rag_errors_are_shown_with_service_text():
    request = httpx.Request("POST", "http://backend.test/chats/x/messages")
    response = httpx.Response(503, json={"error": {"code": "rag_not_indexed",
                                                   "message": "База знаний ещё индексируется — повторите вопрос "
                                                              "через минуту."}}, request=request)
    error = httpx.HTTPStatusError("503", request=request, response=response)
    assert texts.user_message(error) == "База знаний ещё индексируется — повторите вопрос через минуту."
