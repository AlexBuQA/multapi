"""Показ ответа потоком (блок 4.2): частота правок, длинные ответы, ответы Telegram 400/429."""
from __future__ import annotations

from uuid import uuid4

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.methods import EditMessageText, SendMessage

from bot import texts
from bot.services import streaming
from bot.services.streaming import StreamRenderer, answer_with_stream, split_point
from bot_fakes import FakeBackend, message_update, new_chat_id


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def source(bot, chat: int):
    return message_update("вопрос", chat).message.as_(bot)


async def test_edits_are_throttled_but_final_text_is_complete(bot, session):
    chat, clock = new_chat_id(), Clock()
    renderer = StreamRenderer(source(bot, chat), interval=1.0, clock=clock)
    for at, chunk in ((0.0, "Раз "), (0.3, "два "), (0.6, "три "), (1.2, "четыре "), (1.5, "пять")):
        clock.now = 100.0 + at
        await renderer.feed(chunk)
    await renderer.finish()
    edits = [m.text for m in session.of(EditMessageText)]
    assert session.sent_texts(chat) == ["Раз "]                       # первое сообщение — сразу
    assert edits == ["Раз два три четыре ", "Раз два три четыре пять"]  # правка через 1 с + финальная
    assert session.on_screen(chat) == ["Раз два три четыре пять"]


async def test_long_answer_continues_in_new_message(bot, session):
    chat = new_chat_id()
    words = [f"слово{i}" for i in range(40)]
    renderer = StreamRenderer(source(bot, chat), interval=0, limit=60)
    for word in words:
        await renderer.feed(word + " ")
    await renderer.finish()
    shown = session.on_screen(chat)
    assert len(shown) >= 4 and all(len(text) <= 60 for text in shown)
    assert "".join(shown) == " ".join(words) + " "                  # ничего не потеряно и не повторено
    assert all(text.endswith(" ") for text in shown[:-1])             # разрыв по пробелу, не посреди слова


def test_split_point():
    assert split_point("а" * 30 + "\n" + "б" * 30, 40) == 31
    assert split_point("а" * 30 + " " + "б" * 30, 40) == 31
    assert split_point("а" * 100, 40) == 40                           # без пробелов — ровно по пределу


async def test_not_modified_is_ignored(bot, session):
    """Telegram обрезает пробелы в конце: «текст » для него то же, что «текст» — ответ 400."""
    chat = new_chat_id()
    renderer = StreamRenderer(source(bot, chat), interval=0)
    await renderer.feed("текст")
    session.fail[EditMessageText] = [TelegramBadRequest(EditMessageText(text="x"),
                                                        "Bad Request: message is not modified")]
    await renderer.feed(" ")
    await renderer.feed("ещё")
    await renderer.finish()
    assert session.on_screen(chat) == ["текст ещё"]


async def test_other_bad_request_propagates(bot, session):
    renderer = StreamRenderer(source(bot, new_chat_id()), interval=0)
    await renderer.feed("текст")
    session.fail[EditMessageText] = [TelegramBadRequest(EditMessageText(text="x"), "Bad Request: chat not found")]
    with pytest.raises(TelegramBadRequest):
        await renderer.feed(" ещё")


async def test_flood_control_skips_intermediate_and_retries_final(bot, session, monkeypatch):
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(streaming.asyncio, "sleep", fake_sleep)
    chat = new_chat_id()
    renderer = StreamRenderer(source(bot, chat), interval=0)
    await renderer.feed("Раз")
    flood = TelegramRetryAfter(EditMessageText(text="x"), "Too Many Requests", retry_after=3)
    session.fail[EditMessageText] = [flood]
    await renderer.feed(" два")                         # промежуточная правка пропущена
    session.fail[EditMessageText] = [flood]
    await renderer.finish()                              # финальная — после паузы и повтора
    assert slept == [3]
    assert session.on_screen(chat) == ["Раз два"]


async def test_empty_answer(bot, session):
    chat = new_chat_id()
    await answer_with_stream(source(bot, chat), FakeBackend(answer=""), uuid4(), "вопрос")
    assert session.on_screen(chat) == [texts.EMPTY_ANSWER]


async def test_whitespace_only_chunks_wait_for_text(bot, session):
    chat = new_chat_id()
    renderer = StreamRenderer(source(bot, chat), interval=0)
    await renderer.feed("\n")                             # Telegram не принимает пустое сообщение
    assert session.of(SendMessage) == []
    await renderer.feed("Ответ")
    await renderer.finish()
    assert session.on_screen(chat) == ["\nОтвет"]


async def test_flood_wait_blocks_intermediate_edits_until_it_passes(bot, session):
    chat, clock = new_chat_id(), Clock()
    renderer = StreamRenderer(source(bot, chat), interval=0, clock=clock)
    await renderer.feed("Раз")
    session.fail[EditMessageText] = [TelegramRetryAfter(EditMessageText(text="x"), "Too Many Requests", retry_after=10)]
    await renderer.feed(" два")                                   # 429: пауза 10 с
    edits_before = len(session.of(EditMessageText))
    for step in range(1, 10):                                     # 9 фрагментов внутри паузы — без правок
        clock.now = 100.0 + step
        await renderer.feed(f" {step}")
    assert len(session.of(EditMessageText)) == edits_before
    clock.now = 111.0                                             # пауза прошла
    await renderer.feed(" конец")
    assert len(session.of(EditMessageText)) == edits_before + 1
    assert session.on_screen(chat)[0].endswith("конец")


async def test_flood_wait_on_new_message_waits_and_retries(bot, session, monkeypatch):
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(streaming.asyncio, "sleep", fake_sleep)
    chat = new_chat_id()
    session.fail[SendMessage] = [TelegramRetryAfter(SendMessage(chat_id=chat, text="x"), "Too Many Requests",
                                                    retry_after=2)]
    await answer_with_stream(source(bot, chat), FakeBackend(answer="Ответ"), uuid4(), "вопрос")
    assert slept == [2] and session.on_screen(chat) == ["Ответ"]


async def test_blank_answer_is_reported(bot, session):
    chat = new_chat_id()
    await answer_with_stream(source(bot, chat), FakeBackend(answer="  \n "), uuid4(), "вопрос")
    assert session.on_screen(chat) == [texts.EMPTY_ANSWER]
