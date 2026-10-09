"""
Показ ответа потоком: правки сообщения (блок 4.2) и черновик sendMessageDraft (блок 4.3) —
частота обновлений, длинные ответы, ответы Telegram 400/429.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.methods import EditMessageText, SendMessage, SendMessageDraft

from bot import texts
from bot.services import streaming
from bot.services.streaming import DraftRenderer, StreamRenderer, answer_with_stream, make_renderer, split_point
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
    await answer_with_stream(source(bot, chat), FakeBackend(answer="Ответ"), uuid4(), "вопрос", mode="edit")
    assert slept == [2] and session.on_screen(chat) == ["Ответ"]


async def test_blank_answer_is_reported(bot, session):
    chat = new_chat_id()
    await answer_with_stream(source(bot, chat), FakeBackend(answer="  \n "), uuid4(), "вопрос")
    assert session.on_screen(chat) == [texts.EMPTY_ANSWER]


# ---------------------------------------------------------------- черновик (блок 4.3)
def drafts(session, chat: int) -> list[tuple[int, str]]:
    return session.drafts.get(chat, [])


async def test_draft_starts_with_thinking_and_grows_with_one_id(bot, session):
    chat, clock = new_chat_id(), Clock()
    renderer = DraftRenderer(source(bot, chat), interval=1.0, keepalive=0, clock=clock)
    await renderer.start()                                       # пустой текст — «Thinking…»
    await renderer.feed("Раз")                                   # первый текст — сразу
    await renderer.feed(" два")                                  # та же секунда — без обновления
    clock.now += 1.5
    await renderer.feed(" три")
    await renderer.feed(" четыре")                               # снова слишком часто
    await renderer.finish()
    assert [text for _, text in drafts(session, chat)] == ["", "Раз", "Раз два три"]
    assert len({draft_id for draft_id, _ in drafts(session, chat)}) == 1          # один draft_id — анимация
    assert all(draft_id > 0 for draft_id, _ in drafts(session, chat))
    assert session.on_screen(chat) == ["Раз два три четыре"]     # черновик эфемерный — ответ сохраняет sendMessage
    assert not session.of(EditMessageText)


async def test_long_draft_continues_with_new_message_and_new_id(bot, session):
    chat = new_chat_id()
    renderer = DraftRenderer(source(bot, chat), interval=0, limit=25, keepalive=0)
    await renderer.start()
    await renderer.feed("первая строка ответа\nвторая строка ответа")
    await renderer.finish()
    assert session.on_screen(chat) == ["первая строка ответа\n", "вторая строка ответа"]
    ids = [draft_id for draft_id, text in drafts(session, chat) if text]
    assert len(set(ids)) == 1 and ids[0] != drafts(session, chat)[0][0]       # остаток — новый черновик


def bad_request(chat: int) -> TelegramBadRequest:
    return TelegramBadRequest(SendMessageDraft(chat_id=chat, draft_id=1), "Bad Request: method is not available")


async def test_draft_rejected_falls_back_to_edits(bot, session):
    """Telegram отклонил черновик с текстом (старый Bot API) — ответ показывают правки, как в 4.2."""
    chat = new_chat_id()
    renderer = DraftRenderer(source(bot, chat), interval=0, keepalive=0)
    await renderer.start()
    session.fail[SendMessageDraft] = [bad_request(chat)]
    await renderer.feed("Ответ ")
    await renderer.feed("целиком")
    await renderer.finish()
    assert renderer.fallback is not None and session.on_screen(chat) == ["Ответ целиком"]
    assert len(session.of(SendMessageDraft)) == 2                # «Thinking…» и отклонённый; дальше не пробуем


async def test_rejected_placeholder_does_not_disable_drafts(bot, session):
    """Пустой черновик aiogram шлёт без text; не принят — живём без «Thinking…», черновики с текстом идут."""
    chat = new_chat_id()
    session.fail[SendMessageDraft] = [bad_request(chat)]
    renderer = DraftRenderer(source(bot, chat), interval=0, keepalive=0)
    await renderer.start()
    await renderer.feed("Ответ")
    await renderer.finish()
    assert renderer.fallback is None and renderer.placeholder is False
    assert [text for _, text in drafts(session, chat)] == ["Ответ"] and session.on_screen(chat) == ["Ответ"]


async def test_draft_flood_wait_skips_updates_but_keeps_answer(bot, session):
    chat, clock = new_chat_id(), Clock()
    renderer = DraftRenderer(source(bot, chat), interval=0, keepalive=0, clock=clock)
    await renderer.start()
    session.fail[SendMessageDraft] = [TelegramRetryAfter(SendMessageDraft(chat_id=chat, draft_id=1),
                                                         "Too Many Requests", retry_after=5)]
    await renderer.feed("Раз")                                   # 429: пауза 5 с
    clock.now += 1
    await renderer.feed(" два")                                  # внутри паузы — без черновика
    assert [text for _, text in drafts(session, chat)] == [""]
    clock.now += 5
    await renderer.feed(" три")
    await renderer.finish()
    assert [text for _, text in drafts(session, chat)] == ["", "Раз два три"]
    assert session.on_screen(chat) == ["Раз два три"]


async def test_draft_keepalive_while_model_thinks(bot, session):
    """Черновик живёт ~30 с: пока нет текста, «Thinking…» обновляется."""
    chat = new_chat_id()
    renderer = DraftRenderer(source(bot, chat), interval=0, keepalive=0.04)
    await renderer.start()
    await asyncio.sleep(0.15)
    await renderer.close()
    assert len(drafts(session, chat)) >= 2 and {text for _, text in drafts(session, chat)} == {""}
    count = len(drafts(session, chat))
    await asyncio.sleep(0.1)
    assert len(drafts(session, chat)) == count                   # после close — тишина


async def test_answer_with_stream_draft_end_to_end(bot, session):
    chat = new_chat_id()
    backend = FakeBackend("Ссылка для сброса действует 30 минут.", chunk=5, delay=0.01)
    renderer = await answer_with_stream(source(bot, chat), backend, uuid4(), "вопрос")
    assert renderer.mode == "draft" and session.on_screen(chat) == [backend.answer]
    assert drafts(session, chat)[0][1] == "" and drafts(session, chat)[-1][1] in backend.answer


def test_make_renderer_depends_on_chat_type(bot):
    private = message_update("вопрос", new_chat_id()).message.as_(bot)
    group = message_update("вопрос", -new_chat_id(), chat_type="supergroup").message.as_(bot)
    assert isinstance(make_renderer(private, "draft"), DraftRenderer)
    assert isinstance(make_renderer(group, "draft"), StreamRenderer)              # черновик — только в личке
    assert isinstance(make_renderer(private, "edit"), StreamRenderer)


async def test_draft_network_error_does_not_lose_answer(bot, session):
    """Черновик — только показ: сбой сети на нём не роняет ответ, его сохраняет sendMessage."""
    from aiogram.exceptions import TelegramNetworkError

    chat = new_chat_id()
    session.fail[SendMessageDraft] = [TelegramNetworkError(SendMessageDraft(chat_id=chat, draft_id=1), "timeout"),
                                      TelegramNetworkError(SendMessageDraft(chat_id=chat, draft_id=1), "timeout")]
    backend = FakeBackend("Ссылка действует 30 минут.", chunk=5)
    renderer = await answer_with_stream(source(bot, chat), backend, uuid4(), "вопрос")
    assert renderer.fallback is None and session.on_screen(chat) == [backend.answer]


async def test_keepalive_never_switches_to_edits_and_waits_for_flood(bot, session):
    chat, clock = new_chat_id(), Clock()
    renderer = DraftRenderer(source(bot, chat), interval=0, keepalive=10, clock=clock)
    await renderer.start()
    await renderer.feed("Начало")
    session.fail[SendMessageDraft] = [bad_request(chat)]
    clock.now += 11
    await renderer._draft(renderer.buffer, force=True, origin="keepalive")        # отклонён — правок нет
    assert renderer.fallback is None
    renderer.blocked_until = clock.now + 30                                      # Telegram просил паузу
    before = len(session.of(SendMessageDraft))
    clock.now += 11
    await renderer._draft(renderer.buffer, force=True, origin="keepalive")
    assert len(session.of(SendMessageDraft)) == before                           # пауза соблюдена
    await renderer.finish()
    assert session.on_screen(chat) == ["Начало"] and not session.of(EditMessageText)
