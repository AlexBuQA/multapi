"""Команды и обычный текст (блок 4.2) через Dispatcher, с FakeBackend вместо сервиса."""
from __future__ import annotations

import asyncio

import httpx
from aiogram.methods import SendChatAction, SendMessage
from aiogram.types import PhotoSize

from bot import texts
from bot.services.backend_client import BackendStreamError
from bot_fakes import ADMIN_ID, FakeBackend, message_update, new_chat_id


def connect_error() -> httpx.ConnectError:
    return httpx.ConnectError("connection refused", request=httpx.Request("POST", "http://backend.test/chats"))


async def test_start_creates_chat_in_backend(dp, bot, session, backend, settings):
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("/start", chat))
    assert (str(chat), "telegram") in backend.chats
    assert session.sent_texts(chat) == [texts.start_text(settings.bot_product_name)]


async def test_same_chat_for_every_message(dp, bot, backend):
    chat = new_chat_id()
    for text in ("/start", "Привет, меня зовут Аня", "Как меня зовут?"):
        await dp.feed_update(bot, message_update(text, chat))
    assert len(backend.chats) == 1
    assert [chat_id for chat_id, _ in backend.sent] == [backend.chats[(str(chat), "telegram")]] * 2


async def test_text_goes_to_backend_and_streams(dp, bot, session):
    backend = dp["backend"] = FakeBackend(delay=0.05)
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("Как меня зовут?", chat))
    assert backend.sent[-1][1] == "Как меня зовут?"
    assert session.on_screen(chat) == [backend.answer]
    assert session.of(SendChatAction)                              # «печатает…», пока ждём первый фрагмент
    assert backend.closed_streams == 1


async def test_clear(dp, bot, session, backend):
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("/clear", chat))
    assert backend.cleared == [backend.chats[(str(chat), "telegram")]]
    assert session.sent_texts(chat) == [texts.HISTORY_CLEARED]


async def test_help_lists_commands(dp, bot, session):
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("/help", chat))
    help_text = session.sent_texts(chat)[0]
    assert all(f"/{name}" in help_text for name in ("start", "ask", "clear", "cancel", "help"))


async def test_backend_down_is_a_message_not_a_traceback(dp, bot, session):
    dp["backend"] = FakeBackend(chat_error=connect_error())
    for text in ("/start", "/clear", "вопрос"):
        chat = new_chat_id()
        await dp.feed_update(bot, message_update(text, chat))
        assert session.sent_texts(chat) == [texts.UNAVAILABLE]


async def test_error_before_first_chunk(dp, bot, session):
    request = httpx.Request("POST", "http://backend.test/chats/x/messages")
    response = httpx.Response(429, json={"error": {"code": "rate_limited", "message": "..."}}, request=request)
    dp["backend"] = FakeBackend(error=httpx.HTTPStatusError("429", request=request, response=response), error_after=0)
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("вопрос", chat))
    assert session.on_screen(chat) == [texts.RATE_LIMITED]


async def test_error_mid_stream_keeps_partial_answer(dp, bot, session):
    backend = FakeBackend("Проверьте папку «Спам» и адрес в профиле.", chunk=8,
                          error=BackendStreamError("llm_unavailable", "provider dropped"), error_after=2)
    dp["backend"] = backend
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("Не приходит письмо", chat))
    assert session.on_screen(chat) == ["Проверьте папку " + texts.INTERRUPTED.format(reason=texts.MODEL_DOWN)]
    assert backend.closed_streams == 1


async def test_admin_status(dp, bot, session):
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("/status", chat, user_id=ADMIN_ID))
    assert session.sent_texts(chat)[-1].startswith("Сервис http://backend.test — отвечает: ok")
    other = new_chat_id()
    await dp.feed_update(bot, message_update("/status", other))
    assert session.sent_texts(other) == [texts.ADMIN_ONLY]


async def test_unknown_command_and_non_text(dp, bot, session, backend):
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("/weather", chat))
    await dp.feed_update(bot, message_update(None, chat, photo=[PhotoSize(file_id="f", file_unique_id="u",
                                                                          width=1, height=1)]))
    assert session.sent_texts(chat) == [texts.UNKNOWN_COMMAND, texts.ONLY_TEXT]
    assert backend.sent == []


async def test_unexpected_error_is_caught(dp, bot, session):
    class Broken(FakeBackend):
        async def clear_messages(self, chat_id):
            raise RuntimeError("баг")

    dp["backend"] = Broken()
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("/clear", chat))
    assert session.sent_texts(chat) == [texts.UNEXPECTED]
    assert all(isinstance(m, SendMessage) for m in session.requests)


async def test_questions_of_one_chat_go_one_by_one(dp, bot, session):
    """Второй вопрос уходит в сервис, когда показан первый ответ: в очереди сервиса он ждал бы
    без единого байта и упирался в таймаут бота."""
    backend = dp["backend"] = FakeBackend(delay=0.05)
    chat = new_chat_id()
    await asyncio.gather(dp.feed_update(bot, message_update("Первый вопрос", chat)),
                         dp.feed_update(bot, message_update("Второй вопрос", chat)),
                         dp.feed_update(bot, message_update("/clear", chat)))
    assert backend.max_active == 1                              # ответы не шли одновременно
    assert sorted(content for _, content in backend.sent) == ["Второй вопрос", "Первый вопрос"]
    assert texts.HISTORY_CLEARED in session.on_screen(chat) and len(backend.cleared) == 1
    assert session.on_screen(chat).count(backend.answer) == 2   # оба ответа показаны целиком


async def test_different_chats_are_answered_in_parallel(dp, bot):
    backend = dp["backend"] = FakeBackend(delay=0.05)
    await asyncio.gather(*(dp.feed_update(bot, message_update("вопрос", new_chat_id())) for _ in range(3)))
    assert backend.max_active == 3
