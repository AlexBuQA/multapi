"""
Оценки 👍/👎 и модерация в боте (блок 4.4) через Dispatcher, с FakeBackend вместо сервиса.

- после ответа под его последним сообщением — кнопки fb:up:<id> и fb:down:<id>, id — из
  события done; нажатие — POST .../feedback, кнопки убираются edit_reply_markup(None);
- вопрос не прошёл модерацию (403) — понятный текст, а не ошибка;
- ответ не прошёл модерацию (событие moderation) — показанные части удалены, вместо них —
  текст сервиса; кнопок оценки под ним нет.
"""
from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
from aiogram.methods import AnswerCallbackQuery, DeleteMessage, EditMessageReplyMarkup

from bot import texts
from bot.keyboards.inline import feedback_kb, parse_feedback
from bot.services.backend_client import BackendStreamError
from bot_fakes import FakeBackend, callback_update, message_update, new_chat_id

REFUSAL = "Не могу показать ответ — он мог нарушить правила."


def status_error(status: int, body: dict) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://backend.test/chats/x/messages")
    response = httpx.Response(status, json=body, request=request)
    return httpx.HTTPStatusError(str(status), request=request, response=response)


async def ask(dp, bot, chat: int, text: str = "Как сменить пароль?") -> None:
    await dp.feed_update(bot, message_update(text, chat))


def answer_buttons(session, chat: int) -> tuple[int, list[tuple[str, str]]]:
    """(message_id, кнопки) единственного сообщения бота с кнопками."""
    [(key, markup)] = [(k, m) for k, m in session.markups.items() if k[0] == chat]
    return key[1], [(b.text, b.callback_data) for row in markup.inline_keyboard for b in row]


# ---------------------------------------------------------------- клавиатура
def test_feedback_keyboard_fits_telegram_limit():
    message_id = uuid4()
    buttons = [b for row in feedback_kb(message_id).inline_keyboard for b in row]
    assert [b.text for b in buttons] == ["👍", "👎"]
    assert [b.callback_data for b in buttons] == [f"fb:up:{message_id}", f"fb:down:{message_id}"]
    assert all(len(b.callback_data.encode()) <= 64 for b in buttons)          # предел Telegram — 64 байта


@pytest.mark.parametrize("data", [None, "", "fb:", "fb:up", "fb:like:" + str(uuid4()), "fb:up:не-uuid",
                                  "topic:billing"])
def test_parse_feedback_rejects_garbage(data):
    assert parse_feedback(data) is None


# ---------------------------------------------------------------- кнопки под ответом
@pytest.mark.parametrize("mode", ["draft", "edit"])
async def test_buttons_under_the_answer(dp, bot, session, settings, mode):
    settings.bot_streaming = mode
    backend = dp["backend"] = FakeBackend()
    chat = new_chat_id()
    await ask(dp, bot, chat)
    message_id, buttons = answer_buttons(session, chat)
    answer_id = backend.answers[-1].message_id
    assert buttons == [("👍", f"fb:up:{answer_id}"), ("👎", f"fb:down:{answer_id}")]
    assert session.screen[chat][-1][0] == message_id                    # под последним сообщением ответа


async def test_long_answer_has_buttons_only_under_last_part(dp, bot, session, settings):
    settings.bot_streaming = "edit"
    dp["backend"] = FakeBackend("слово " * 1200, chunk=500)
    chat = new_chat_id()
    await ask(dp, bot, chat)
    assert len(session.on_screen(chat)) == 2
    message_id, _ = answer_buttons(session, chat)
    assert message_id == session.screen[chat][-1][0]


async def test_no_buttons_when_answer_failed(dp, bot, session):
    dp["backend"] = FakeBackend(error=BackendStreamError("llm_unavailable", "x"), error_after=2)
    chat = new_chat_id()
    await ask(dp, bot, chat)
    assert session.buttons(chat) == []


# ---------------------------------------------------------------- нажатие
async def test_vote_is_saved_and_buttons_removed(dp, bot, session):
    backend = dp["backend"] = FakeBackend()
    chat = new_chat_id()
    await ask(dp, bot, chat)
    message_id, buttons = answer_buttons(session, chat)
    await dp.feed_update(bot, callback_update(buttons[1][1], chat, message_id=message_id, text=backend.answer))
    chat_id = backend.chats[(str(chat), "telegram")]
    assert backend.feedback == {(chat_id, backend.answers[-1].message_id): "down"}
    edit = session.of(EditMessageReplyMarkup)[-1]
    assert (edit.message_id, edit.reply_markup) == (message_id, None)
    assert session.buttons(chat) == []
    assert session.of(AnswerCallbackQuery)[-1].text == texts.FEEDBACK_THANKS


async def test_second_vote_is_not_counted(dp, bot, session):
    """Кнопки убраны, но старый клиент мог нажать дважды: сервис хранит первую оценку."""
    backend = dp["backend"] = FakeBackend()
    chat = new_chat_id()
    await ask(dp, bot, chat)
    message_id, buttons = answer_buttons(session, chat)
    for data in (buttons[0][1], buttons[1][1]):
        await dp.feed_update(bot, callback_update(data, chat, message_id=message_id, text=backend.answer))
    assert list(backend.feedback.values()) == ["up"]
    assert [a.text for a in session.of(AnswerCallbackQuery)] == [texts.FEEDBACK_THANKS, texts.FEEDBACK_ALREADY]


async def test_backend_down_keeps_buttons(dp, bot, session):
    dp["backend"] = FakeBackend(feedback_error=httpx.ConnectError("refused", request=httpx.Request("POST", "http://b/")))
    chat = new_chat_id()
    await ask(dp, bot, chat)
    message_id, buttons = answer_buttons(session, chat)
    await dp.feed_update(bot, callback_update(buttons[0][1], chat, message_id=message_id))
    assert session.of(AnswerCallbackQuery)[-1].text == texts.FEEDBACK_FAILED
    assert len(session.buttons(chat)) == 1                              # можно нажать ещё раз


@pytest.mark.parametrize("status, code", [(404, "message_not_found"), (422, "not_an_answer")])
async def test_answer_that_cannot_be_rated_drops_buttons(dp, bot, session, status, code):
    dp["backend"] = FakeBackend(feedback_error=status_error(status, {"error": {"code": code, "message": "..."}}))
    chat = new_chat_id()
    await ask(dp, bot, chat)
    message_id, buttons = answer_buttons(session, chat)
    await dp.feed_update(bot, callback_update(buttons[0][1], chat, message_id=message_id))
    assert session.of(AnswerCallbackQuery)[-1].text == texts.FEEDBACK_GONE and session.buttons(chat) == []


async def test_garbage_callback_is_answered(dp, bot, session):
    backend = dp["backend"] = FakeBackend()
    chat = new_chat_id()
    await dp.feed_update(bot, callback_update("fb:up:не-uuid", chat))
    assert session.of(AnswerCallbackQuery)[-1].text == texts.FEEDBACK_GONE and backend.feedback == {}


# ---------------------------------------------------------------- модерация
async def test_blocked_question_is_a_friendly_text(dp, bot, session):
    body = {"detail": {"code": "moderation_blocked", "categories": ["violence"]},
            "error": {"code": "moderation_blocked", "message": "Сообщение не прошло модерацию.",
                      "categories": ["violence"]}}
    dp["backend"] = FakeBackend(error=status_error(403, body), error_after=0)
    chat = new_chat_id()
    await ask(dp, bot, chat, "Я тебя убью")
    assert session.on_screen(chat) == [texts.MODERATION_BLOCKED.format(topics="угрозы и насилие")]
    assert session.buttons(chat) == []


async def test_blocked_question_is_logged_as_info_not_error(dp, bot, session, caplog):
    import logging

    body = {"error": {"code": "moderation_blocked", "message": "...", "categories": ["violence"]}}
    dp["backend"] = FakeBackend(error=status_error(403, body), error_after=0)
    chat = new_chat_id()
    with caplog.at_level(logging.INFO, logger="bot.services.streaming"):
        await ask(dp, bot, chat, "Я тебя убью")
    lines = [(r.levelname, r.getMessage()) for r in caplog.records if r.name == "bot.services.streaming"]
    assert lines == [("INFO", f"question_blocked chat={chat} categories=violence")]       # без текста вопроса


async def test_blocked_self_harm_question_gets_support_not_refusal(dp, bot, session):
    body = {"error": {"code": "moderation_blocked", "message": "...", "categories": ["self_harm"]}}
    dp["backend"] = FakeBackend(error=status_error(403, body), error_after=0)
    chat = new_chat_id()
    await ask(dp, bot, chat, "не хочу жить")
    assert session.on_screen(chat) == [texts.SELF_HARM_SUPPORT] and "112" in texts.SELF_HARM_SUPPORT


@pytest.mark.parametrize("mode", ["draft", "edit"])
async def test_replaced_answer_removes_shown_parts(dp, bot, session, settings, mode):
    settings.bot_streaming = mode
    backend = dp["backend"] = FakeBackend("Понимаю раздражение. Но я тебя ", chunk=8, replacement=REFUSAL,
                                          replaced_after=3)
    chat = new_chat_id()
    await ask(dp, bot, chat)
    assert session.on_screen(chat) == [REFUSAL]                         # части ответа не остались
    assert session.buttons(chat) == []                                  # отказ не оценивают
    assert backend.closed_streams == 1


async def test_long_replaced_answer_deletes_every_part(dp, bot, session, settings):
    settings.bot_streaming = "edit"
    dp["backend"] = FakeBackend("слово " * 1200, chunk=500, replacement=REFUSAL)
    chat = new_chat_id()
    await ask(dp, bot, chat)
    assert session.on_screen(chat) == [REFUSAL] and len(session.of(DeleteMessage)) == 2


async def test_replaced_before_first_chunk(dp, bot, session):
    dp["backend"] = FakeBackend(replacement=REFUSAL, replaced_after=0)
    chat = new_chat_id()
    await ask(dp, bot, chat)
    assert session.on_screen(chat) == [REFUSAL] and not session.of(DeleteMessage)


async def test_replaced_self_harm_answer_gets_support(dp, bot, session):
    class SelfHarm(FakeBackend):
        async def _chunks(self, answer):
            async for chunk in super()._chunks(answer):
                yield chunk
            answer.categories = ["self_harm_instructions"]

    dp["backend"] = SelfHarm(replacement=REFUSAL)
    chat = new_chat_id()
    await ask(dp, bot, chat)
    assert session.on_screen(chat) == [texts.SELF_HARM_SUPPORT]

