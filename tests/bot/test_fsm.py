"""
Сценарий /ask (блок 4.2) через настоящий Dispatcher: апдейты Telegram подаются в
dp.feed_update, Bot API подменён MockedSession, сервис — FakeBackend.
"""
from __future__ import annotations

from aiogram.methods import AnswerCallbackQuery, EditMessageReplyMarkup, EditMessageText, SendMessage
from aiogram.types import InlineKeyboardMarkup

from bot import texts
from bot.keyboards.inline import TOPICS, topics_kb
from bot.states import AskFlow
from bot_fakes import callback_update, fsm, message_update, new_chat_id


async def test_ask_topic_then_question(dp, bot, session, backend):
    chat = new_chat_id()
    state = fsm(dp, bot, chat)

    await dp.feed_update(bot, message_update("/ask", chat))
    assert await state.get_state() == AskFlow.waiting_for_topic.state
    menu = session.of(SendMessage)[-1]
    assert menu.text == texts.ASK_TOPIC and isinstance(menu.reply_markup, InlineKeyboardMarkup)

    await dp.feed_update(bot, callback_update("topic:billing", chat))
    assert await state.get_state() == AskFlow.waiting_for_question.state
    assert (await state.get_data())["topic"] == "Оплата и документы"
    assert session.of(EditMessageText)[-1].text == texts.ASK_QUESTION.format(topic="Оплата и документы")
    assert session.of(AnswerCallbackQuery)                         # «часики» на кнопке сняты

    await dp.feed_update(bot, message_update("Как получить закрывающие документы?", chat))
    assert backend.sent == [(backend.chats[(str(chat), "telegram")],
                             "Тема: Оплата и документы. Вопрос: Как получить закрывающие документы?")]
    assert await state.get_state() is None
    assert session.on_screen(chat)[-1] == backend.answer          # ответ показан целиком


async def test_cancel_resets_state_on_every_step(dp, bot, session, backend):
    for steps in ([], ["topic:api"]):
        chat = new_chat_id()
        state = fsm(dp, bot, chat)
        await dp.feed_update(bot, message_update("/ask", chat))
        for data in steps:
            await dp.feed_update(bot, callback_update(data, chat))
        await dp.feed_update(bot, message_update("/cancel", chat))
        assert await state.get_state() is None
        assert session.sent_texts(chat)[-1] == texts.CANCELLED
    assert backend.sent == []                                    # «/cancel» не ушёл в сервис вопросом
    assert session.of(EditMessageReplyMarkup)                    # кнопки старого меню убраны


async def test_cancel_button(dp, bot, session):
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("/ask", chat))
    await dp.feed_update(bot, callback_update("topic:cancel", chat))
    assert await fsm(dp, bot, chat).get_state() is None
    assert session.of(EditMessageText)[-1].text == texts.CANCELLED


async def test_cancel_without_scenario(dp, bot, session):
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("/cancel", chat))
    assert session.sent_texts(chat) == [texts.NOTHING_TO_CANCEL]


async def test_text_instead_of_topic_is_not_sent(dp, bot, session, backend):
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("/ask", chat))
    await dp.feed_update(bot, message_update("у меня вопрос про оплату", chat))
    assert backend.sent == [] and session.sent_texts(chat)[-1] == texts.PICK_TOPIC
    assert await fsm(dp, bot, chat).get_state() == AskFlow.waiting_for_topic.state


async def test_stale_topic_button(dp, bot, session):
    chat = new_chat_id()
    await dp.feed_update(bot, callback_update("topic:login", chat))
    assert await fsm(dp, bot, chat).get_state() is None
    assert session.of(AnswerCallbackQuery)[-1].text == texts.TOPIC_STALE
    assert session.of(EditMessageReplyMarkup) == []        # в группе это может быть чужое действующее меню


async def test_other_group_member_cannot_break_menu(dp, bot, session):
    group, author, other = -new_chat_id(), new_chat_id(), new_chat_id()
    await dp.feed_update(bot, message_update("/ask", group, user_id=author))
    await dp.feed_update(bot, callback_update("topic:api", group, user_id=other))
    author_state = dp.fsm.get_context(bot=bot, chat_id=group, user_id=author)
    assert await author_state.get_state() == AskFlow.waiting_for_topic.state
    await dp.feed_update(bot, callback_update("topic:api", group, user_id=author))
    assert (await author_state.get_data())["topic"] == "API и ключи"


async def test_second_ask_removes_old_menu(dp, bot, session):
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("/ask", chat))
    await dp.feed_update(bot, message_update("/ask", chat))
    assert len(session.of(EditMessageReplyMarkup)) == 1
    assert await fsm(dp, bot, chat).get_state() == AskFlow.waiting_for_topic.state


async def test_unknown_topic_keeps_waiting(dp, bot, session):
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("/ask", chat))
    await dp.feed_update(bot, callback_update("topic:python", chat))
    assert await fsm(dp, bot, chat).get_state() == AskFlow.waiting_for_topic.state
    assert session.of(AnswerCallbackQuery)[-1].text == texts.UNKNOWN_TOPIC


async def test_commands_work_inside_scenario(dp, bot, session, backend):
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("/ask", chat))
    await dp.feed_update(bot, callback_update("topic:mobile", chat))
    await dp.feed_update(bot, message_update("/help", chat))
    assert session.sent_texts(chat)[-1] == texts.help_text()
    assert await fsm(dp, bot, chat).get_state() == AskFlow.waiting_for_question.state


def test_topics_keyboard_from_diploma_domain():
    rows = topics_kb().inline_keyboard
    buttons = [button for row in rows for button in row]
    assert 3 <= len(TOPICS) <= 5
    assert [b.callback_data for b in buttons] == [f"topic:{slug}" for slug in TOPICS] + ["topic:cancel"]
    assert rows[-1][0].text == "Отмена" and len(rows[-1]) == 1
    assert all(len(b.callback_data.encode()) <= 64 for b in buttons)
    assert not {"python", "sql", "math"} & set(TOPICS)
