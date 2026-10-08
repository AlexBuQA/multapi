"""
Сценарий /ask (блок 4.2): раздел кнопкой -> вопрос текстом -> ответ потоком.

    /ask                      клавиатура разделов, состояние waiting_for_topic
    кнопка "topic:<slug>"     data["topic"] — название раздела, состояние waiting_for_question
    кнопка "topic:cancel"     state.clear()
    текст в waiting_for_question
                              «Тема: <раздел>. Вопрос: <текст>» уходит в сервис обычным
                              сообщением чата, ответ — потоком, в конце state.clear()

Состояние — в MemoryStorage (подключён в bot/__main__.py): после перезапуска бота начатый
сценарий забывается, история разговора — нет, она в сервисе.

Кнопка старой клавиатуры (сценарий уже отменён или закончен) не меняет состояние: бот
отвечает «выбор неактуален». Кнопки старых меню убирают выбор раздела, «Отмена», /cancel и
новый /ask. Текст вместо кнопки на шаге выбора раздела — подсказка, в сервис он не уходит.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot import texts
from bot.handlers.common import backend_chat_id
from bot.keyboards.inline import CANCEL, TOPIC_PREFIX, TOPICS, topics_kb
from bot.services.backend_client import BACKEND_ERRORS, BackendClient
from bot.services.chat_queue import ChatQueue
from bot.services.streaming import answer_with_stream
from bot.states import AskFlow

router = Router(name="fsm")
log = logging.getLogger(__name__)
NOT_COMMAND = F.text & ~F.text.startswith("/")


def build_prompt(topic: str, question: str) -> str:
    return f"Тема: {topic}. Вопрос: {question}"


async def drop_menu(bot: Bot, chat_id: int, data: dict[str, Any]) -> None:
    """Убрать кнопки меню разделов из прошлого /ask — чтобы их не нажали позже."""
    if menu := data.get("menu_message_id"):
        with contextlib.suppress(TelegramBadRequest):   # сообщение уже без кнопок или удалено
            await bot.edit_message_reply_markup(chat_id=chat_id, message_id=menu, reply_markup=None)


@router.message(Command("ask"))
async def cmd_ask(message: Message, state: FSMContext, bot: Bot) -> None:
    await drop_menu(bot, message.chat.id, await state.get_data())
    await state.clear()                                  # /ask посреди сценария — начать заново
    menu = await message.answer(texts.ASK_TOPIC, reply_markup=topics_kb())
    await state.set_state(AskFlow.waiting_for_topic)
    await state.update_data(menu_message_id=menu.message_id)


@router.callback_query(AskFlow.waiting_for_topic, F.data == TOPIC_PREFIX + CANCEL)
async def topic_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    if isinstance(callback.message, Message):
        await callback.message.edit_text(texts.CANCELLED)
    await callback.answer()


@router.callback_query(AskFlow.waiting_for_topic, F.data.startswith(TOPIC_PREFIX))
async def topic_chosen(callback: CallbackQuery, state: FSMContext) -> None:
    topic = TOPICS.get((callback.data or "").removeprefix(TOPIC_PREFIX))
    if topic is None:
        await callback.answer(texts.UNKNOWN_TOPIC, show_alert=True)
        return
    await state.update_data(topic=topic.title, topic_slug=topic.slug)
    await state.set_state(AskFlow.waiting_for_question)
    if isinstance(callback.message, Message):           # кнопки убираются: выбор сделан
        await callback.message.edit_text(texts.ASK_QUESTION.format(topic=topic.title))
    await callback.answer()


@router.callback_query(F.data.startswith(TOPIC_PREFIX))
async def topic_stale(callback: CallbackQuery) -> None:
    # Кнопки здесь не убираются: в группе это может быть действующее меню другого участника
    # (состояние сценария у каждого своё). Старые меню и так без кнопок — их убирают выбор
    # раздела, «Отмена», /cancel и новый /ask.
    await callback.answer(texts.TOPIC_STALE)


@router.message(AskFlow.waiting_for_topic, NOT_COMMAND)
async def topic_expected(message: Message) -> None:
    await message.answer(texts.PICK_TOPIC)


@router.message(AskFlow.waiting_for_question, NOT_COMMAND)
async def question_received(message: Message, state: FSMContext, backend: BackendClient,
                            chat_queue: ChatQueue) -> None:
    data = await state.get_data()
    prompt = build_prompt(data.get("topic", ""), message.text or "")
    await state.clear()                     # ответ ниже может идти долго — сценарий уже завершён
    async with chat_queue.turn(message.chat.id):
        try:
            chat_id = await backend_chat_id(backend, message.chat)
        except BACKEND_ERRORS as exc:
            log.warning("backend_error chat=%s during=ask error=%r", message.chat.id, exc)
            await message.answer(texts.user_message(exc))
            return
        await answer_with_stream(message, backend, chat_id, prompt)
