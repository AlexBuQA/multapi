"""
Команды (блок 4.2): /cancel, /start, /help, /clear. Команды администраторов — /status, /stats,
/users, /broadcast — с блока 4.4 в bot/handlers/admin.py под фильтром роутера; здесь — ответ
«только для администраторов» всем остальным.

Роутер подключается первым (bot/handlers/__init__.py), а /cancel — первый handler в нём:
иначе в сценарии /ask текст «/cancel» в состоянии waiting_for_question забрал бы handler
FSM и отправил его в сервис как вопрос. Остальные команды тоже работают на любом шаге
сценария, не сбрасывая его.
"""
from __future__ import annotations

import logging

from aiogram import Bot, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot import texts
from bot.config import BotSettings
from bot.handlers.common import backend_chat_id
from bot.handlers.fsm import drop_menu
from bot.services.backend_client import BACKEND_ERRORS, BackendClient
from bot.services.chat_queue import ChatQueue

router = Router(name="commands")
log = logging.getLogger(__name__)


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext, bot: Bot) -> None:
    if await state.get_state() is None:
        await message.answer(texts.NOTHING_TO_CANCEL)
        return
    data = await state.get_data()
    await state.clear()
    await drop_menu(bot, message.chat.id, data)
    await message.answer(texts.CANCELLED)


@router.message(CommandStart())
async def cmd_start(message: Message, backend: BackendClient, settings: BotSettings) -> None:
    try:
        chat_id = await backend_chat_id(backend, message.chat)
    except BACKEND_ERRORS as exc:
        log.warning("backend_error chat=%s during=start error=%r", message.chat.id, exc)
        await message.answer(texts.user_message(exc))
        return
    log.info("chat_ready telegram_chat=%s chat_id=%s", message.chat.id, chat_id)
    await message.answer(texts.start_text(settings.bot_product_name))


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(texts.help_text())


@router.message(Command("clear"))
async def cmd_clear(message: Message, backend: BackendClient, chat_queue: ChatQueue) -> None:
    async with chat_queue.turn(message.chat.id):        # после ответа, который сейчас идёт
        try:
            await backend.clear_messages(await backend_chat_id(backend, message.chat))
        except BACKEND_ERRORS as exc:
            log.warning("backend_error chat=%s during=clear error=%r", message.chat.id, exc)
            await message.answer(texts.user_message(exc))
            return
    await message.answer(texts.HISTORY_CLEARED)


@router.message(Command("status", "stats", "users", "broadcast"))
async def cmd_admin_denied(message: Message, settings: BotSettings) -> None:
    """Сюда доходят не-администраторы и администраторы в группе: личные чаты администраторов
    забирает роутер admin."""
    is_admin = message.from_user is not None and message.from_user.id in settings.bot_admin_ids
    await message.answer(texts.ADMIN_PRIVATE_ONLY if is_admin else texts.ADMIN_ONLY)
