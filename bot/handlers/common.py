"""Общее для handlers (блок 4.2): чат клиента в сервисе и проверка администратора."""
from __future__ import annotations

from uuid import UUID

from aiogram.filters import Filter
from aiogram.types import Chat, Message

from bot.config import BotSettings
from bot.services.backend_client import BackendClient

INTERFACE = "telegram"


async def backend_chat_id(backend: BackendClient, chat: Chat) -> UUID:
    """Чат клиента в сервисе. owner_external_id — Telegram chat.id строкой: в личном чате это
    id пользователя, в группе — id группы (история общая на группу). POST /chats идемпотентен,
    поэтому бот не хранит соответствие «чат Telegram -> chat_id» и спрашивает сервис на
    каждое сообщение: это один короткий запрос, а бот после перезапуска ничего не теряет."""
    return await backend.get_or_create_chat(str(chat.id), INTERFACE)


class IsAdmin(Filter):
    """Отправитель — в BOT_ADMIN_IDS. Настройки приходят из dp["settings"]."""

    async def __call__(self, message: Message, settings: BotSettings) -> bool:
        return message.from_user is not None and message.from_user.id in settings.bot_admin_ids
