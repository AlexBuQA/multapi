"""
Помощники тестов Telegram-бота (блок 4.2). Отдельный модуль, а не conftest.py: так их можно
импортировать в тестах (from bot_fakes import ...), а conftest.py только объявляет фикстуры.
Сеть не нужна — ни Telegram, ни сервис.

- MockedSession — сессия aiogram без HTTP: запоминает вызванные методы Bot API и отвечает
  правдоподобными объектами (sendMessage -> Message и т. д.). Так апдейты проходят через
  настоящий Dispatcher: фильтры, порядок роутеров, FSM, передачу dp["backend"] в handlers.
- FakeBackend — подмена BackendClient: чаты по owner_external_id, ответ фрагментами,
  ошибки по заказу.
"""
from __future__ import annotations

import asyncio
import itertools
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiogram import Bot, Dispatcher  # noqa: E402
from aiogram.client.session.base import BaseSession  # noqa: E402
from aiogram.fsm.context import FSMContext  # noqa: E402
from aiogram.methods import (  # noqa: E402
    AnswerCallbackQuery,
    EditMessageReplyMarkup,
    EditMessageText,
    GetMe,
    SendChatAction,
    SendMessage,
    SetMyCommands,
    TelegramMethod,
)
from aiogram.types import CallbackQuery, Chat, Message, Update, User  # noqa: E402


BOT_USER = User(id=42, is_bot=True, first_name="Тестовый бот", username="test_bot")
ADMIN_ID = 777
_chat_ids = itertools.count(10_000)
_update_ids = itertools.count(1)


def new_chat_id() -> int:
    return next(_chat_ids)


def now() -> datetime:
    return datetime.now(UTC)


class MockedSession(BaseSession):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[TelegramMethod[Any]] = []
        self.fail: dict[type, list[Exception]] = {}       # метод -> исключения по очереди
        self._message_ids = itertools.count(500)
        self.screen: dict[int, list[tuple[int, str]]] = {}   # chat_id -> [(message_id, текст сейчас)]

    async def close(self) -> None:
        pass

    async def stream_content(self, *args: Any, **kwargs: Any) -> AsyncIterator[bytes]:  # pragma: no cover
        raise NotImplementedError
        yield b""

    async def make_request(self, bot: Bot, method: TelegramMethod[Any], timeout: int | None = None) -> Any:
        self.requests.append(method)
        errors = self.fail.get(type(method))
        if errors:
            raise errors.pop(0)
        if isinstance(method, SendMessage):
            message_id = next(self._message_ids)
            self.screen.setdefault(int(method.chat_id), []).append((message_id, method.text))
            return Message(message_id=message_id, date=now(), from_user=BOT_USER, text=method.text,
                           chat=Chat(id=int(method.chat_id), type="private"),
                           reply_markup=method.reply_markup).as_(bot)
        if isinstance(method, EditMessageText):
            chat_id = int(method.chat_id or 0)
            self.screen[chat_id] = [(mid, method.text if mid == method.message_id else text)
                                    for mid, text in self.screen.get(chat_id, [])]
            return Message(message_id=method.message_id or 0, date=now(), from_user=BOT_USER, text=method.text,
                           chat=Chat(id=chat_id, type="private")).as_(bot)
        if isinstance(method, (AnswerCallbackQuery, SendChatAction, SetMyCommands, EditMessageReplyMarkup)):
            return True
        if isinstance(method, GetMe):
            return BOT_USER
        raise AssertionError(f"MockedSession: неожиданный метод {type(method).__name__}")

    # --- что бот сделал
    def of(self, kind: type) -> list[Any]:
        return [m for m in self.requests if isinstance(m, kind)]

    def sent_texts(self, chat_id: int | None = None) -> list[str]:
        return [m.text for m in self.of(SendMessage) if chat_id is None or int(m.chat_id) == chat_id]

    def on_screen(self, chat_id: int) -> list[str]:
        """Сообщения бота в чате так, как их видит пользователь: с учётом всех правок."""
        return [text for _, text in self.screen.get(chat_id, [])]


class FakeBackend:
    """Подмена BackendClient. answer — текст ответа; chunk — длина фрагмента; delay — пауза
    перед первым фрагментом (модель думает)."""

    base_url = "http://backend.test"

    def __init__(self, answer: str = "Вас зовут Аня. Чем ещё помочь?", *, chunk: int = 6,
                 error: Exception | None = None, error_after: int | None = None,
                 chat_error: Exception | None = None, delay: float = 0.0) -> None:
        self.answer, self.chunk, self.delay = answer, chunk, delay
        self.error, self.error_after, self.chat_error = error, error_after, chat_error
        self.chats: dict[tuple[str, str], UUID] = {}
        self.sent: list[tuple[UUID, str]] = []
        self.cleared: list[UUID] = []
        self.closed_streams = 0
        self.active = self.max_active = 0                 # одновременных ответов сейчас и максимум

    async def get_or_create_chat(self, owner_external_id: str, interface: str) -> UUID:
        if self.chat_error is not None:
            raise self.chat_error
        return self.chats.setdefault((owner_external_id, interface), uuid4())

    async def send_message(self, chat_id: UUID, content: str) -> AsyncIterator[str]:
        self.sent.append((chat_id, content))
        parts = [self.answer[i:i + self.chunk] for i in range(0, len(self.answer), self.chunk)]
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            for number, part in enumerate(parts):
                if self.error is not None and self.error_after == number:
                    raise self.error
                yield part
            if self.error is not None and (self.error_after is None or self.error_after >= len(parts)):
                raise self.error
        finally:
            self.active -= 1
            self.closed_streams += 1

    async def clear_messages(self, chat_id: UUID) -> None:
        if self.chat_error is not None:
            raise self.chat_error
        self.cleared.append(chat_id)

    async def health(self) -> dict[str, Any]:
        if self.chat_error is not None:
            raise self.chat_error
        return {"status": "ok"}


# ---------------------------------------------------------------- апдейты
def user(user_id: int) -> User:
    return User(id=user_id, is_bot=False, first_name="Аня")


def message_update(text: str | None, chat_id: int, *, user_id: int | None = None, **extra: Any) -> Update:
    msg = Message(message_id=next(_update_ids), date=now(), chat=Chat(id=chat_id, type="private"),
                  from_user=user(user_id or chat_id), text=text, **extra)
    return Update(update_id=next(_update_ids), message=msg)


def callback_update(data: str, chat_id: int, *, message_id: int = 1, user_id: int | None = None) -> Update:
    menu = Message(message_id=message_id, date=now(), chat=Chat(id=chat_id, type="private"),
                   from_user=BOT_USER, text="Выберите раздел:")
    query = CallbackQuery(id=str(next(_update_ids)), from_user=user(user_id or chat_id), chat_instance="ci",
                          data=data, message=menu)
    return Update(update_id=next(_update_ids), callback_query=query)




def fsm(dp: Dispatcher, bot: Bot, chat_id: int) -> FSMContext:
    return dp.fsm.get_context(bot=bot, chat_id=chat_id, user_id=chat_id)
