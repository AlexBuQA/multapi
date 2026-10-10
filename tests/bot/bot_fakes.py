"""
Помощники тестов Telegram-бота (блоки 4.2–4.4). Отдельный модуль, а не conftest.py: так их
можно импортировать в тестах (from bot_fakes import ...), а conftest.py только объявляет
фикстуры. Сеть не нужна — ни Telegram, ни сервис.

- MockedSession — сессия aiogram без HTTP: запоминает вызванные методы Bot API и отвечает
  правдоподобными объектами (sendMessage -> Message и т. д.). Так апдейты проходят через
  настоящий Dispatcher: фильтры, порядок роутеров, FSM, передачу dp["backend"] в handlers.
  Черновики sendMessageDraft — в drafts; файлы для getFile и скачивания — в files;
  inline-кнопки под сообщениями — в markups (блок 4.4).
- FakeBackend — подмена BackendClient: чаты по owner_external_id, ответ фрагментами,
  ошибки по заказу; файлы, пришедшие в send_message, — в media. Блок 4.4: send_message
  возвращает FakeAnswer с message_id и replacement, как AnswerStream; оценки, admin API и
  очередь рассылок — в памяти.
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
    DeleteMessage,
    EditMessageReplyMarkup,
    EditMessageText,
    GetFile,
    GetMe,
    SendChatAction,
    SendMessage,
    SendMessageDraft,
    SetMyCommands,
    TelegramMethod,
)
from aiogram.types import CallbackQuery, Chat, File, Message, Update, User  # noqa: E402


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
        self.drafts: dict[int, list[tuple[int, str]]] = {}   # chat_id -> [(draft_id, текст)] по порядку
        self.files: dict[str, bytes] = {}                    # file_id -> содержимое (getFile, скачивание)
        self.markups: dict[tuple[int, int], Any] = {}        # (chat_id, message_id) -> кнопки сейчас

    async def close(self) -> None:
        pass

    async def stream_content(self, url: str, *args: Any, **kwargs: Any) -> AsyncIterator[bytes]:
        file_id = url.rsplit("/", 1)[-1]
        data = self.files[file_id]
        for i in range(0, len(data), 1000):
            yield data[i:i + 1000]

    async def make_request(self, bot: Bot, method: TelegramMethod[Any], timeout: int | None = None) -> Any:
        self.requests.append(method)
        errors = self.fail.get(type(method))
        if errors:
            raise errors.pop(0)
        if isinstance(method, SendMessage):
            message_id = next(self._message_ids)
            self.screen.setdefault(int(method.chat_id), []).append((message_id, method.text))
            if method.reply_markup is not None:
                self.markups[(int(method.chat_id), message_id)] = method.reply_markup
            return Message(message_id=message_id, date=now(), from_user=BOT_USER, text=method.text,
                           chat=Chat(id=int(method.chat_id), type="private"),
                           reply_markup=method.reply_markup).as_(bot)
        if isinstance(method, EditMessageText):
            chat_id = int(method.chat_id or 0)
            self.screen[chat_id] = [(mid, method.text if mid == method.message_id else text)
                                    for mid, text in self.screen.get(chat_id, [])]
            return Message(message_id=method.message_id or 0, date=now(), from_user=BOT_USER, text=method.text,
                           chat=Chat(id=chat_id, type="private")).as_(bot)
        if isinstance(method, SendMessageDraft):
            self.drafts.setdefault(int(method.chat_id), []).append((method.draft_id, method.text or ""))
            return True
        if isinstance(method, GetFile):
            if method.file_id not in self.files:
                raise AssertionError(f"MockedSession: нет файла {method.file_id}")
            return File(file_id=method.file_id, file_unique_id=f"u-{method.file_id}",
                        file_size=len(self.files[method.file_id]), file_path=f"documents/{method.file_id}")
        if isinstance(method, EditMessageReplyMarkup):
            key = (int(method.chat_id or 0), int(method.message_id or 0))
            if method.reply_markup is None:
                self.markups.pop(key, None)
            else:
                self.markups[key] = method.reply_markup
            return True
        if isinstance(method, DeleteMessage):
            chat_id = int(method.chat_id)
            self.screen[chat_id] = [(mid, text) for mid, text in self.screen.get(chat_id, [])
                                    if mid != method.message_id]
            self.markups.pop((chat_id, method.message_id), None)
            return True
        if isinstance(method, (AnswerCallbackQuery, SendChatAction, SetMyCommands)):
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

    def draft_texts(self, chat_id: int) -> list[str]:
        return [text for _, text in self.drafts.get(chat_id, [])]

    def buttons(self, chat_id: int) -> list[list[tuple[str, str]]]:
        """Кнопки под сообщениями бота в чате сейчас: по сообщению — [(текст, callback_data)]."""
        return [[(b.text, b.callback_data or "") for row in markup.inline_keyboard for b in row]
                for (chat, _), markup in sorted(self.markups.items()) if chat == chat_id]


class FakeAnswer:
    """Как AnswerStream: фрагменты, а после конца потока — message_id и replacement."""

    def __init__(self) -> None:
        self.message_id: UUID | None = None
        self.replacement: str | None = None
        self.categories: list[str] = []
        self.chunks: AsyncIterator[str] | None = None

    def __aiter__(self) -> FakeAnswer:
        return self

    async def __anext__(self) -> str:
        assert self.chunks is not None
        return await anext(self.chunks)

    async def aclose(self) -> None:
        await self.chunks.aclose()  # type: ignore[union-attr]


class FakeBackend:
    """Подмена BackendClient. answer — текст ответа; chunk — длина фрагмента; delay — пауза
    перед первым фрагментом (модель думает)."""

    base_url = "http://backend.test"

    def __init__(self, answer: str = "Вас зовут Аня. Чем ещё помочь?", *, chunk: int = 6,
                 error: Exception | None = None, error_after: int | None = None,
                 chat_error: Exception | None = None, delay: float = 0.0,
                 replacement: str | None = None, replaced_after: int | None = None,
                 admin_token: str | None = "admin-token-for-tests", admin_error: Exception | None = None,
                 feedback_error: Exception | None = None) -> None:
        self.answer, self.chunk, self.delay = answer, chunk, delay
        self.error, self.error_after, self.chat_error = error, error_after, chat_error
        # блок 4.4: ответ заменён модерацией после replaced_after фрагментов (None — после всех)
        self.replacement, self.replaced_after = replacement, replaced_after
        self.admin_token, self.admin_error, self.feedback_error = admin_token, admin_error, feedback_error
        self.answers: list[FakeAnswer] = []
        self.feedback: dict[tuple[UUID, UUID], str] = {}
        self.stats: dict[str, Any] = {"period_hours": 24, "total_messages": 12, "active_users": 3,
                                      "avg_latency_ms": 2450.0, "moderation_blocks": 1, "moderation_block_rate": 0.1667,
                                      "feedback_votes": 4, "feedback_up_ratio": 0.75,
                                      "top_questions": [{"question": "как сбросить <пароль>", "count": 3}]}
        self.users: list[dict[str, Any]] = []
        self.broadcasts: list[dict[str, Any]] = []       # очередь: {"id", "message", "interface", "recipients"}
        self.results: list[tuple[int, int, int]] = []    # finish_broadcast: (id, sent, failed)
        self.chats: dict[tuple[str, str], UUID] = {}
        self.sent: list[tuple[UUID, str]] = []
        self.media: list[dict[str, Any]] = []                # файлы из send_message: media, mime, filename
        self.cleared: list[UUID] = []
        self.closed_streams = 0
        self.active = self.max_active = 0                 # одновременных ответов сейчас и максимум

    async def get_or_create_chat(self, owner_external_id: str, interface: str) -> UUID:
        if self.chat_error is not None:
            raise self.chat_error
        return self.chats.setdefault((owner_external_id, interface), uuid4())

    def send_message(self, chat_id: UUID, content: str, media: bytes | None = None, mime: str | None = None,
                     *, filename: str | None = None) -> FakeAnswer:
        self.sent.append((chat_id, content))
        if media is not None:
            self.media.append({"media": media, "mime": mime, "filename": filename})
        answer = FakeAnswer()
        answer.chunks = self._chunks(answer)
        self.answers.append(answer)
        return answer

    async def _chunks(self, answer: FakeAnswer) -> AsyncIterator[str]:
        parts = [self.answer[i:i + self.chunk] for i in range(0, len(self.answer), self.chunk)]
        if self.replacement is not None and self.replaced_after is not None:
            parts = parts[:self.replaced_after]
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
            if self.replacement is not None:
                answer.replacement, answer.categories = self.replacement, ["violence"]
            answer.message_id = uuid4()                    # событие done
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

    # ------------------------------------------------------------------ блок 4.4
    async def send_feedback(self, chat_id: UUID, message_id: UUID, value: str) -> dict[str, Any]:
        if self.feedback_error is not None:
            raise self.feedback_error
        saved = (chat_id, message_id) not in self.feedback
        self.feedback.setdefault((chat_id, message_id), value)
        return {"message_id": str(message_id), "value": self.feedback[(chat_id, message_id)], "saved": saved}

    def _admin(self) -> None:
        from bot.services.backend_client import AdminNotConfigured

        if not self.admin_token:
            raise AdminNotConfigured("ADMIN_TOKEN не задан в .env бота")
        if self.admin_error is not None:
            raise self.admin_error

    async def admin_stats(self, hours: int = 24, top: int = 5) -> dict[str, Any]:
        self._admin()
        return self.stats

    async def admin_users(self, limit: int = 10) -> list[dict[str, Any]]:
        self._admin()
        return self.users[:limit]

    async def admin_broadcast(self, message: str, interface: str = "telegram") -> dict[str, Any]:
        self._admin()
        recipients = sorted({owner for owner, kind in self.chats if kind == interface})
        item = {"id": len(self.broadcasts) + 1, "message": message, "interface": interface, "recipients": recipients}
        self.broadcasts.append(item)
        return {"id": item["id"], "status": "pending", "interface": interface, "recipients": len(recipients)}

    async def claim_broadcast(self) -> dict[str, Any] | None:
        self._admin()
        return self.broadcasts.pop(0) if self.broadcasts else None

    async def finish_broadcast(self, broadcast_id: int, sent: int, failed: int) -> dict[str, Any]:
        self._admin()
        self.results.append((broadcast_id, sent, failed))
        return {"id": broadcast_id, "status": "sent" if sent else "failed", "sent": sent, "failed": failed}


# ---------------------------------------------------------------- апдейты
def user(user_id: int) -> User:
    return User(id=user_id, is_bot=False, first_name="Аня")


def message_update(text: str | None, chat_id: int, *, user_id: int | None = None, chat_type: str = "private",
                   **extra: Any) -> Update:
    msg = Message(message_id=next(_update_ids), date=now(), chat=Chat(id=chat_id, type=chat_type),
                  from_user=user(user_id or chat_id), text=text, **extra)
    return Update(update_id=next(_update_ids), message=msg)


def callback_update(data: str, chat_id: int, *, message_id: int = 1, user_id: int | None = None,
                    text: str = "Выберите раздел:") -> Update:
    menu = Message(message_id=message_id, date=now(), chat=Chat(id=chat_id, type="private"),
                   from_user=BOT_USER, text=text)
    query = CallbackQuery(id=str(next(_update_ids)), from_user=user(user_id or chat_id), chat_instance="ci",
                          data=data, message=menu)
    return Update(update_id=next(_update_ids), callback_query=query)




def fsm(dp: Dispatcher, bot: Bot, chat_id: int) -> FSMContext:
    return dp.fsm.get_context(bot=bot, chat_id=chat_id, user_id=chat_id)
