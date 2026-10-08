"""
PostgresChatRepository (блок 4.1): история в Postgres через async SQLAlchemy 2.x (asyncpg).

- Сессия AsyncSession передаётся в конструктор: её создаёт app/chat/deps.py на время
  запроса. Глобальных сессий и движков внутри класса нет.
- Каждый метод — отдельная транзакция (session.begin()). Соединение возвращается в пул
  сразу после неё, а не держится открытым, пока модель генерирует ответ: поток в
  POST /chats/{id}/messages идёт десятки секунд.
- ORM-строки (pg_models.py) наружу не выходят: граница — ChatMessage.model_validate(row,
  from_attributes=True).
- list_messages: SELECT ... WHERE chat_id = ? AND deleted_at IS NULL ORDER BY created_at
  DESC LIMIT N — последние N по частичному индексу — и reversed(): модели нужен
  хронологический порядок. При равном created_at порядок задаёт seq.
- get_or_create_chat (блок 4.2): в одной транзакции pg_advisory_xact_lock по ключу
  interface:owner_external_id, SELECT самого раннего чата с этой парой и, если его нет,
  INSERT. Одновременные запросы с одним ключом — даже из разных копий сервиса — идут по
  очереди, а уникальный индекс не нужен: в базе уже могут быть чаты-дубли, созданные
  create_chat до блока 4.2, — из них берётся самый ранний. Поиск — по индексу
  ix_chats_owner_interface.
- soft_delete_messages: UPDATE ... SET deleted_at = NOW() WHERE chat_id = ? AND
  deleted_at IS NULL — строки остаются в таблице.
- Ошибки SQLAlchemy и сети (нет соединения, нет таблиц) -> ChatStorageError; причина — в
  __cause__ и в логе.
"""
from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.domain import Chat, ChatMessage, ChatNotFoundError, ChatStorageError
from app.chat.repositories.pg_models import ChatMessageRow, ChatRow
from app.observability.logging import get_logger

log = get_logger()


class PostgresChatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @contextlib.asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        try:
            async with self.session.begin():
                yield
        except (SQLAlchemyError, OSError) as exc:
            log.warning("chat_storage_failed", storage="postgres", error=repr(exc)[:300])
            raise ChatStorageError() from exc

    async def create_chat(self, owner_external_id: str, interface: str,
                          system_prompt: str | None = None) -> Chat:
        chat = Chat(owner_external_id=owner_external_id, interface=interface, system_prompt=system_prompt)
        async with self._transaction():
            self.session.add(ChatRow(**chat.model_dump()))
        return chat

    async def get_or_create_chat(self, owner_external_id: str, interface: str,
                                 system_prompt: str | None = None) -> tuple[Chat, bool]:
        lock_key = func.hashtextextended(f"{interface}:{owner_external_id}", 0)
        stmt = (
            select(ChatRow)
            .where(ChatRow.owner_external_id == owner_external_id, ChatRow.interface == interface)
            .order_by(ChatRow.created_at, ChatRow.id)
            .limit(1)
        )
        async with self._transaction():
            # Замок до конца транзакции: второй такой же запрос ждёт здесь и после COMMIT
            # первого уже находит его чат.
            await self.session.execute(select(func.pg_advisory_xact_lock(lock_key)))
            row = await self.session.scalar(stmt)
            if row is not None:
                return Chat.model_validate(row, from_attributes=True), False
            chat = Chat(owner_external_id=owner_external_id, interface=interface, system_prompt=system_prompt)
            self.session.add(ChatRow(**chat.model_dump()))
        return chat, True

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        async with self._transaction():
            row = await self.session.get(ChatRow, chat_id)
            return Chat.model_validate(row, from_attributes=True) if row is not None else None

    async def append_message(self, chat_id: UUID, message: ChatMessage) -> ChatMessage:
        if message.chat_id != chat_id:
            raise ValueError(f"message.chat_id={message.chat_id} не совпадает с chat_id={chat_id}")
        async with self._transaction():
            if await self.session.get(ChatRow, chat_id) is None:
                raise ChatNotFoundError(chat_id)
            self.session.add(ChatMessageRow(**message.model_dump()))
        return message

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        if limit <= 0:
            return []
        stmt = (
            select(ChatMessageRow)
            .where(ChatMessageRow.chat_id == chat_id, ChatMessageRow.deleted_at.is_(None))
            .order_by(ChatMessageRow.created_at.desc(), ChatMessageRow.seq.desc())
            .limit(limit)
        )
        async with self._transaction():
            rows = (await self.session.scalars(stmt)).all()
            return [ChatMessage.model_validate(row, from_attributes=True) for row in reversed(rows)]

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        stmt = (
            update(ChatMessageRow)
            .where(ChatMessageRow.chat_id == chat_id, ChatMessageRow.deleted_at.is_(None))
            .values(deleted_at=func.now())
            .execution_options(synchronize_session=False)
        )
        async with self._transaction():
            await self.session.execute(stmt)
