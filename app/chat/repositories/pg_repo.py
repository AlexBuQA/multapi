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

Блок 4.4 (OpsRepository):
- add_feedback: INSERT … ON CONFLICT ON CONSTRAINT uq_message_feedback_owner_message DO NOTHING
  RETURNING value. Ничего не вернулось — оценка уже есть, она и возвращается (saved=False).
  Два одновременных клика 👍 и 👎 не дадут двух строк: решает уникальный индекс, а не
  проверка «есть ли уже» в приложении;
- claim_broadcast: UPDATE … WHERE id = (SELECT id … ORDER BY created_at LIMIT 1 FOR UPDATE
  SKIP LOCKED) RETURNING — две копии бота не заберут одну рассылку;
- stats — пять запросов в одной транзакции. Задержка ответа — LAG() по сообщениям чата:
  от вопроса пользователя до записи ответа ассистента. Частые вопросы —
  lower(regexp_replace(content, …)) + GROUP BY, шаблоны — QUESTION_PUNCTUATION и QUESTION_SPACES.
"""
from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from datetime import datetime
from typing import cast
from uuid import UUID

from sqlalchemy import func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.domain import (Broadcast, BroadcastNotClaimedError, Chat, ChatMessage, ChatNotFoundError, ChatStats,
                             ChatStorageError, FeedbackResult, FeedbackValue, ModerationIncident, TopQuestion, UserActivity, utc_now)
from app.chat.repositories.pg_models import (BroadcastRow, ChatMessageRow, ChatRow, MessageFeedbackRow,
                                             ModerationIncidentRow)
from app.chat.repository import QUESTION_PUNCTUATION, QUESTION_SPACES
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

    # ------------------------------------------------------------------ блок 4.4
    async def get_message(self, chat_id: UUID, message_id: UUID) -> ChatMessage | None:
        async with self._transaction():
            row = await self.session.get(ChatMessageRow, message_id)
            if row is None or row.chat_id != chat_id:
                return None
            return ChatMessage.model_validate(row, from_attributes=True)

    async def add_feedback(self, chat_id: UUID, message_id: UUID, owner_external_id: str,
                           value: FeedbackValue) -> FeedbackResult:
        insert = (
            pg_insert(MessageFeedbackRow)
            .values(message_id=message_id, chat_id=chat_id, owner_external_id=owner_external_id, value=value,
                    created_at=utc_now())
            .on_conflict_do_nothing(constraint="uq_message_feedback_owner_message")
            .returning(MessageFeedbackRow.value)
        )
        existing = select(MessageFeedbackRow.value).where(MessageFeedbackRow.owner_external_id == owner_external_id,
                                                         MessageFeedbackRow.message_id == message_id)
        async with self._transaction():
            saved = await self.session.scalar(insert)
            if saved is not None:
                return FeedbackResult(saved=True, value=cast(FeedbackValue, saved))
            return FeedbackResult(saved=False, value=cast(FeedbackValue, await self.session.scalar(existing)))

    async def record_moderation(self, incident: ModerationIncident) -> None:
        async with self._transaction():
            self.session.add(ModerationIncidentRow(**incident.model_dump()))

    async def enqueue_broadcast(self, message: str, interface: str) -> Broadcast:
        row = BroadcastRow(message=message, interface=interface, status="pending", created_at=utc_now())
        async with self._transaction():
            self.session.add(row)
            await self.session.flush()
            return Broadcast.model_validate(row, from_attributes=True)

    async def claim_broadcast(self, stale_before: datetime, interface: str = "telegram") -> Broadcast | None:
        oldest = (
            select(BroadcastRow.id)
            .where(BroadcastRow.interface == interface,
                   or_(BroadcastRow.status == "pending",
                       (BroadcastRow.status == "sending") & (BroadcastRow.claimed_at < stale_before)))
            .order_by(BroadcastRow.created_at, BroadcastRow.id)
            .limit(1)
            .with_for_update(skip_locked=True)
            .scalar_subquery()
        )
        stmt = (update(BroadcastRow).where(BroadcastRow.id == oldest)
                .values(status="sending", claimed_at=utc_now()).returning(BroadcastRow))
        async with self._transaction():
            row = (await self.session.scalars(stmt, execution_options={"synchronize_session": False})).first()
            return Broadcast.model_validate(row, from_attributes=True) if row is not None else None

    async def finish_broadcast(self, broadcast_id: int, sent: int, failed: int) -> Broadcast | None:
        stmt = (update(BroadcastRow).where(BroadcastRow.id == broadcast_id, BroadcastRow.status == "sending")
                .values(status="sent" if sent or not failed else "failed", sent=sent, failed=failed,
                        recipients=sent + failed, finished_at=utc_now())
                .returning(BroadcastRow))
        async with self._transaction():
            row = (await self.session.scalars(stmt, execution_options={"synchronize_session": False})).first()
            if row is not None:
                return Broadcast.model_validate(row, from_attributes=True)
            status = await self.session.scalar(select(BroadcastRow.status).where(BroadcastRow.id == broadcast_id))
        if status is None:
            return None
        raise BroadcastNotClaimedError(broadcast_id, status)

    async def broadcast_recipients(self, interface: str) -> list[str]:
        stmt = (select(ChatRow.owner_external_id).where(ChatRow.interface == interface)
                .group_by(ChatRow.owner_external_id).order_by(func.min(ChatRow.created_at), ChatRow.owner_external_id))
        async with self._transaction():
            return list((await self.session.scalars(stmt)).all())

    async def stats(self, since: datetime, top_n: int = 5) -> ChatStats:
        params = {"since": since, "punct": QUESTION_PUNCTUATION, "spaces": QUESTION_SPACES, "top": top_n}
        async with self._transaction():
            total, users = (await self.session.execute(text(
                "SELECT count(*), count(*) FILTER (WHERE role = 'user') FROM chat_messages "
                "WHERE created_at >= :since"), params)).one()
            active = await self.session.scalar(text(
                "SELECT count(DISTINCT (c.interface, c.owner_external_id)) FROM chat_messages m "
                "JOIN chats c ON c.id = m.chat_id WHERE m.role = 'user' AND m.created_at >= :since"), params)
            latency = await self.session.scalar(text(
                "SELECT avg(EXTRACT(EPOCH FROM (created_at - prev_at)) * 1000) FROM ("
                "  SELECT role, created_at,"
                "         lag(role) OVER w AS prev_role, lag(created_at) OVER w AS prev_at"
                "  FROM chat_messages WINDOW w AS (PARTITION BY chat_id ORDER BY created_at, seq)"
                ") t WHERE role = 'assistant' AND prev_role = 'user' AND created_at >= :since"), params)
            blocks, input_blocks = (await self.session.execute(text(
                "SELECT count(*), count(*) FILTER (WHERE direction = 'input') FROM moderation_incidents "
                "WHERE created_at >= :since"), params)).one()
            up, down = (await self.session.execute(text(
                "SELECT count(*) FILTER (WHERE value = 'up'), count(*) FILTER (WHERE value = 'down') "
                "FROM message_feedback WHERE created_at >= :since"), params)).one()
            top = (await self.session.execute(text(
                "SELECT question, count(*) AS n FROM ("
                "  SELECT btrim(regexp_replace(regexp_replace(lower(content), :punct, ' ', 'g'), :spaces, ' ', 'g'))"
                "         AS question"
                "  FROM chat_messages WHERE role = 'user' AND created_at >= :since"
                ") q WHERE question <> '' GROUP BY question ORDER BY n DESC, question COLLATE \"C\" LIMIT :top"), params)).all()
        return ChatStats(since=since, total_messages=total, user_messages=users, active_users=active or 0,
                         avg_latency_ms=round(float(latency), 1) if latency is not None else None,
                         moderation_blocks=blocks, moderation_input_blocks=input_blocks,
                         feedback_up=up, feedback_down=down,
                         top_questions=[TopQuestion(question=q, count=n) for q, n in top])

    async def recent_users(self, limit: int = 50) -> list[UserActivity]:
        stmt = text(
            "SELECT c.owner_external_id, c.interface, count(DISTINCT c.id) AS chats,"
            "       greatest(max(c.created_at), coalesce(max(m.created_at), max(c.created_at))) AS last_seen_at"
            " FROM chats c LEFT JOIN chat_messages m ON m.chat_id = c.id"
            " GROUP BY c.owner_external_id, c.interface"
            " ORDER BY last_seen_at DESC, c.owner_external_id LIMIT :limit")
        async with self._transaction():
            rows = (await self.session.execute(stmt, {"limit": limit})).all()
        return [UserActivity(owner_external_id=o, interface=i, chats=n, last_seen_at=seen) for o, i, n, seen in rows]
