"""
ORM-модели Postgres для истории чатов (блок 4.1): ChatRow и ChatMessageRow.

Это отдельные классы, а не доменные Chat и ChatMessage: наружу модуля
app/chat/repositories/ они не выходят. Граница — ChatMessage.model_validate(row,
from_attributes=True) в pg_repo.py.

Схема — как в задании, плюс один столбец:
- chats — метаданные чата;
- chat_messages — сообщения; deleted_at — мягкое удаление (/clear), строки не стираются;
- частичный индекс ix_chat_messages_chat_created (chat_id, created_at DESC) WHERE
  deleted_at IS NULL — под запрос «последние N живых сообщений чата»;
- ix_chats_owner_interface (owner_external_id, interface) — блок 4.2: поиск чата
  клиента в get_or_create_chat (POST /chats идемпотентен). Не уникальный: в базе могут
  быть дубли, созданные до блока 4.2;
- media_refs (блок 4.3) — вложение сообщения в JSONB: тип, MIME, размер, имя файла и готовый
  content-part (у картинки — base64 целиком). NULL у сообщений без вложения;
- seq — добавлен к схеме задания: порядковый номер вставки (IDENTITY). created_at задаёт
  приложение, и у двух сообщений он может совпасть: на Windows до Python 3.13 часы
  datetime.now() идут шагом около 15 мс. Тогда порядок по created_at не определён, а
  ORDER BY created_at DESC, seq DESC сохраняет порядок записи.

Блок 4.4 — три таблицы production-обвязки:
- message_feedback — оценки ответов 👍/👎: UNIQUE (owner_external_id, message_id), повторная
  оценка того же ответа тем же клиентом не вставляется (INSERT … ON CONFLICT DO NOTHING);
- broadcast_queue — рассылки из /chats/admin/broadcast; бот забирает их по одной
  (SELECT … FOR UPDATE SKIP LOCKED) и отчитывается, сколько дошло;
- moderation_incidents — заблокированные вопросы и ответы: направление, слой, категории и
  sha256 текста. Самого текста нет.

Миграции — migrations/versions/ (alembic revision --autogenerate): *_chat_tables.py —
таблицы блока 4.1, *_chat_owner_index.py — индекс блока 4.2, *_message_media_refs.py — столбец
media_refs блока 4.3, *_production_tables.py — таблицы блока 4.4.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from typing import Any

from sqlalchemy import (BigInteger, CheckConstraint, DateTime, ForeignKey, Identity, Index, Integer, Text,
                        UniqueConstraint, Uuid, func)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ChatRow(Base):
    __tablename__ = "chats"
    __table_args__ = (Index("ix_chats_owner_interface", "owner_external_id", "interface"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    owner_external_id: Mapped[str] = mapped_column(Text, nullable=False)
    interface: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class ChatMessageRow(Base):
    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    chat_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    seq: Mapped[int] = mapped_column(BigInteger, Identity(always=True), nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # none_as_null: у сообщения без вложения — SQL NULL, а не JSON-значение null.
    media_refs: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True), nullable=True)


Index(
    "ix_chat_messages_chat_created",
    ChatMessageRow.chat_id,
    ChatMessageRow.created_at.desc(),
    postgresql_where=ChatMessageRow.deleted_at.is_(None),
)


class MessageFeedbackRow(Base):
    __tablename__ = "message_feedback"
    __table_args__ = (
        UniqueConstraint("owner_external_id", "message_id", name="uq_message_feedback_owner_message"),
        CheckConstraint("value IN ('up', 'down')", name="ck_message_feedback_value"),
        Index("ix_message_feedback_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    message_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("chat_messages.id", ondelete="CASCADE"),
                                                  nullable=False)
    chat_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("chats.id", ondelete="CASCADE"), nullable=False)
    owner_external_id: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class BroadcastRow(Base):
    __tablename__ = "broadcast_queue"
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'sending', 'sent', 'failed')", name="ck_broadcast_queue_status"),
        Index("ix_broadcast_queue_status_created", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    interface: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    recipients: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sent: Mapped[int | None] = mapped_column(Integer, nullable=True)
    failed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ModerationIncidentRow(Base):
    __tablename__ = "moderation_incidents"
    __table_args__ = (Index("ix_moderation_incidents_created", "created_at"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    # Чат удалён — инцидент остаётся в статистике.
    chat_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("chats.id", ondelete="SET NULL"),
                                                      nullable=True)
    direction: Mapped[str] = mapped_column(Text, nullable=False)
    blocked_by: Mapped[str] = mapped_column(Text, nullable=False)
    categories: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    text_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
