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

Миграции — migrations/versions/ (alembic revision --autogenerate): *_chat_tables.py —
таблицы блока 4.1, *_chat_owner_index.py — индекс блока 4.2, *_message_media_refs.py — столбец
media_refs блока 4.3.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from typing import Any

from sqlalchemy import BigInteger, DateTime, ForeignKey, Identity, Index, Integer, Text, Uuid, func
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
