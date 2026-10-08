"""
Доменные модели чата (блок 4.1): Chat и ChatMessage на Pydantic v2.

Модели не знают про инфраструктуру — здесь только Pydantic и стандартная библиотека.
Хранилища (app/chat/repositories/) переводят их в свои форматы и обратно: JSONL —
через model_dump_json / model_validate_json, Postgres — через
model_validate(row, from_attributes=True).

Здесь же доменные ошибки: их поднимают репозитории и сервис, а app/main.py переводит в
HTTP-ответы 404 и 503.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, Field

Role = Literal["user", "assistant", "system"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class ChatMessage(BaseModel):
    """Одно сообщение диалога. tokens — длина текста в токенах (для ответа модели — по
    usage провайдера), None — не посчитано."""

    id: UUID = Field(default_factory=uuid4)
    chat_id: UUID
    role: Role
    content: str
    tokens: int | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)


class Chat(BaseModel):
    """Диалог. owner_external_id — идентификатор клиента во внешней системе: chat.id в
    Telegram (строкой), email пользователя веб-интерфейса, UUID мобильного устройства.
    interface — откуда пришёл клиент: telegram, web, cli."""

    id: UUID = Field(default_factory=uuid4)
    owner_external_id: str
    interface: str
    system_prompt: str | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)


class ChatNotFoundError(LookupError):
    """Чата с таким id нет. В HTTP — 404 chat_not_found."""

    code = "chat_not_found"

    def __init__(self, chat_id: UUID) -> None:
        self.chat_id = chat_id
        super().__init__(f"Чат {chat_id} не найден.")


class ChatInputError(ValueError):
    """Данные чата не прошли проверку сервиса (например, системный промпт чата — проверку
    входа блока 3.8). В HTTP — 422 validation_error с полем."""

    code = "validation_error"

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


class ChatStorageError(RuntimeError):
    """Хранилище истории недоступно: нет соединения с Postgres, нет таблиц, ошибка диска.
    В HTTP — 503 chat_storage_unavailable; причина — в __cause__ и в логе сервиса."""

    code = "chat_storage_unavailable"
    default_message = "Хранилище истории чатов недоступно. Попробуйте позже."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)
