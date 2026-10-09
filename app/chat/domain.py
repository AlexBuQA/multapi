"""
Доменные модели чата (блок 4.1): Chat и ChatMessage на Pydantic v2.

Модели не знают про инфраструктуру — здесь только Pydantic и стандартная библиотека.
Хранилища (app/chat/repositories/) переводят их в свои форматы и обратно: JSONL —
через model_dump_json / model_validate_json, Postgres — через
model_validate(row, from_attributes=True).

Здесь же доменные ошибки: их поднимают репозитории и сервис, а app/main.py переводит в
HTTP-ответы 404, 413, 415, 422 и 503.

Медиа (блок 4.3): у сообщения пользователя с фото, голосом или документом есть media_refs —
MediaRef с готовым content-part для модели (part). content при этом — текстовая копия для
истории и интерфейса: подпись пользователя или пометка вида «[фото]». Модель получает и то,
и другое: [{"type": "text", "text": content}, part] — так фото видно и на следующих
репликах чата, без повторной загрузки.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, Field

Role = Literal["user", "assistant", "system"]
MediaKind = Literal["image", "audio", "document"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class MediaRef(BaseModel):
    """Вложение сообщения (блок 4.3). part — готовый content-part OpenAI Chat Completions из
    app/chat/media.py: {"type": "image_url", ...} для картинки, {"type": "text", ...} с
    расшифровкой голоса или текстом документа. filename — имя файла от клиента: в лог оно не
    пишется."""

    kind: MediaKind
    mime: str
    size: int = Field(ge=0)
    filename: str | None = None
    part: dict[str, Any]

    def summary(self) -> MediaInfo:
        return MediaInfo(kind=self.kind, mime=self.mime, size=self.size, filename=self.filename)


class MediaInfo(BaseModel):
    """Вложение в ответе GET /chats/{id}/messages: без part — в нём картинка целиком в base64."""

    kind: MediaKind
    mime: str
    size: int
    filename: str | None = None


class ChatMessage(BaseModel):
    """Одно сообщение диалога. tokens — длина в токенах (для ответа модели — по usage
    провайдера, для вложения — с текстом документа или оценкой картинки), None — не
    посчитано."""

    id: UUID = Field(default_factory=uuid4)
    chat_id: UUID
    role: Role
    content: str
    tokens: int | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    media_refs: MediaRef | None = None


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


class RequestError(Exception):
    """Запрос к /chats не выполнен по понятной причине (блок 4.3). status — HTTP-код ответа,
    code — код ошибки, message — текст для пользователя: бот показывает его как есть."""

    def __init__(self, status: int, code: str, message: str) -> None:
        self.status = status
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class MediaError(RequestError, ValueError):
    """Вложение не принято: тип не поддерживается, файл велик или не читается, нет модели для
    картинок или расшифровки голоса."""


class ChatStorageError(RuntimeError):
    """Хранилище истории недоступно: нет соединения с Postgres, нет таблиц, ошибка диска.
    В HTTP — 503 chat_storage_unavailable; причина — в __cause__ и в логе сервиса."""

    code = "chat_storage_unavailable"
    default_message = "Хранилище истории чатов недоступно. Попробуйте позже."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)
