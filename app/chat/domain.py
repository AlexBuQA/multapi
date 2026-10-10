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

Production-обвязка (блок 4.4): оценки ответов (Feedback), инциденты модерации
(ModerationIncident), очередь рассылки (Broadcast) и сводки для /chats/admin/* (ChatStats,
UserActivity) — их тоже хранят репозитории (OpsRepository в app/chat/repository.py).
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


FeedbackValue = Literal["up", "down"]
BroadcastStatus = Literal["pending", "sending", "sent", "failed"]


class FeedbackResult(BaseModel):
    """Итог оценки ответа: saved=False — этот пользователь уже оценил ответ, value — его оценка."""

    saved: bool
    value: FeedbackValue


class ModerationIncident(BaseModel):
    """Заблокированный вопрос или ответ (блок 4.4) — без текста: только отпечаток."""

    chat_id: UUID | None = None
    direction: Literal["input", "output"]
    blocked_by: str
    categories: list[str] = Field(default_factory=list)
    text_hash: str
    created_at: AwareDatetime = Field(default_factory=utc_now)


class Broadcast(BaseModel):
    """Рассылка из POST /chats/admin/broadcast. pending — ждёт бота, sending — бот рассылает,
    sent / failed — готово (failed — ни одно сообщение не дошло)."""

    id: int
    message: str
    interface: str
    status: BroadcastStatus = "pending"
    recipients: int | None = None
    sent: int | None = None
    failed: int | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    claimed_at: AwareDatetime | None = None
    finished_at: AwareDatetime | None = None


class UserActivity(BaseModel):
    owner_external_id: str
    interface: str
    chats: int
    last_seen_at: AwareDatetime


class TopQuestion(BaseModel):
    question: str
    count: int


class ChatStats(BaseModel):
    """Сводка за период для GET /chats/admin/stats; определения — в app/admin/routes.py."""

    since: AwareDatetime
    total_messages: int = 0
    user_messages: int = 0
    active_users: int = 0
    avg_latency_ms: float | None = None
    moderation_blocks: int = 0
    moderation_input_blocks: int = 0
    feedback_up: int = 0
    feedback_down: int = 0
    top_questions: list[TopQuestion] = Field(default_factory=list)

    @property
    def moderation_block_rate(self) -> float:
        """Доля вопросов, закончившихся блокировкой: блокировки / (принятые вопросы + отклонённые)."""
        asked = self.user_messages + self.moderation_input_blocks
        return round(self.moderation_blocks / asked, 4) if asked else 0.0

    @property
    def feedback_up_ratio(self) -> float | None:
        votes = self.feedback_up + self.feedback_down
        return round(self.feedback_up / votes, 4) if votes else None


class BroadcastNotClaimedError(Exception):
    """Итог рассылки, которую сейчас никто не отправляет (не забрана или уже завершена).
    В HTTP — 409 broadcast_not_sending."""

    code = "broadcast_not_sending"

    def __init__(self, broadcast_id: int, status: str) -> None:
        self.broadcast_id, self.status = broadcast_id, status
        super().__init__(f"рассылка {broadcast_id} в статусе {status}")


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
