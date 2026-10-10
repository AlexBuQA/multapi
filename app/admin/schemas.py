"""Схемы admin API (блок 4.4)."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.chat.domain import Broadcast, ChatStats


class TopQuestionOut(BaseModel):
    question: str = Field(description="Вопрос в нижнем регистре, без знаков препинания")
    count: int


class StatsOut(BaseModel):
    period_hours: int
    since: datetime
    total_messages: int = Field(description="Сообщения пользователей и ответы за период, и скрытые /clear")
    active_users: int = Field(description="DAU: клиенты (owner_external_id + interface), задавшие вопрос")
    avg_latency_ms: float | None = Field(description="Средняя задержка ответа: от вопроса до записи ответа")
    moderation_blocks: int = Field(description="Заблокированные вопросы и ответы")
    moderation_block_rate: float = Field(description="Блокировки / (принятые вопросы + отклонённые модерацией)")
    feedback_votes: int
    feedback_up_ratio: float | None = Field(description="👍 / (👍 + 👎); null — оценок не было")
    top_questions: list[TopQuestionOut]

    @classmethod
    def of(cls, stats: ChatStats, hours: int) -> StatsOut:
        return cls(period_hours=hours, since=stats.since, total_messages=stats.total_messages,
                   active_users=stats.active_users, avg_latency_ms=stats.avg_latency_ms,
                   moderation_blocks=stats.moderation_blocks, moderation_block_rate=stats.moderation_block_rate,
                   feedback_votes=stats.feedback_up + stats.feedback_down, feedback_up_ratio=stats.feedback_up_ratio,
                   top_questions=[TopQuestionOut(question=q.question, count=q.count) for q in stats.top_questions])


class UserOut(BaseModel):
    owner_external_id: str
    interface: str
    chats: int
    last_seen_at: datetime


class BroadcastIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000, examples=["Сегодня с 23:00 до 23:30 — плановые работы."])
    interface_filter: str = Field(default="telegram", pattern=r"^[a-z][a-z0-9_-]*$", max_length=32,
                                  description="Кому: owner_external_id всех чатов этого интерфейса")

    @field_validator("message")
    @classmethod
    def _text(cls, value: str) -> str:
        if "\x00" in value or not value.strip():
            raise ValueError("пустой текст или символ NUL")
        return value.strip()


class BroadcastOut(BaseModel):
    id: int
    status: Literal["pending", "sending", "sent", "failed"]
    interface: str
    recipients: int | None = Field(description="Сколько получателей (для pending — сколько их сейчас)")
    sent: int | None = None
    failed: int | None = None
    created_at: datetime
    finished_at: datetime | None = None

    @classmethod
    def of(cls, item: Broadcast, recipients: int | None = None) -> BroadcastOut:
        return cls(id=item.id, status=item.status, interface=item.interface,
                   recipients=item.recipients if item.recipients is not None else recipients,
                   sent=item.sent, failed=item.failed, created_at=item.created_at, finished_at=item.finished_at)


class BroadcastClaimOut(BaseModel):
    id: int
    message: str
    interface: str
    recipients: list[str] = Field(description="owner_external_id получателей; для telegram — chat.id")


class BroadcastResultIn(BaseModel):
    sent: int = Field(ge=0)
    failed: int = Field(ge=0)
