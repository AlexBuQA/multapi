"""
Оценки ответов 👍/👎 (блок 4.4).

    POST /chats/{chat_id}/messages/{message_id}/feedback     {"value": "up" | "down"}
    -> 200 {"message_id", "value", "saved"}

message_id — id ответа ассистента из события done потока POST /chats/{id}/messages
({"type": "done", "message_id": "…"}): Telegram-бот кладёт его в callback_data кнопок
fb:up:<id> и fb:down:<id>. Оценивает владелец чата (owner_external_id); хранилище держит
UNIQUE (owner_external_id, message_id): повторная оценка не создаёт вторую строку и не
меняет первую — ответ 200 с saved=false и сохранённой оценкой. Так двойной клик или
повтор запроса ботом безопасны.

Ошибки: чата нет — 404 chat_not_found, сообщения в этом чате нет — 404 message_not_found,
оценивают вопрос, а не ответ — 422 not_an_answer.
"""
from __future__ import annotations

from typing import Literal
from uuid import UUID

import structlog
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.chat.deps import ChatServiceDep
from app.schemas.errors import ErrorResponse

router = APIRouter(prefix="/chats", tags=["feedback"])


class FeedbackIn(BaseModel):
    value: Literal["up", "down"] = Field(description="up — 👍, down — 👎", examples=["up"])


class FeedbackOut(BaseModel):
    message_id: UUID
    value: Literal["up", "down"] = Field(description="Сохранённая оценка (при повторе — первая)")
    saved: bool = Field(description="false — этот пользователь уже оценил ответ, оценка не изменилась")


@router.post(
    "/{chat_id}/messages/{message_id}/feedback",
    response_model=FeedbackOut,
    summary="Оценка ответа 👍/👎",
    responses={404: {"model": ErrorResponse, "description": "Нет чата или сообщения в нём"},
               422: {"model": ErrorResponse, "description": "Оценивают не ответ ассистента или value не up/down"},
               503: {"model": ErrorResponse, "description": "Хранилище недоступно"}},
)
async def save_feedback(chat_id: UUID, message_id: UUID, body: FeedbackIn, service: ChatServiceDep) -> FeedbackOut:
    structlog.contextvars.bind_contextvars(chat_id=str(chat_id))
    result = await service.add_feedback(chat_id, message_id, body.value)
    return FeedbackOut(message_id=message_id, value=result.value, saved=result.saved)
