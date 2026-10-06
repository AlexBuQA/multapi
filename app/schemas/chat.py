"""
Схемы чата (блок 3.4): запрос, ответ, кадр потока и usage.

ChatResponse не повторяет структуру ответа SDK: from_openai() переводит её в единую
модель сервиса. Её же сервис кладёт в кеш и отдаёт фронтенду.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role = Field(description="Автор сообщения: system — инструкция модели, user — пользователь, assistant — прошлый ответ модели")
    content: str = Field(min_length=1, max_length=32_000, description="Текст сообщения")


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "messages": [
                        {"role": "user", "content": "Как сбросить пароль от личного кабинета?"}
                    ]
                },
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Ты — ассистент техподдержки «Личного кабинета». "
                                       "Отвечай по-русски, коротко и по шагам.",
                        },
                        {"role": "user", "content": "Не приходит письмо для подтверждения email. Что делать?"},
                    ],
                    "temperature": 0,
                    "max_tokens": 300,
                    "user_id": "u-42",
                    "session_id": "s-0001",
                },
            ]
        }
    )

    messages: list[Message] = Field(
        min_length=1, max_length=50, description="История диалога, последнее сообщение — вопрос пользователя"
    )
    model: str | None = Field(
        default=None,
        min_length=1,
        description="Модель провайдера. Не задана — берётся LLM__DEFAULT_MODEL из настроек сервиса",
    )
    temperature: float = Field(default=0.3, ge=0, le=2, description="0 — самые предсказуемые ответы")
    max_tokens: int = Field(default=1024, ge=1, le=16_000, description="Предел длины ответа в токенах")
    user_id: str | None = Field(default=None, max_length=128, description="Кто спрашивает; в ключ кеша не входит")
    session_id: str | None = Field(default=None, max_length=128, description="Диалог; в ключ кеша не входит")

    @model_validator(mode="after")
    def _dialog_starts_correctly(self) -> ChatRequest:
        if self.messages[0].role == "assistant":
            raise ValueError("Диалог не может начинаться с сообщения assistant")
        return self


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_openai(cls, usage: Any) -> Usage:
        if usage is None:
            return cls()
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or 0) or prompt + completion
        return cls(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


class ChatResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "content": "Нажмите «Забыли пароль?» на странице входа и укажите email ...",
                    "model": "llama3.2",
                    "usage": {"prompt_tokens": 31, "completion_tokens": 58, "total_tokens": 89},
                    "finish_reason": "stop",
                    "cached": False,
                }
            ]
        }
    )

    content: str = Field(description="Текст ответа модели")
    model: str = Field(description="Модель, которая ответила (по данным провайдера)")
    usage: Usage
    finish_reason: str | None = Field(default=None, description="stop — ответ закончен, length — упёрся в max_tokens")
    cached: bool = Field(default=False, description="true — ответ взят из кеша Redis, модель не вызывалась")

    @classmethod
    def from_openai(cls, raw: Any) -> ChatResponse:
        """Ответ AsyncOpenAI (chat.completions.create) -> модель сервиса."""
        choice = raw.choices[0]
        return cls(
            content=choice.message.content or "",
            model=raw.model,
            usage=Usage.from_openai(raw.usage),
            finish_reason=choice.finish_reason,
        )


class ChatDelta(BaseModel):
    """Кадр потока /chat/stream: либо фрагмент текста, либо итоговый usage."""

    content: str | None = None
    usage: Usage | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> ChatDelta:
        if (self.content is None) == (self.usage is None):
            raise ValueError("ChatDelta: нужно ровно одно поле — content или usage")
        return self
