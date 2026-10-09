"""
Схемы чата (блок 3.4): запрос, ответ, кадр потока и usage.

ChatResponse не повторяет структуру ответа SDK: from_openai() переводит её в единую
модель сервиса. Её же сервис кладёт в кеш и отдаёт фронтенду.

repr и str сообщения (блок 3.7) маскируют персональные данные: модель запроса попадает
в отладчик, трейсбеки и сообщения об ошибках, а сырой текст пользователя туда попадать
не должен. model_dump() возвращает текст как есть — он нужен модели.

Мультимодальные сообщения (блок 4.3) — MediaMessage и MediaChatRequest: content — строка
или список content-part OpenAI ({"type": "text"} и {"type": "image_url"}). Их собирает
ChatService для чатов (/chats); публичный POST /chat по-прежнему принимает только текст —
его схема ChatRequest не меняется. У text-части есть служебная пометка media ("audio" —
расшифровка голоса, "document" — текст PDF/DOCX): по ней защитный слой и бюджет контекста
отличают вложение от того, что пользователь напечатал сам. Провайдеру пометка не уходит —
provider_messages() её убирает.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.observability.pii import redact_pii

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role = Field(description="Автор сообщения: system — инструкция модели, user — пользователь, assistant — прошлый ответ модели")
    content: str = Field(min_length=1, max_length=32_000, description="Текст сообщения")

    def __repr_args__(self) -> Iterator[tuple[str | None, Any]]:
        # repr/str: email, телефон, карта, ИНН и паспорт — плейсхолдерами
        for name, value in super().__repr_args__():
            yield name, redact_pii(value) if name == "content" and isinstance(value, str) else value


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str = Field(max_length=40_000)
    media: Literal["audio", "document"] | None = Field(
        default=None, description="Служебная пометка сервиса: текст вложения (провайдеру не уходит)")

    def __repr_args__(self) -> Iterator[tuple[str | None, Any]]:
        for name, value in super().__repr_args__():
            yield name, redact_pii(value) if name == "text" else value


class ImageURL(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = None

    def __repr_args__(self) -> Iterator[tuple[str | None, Any]]:
        # data:-URL картинки — сотни тысяч символов base64: в repr и трейсбеках только начало.
        for name, value in super().__repr_args__():
            yield name, (value[:40] + f"…({len(value)} симв.)") if name == "url" and len(value) > 60 else value


class ImagePart(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]
Content = Annotated[str, Field(min_length=1, max_length=32_000)] | Annotated[
    list[ContentPart], Field(min_length=1, max_length=8)]


class MediaMessage(Message):
    """Сообщение с вложением (блок 4.3): content — текст или список content-part."""

    content: Content = Field(description="Текст или content-part: text, image_url")  # type: ignore[assignment]


def _part(part: Any) -> dict[str, Any]:
    return part if isinstance(part, Mapping) else part.model_dump()


def content_parts(content: str | Sequence[Any]) -> list[dict[str, Any]]:
    """content как список content-part (строка — одна text-часть)."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [_part(p) for p in content]


def content_text(content: str | Sequence[Any], *, typed_only: bool = False) -> str:
    """Текст сообщения: text-части через пустую строку. typed_only — без текста вложений
    (только то, что пользователь напечатал сам)."""
    if isinstance(content, str):
        return content
    return "\n\n".join(p["text"] for p in content_parts(content)
                       if p.get("type") == "text" and not (typed_only and p.get("media")))


def has_image(content: str | Sequence[Any]) -> bool:
    return not isinstance(content, str) and any(p.get("type") == "image_url" for p in content_parts(content))


def map_text(content: str | Sequence[Any], fn: Callable[[str], str]) -> str | list[dict[str, Any]]:
    """fn к каждой text-части; картинки — как есть."""
    if isinstance(content, str):
        return fn(content)
    return [{**p, "text": fn(p["text"])} if p.get("type") == "text" else p for p in content_parts(content)]


def provider_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Сообщения в формате OpenAI Chat Completions: без служебной пометки media и пустых полей."""
    result = []
    for message in messages:
        content = message["content"]
        if not isinstance(content, str):
            parts = []
            for p in content_parts(content):
                if p.get("type") == "image_url":
                    image = {k: v for k, v in dict(p["image_url"]).items() if v is not None}
                    parts.append({"type": "image_url", "image_url": image})
                else:
                    parts.append({"type": "text", "text": p["text"]})
            content = parts
        result.append({**message, "content": content})
    return result


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


class MediaChatRequest(ChatRequest):
    """Запрос ChatService к модели (блок 4.3): сообщения могут быть мультимодальными."""

    messages: list[MediaMessage] = Field(min_length=1, max_length=50)  # type: ignore[assignment]


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
