"""
Эндпоинты чатов (блок 4.1). Контракт /chats/{id}/messages стабилен на весь курс: поверх
него работают Telegram-бот (M4Б2), RAG (M5) и инструменты агентов (M6).

    POST   /chats                       создать чат -> {"chat_id": ...}
    GET    /chats/{chat_id}             метаданные чата (404, если нет)
    POST   /chats/{chat_id}/messages    вопрос -> ответ потоком SSE
    GET    /chats/{chat_id}/messages    история, от старых к новым (?limit=50)
    DELETE /chats/{chat_id}/messages    очистить историю (мягкое удаление)

Поток POST /chats/{chat_id}/messages — text/event-stream:
    data: <фрагмент ответа>\\n\\n      по мере генерации
    data: [DONE]\\n\\n                 конец ответа
Фрагмент с переводами строк передаётся несколькими строками data: одного события — так
требует формат SSE; клиент склеивает их через \\n. Ошибка до первого фрагмента (чата нет,
провайдер недоступен, 429) — обычный JSON-ответ с кодом 404/429/502/503/504. Ошибка
посреди ответа — событие error с JSON {"error": {"code", "message"}}, после него [DONE]
не приходит.

Сервис чата приходит через Depends(get_chat_service); глобальных сервисов в модуле нет.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.chat.deps import ChatServiceDep
from app.chat.domain import Chat, ChatMessage, ChatStorageError
from app.core.exceptions import LLMError
from app.observability.logging import get_logger
from app.schemas.errors import LLM_ERROR_RESPONSES, ErrorResponse

router = APIRouter(prefix="/chats", tags=["chats"])
log = get_logger()

DONE = "data: [DONE]\n\n"
NOT_FOUND = {404: {"model": ErrorResponse, "description": "Чата с таким id нет (chat_not_found)"}}
STORAGE = {503: {"model": ErrorResponse, "description": "Хранилище истории недоступно (chat_storage_unavailable)"}}


def no_nul(value: str | None) -> str | None:
    """Символ NUL (\\u0000) Postgres в тексте не хранит — без этой проверки вместо 422 был
    бы 503, и хранилища вели бы себя по-разному."""
    if value is not None and "\x00" in value:
        raise ValueError("символ NUL (\\u0000) в тексте не допускается")
    return value


class CreateChatIn(BaseModel):
    owner_external_id: str = Field(min_length=1, max_length=256, examples=["test-1"],
                                   description="Клиент во внешней системе: Telegram chat.id, email, UUID устройства")
    interface: str = Field(min_length=1, max_length=32, pattern=r"^[a-z][a-z0-9_-]*$", examples=["cli"],
                           description="Откуда клиент: telegram, web, cli")
    system_prompt: str | None = Field(default=None, max_length=8000,
                                      description="Свой системный промпт чата; не задан — CHAT_SYSTEM_PROMPT сервиса")

    _no_nul = field_validator("owner_external_id", "system_prompt")(no_nul)


class CreateChatOut(BaseModel):
    chat_id: UUID


class MessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=32_000, examples=["Привет, меня зовут Аня"])

    _no_nul = field_validator("content")(no_nul)


class StatusOut(BaseModel):
    status: str = "ok"


def sse_data(chunk: str) -> str:
    """Одно событие SSE. Переводы строк внутри фрагмента — отдельными строками data:,
    иначе пустая строка в ответе модели закончила бы событие раньше времени."""
    lines = chunk.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "".join(f"data: {line}\n" for line in lines) + "\n"


def sse_error(code: str, message: str) -> str:
    payload = {"error": {"code": code, "message": message}}
    return f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _events(first: str | None, chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    try:
        if first is not None:
            yield sse_data(first)
        async for chunk in chunks:
            yield sse_data(chunk)
    except (LLMError, ChatStorageError) as exc:
        # Заголовки с кодом 200 уже ушли: об ошибке посреди ответа сообщаем событием error.
        yield sse_error(exc.code, exc.message)
        return
    except Exception:  # noqa: BLE001 — непредвиденное: клиент всё равно узнаёт, что ответ оборван
        log.exception("chat_stream_failed")
        yield sse_error("internal_error", "Внутренняя ошибка сервиса.")
        return
    finally:
        await chunks.aclose()   # type: ignore[attr-defined]
    yield DONE


@router.post("", response_model=CreateChatOut, summary="Создать чат", responses=STORAGE)
async def create_chat(body: CreateChatIn, service: ChatServiceDep) -> CreateChatOut:
    chat = await service.create_chat(body.owner_external_id, body.interface, body.system_prompt)
    return CreateChatOut(chat_id=chat.id)


@router.get("/{chat_id}", response_model=Chat, summary="Метаданные чата", responses={**NOT_FOUND, **STORAGE})
async def get_chat(chat_id: UUID, service: ChatServiceDep) -> Chat:
    return await service.get_chat(chat_id)


@router.post(
    "/{chat_id}/messages",
    response_class=StreamingResponse,
    summary="Сообщение в чат — ответ потоком (SSE)",
    description=(
        "Сохраняет вопрос, собирает контекст из истории чата (последние CHAT_CONTEXT_WINDOW "
        "сообщений в пределах бюджета токенов) и отдаёт ответ модели по мере генерации: "
        "`data: <фрагмент>`, в конце `data: [DONE]`. Ответ сохраняется в историю целиком. "
        "Ошибка посреди ответа — событие `error`. В Swagger поток виден только после "
        "завершения — по кускам его показывает `curl -N`."
    ),
    responses={
        200: {"description": "Поток событий SSE",
              "content": {"text/event-stream": {"example": "data: Вас зовут\n\ndata:  Аня.\n\ndata: [DONE]\n\n"}}},
        **NOT_FOUND, **STORAGE, **LLM_ERROR_RESPONSES,
    },
)
async def send_message(chat_id: UUID, body: MessageIn, service: ChatServiceDep) -> StreamingResponse:
    structlog.contextvars.bind_contextvars(chat_id=str(chat_id))
    chunks = service.send_message(chat_id, body.content)
    # Первый фрагмент — до ответа клиенту: «чата нет», недоступная база или провайдер
    # возвращаются обычным JSON с нужным HTTP-кодом, а не событием посреди потока.
    try:
        first: str | None = await anext(chunks)
    except StopAsyncIteration:
        first = None
    return StreamingResponse(
        _events(first, chunks),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{chat_id}/messages", response_model=list[ChatMessage], summary="История чата",
            responses={**NOT_FOUND, **STORAGE})
async def list_messages(chat_id: UUID, service: ChatServiceDep,
                        limit: Annotated[int, Query(ge=1, le=500)] = 50) -> list[ChatMessage]:
    return await service.list_messages(chat_id, limit=limit)


@router.delete("/{chat_id}/messages", response_model=StatusOut, summary="Очистить историю (/clear)",
               responses={**NOT_FOUND, **STORAGE})
async def clear_history(chat_id: UUID, service: ChatServiceDep) -> StatusOut:
    structlog.contextvars.bind_contextvars(chat_id=str(chat_id))
    await service.clear_history(chat_id)
    return StatusOut()
