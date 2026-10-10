"""
Эндпоинты чатов (блок 4.1). Контракт /chats/{id}/messages стабилен на весь курс: поверх
него работают Telegram-бот (M4Б2), RAG (M5) и инструменты агентов (M6).

    POST   /chats                       чат клиента: найти или создать -> {"chat_id", "created"}
                                        (блок 4.2: идемпотентен по owner_external_id + interface)
    GET    /chats/{chat_id}             метаданные чата (404, если нет)
    POST   /chats/{chat_id}/messages    вопрос -> ответ потоком SSE; multipart/form-data:
                                        content (текст) и необязательный файл media (блок 4.3)
    GET    /chats/{chat_id}/messages    история, от старых к новым (?limit=50)
    DELETE /chats/{chat_id}/messages    очистить историю (мягкое удаление)
    POST   /chats/{chat_id}/system-message   сообщение от системы в чат и, по желанию,
                                        уведомление в Telegram (блок 4.3; X-Internal-Token)

Тело POST /chats/{chat_id}/messages с блока 4.3 — форма (multipart/form-data, без файла
подойдёт и application/x-www-form-urlencoded): content — текст вопроса или подпись к файлу,
обязателен; media — необязательный файл: фото, голос, PDF или DOCX (app/chat/media.py);
user_name — необязательное имя, как обращаться к пользователю, пока он не представился.
Отдельного /messages/with-media нет. JSON {"content": ...} блока 4.1 получает 415 с
подсказкой. Ошибки вложения — до начала потока, обычным JSON: 413 файл велик, 415 тип не
поддерживается, 422 файл не читается или фото некому показать, 502/503/504 — расшифровка
голоса.

Поток POST /chats/{chat_id}/messages — text/event-stream, в каждом событии одна строка
data: с JSON (блок 4.3; до него фрагменты шли текстом и в конце data: [DONE]):
    data: {"type": "token", "delta": "<фрагмент>"}\\n\\n      по мере генерации
    data: {"type": "moderation", "code": "moderation_blocked", "categories": [...],
           "message": "Не могу показать ответ — он мог нарушить правила."}\\n\\n
                                                        ответ не прошёл модерацию (блок 4.4):
                                                        показанный текст заменить на message
    data: {"type": "done", "message_id": "<uuid>"}\\n\\n    конец ответа; message_id — id
                                                        сохранённого ответа для оценки (4.4)
    data: {"type": "error", "code": "...", "message": "..."}\\n\\n   ошибка посреди ответа
Переводы строк внутри фрагмента экранирует JSON, поэтому событие всегда в одну строку.
Ошибка до первого фрагмента (чата нет, провайдер недоступен, 429) — обычный JSON-ответ с
кодом 404/429/502/503/504; вопрос не прошёл модерацию — 403 с detail.code =
moderation_blocked (блок 4.4). После события error события done нет. Клиент ушёл, не дождавшись
первого фрагмента (бот получил ReadTimeout), — запрос к модели обрывается, замок чата
отпускается, в лог — chat_client_gone, ответ 499 (его уже некому читать).

Сервис чата приходит через Depends(get_chat_service); глобальных сервисов в модуле нет.
"""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from typing import Annotated, Any, TypeVar
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, File, Form, Header, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.chat.deps import AudioClientDep, ChatServiceDep, SettingsDep
from app.chat.domain import Chat, ChatInputError, ChatMessage, ChatStorageError, MediaError, MediaInfo, \
    RequestError, Role
from app.chat.media import normalize_mime, read_media
from app.chat.service import AnswerReplaced, AnswerSaved, StreamItem, clean_user_name
from app.core.exceptions import LLMError
from app.observability.logging import get_logger
from app.schemas.errors import LLM_ERROR_RESPONSES, ErrorResponse
from app.services.notifier import NotifyError, notify_user

router = APIRouter(prefix="/chats", tags=["chats"])
log = get_logger()

def sse_event(payload: dict[str, Any]) -> str:
    """Одно событие SSE: JSON в одной строке data:. ensure_ascii=False — кириллица как есть."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


DONE = sse_event({"type": "done"})
NOT_FOUND = {404: {"model": ErrorResponse, "description": "Чата с таким id нет (chat_not_found)"}}
STORAGE = {503: {"model": ErrorResponse, "description": "Хранилище истории недоступно (chat_storage_unavailable)"}}
MEDIA = {
    413: {"model": ErrorResponse, "description": "Файл больше предела MEDIA__MAX_*_BYTES (media_too_large)"},
    415: {"model": ErrorResponse, "description": "Тело не форма или тип файла не поддерживается"},
    422: {"model": ErrorResponse, "description": "Пустое сообщение, файл не читается, фото некому показать"},
}
FORM_TYPES = {"multipart/form-data", "application/x-www-form-urlencoded"}
DISCONNECT_POLL = 0.5     # с: как часто, пока модель молчит, проверять, ждёт ли ещё клиент
CLIENT_CLOSED = 499       # как у nginx: клиент закрыл соединение, ответ некому отдать
T = TypeVar("T")


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
    created: bool = Field(description="true — чат создан этим запросом, false — вернулся существующий")


class MessageOut(BaseModel):
    """Сообщение истории. Вложение — без part: в нём картинка целиком в base64."""

    id: UUID
    chat_id: UUID
    role: Role
    content: str
    tokens: int | None = None
    created_at: datetime
    media: MediaInfo | None = None

    @classmethod
    def of(cls, message: ChatMessage) -> MessageOut:
        return cls(**message.model_dump(exclude={"media_refs"}),
                   media=message.media_refs.summary() if message.media_refs else None)


class SystemMessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=4096, examples=["Ваша заявка №123 решена: доступ восстановлен."])
    notify: bool = Field(default=False, description="Отправить текст пользователю в Telegram (чаты interface=telegram)")

    _no_nul = field_validator("text")(no_nul)


class SystemMessageOut(BaseModel):
    message_id: UUID
    notified: bool = Field(description="true — бот принял уведомление и отправил его в Telegram")
    detail: str | None = Field(default=None, description="Почему уведомление не отправлено")


class StatusOut(BaseModel):
    status: str = "ok"


async def require_form(request: Request) -> None:
    """Тело POST /chats/{id}/messages — форма. Зависимость выполняется до разбора полей формы,
    поэтому JSON блока 4.1 получает понятный 415, а не 422 «content: Field required»."""
    if normalize_mime(request.headers.get("content-type")) not in FORM_TYPES:
        raise MediaError(415, "unsupported_content_type",
                         "Тело запроса — форма multipart/form-data: поле content (текст) и необязательный файл media. "
                         "JSON с блока 4.3 не принимается: curl -F content=Привет …")


def require_internal_token(settings: SettingsDep,
                           x_internal_token: Annotated[str | None, Header()] = None) -> None:
    """Служебные запросы (блок 4.3) — только с X-Internal-Token, равным INTERNAL_TOKEN из .env."""
    if settings.internal_token is None:
        raise RequestError(503, "internal_token_not_configured",
                           "Служебные запросы выключены: в .env сервиса не задан INTERNAL_TOKEN.")
    expected = settings.internal_token.get_secret_value().encode()
    if x_internal_token is None or not hmac.compare_digest(x_internal_token.encode(), expected):
        raise RequestError(401, "unauthorized", "Нужен заголовок X-Internal-Token с INTERNAL_TOKEN сервиса.")


def sse_token(delta: str) -> str:
    return sse_event({"type": "token", "delta": delta})


def sse_item(item: StreamItem) -> str:
    """Событие потока ChatService.stream_message -> строка SSE."""
    if isinstance(item, AnswerSaved):
        return sse_event({"type": "done", "message_id": str(item.message_id) if item.message_id else None})
    if isinstance(item, AnswerReplaced):
        return sse_event({"type": "moderation", "code": "moderation_blocked", "categories": item.categories,
                          "message": item.text})
    return sse_token(item)


def sse_error(code: str, message: str) -> str:
    return sse_event({"type": "error", "code": code, "message": message})


class ClientGone(Exception):
    """Клиент закрыл соединение, не дождавшись первого фрагмента ответа."""


async def first_chunk(chunks: AsyncIterator[T], is_disconnected: Callable[[], Awaitable[bool]],
                      poll: float | None = None) -> T | None:
    """Первый фрагмент ответа, пока клиент ещё ждёт; None — ответ пустой.

    До первого фрагмента StreamingResponse ещё нет, и разрыв соединения слушать некому.
    Картинку модель на CPU разбирает минутами: бот уходит по ReadTimeout, а запрос к модели
    идёт дальше и держит замок чата — /clear и следующий вопрос ждут его конца. Поэтому раз
    в poll секунд проверяем соединение; клиент ушёл — генератор отменяется (запрос к модели
    обрывается, замок отпускается) и поднимается ClientGone.
    Ошибки генератора, пока клиент на связи, поднимаются как есть."""
    async def step() -> T | None:
        try:
            return await anext(chunks)
        except StopAsyncIteration:
            return None

    task = asyncio.create_task(step())
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=DISCONNECT_POLL if poll is None else poll)
            if not task.done() and await is_disconnected():
                raise ClientGone
        return task.result()
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):   # ответ уже некому отдать
                await task


async def _events(first: StreamItem | None, chunks: AsyncIterator[StreamItem]) -> AsyncIterator[str]:
    done = False
    try:
        if first is not None:
            done = isinstance(first, AnswerSaved)
            yield sse_item(first)
        async for item in chunks:
            done = done or isinstance(item, AnswerSaved)
            yield sse_item(item)
    except (LLMError, ChatStorageError) as exc:
        # Заголовки с кодом 200 уже ушли: об ошибке посреди ответа сообщаем событием error.
        yield sse_error(exc.code, exc.message)
        return
    except Exception:  # непредвиденное: клиент всё равно узнаёт, что ответ оборван
        log.exception("chat_stream_failed")
        yield sse_error("internal_error", "Внутренняя ошибка сервиса.")
        return
    finally:
        await chunks.aclose()   # type: ignore[attr-defined]
    if not done:                # поток без AnswerSaved (не ChatService) — done без message_id
        yield DONE


@router.post("", response_model=CreateChatOut, summary="Чат клиента: найти или создать", responses=STORAGE)
async def create_chat(body: CreateChatIn, service: ChatServiceDep) -> CreateChatOut:
    """Идемпотентен (блок 4.2): повторный запрос с теми же owner_external_id и interface
    возвращает тот же chat_id и created=false. system_prompt учитывается только при
    создании чата."""
    chat, created = await service.get_or_create_chat(body.owner_external_id, body.interface, body.system_prompt)
    return CreateChatOut(chat_id=chat.id, created=created)


@router.get("/{chat_id}", response_model=Chat, summary="Метаданные чата", responses={**NOT_FOUND, **STORAGE})
async def get_chat(chat_id: UUID, service: ChatServiceDep) -> Chat:
    return await service.get_chat(chat_id)


@router.post(
    "/{chat_id}/messages",
    response_class=StreamingResponse,
    summary="Сообщение в чат — ответ потоком (SSE)",
    description=(
        "Тело — форма `multipart/form-data`: `content` — текст вопроса или подпись к файлу, "
        "`media` — необязательный файл: фото (JPEG, PNG, WEBP, GIF), голос (ogg, mp3, m4a, wav), "
        "PDF или DOCX. Сохраняет вопрос, собирает контекст из истории чата (последние "
        "CHAT_CONTEXT_WINDOW сообщений в пределах бюджета токенов) и отдаёт ответ модели по мере "
        "генерации: `data: {\"type\": \"token\", \"delta\": \"...\"}`, в конце "
        "`data: {\"type\": \"done\"}`. Ответ сохраняется в историю целиком. Ошибка посреди ответа — "
        "событие `{\"type\": \"error\", \"code\", \"message\"}`. В Swagger поток виден только после "
        "завершения — по кускам его показывает `curl -N`."
    ),
    dependencies=[Depends(require_form)],
    responses={
        200: {"description": "Поток событий SSE",
              "content": {"text/event-stream": {"example": (
                  'data: {"type": "token", "delta": "Вас зовут"}\n\n'
                  'data: {"type": "token", "delta": " Аня."}\n\n'
                  'data: {"type": "done"}\n\n')}}},
        **NOT_FOUND, **MEDIA, **STORAGE, **LLM_ERROR_RESPONSES,
    },
)
async def send_message(
    request: Request,
    chat_id: UUID,
    content: Annotated[str, Form(max_length=32_000, description="Текст вопроса или подпись к файлу",
                                 examples=["Привет, меня зовут Аня"])],
    service: ChatServiceDep,
    settings: SettingsDep,
    audio: AudioClientDep,
    media: Annotated[UploadFile | None, File(description="Фото, голос, PDF или DOCX")] = None,
    user_name: Annotated[str | None, Form(max_length=200, description=(
        "Как обращаться к пользователю, пока он сам не представился: одно-три слова из букв, "
        "до 40 знаков. Попадает только в запрос к модели (системный промпт и начало диалога) — не в историю и не в лог. "
        "Telegram-бот присылает BOT_DEFAULT_USER_NAME."), examples=["Александра"])] = None,
) -> Response:
    structlog.contextvars.bind_contextvars(chat_id=str(chat_id))
    if "\x00" in content:
        raise ChatInputError("content", "символ NUL (\\u0000) в тексте не допускается")
    user_name = clean_user_name(user_name)           # 422 — до разбора файла и вызова Whisper
    if not content.strip() and media is None:
        raise ChatInputError("content", "пустое сообщение: нужен текст вопроса")
    media_ref = None
    if media is not None:
        await service.get_chat(chat_id)          # «чата нет» — до разбора файла и вызова Whisper
        started = time.perf_counter()
        try:
            media_ref = await read_media(media, settings=settings.media, audio_client=audio,
                                         whisper_model=settings.whisper_model, language=settings.audio_language)
        finally:
            await media.close()
        part = media_ref.part
        log.info("chat_media_received", kind=media_ref.kind, mime=media_ref.mime, bytes=media_ref.size,
                 text_chars=len(part["text"]) if part.get("type") == "text" else None,
                 latency_ms=round((time.perf_counter() - started) * 1000, 1))
    chunks = service.stream_message(chat_id, content, media=media_ref, user_name=user_name)
    # Первый фрагмент — до ответа клиенту: «чата нет», недоступная база или провайдер
    # возвращаются обычным JSON с нужным HTTP-кодом, а не событием посреди потока.
    waiting = time.perf_counter()
    try:
        first = await first_chunk(chunks, request.is_disconnected)
    except ClientGone:
        log.warning("chat_client_gone", chat_id=str(chat_id), stage="before_first_chunk",
                    waited_ms=round((time.perf_counter() - waiting) * 1000, 1))
        return Response(status_code=CLIENT_CLOSED)
    return StreamingResponse(
        _events(first, chunks),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{chat_id}/messages", response_model=list[MessageOut], summary="История чата",
            responses={**NOT_FOUND, **STORAGE})
async def list_messages(chat_id: UUID, service: ChatServiceDep,
                        limit: Annotated[int, Query(ge=1, le=500)] = 50) -> list[MessageOut]:
    return [MessageOut.of(m) for m in await service.list_messages(chat_id, limit=limit)]


@router.post(
    "/{chat_id}/system-message",
    response_model=SystemMessageOut,
    summary="Сообщение от системы и уведомление в Telegram",
    description=(
        "Дописывает в историю чата сообщение ассистента — например, «заявка решена» от фоновой "
        "задачи — и, если `notify: true`, отправляет текст пользователю через HTTP-API бота "
        "(BOT_URL/notify). Только с заголовком `X-Internal-Token` = INTERNAL_TOKEN. Уведомление "
        "не дошло — сообщение всё равно сохранено: `notified: false` и причина в `detail`."
    ),
    dependencies=[Depends(require_internal_token)],
    responses={401: {"model": ErrorResponse, "description": "Нет или неверный X-Internal-Token"},
               **NOT_FOUND, **STORAGE},
)
async def system_message(chat_id: UUID, body: SystemMessageIn, service: ChatServiceDep,
                         settings: SettingsDep) -> SystemMessageOut:
    structlog.contextvars.bind_contextvars(chat_id=str(chat_id))
    chat, message = await service.add_system_message(chat_id, body.text)
    out = SystemMessageOut(message_id=message.id, notified=False)
    if body.notify:
        owner = chat.owner_external_id
        if chat.interface != "telegram" or not owner.lstrip("-").isdigit():
            out.detail = f"Уведомления — только для чатов Telegram, а у этого interface={chat.interface}."
        else:
            try:
                await notify_user(int(owner), body.text, settings=settings)
                out.notified = True
            except NotifyError as exc:
                out.detail = exc.message
    log.info("chat_system_message", notify=body.notify, notified=out.notified, chars=len(body.text))
    return out


@router.delete("/{chat_id}/messages", response_model=StatusOut, summary="Очистить историю (/clear)",
               responses={**NOT_FOUND, **STORAGE})
async def clear_history(chat_id: UUID, service: ChatServiceDep) -> StatusOut:
    structlog.contextvars.bind_contextvars(chat_id=str(chat_id))
    await service.clear_history(chat_id)
    return StatusOut()
