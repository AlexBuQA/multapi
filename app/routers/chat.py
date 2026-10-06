"""
POST /chat и POST /chat/stream (блок 3.4).

Поток — StreamingResponse(media_type="text/event-stream"), кадры собираются вручную:
    data: {"content": "Раз"}\n\n          фрагменты текста
    data: {"usage": {...}}\n\n             итоговый usage (stream_options.include_usage)
    data: [DONE]\n\n                       конец потока
Текст передаётся внутри JSON: перевод строки в ответе модели (списки, код) не ломает
разметку SSE, где пустая строка означает конец события.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse

from app.core.exceptions import LLMError
from app.deps.providers import LLMServiceDep
from app.schemas.chat import ChatDelta, ChatRequest, ChatResponse
from app.schemas.errors import LLM_ERROR_RESPONSES, ErrorResponse

router = APIRouter(prefix="/chat", tags=["chat"])

OPENAPI_EXAMPLES = {
    "simple": {
        "summary": "Один вопрос",
        "value": ChatRequest.model_config["json_schema_extra"]["examples"][0],
    },
    "dialog": {
        "summary": "System prompt, temperature 0 и идентификаторы",
        "value": ChatRequest.model_config["json_schema_extra"]["examples"][1],
    },
}
ChatBody = Body(openapi_examples=OPENAPI_EXAMPLES)

SSE_EXAMPLE = (
    'data: {"content":"Раз"}\n\n'
    'data: {"content":", два, три"}\n\n'
    'data: {"usage":{"prompt_tokens":14,"completion_tokens":9,"total_tokens":23}}\n\n'
    "data: [DONE]\n\n"
)


@router.post(
    "",
    response_model=ChatResponse,
    summary="Ответ модели целиком",
    description=(
        "Отправляет диалог модели и возвращает ответ целиком. Одинаковый запрос "
        "(без учёта user_id и session_id) в пределах CACHE_TTL_SECONDS отдаётся из Redis "
        "с `cached: true` — без обращения к модели."
    ),
    responses={200: {"description": "Ответ модели"}, **LLM_ERROR_RESPONSES},
)
async def chat(service: LLMServiceDep, req: ChatRequest = ChatBody) -> ChatResponse:
    return await service.complete(req)


def _frame(payload: str) -> str:
    return f"data: {payload}\n\n"


async def _sse(first: ChatDelta | None, deltas: AsyncIterator[ChatDelta]) -> AsyncIterator[str]:
    try:
        if first is not None:
            yield _frame(first.model_dump_json(exclude_none=True))
        async for delta in deltas:
            yield _frame(delta.model_dump_json(exclude_none=True))
    except LLMError as exc:
        # Заголовки с кодом 200 уже ушли: об обрыве на середине сообщаем кадром error.
        yield _frame(ErrorResponse.model_validate(
            {"error": {"code": exc.code, "message": exc.message}}
        ).model_dump_json(exclude_none=True))
    finally:
        await deltas.aclose()   # type: ignore[attr-defined]
    yield _frame("[DONE]")


@router.post(
    "/stream",
    response_class=StreamingResponse,
    summary="Потоковый ответ (SSE)",
    description=(
        "Тот же запрос, что у POST /chat, но ответ приходит по мере генерации: кадры "
        "`data: {\"content\": ...}`, затем `data: {\"usage\": ...}` и `data: [DONE]`. "
        "Ошибка до первого фрагмента возвращается обычным JSON с кодом 429/502/504; "
        "обрыв посреди ответа — кадром `data: {\"error\": ...}` перед `[DONE]`. "
        "Кеш не используется. В Swagger поток показывается целиком после завершения — "
        "для проверки по кускам используйте `curl -N`."
    ),
    responses={
        200: {
            "description": "Поток событий SSE",
            "content": {"text/event-stream": {"example": SSE_EXAMPLE}},
        },
        **LLM_ERROR_RESPONSES,
    },
)
async def chat_stream(service: LLMServiceDep, req: ChatRequest = ChatBody) -> StreamingResponse:
    deltas = service.stream(req)
    # Первый фрагмент получаем до ответа клиенту: ошибка подключения к провайдеру,
    # неверный ключ или 429 превращаются в обычный JSON с нужным HTTP-кодом.
    try:
        first: ChatDelta | None = await anext(deltas)
    except StopAsyncIteration:
        first = None
    return StreamingResponse(
        _sse(first, deltas),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
