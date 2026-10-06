"""
FastAPI-приложение (блок 3.3): ответ целиком и потоковый ответ через SSE.

Запуск из корня проекта:
    uvicorn app.main:app --port 8000

Проверка потока (Linux/macOS):
    curl -N -X POST http://localhost:8000/chat/stream -H "Content-Type: application/json" \
         -d '{"prompt": "Что такое event loop?"}'
В PowerShell — curl.exe с файлом запроса (см. README):
    curl.exe -N -X POST http://localhost:8000/chat/stream -H "Content-Type: application/json" -d "@scripts/stream_request.json"

Клиент AsyncLLMClient создаётся один раз при старте (lifespan) и закрывается при
остановке; ручки получают его из app.state. В Б3.4 приложение обрастёт роутерами,
внедрением зависимостей, кешем в Redis и обработчиками ошибок.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.services.llm_client import AsyncLLMClient

USER_FACING_FAILURE = "Сервис временно недоступен. Попробуйте позже."


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.llm = AsyncLLMClient()
    yield
    await app.state.llm.aclose()


app = FastAPI(title="multapi — ассистент техподдержки", lifespan=lifespan)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_tokens: int | None = Field(default=None, ge=1, le=2000)


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    llm: AsyncLLMClient = request.app.state.llm
    try:
        answer = await llm.complete(body.prompt, max_tokens=body.max_tokens)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=USER_FACING_FAILURE) from exc
    except Exception as exc:  # noqa: BLE001 — наружу не отдаём детали провайдера
        raise HTTPException(status_code=503, detail=USER_FACING_FAILURE) from exc
    return ChatResponse(answer=answer)


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request) -> EventSourceResponse:
    """SSE: события без имени — фрагменты ответа, `done` — конец, `error` — сбой."""
    llm: AsyncLLMClient = request.app.state.llm

    async def events() -> AsyncIterator[dict[str, str]]:
        try:
            # aclosing: при уходе клиента генератор закрывается сразу — запрос к модели
            # прерывается, слот семафора освобождается, в лог пишется status=cancelled.
            async with aclosing(llm.stream_chat(body.prompt, max_tokens=body.max_tokens)) as stream:
                async for delta in stream:
                    if await request.is_disconnected():
                        return
                    yield {"data": delta}
            yield {"event": "done", "data": "[DONE]"}
        except Exception:  # noqa: BLE001 — поток уже начат: сообщаем об ошибке событием
            yield {"event": "error", "data": USER_FACING_FAILURE}

    return EventSourceResponse(events(), ping=15)
