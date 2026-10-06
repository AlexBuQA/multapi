"""
Мок OpenAI-совместимого API для бенчмарка и демо стриминга (блок 3.3).

Отвечает с фиксированной задержкой через asyncio.sleep — так ведёт себя облачный
провайдер с точки зрения клиента: запрос ждёт сеть и чужие GPU, а не процессор
этой машины. Поддерживает обычный ответ, stream=True (SSE-чанки и usage) и
ошибку 404 для модели «invalid-model» (проверка batch_chat и batch_chat_strict).

Отдельный запуск (порт 8001, задержка 1 с):
    python scripts/mock_llm_server.py --port 8001 --latency 1.0
Бенчмарк и демо стриминга поднимают его сами в фоновом потоке (--target mock).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

INVALID_MODEL = "invalid-model"
ANSWER = (
    "Это ответ мок-провайдера: он ждёт заданную задержку, как ждал бы облачный API, "
    "и возвращает текст фиксированной длины, чтобы бенчмарк измерял только время "
    "ожидания ответа, а не скорость генерации на этой машине."
)


def create_app(latency: float) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        model = body.get("model", "mock-model")
        if model == INVALID_MODEL:
            return JSONResponse(status_code=404, content={"error": {
                "message": f"model '{model}' not found", "type": "invalid_request_error",
                "code": "model_not_found"}})

        prompt_tokens = sum(len(m.get("content") or "") for m in body["messages"]) // 4
        words = ANSWER.split(" ")
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": len(words),
                 "total_tokens": prompt_tokens + len(words)}
        created = int(time.time())

        if not body.get("stream"):
            await asyncio.sleep(latency)
            return {"id": "mock", "object": "chat.completion", "created": created, "model": model,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": ANSWER}}],
                    "usage": usage}

        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        async def events():
            def chunk(delta: dict, finish: str | None = None) -> str:
                data = {"id": "mock", "object": "chat.completion.chunk", "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            await asyncio.sleep(latency * 0.3)            # время до первого токена
            step = latency * 0.7 / len(words)
            for i, word in enumerate(words):
                yield chunk({"role": "assistant", "content": word if i == 0 else " " + word})
                await asyncio.sleep(step)
            yield chunk({}, "stop")
            if include_usage:
                data = {"id": "mock", "object": "chat.completion.chunk", "created": created,
                        "model": model, "choices": [], "usage": usage}
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_in_background(latency: float, port: int | None = None) -> str:
    """Поднимает мок в фоновом потоке и возвращает base_url для OpenAI SDK."""
    port = port or _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(latency), host="127.0.0.1", port=port,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("Мок-сервер не запустился за 10 секунд")
        time.sleep(0.05)   # синхронный код вне event loop: ждём старта потока сервера
    return f"http://127.0.0.1:{port}/v1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--latency", type=float, default=1.0, help="задержка ответа, с")
    args = parser.parse_args()
    uvicorn.run(create_app(args.latency), host="127.0.0.1", port=args.port)
