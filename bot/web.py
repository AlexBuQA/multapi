"""
HTTP-API бота (блок 4.3): обратный канал backend -> bot для проактивных уведомлений.

    POST /notify   {"chat_id": <Telegram chat id>, "text": "..."}
                   заголовок X-Internal-Token: <INTERNAL_TOKEN>
    GET  /health   {"status": "ok"} — без токена: проверить, что API бота поднят

Сервис зовёт /notify (app/services/notifier.py), когда пользователю нужно написать первым:
статус заявки изменился, долгая фоновая задача закончилась, истекает подписка. Бот
отправляет текст в Telegram как есть — без разметки (parse_mode не задан).

API поднимается в bot/__main__.py рядом с polling, в том же цикле событий: uvicorn.Server
и dp.start_polling работают одновременно. Слушает BOT_API_HOST:BOT_API_PORT — по умолчанию
127.0.0.1:9000, то есть только этот компьютер: снаружи /notify недоступен, даже если
токен утёк. Без INTERNAL_TOKEN API не запускается.

Ответы:
- 200 {"ok": true} — Telegram принял сообщение;
- 401 — нет заголовка X-Internal-Token или он не совпал с INTERNAL_TOKEN. Сравнение —
  hmac.compare_digest: время ответа не подсказывает, сколько символов токена угадано.
  Токен проверяется раньше тела: без токена — 401 на любое тело, даже не JSON, и о
  формате запроса посторонний ничего не узнаёт;
- 422 — нет chat_id, текст пустой (или из одних пробелов) либо длиннее 4096 знаков в UTF-16
  — так Telegram считает предел: эмодзи занимает два знака;
- 403 — пользователь заблокировал бота; 404 — Telegram не знает такого чата (пользователь
  ни разу не писал боту); 400 — Telegram отклонил сообщение по другой причине; 429 —
  Telegram просит подождать (Retry-After); 502 — Telegram недоступен или ответил ошибкой.
Swagger у этого API выключен: он служебный, а не для людей.
"""
from __future__ import annotations

import hmac
import logging
from typing import Annotated

from aiogram import Bot
from aiogram.exceptions import (
    TelegramAPIError,
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramNotFound,
    TelegramRetryAfter,
)
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError, field_validator

log = logging.getLogger(__name__)

TEXT_LIMIT = 4096


class NotifyRequest(BaseModel):
    chat_id: int = Field(description="Telegram chat id: у личного чата он равен id пользователя")
    text: str = Field(min_length=1, max_length=TEXT_LIMIT)

    @field_validator("text")
    @classmethod
    def _text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("текст из одних пробелов Telegram не отправит")
        if len(value.encode("utf-16-le")) // 2 > TEXT_LIMIT:
            raise ValueError(f"длиннее {TEXT_LIMIT} знаков UTF-16 — предел сообщения Telegram")
        return value


def build_api(bot: Bot, internal_token: str) -> FastAPI:
    if not internal_token:
        raise ValueError("internal_token пустой: без него /notify открыт любому")
    expected = internal_token.encode()
    api = FastAPI(title="multapi bot API", docs_url=None, redoc_url=None, openapi_url=None)

    def check_token(x_internal_token: str | None) -> None:
        if x_internal_token is None or not hmac.compare_digest(x_internal_token.encode(), expected):
            log.warning("notify_unauthorized token=%s", "missing" if x_internal_token is None else "wrong")
            raise HTTPException(status_code=401, detail="Нужен заголовок X-Internal-Token с INTERNAL_TOKEN.")

    @api.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.post("/notify")
    async def notify(request: Request, x_internal_token: Annotated[str | None, Header()] = None) -> dict[str, bool]:
        # Тело разбирается здесь, после токена: параметр-модель FastAPI разобрал бы его раньше,
        # и на не-JSON без токена посторонний получил бы 422 с описанием полей.
        check_token(x_internal_token)
        try:
            req = NotifyRequest.model_validate_json(await request.body())
        except ValidationError as exc:
            raise RequestValidationError(exc.errors(include_url=False, include_input=False)) from exc
        try:
            await bot.send_message(chat_id=req.chat_id, text=req.text)
        except TelegramForbiddenError as exc:
            raise HTTPException(status_code=403, detail=f"Telegram: {exc.message}") from exc
        except (TelegramBadRequest, TelegramNotFound) as exc:
            status = 404 if "not found" in exc.message.lower() else 400       # chat not found — 404
            raise HTTPException(status_code=status, detail=f"Telegram: {exc.message}") from exc
        except TelegramRetryAfter as exc:
            raise HTTPException(status_code=429, detail="Telegram просит подождать",
                                headers={"Retry-After": str(exc.retry_after)}) from exc
        except TelegramNetworkError as exc:
            log.warning("notify_failed telegram_chat=%s error=%r", req.chat_id, exc)
            raise HTTPException(status_code=502, detail="Telegram недоступен") from exc
        except TelegramAPIError as exc:                        # прочие ответы Telegram с ошибкой
            log.warning("notify_failed telegram_chat=%s error=%r", req.chat_id, exc)
            raise HTTPException(status_code=502, detail=f"Telegram: {exc.message}") from exc
        # Текст уведомления в лог не пишется — в нём могут быть данные пользователя.
        log.info("notify_sent telegram_chat=%s chars=%d", req.chat_id, len(req.text))
        return {"ok": True}

    return api
