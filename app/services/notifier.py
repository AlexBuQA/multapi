"""
Уведомления из сервиса в Telegram (блок 4.3): обратный канал backend -> bot.

    await notify_user(chat_id_tg, text)

Бот поднимает рядом с polling маленький HTTP-API (bot/web.py): POST /notify с заголовком
X-Internal-Token. Сервис зовёт его, когда пользователю надо написать первым: статус заявки
изменился, долгая фоновая задача закончилась, истекает подписка. В демо это делает
POST /chats/{chat_id}/system-message с notify=true (app/chat/routes.py).

- BOT_URL — адрес HTTP-API бота (по умолчанию http://127.0.0.1:9000: бот на этом же
  компьютере; в compose — http://bot:9000);
- INTERNAL_TOKEN — общий секрет сервиса и бота, только в .env. Не задан — уведомления
  выключены (NotifyError notify_not_configured), сервис не стучится к боту без пароля;
- trust_env=False: HTTP(S)_PROXY из окружения к боту не применяется — он в той же сети;
- таймаут 5 с: уведомление — не ответ модели, ждать его долго незачем.

Ошибки — NotifyError с кодом и текстом: бот недоступен, отклонил токен, Telegram не
доставил сообщение (пользователь заблокировал бота).
"""
from __future__ import annotations

import httpx

from app.core.config import Settings, get_settings
from app.observability.logging import get_logger

log = get_logger()

NOTIFY_TIMEOUT = 5.0


class NotifyError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


async def notify_user(chat_id_tg: int, text: str, *, settings: Settings | None = None,
                      transport: httpx.AsyncBaseTransport | None = None) -> None:
    settings = settings or get_settings()
    if settings.internal_token is None:
        raise NotifyError("notify_not_configured", "Уведомления выключены: в .env не задан INTERNAL_TOKEN.")
    async with httpx.AsyncClient(timeout=NOTIFY_TIMEOUT, trust_env=False, transport=transport) as c:
        try:
            r = await c.post(
                f"{settings.bot_url}/notify",
                json={"chat_id": chat_id_tg, "text": text},
                headers={"X-Internal-Token": settings.internal_token.get_secret_value()},
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            message = {401: "бот отклонил INTERNAL_TOKEN — у сервиса и бота он должен совпадать",
                       403: "пользователь заблокировал бота или не начинал с ним диалог",
                       404: "Telegram не знает такого чата"}.get(status, f"бот ответил кодом {status}")
            log.warning("notify_failed", status=status, bot_url=settings.bot_url)
            raise NotifyError("notify_rejected", f"Уведомление не доставлено: {message}.") from exc
        except httpx.HTTPError as exc:
            log.warning("notify_failed", error=repr(exc)[:200], bot_url=settings.bot_url)
            raise NotifyError("bot_unavailable",
                              f"Уведомление не доставлено: бот недоступен по адресу {settings.bot_url} (BOT_URL).") from exc
    log.info("notify_sent", telegram_chat=chat_id_tg, chars=len(text))
