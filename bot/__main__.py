"""
Точка входа бота (блок 4.2):

    uvicorn app.main:app --port 8000      # сначала chat-сервис блока 4.1
    python -m bot

1. Настройки из .env (bot/config.py): нет BOT_TOKEN — понятная ошибка, бот не стартует.
2. Bot с TelegramSession (HTTPS по хранилищу сертификатов ОС, прокси), Dispatcher с
   MemoryStorage для сценария /ask, роутеры из bot/handlers.
3. BackendClient кладётся в dp["backend"], настройки — в dp["settings"], очередь вопросов
   чата — в dp["chat_queue"]: aiogram передаёт данные диспетчера в handlers и фильтры
   параметрами с теми же именами (backend, settings, chat_queue) — отдельный middleware не
   нужен.
4. Проверки до polling: getMe (токен и доступ к Telegram) — без них бот не стартует;
   GET /health сервиса — если он недоступен, только предупреждение: сервис можно
   поднять и после бота, сообщения пойдут, как только он ответит.
5. Меню команд (setMyCommands) и long polling.
"""
from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramNetworkError, TelegramUnauthorizedError
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand
from aiogram.utils.token import TokenValidationError
from pydantic import ValidationError

from bot import texts
from bot.config import BotSettings
from bot.handlers import setup_routers
from bot.services.backend_client import BACKEND_ERRORS, BackendClient
from bot.services.chat_queue import ChatQueue
from bot.services.telegram import TelegramSession, proxy_errors

log = logging.getLogger("bot")


def create_dispatcher(backend: BackendClient, settings: BotSettings) -> Dispatcher:
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(setup_routers())
    dp["backend"] = backend
    dp["settings"] = settings
    dp["chat_queue"] = ChatQueue()
    return dp


def create_bot(settings: BotSettings) -> Bot:
    session = TelegramSession(proxy=settings.proxy(), use_system_certs=settings.bot_use_system_certs)
    return Bot(token=settings.bot_token.get_secret_value(), session=session)


async def check_backend(backend: BackendClient) -> None:
    try:
        await backend.health()
        log.info("backend_ready url=%s", backend.base_url)
    except BACKEND_ERRORS as exc:
        log.warning("backend_unavailable url=%s error=%r — бот запущен, сообщения пойдут, когда сервис ответит "
                    "(uvicorn app.main:app --port 8000)", backend.base_url, exc)


async def run(settings: BotSettings) -> None:
    bot = create_bot(settings)
    async with BackendClient(settings.backend_url, timeout=settings.backend_timeout) as backend:
        try:
            me = await bot.get_me()
            await check_backend(backend)
            await bot.set_my_commands([BotCommand(command=name, description=description)
                                       for name, description in texts.COMMANDS])
            log.info("bot_started username=@%s backend=%s timeout=%ss", me.username, backend.base_url,
                     settings.backend_timeout)
            await create_dispatcher(backend, settings).start_polling(bot, handle_signals=False)
        finally:
            await bot.session.close()


def network_hint(error: str, settings: BotSettings) -> str:
    """Что делать, если до api.telegram.org не достучаться: сертификат или сама связь."""
    text = f"Нет связи с api.telegram.org: {error}\n"
    if "certificate" in error.lower() or "CERTIFICATE_VERIFY_FAILED" in error:
        return text + ("Сеть подменяет HTTPS-сертификат: нужны BOT_USE_SYSTEM_CERTS=true (по умолчанию) и пакет "
                       "truststore (pip install -r requirements.txt).")
    if settings.bot_proxy_url is not None:
        return text + "Через прокси BOT_PROXY_URL соединиться не удалось: проверьте адрес, порт и доступ прокси к Telegram."
    return text + ("Сеть не пускает к Telegram напрямую (таймаут или отказ соединения). Задайте прокси в .env: "
                   "BOT_PROXY_URL=http://логин:пароль@хост:порт — например, тот же, что в LLM__PROXY_URL, "
                   "если он пропускает к api.telegram.org.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = BotSettings()  # type: ignore[call-arg]  # bot_token — из окружения или .env
    except ValidationError as exc:
        fields = ", ".join(str(err["loc"][0]).upper() for err in exc.errors())
        print(f"Настройки бота не прошли проверку: {fields}. Токен от @BotFather — строкой BOT_TOKEN=... "
              "в .env в корне проекта; остальное — в .env.example, раздел «Telegram-бот».", file=sys.stderr)
        return 2
    try:
        asyncio.run(run(settings))
    except (TelegramUnauthorizedError, TokenValidationError):
        print("Telegram отклонил BOT_TOKEN или он не похож на токен (вид 123456789:AA...): "
              "проверьте токен у @BotFather.", file=sys.stderr)
        return 1
    except proxy_errors() as exc:
        print(f"Прокси BOT_PROXY_URL не пропустил к api.telegram.org: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except TelegramNetworkError as exc:
        print(network_hint(exc.message, settings), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        log.info("bot_stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
