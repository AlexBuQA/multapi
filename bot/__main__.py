"""
Точка входа бота (блоки 4.2–4.3):

    uvicorn app.main:app --port 8000      # сначала chat-сервис
    python -m bot

1. Настройки из .env (bot/config.py): нет BOT_TOKEN — понятная ошибка, бот не стартует.
2. Bot с TelegramSession (HTTPS по хранилищу сертификатов ОС, прокси), Dispatcher с
   MemoryStorage для сценария /ask, роутеры из bot/handlers.
3. HTTP-клиент сервиса — один httpx.AsyncClient на всё приложение (make_http), создаётся
   здесь и закрывается в finally: await http.aclose(). BackendClient пользуется им.
4. BackendClient кладётся в dp["backend"], настройки — в dp["settings"], очередь вопросов
   чата — в dp["chat_queue"]: aiogram передаёт данные диспетчера в handlers и фильтры
   параметрами с теми же именами (backend, settings, chat_queue) — отдельный middleware не
   нужен.
5. Проверки до polling: getMe (токен и доступ к Telegram) — без них бот не стартует;
   GET /health сервиса — если он недоступен, только предупреждение: сервис можно
   поднять и после бота, сообщения пойдут, как только он ответит.
6. HTTP-API для уведомлений из сервиса (блок 4.3, bot/web.py: POST /notify) — uvicorn.Server
   в том же цикле событий, рядом с polling. Запускается, только если задан INTERNAL_TOKEN.
   Порт занят — ошибка в логе, а бот отвечает как обычно. Сигналы (Ctrl+C) uvicorn не
   перехватывает (ApiServer): их обрабатывает asyncio.run, и в finally API
   останавливается штатно.
7. Меню команд (setMyCommands) и long polling.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import sys
from collections.abc import Iterator

import httpx
import uvicorn
from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramNetworkError, TelegramUnauthorizedError
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand
from aiogram.utils.token import TokenValidationError
from pydantic import ValidationError

from bot import texts
from bot.config import BotSettings
from bot.handlers import setup_routers
from bot.services.backend_client import BACKEND_ERRORS, BackendClient, make_http
from bot.services.chat_queue import ChatQueue
from bot.services.telegram import TelegramSession, proxy_errors
from bot.web import build_api

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


class ApiServer(uvicorn.Server):
    """uvicorn рядом с polling. Свои обработчики SIGINT/SIGTERM uvicorn не ставит: Ctrl+C
    ловит asyncio.run, отменяет run(), и в его finally API останавливается (stop_api)."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def bind_socket(host: str, port: int) -> socket.socket:
    """Сокет API заранее: занятый порт — OSError здесь, а не sys.exit внутри uvicorn, который
    остановил бы и бота. listen() сразу: запрос, пришедший, пока uvicorn запускается, ждёт в
    очереди сокета, а не получает отказ."""
    sock = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
    try:
        if sys.platform != "win32":       # на Windows SO_REUSEADDR позволил бы двум процессам делить порт
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(128)
    except OSError:
        sock.close()
        raise
    return sock


async def start_api(bot: Bot, settings: BotSettings) -> tuple[ApiServer, asyncio.Task[None]] | None:
    if settings.internal_token is None:
        log.info("notify_api_disabled reason=INTERNAL_TOKEN не задан — уведомления из сервиса не принимаются")
        return None
    host, port = settings.bot_api_host, settings.bot_api_port
    try:
        sock = bind_socket(host, port)
    except OSError as exc:
        log.error("notify_api_failed host=%s port=%s error=%s — порт занят (второй бот?); /notify не работает, "
                  "бот отвечает как обычно", host, port, exc)
        return None
    api = build_api(bot, settings.internal_token.get_secret_value())
    server = ApiServer(uvicorn.Config(api, host=host, port=port, log_config=None))
    task = asyncio.create_task(server.serve(sockets=[sock]), name="notify-api")
    task.add_done_callback(_api_finished)
    for _ in range(100):                   # до 5 с: uvicorn поднимается за доли секунды
        if server.started or task.done():
            break
        await asyncio.sleep(0.05)
    if not server.started:
        await stop_api((server, task))
        sock.close()
        log.error("notify_api_failed host=%s port=%s — uvicorn не запустился; /notify не работает", host, port)
        return None
    log.info("notify_api_started url=http://%s:%s/notify", host, port)
    return server, task


def _api_finished(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and task.exception() is not None:
        log.error("notify_api_crashed error=%r", task.exception())


async def stop_api(api: tuple[ApiServer, asyncio.Task[None]] | None) -> None:
    if api is None:
        return
    server, task = api
    server.should_exit = True
    try:
        await asyncio.wait_for(task, timeout=5)
    except Exception as exc:  # noqa: BLE001 — остановка не должна мешать закрыть остальное
        log.warning("notify_api_stop error=%r", exc)


def make_backend(http: httpx.AsyncClient, settings: BotSettings) -> BackendClient:
    """Клиент сервиса с настройками бота: таймаут потока и имя по умолчанию."""
    return BackendClient(http, stream_timeout=settings.backend_stream_timeout, user_name=settings.bot_default_user_name)


async def run(settings: BotSettings) -> None:
    bot = create_bot(settings)
    http = make_http(settings.backend_url, timeout=settings.backend_timeout)   # один на всё приложение
    backend = make_backend(http, settings)
    api = None
    try:
        me = await bot.get_me()
        await check_backend(backend)
        await bot.set_my_commands([BotCommand(command=name, description=description)
                                   for name, description in texts.COMMANDS])
        api = await start_api(bot, settings)
        log.info("bot_started username=@%s backend=%s timeout=%ss stream_timeout=%ss streaming=%s "
                 "default_user_name=%s", me.username, backend.base_url, settings.backend_timeout,
                 settings.backend_stream_timeout, settings.bot_streaming,
                 "on" if settings.bot_default_user_name else "off")
        await create_dispatcher(backend, settings).start_polling(bot, handle_signals=False)
    finally:
        await stop_api(api)
        await http.aclose()
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
