"""
Admin-команды бота (блок 4.4) через Dispatcher, с FakeBackend вместо сервиса: доступ только
для BOT_ADMIN_IDS (фильтр на уровне роутера), /stats, /users, /broadcast, ошибки сервиса —
понятным текстом. И граница ответственности: модерации и ограничения частоты в боте нет.
"""
from __future__ import annotations

import ast
from pathlib import Path

import httpx
import pytest
from aiogram.methods import SendMessage

from bot import texts
from bot.handlers import admin
from bot.handlers.common import IsAdmin
from bot.services.backend_client import AdminNotConfigured
from bot_fakes import ADMIN_ID, FakeBackend, message_update, new_chat_id

BOT_DIR = Path(__file__).resolve().parents[2] / "bot"


def status_error(status: int, code: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://backend.test/chats/admin/stats")
    response = httpx.Response(status, json={"error": {"code": code, "message": "..."}}, request=request)
    return httpx.HTTPStatusError(str(status), request=request, response=response)


class Spy(FakeBackend):
    """Запоминает вызовы admin-методов: не-админ не должен до них дойти."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.admin_calls: list[str] = []

    def _admin(self) -> None:
        self.admin_calls.append("admin")
        super()._admin()


# ---------------------------------------------------------------- доступ
def test_is_admin_filter_is_on_the_router_not_inside_handlers():
    filters = admin.router.message._handler.filters
    assert isinstance(filters[0].callback, IsAdmin) and len(filters) == 2      # второй — личный чат
    for handler in admin.router.message.handlers:
        assert not any(isinstance(f.callback, IsAdmin) for f in handler.filters or [])


@pytest.mark.parametrize("command", ["/stats", "/users", "/broadcast Привет всем", "/status"])
async def test_non_admin_gets_refusal_and_backend_is_not_called(dp, bot, session, command):
    backend = dp["backend"] = Spy()
    chat = new_chat_id()
    await dp.feed_update(bot, message_update(command, chat))
    assert session.sent_texts(chat) == [texts.ADMIN_ONLY]
    assert backend.admin_calls == [] and backend.broadcasts == []


@pytest.mark.parametrize("command", ["/stats", "/users", "/broadcast Привет", "/status"])
async def test_admin_commands_only_in_private_chat(dp, bot, session, command):
    """В группе ответ /stats и /users увидели бы все: чужие id и тексты вопросов."""
    backend = dp["backend"] = Spy()
    chat = -new_chat_id()
    await dp.feed_update(bot, message_update(command, chat, user_id=ADMIN_ID, chat_type="group"))
    assert session.sent_texts(chat) == [texts.ADMIN_PRIVATE_ONLY]
    assert backend.admin_calls == [] and backend.broadcasts == []


# ---------------------------------------------------------------- /stats
async def test_stats_is_html_with_escaped_questions(dp, bot, session):
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("/stats", chat, user_id=ADMIN_ID))
    sent = session.of(SendMessage)[-1]
    assert sent.parse_mode == "HTML"
    assert sent.text.splitlines()[:7] == [
        "<b>Статистика за 24 ч</b>", "Сообщений: 12", "Активных пользователей (DAU): 3",
        "Средняя задержка ответа: 2.5 с", "Заблокировано модерацией: 1 (17% вопросов)",
        "Оценок: 4, доля 👍: 75%", ""]
    assert "1. как сбросить &lt;пароль&gt; — 3" in sent.text             # текст пользователя экранирован


def test_long_questions_are_shortened():
    """Вопрос может быть длиной в целое сообщение — сводка должна влезть в 4096 символов."""
    stats = {"top_questions": [{"question": "я" * 4000, "count": 1} for _ in range(5)]}
    text = admin.format_stats(stats)
    assert len(text) < 1000 and f"1. {'я' * 79}… — 1" in text


async def test_stats_without_data():
    text = admin.format_stats({"period_hours": 24, "total_messages": 0, "active_users": 0, "avg_latency_ms": None,
                               "moderation_blocks": 0, "moderation_block_rate": 0.0, "feedback_votes": 0,
                               "feedback_up_ratio": None, "top_questions": []})
    assert "Средняя задержка ответа: —" in text and "доля 👍: —" in text and "Частые вопросы" not in text


# ---------------------------------------------------------------- /users
async def test_users_table_shows_first_ten(dp, bot, session):
    backend = dp["backend"] = FakeBackend()
    backend.users = [{"owner_external_id": str(1000 + n), "interface": "telegram", "chats": 1,
                      "last_seen_at": f"2026-10-09T12:{n:02d}:00+00:00"} for n in range(15)]
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("/users", chat, user_id=ADMIN_ID))
    sent = session.of(SendMessage)[-1]
    assert sent.parse_mode == "HTML" and sent.text.startswith("<b>Последние пользователи</b> (время UTC)\n<pre>")
    rows = sent.text.split("<pre>")[1].removesuffix("</pre>").splitlines()
    assert rows[0].split() == ["ID", "Канал", "Чаты", "Был"]
    assert len(rows) == 11 and rows[1].split() == ["1000", "telegram", "1", "09.10", "12:00"]


def test_users_table_fits_phone_width():
    """На телефоне строка <pre> длиннее ~40 символов переносится (проверка на Windows)."""
    text = admin.format_users([{"owner_external_id": "scenario-f9ea56d3", "interface": "telegram", "chats": 12,
                                "last_seen_at": "2026-10-09T18:09:06.872406Z"}])
    rows = text.split("<pre>")[1].removesuffix("</pre>").splitlines()
    assert max(len(row) for row in rows) <= 40 and rows[1].startswith("scenario-f9…")


async def test_users_empty(dp, bot, session):
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("/users", chat, user_id=ADMIN_ID))
    assert session.sent_texts(chat) == [texts.USERS_EMPTY]


def test_users_table_escapes_html():
    text = admin.format_users([{"owner_external_id": "<b>x</b>", "interface": "web", "chats": 2,
                                "last_seen_at": "не дата"}])
    assert "&lt;b&gt;x&lt;/b&gt;" in text and "не дата" in text


# ---------------------------------------------------------------- /broadcast
async def test_broadcast_goes_to_queue(dp, bot, session, settings):
    backend = dp["backend"] = FakeBackend()
    for user in (1001, 1002):
        await dp.feed_update(bot, message_update("/start", user))
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("/broadcast  Сегодня с 23:00 — плановые работы ", chat,
                                             user_id=ADMIN_ID))
    assert backend.broadcasts[0]["message"] == "Сегодня с 23:00 — плановые работы"
    assert session.sent_texts(chat)[-1] == texts.BROADCAST_QUEUED.format(id=1, recipients=2, poll=5)
    assert not session.sent_texts(1001)[1:]           # бот сам ничего не разослал: это работа очереди


async def test_broadcast_without_text_shows_usage(dp, bot, session):
    backend = dp["backend"] = FakeBackend()
    chat = new_chat_id()
    await dp.feed_update(bot, message_update("/broadcast", chat, user_id=ADMIN_ID))
    assert session.sent_texts(chat) == [texts.BROADCAST_USAGE] and backend.broadcasts == []


# ---------------------------------------------------------------- ошибки сервиса
@pytest.mark.parametrize("error, expected", [
    (status_error(401, "unauthorized"), texts.ADMIN_UNAUTHORIZED),
    (status_error(503, "admin_token_not_configured"), texts.ADMIN_API_OFF),
    (status_error(500, "internal_error"), texts.SERVICE_ERROR),
    (httpx.ConnectError("refused", request=httpx.Request("GET", "http://backend.test/")), texts.UNAVAILABLE),
    (AdminNotConfigured("нет токена"), texts.ADMIN_TOKEN_MISSING),
])
@pytest.mark.parametrize("command", ["/stats", "/users", "/broadcast текст"])
async def test_backend_errors_are_messages_not_tracebacks(dp, bot, session, error, expected, command):
    dp["backend"] = FakeBackend(admin_error=error) if not isinstance(error, AdminNotConfigured) \
        else FakeBackend(admin_token=None)
    chat = new_chat_id()
    await dp.feed_update(bot, message_update(command, chat, user_id=ADMIN_ID))
    assert session.sent_texts(chat) == [expected]


# ---------------------------------------------------------------- граница ответственности
def bot_sources() -> dict[Path, str]:
    return {path: path.read_text(encoding="utf-8") for path in BOT_DIR.rglob("*.py")}


def test_bot_does_not_moderate_or_rate_limit():
    """Модерация и ограничение частоты — в сервисе; бот только показывает их результат."""
    for path, source in bot_sources().items():
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not {m for m in imported if m.split(".")[0] in ("openai", "app", "slowapi", "limits")}, path
        for forbidden in ("moderations", "omni-moderation", "RateLimit", "Throttl", "TokenBucket", "aiolimiter"):
            assert forbidden not in source, (path, forbidden)


# ---------------------------------------------------------------- меню
async def test_admin_menu_only_in_admin_chats(bot, session, settings):
    from aiogram.exceptions import TelegramBadRequest
    from aiogram.methods import SetMyCommands

    from bot.__main__ import set_commands

    settings.bot_admin_ids = [ADMIN_ID, 888]
    original = session.make_request

    async def chat_not_found(bot_, method, timeout=None):
        if isinstance(method, SetMyCommands) and getattr(method.scope, "chat_id", None) == 888:
            session.requests.append(method)
            raise TelegramBadRequest(method=method, message="Bad Request: chat not found")
        return await original(bot_, method, timeout)

    session.make_request = chat_not_found
    await set_commands(bot, settings)                     # админ 888 не писал боту — не ошибка запуска
    calls = session.of(SetMyCommands)
    assert [c.scope.chat_id if c.scope else None for c in calls] == [None, ADMIN_ID, 888]
    assert [c.command for c in calls[0].commands] == [name for name, _ in texts.COMMANDS]
    assert {"stats", "users", "broadcast", "status"} <= {c.command for c in calls[1].commands}
