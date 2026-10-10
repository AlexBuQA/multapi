"""
Admin-команды бота (блок 4.4): /stats, /users, /broadcast <текст> и /status.

Доступ — фильтр IsAdmin на уровне роутера (admin_router.message.filter(IsAdmin())), а не
проверка внутри каждого handler: отправитель не из BOT_ADMIN_IDS до этих handlers просто не
доходит. Его команду подхватывает роутер commands и отвечает «только для администраторов».
Второй фильтр роутера — личный чат: в группе /stats и /users показали бы всем участникам
чужие Telegram id и тексты вопросов. Администратору в группе commands отвечает «только в
личном чате с ботом».

Данные — из admin API сервиса (/chats/admin/*) с заголовком X-Admin-Token (ADMIN_TOKEN в .env
бота — тот же, что у сервиса). Сам бот ничего не считает и не хранит.

Ошибки сервиса (httpx.HTTPStatusError и сеть) — понятным текстом админу, не трассировкой:
401 — токены бота и сервиса не совпадают, 503 admin_token_not_configured — в сервисе нет
ADMIN_TOKEN, остальное — как у обычных вопросов (bot/texts.py).
"""
from __future__ import annotations

import html
import logging
import time
from datetime import datetime
from typing import Any

import httpx
from aiogram import F, Router
from aiogram.enums import ChatType
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from bot import texts
from bot.config import BotSettings
from bot.handlers.common import IsAdmin
from bot.services.backend_client import BACKEND_ERRORS, AdminNotConfigured, BackendClient, error_code

router = Router(name="admin")
router.message.filter(IsAdmin(), F.chat.type == ChatType.PRIVATE)
log = logging.getLogger(__name__)

USERS_SHOWN = 10
USER_ID_SHOWN = 12       # символов id в /users: таблица шириной ~38 символов не переносится на телефоне
QUESTION_SHOWN = 80      # символов вопроса в /stats: сообщение Telegram — до 4096
ADMIN_ERRORS: tuple[type[Exception], ...] = (*BACKEND_ERRORS, AdminNotConfigured)


def admin_error(exc: Exception) -> str:
    if isinstance(exc, AdminNotConfigured):
        return texts.ADMIN_TOKEN_MISSING
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code == 401:
            return texts.ADMIN_UNAUTHORIZED
        if error_code(exc) == "admin_token_not_configured":
            return texts.ADMIN_API_OFF
    return texts.user_message(exc)


def shorten(text: str, limit: int = QUESTION_SHOWN) -> str:
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def format_stats(stats: dict[str, Any]) -> str:
    latency = stats.get("avg_latency_ms")
    lines = [
        f"<b>Статистика за {stats.get('period_hours', 24)} ч</b>",
        f"Сообщений: {stats.get('total_messages', 0)}",
        f"Активных пользователей (DAU): {stats.get('active_users', 0)}",
        f"Средняя задержка ответа: {'—' if latency is None else f'{latency / 1000:.1f} с'}",
        f"Заблокировано модерацией: {stats.get('moderation_blocks', 0)} "
        f"({percent(stats.get('moderation_block_rate'))} вопросов)",
        f"Оценок: {stats.get('feedback_votes', 0)}, доля 👍: {percent(stats.get('feedback_up_ratio'))}",
    ]
    top = stats.get("top_questions") or []
    if top:
        lines.append("\n<b>Частые вопросы</b>")
        lines += [f"{n}. {html.escape(shorten(str(q.get('question', ''))))} — {q.get('count', 0)}"
                  for n, q in enumerate(top, 1)]
    return "\n".join(lines)


def format_users(users: list[dict[str, Any]]) -> str:
    if not users:
        return texts.USERS_EMPTY
    rows = [("ID", "Канал", "Чаты", "Был")]
    for user in users[:USERS_SHOWN]:
        seen = str(user.get("last_seen_at", ""))
        try:
            seen = datetime.fromisoformat(seen).strftime("%d.%m %H:%M")
        except ValueError:
            pass
        rows.append((shorten(str(user.get("owner_external_id", "")), USER_ID_SHOWN), str(user.get("interface", ""))[:8],
                     str(user.get("chats", "")), seen))
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    table = "\n".join(" ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows)
    return f"<b>Последние пользователи</b> (время UTC)\n<pre>{html.escape(table)}</pre>"


@router.message(Command("stats"))
async def cmd_stats(message: Message, backend: BackendClient) -> None:
    try:
        stats = await backend.admin_stats()
    except ADMIN_ERRORS as exc:
        log.warning("admin_command_failed command=stats error=%r", exc)
        await message.answer(admin_error(exc))
        return
    await message.answer(format_stats(stats), parse_mode="HTML")


@router.message(Command("users"))
async def cmd_users(message: Message, backend: BackendClient) -> None:
    try:
        users = await backend.admin_users(limit=USERS_SHOWN)
    except ADMIN_ERRORS as exc:
        log.warning("admin_command_failed command=users error=%r", exc)
        await message.answer(admin_error(exc))
        return
    await message.answer(format_users(users), parse_mode="HTML")


@router.message(Command("broadcast"))
async def cmd_broadcast(message: Message, command: CommandObject, backend: BackendClient,
                        settings: BotSettings) -> None:
    text = (command.args or "").strip()
    if not text:
        await message.answer(texts.BROADCAST_USAGE)
        return
    try:
        queued = await backend.admin_broadcast(text, interface="telegram")
    except ADMIN_ERRORS as exc:
        log.warning("admin_command_failed command=broadcast error=%r", exc)
        await message.answer(admin_error(exc))
        return
    log.info("broadcast_queued id=%s recipients=%s admin=%s", queued.get("id"), queued.get("recipients"),
             message.from_user.id if message.from_user else None)
    await message.answer(texts.BROADCAST_QUEUED.format(id=queued.get("id"), recipients=queued.get("recipients", 0),
                                                       poll=int(settings.bot_broadcast_poll)))


@router.message(Command("status"))
async def cmd_status(message: Message, backend: BackendClient) -> None:
    started = time.perf_counter()
    try:
        health = await backend.health()
        state = f"отвечает: {health.get('status', '?')}, {(time.perf_counter() - started) * 1000:.0f} мс"
    except BACKEND_ERRORS as exc:
        state = f"недоступен ({type(exc).__name__})"
    await message.answer(f"Сервис {backend.base_url} — {state}.")
