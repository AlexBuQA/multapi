"""Проверка доступа к /chats/admin/* (блок 4.4)."""
from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Header

from app.chat.deps import SettingsDep
from app.chat.domain import RequestError


def require_admin(settings: SettingsDep, x_admin_token: Annotated[str | None, Header()] = None) -> None:
    """X-Admin-Token должен совпасть с ADMIN_TOKEN из .env. Сравнение — hmac.compare_digest:
    время ответа не подсказывает, сколько символов угадано. ADMIN_TOKEN не задан — 503:
    admin API выключено, а не открыто всем."""
    if settings.admin_token is None:
        raise RequestError(503, "admin_token_not_configured", "Admin API выключено: в .env сервиса не задан ADMIN_TOKEN.")
    expected = settings.admin_token.get_secret_value().encode()
    if x_admin_token is None or not hmac.compare_digest(x_admin_token.encode(), expected):
        raise RequestError(401, "unauthorized", "Нужен заголовок X-Admin-Token с ADMIN_TOKEN сервиса.")
