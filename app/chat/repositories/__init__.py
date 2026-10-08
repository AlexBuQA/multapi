"""
Реализации ChatRepository (блок 4.1): JsonChatRepository — файлы JSONL,
PostgresChatRepository — Postgres через async SQLAlchemy. Какая используется, решает
CHAT_REPOSITORY (app/chat/deps.py).

Postgres-реализация импортируется лениво: с CHAT_REPOSITORY=json сервису не нужны
SQLAlchemy и asyncpg.
"""
from __future__ import annotations

from app.chat.repositories.json_repo import JsonChatRepository

__all__ = ["JsonChatRepository"]
