"""
Помощники тестов чата (блок 4.1): настройки, фейковая модель, тестовая база Postgres.
Отдельный модуль, а не conftest.py: в tests/unit есть свой conftest, и «from conftest
import ...» в двух папках без __init__.py легко перепутать.
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from app.chat.domain import ChatMessage
from app.core.config import DatabaseSettings, Settings
from app.core.exceptions import LLMError
from app.schemas.chat import ChatDelta, ChatRequest, Usage

ROOT = Path(__file__).resolve().parents[2]


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {"llm": {"openai_api_key": "k", "default_model": "test-model"},
                                 "chat_repository": "json", "chat_storage_dir": tmp_path / "chats"}
    values.update(overrides)
    return Settings(**values, _env_file=None)  # type: ignore[arg-type]


def remembered_name(req: ChatRequest) -> str:
    """Ответ «модели», которой важна история: имя ищется во всех сообщениях пользователя."""
    last = req.messages[-1].content
    asks = "как меня зовут" in last.lower()
    for item in req.messages:
        if item.role == "user" and "меня зовут" in item.content and asks:
            name = item.content.split("меня зовут", 1)[1].strip(" .,!?")
            if name and "?" not in item.content:
                return f"Вас зовут {name}."
    if asks:
        return "Вы не называли своего имени."
    return f"Приятно познакомиться! Чем помочь? (вопрос: {last})"


class FakeLLM:
    """Клиент модели для ChatService: stream(req) отдаёт ответ по частям и usage.

    reply — функция «запрос -> текст ответа»; fail_after — после скольких фрагментов
    поднять ошибку (0 — до первого); error — какую; delay — пауза перед фрагментом."""

    def __init__(self, reply: Callable[[ChatRequest], str] = remembered_name, *, chunk: int = 7,
                 fail_after: int | None = None, error: Exception | None = None, delay: float = 0.0) -> None:
        self.reply = reply
        self.chunk = chunk
        self.fail_after = fail_after
        self.error = error
        self.delay = delay
        self.requests: list[ChatRequest] = []

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]:
        self.requests.append(req)
        text = self.reply(req)
        parts = [text[i:i + self.chunk] for i in range(0, len(text), self.chunk)]
        for number, part in enumerate(parts):
            if self.fail_after is not None and number == self.fail_after:
                raise self.error or LLMError()
            if self.delay:
                await asyncio.sleep(self.delay)
            yield ChatDelta(content=part)
        if self.fail_after is not None and self.fail_after >= len(parts):
            raise self.error or LLMError()
        yield ChatDelta(usage=Usage(prompt_tokens=50, completion_tokens=len(parts), total_tokens=50 + len(parts)))

    def user_messages(self, call: int = -1) -> list[str]:
        return [m.content for m in self.requests[call].messages if m.role == "user"]


def message(chat_id: object, role: str, content: str, **extra: object) -> ChatMessage:
    return ChatMessage(chat_id=chat_id, role=role, content=content, **extra)  # type: ignore[arg-type]


# ---------------------------------------------------------------- Postgres
def database_url_for_tests() -> str:
    """TEST_DATABASE_URL или DATABASE_URL с базой <имя>_test — данные сервиса тесты не трогают."""
    explicit = os.environ.get("TEST_DATABASE_URL")
    if explicit:
        return explicit
    from sqlalchemy.engine import make_url

    url = make_url(DatabaseSettings().database_url.get_secret_value())
    return url.set(database=f"{url.database}_test").render_as_string(hide_password=False)


async def prepare_database(url: str) -> None:
    """Создаёт тестовую базу, если её нет (подключение к служебной базе postgres)."""
    import asyncpg
    from sqlalchemy.engine import make_url

    target = make_url(url)
    admin = target.set(drivername="postgresql", database="postgres").render_as_string(hide_password=False)
    conn = await asyncpg.connect(admin, timeout=3)
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", target.database)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{target.database}"')
    finally:
        await conn.close()


async def truncate_tables(url: str) -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE chats CASCADE"))
    await engine.dispose()


def migrate(url: str) -> None:
    """alembic upgrade head для тестовой базы — так проверяется и сама миграция."""
    from alembic import command
    from alembic.config import Config

    config = Config(str(ROOT / "alembic.ini"))
    config.attributes["database_url"] = url
    config.attributes["configure_logging"] = False
    command.upgrade(config, "head")
