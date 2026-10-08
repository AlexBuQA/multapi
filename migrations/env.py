"""
Окружение Alembic (блок 4.1): async-движок SQLAlchemy поверх asyncpg.

Адрес базы, по порядку:
1. config.attributes["database_url"] — так тесты (tests/chat/conftest.py) применяют
   миграции к тестовой базе;
2. DATABASE_URL из окружения или .env — через DatabaseSettings, которым не нужен ключ
   провайдера LLM;
3. значение по умолчанию — Postgres из compose.yaml на 127.0.0.1:5433.

target_metadata — ORM-модели чата: по ним alembic revision --autogenerate находит
изменения схемы.

alembic.ini — только ASCII: Alembic читает его в кодировке системы (cp1251 на русской
Windows, ascii в контейнере без локали), и кириллица в комментариях ломала бы запуск.
"""
from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from app.chat.repositories.pg_models import Base
from app.core.config import DatabaseSettings, async_database_url

config = context.config

if config.config_file_name is not None and config.attributes.get("configure_logging", True):
    # disable_existing_loggers=False: при вызове из тестов логгеры сервиса не отключаются.
    fileConfig(config.config_file_name, disable_existing_loggers=False, encoding="utf-8")

target_metadata = Base.metadata


def database_url() -> str:
    raw = config.attributes.get("database_url") or DatabaseSettings().database_url.get_secret_value()
    return async_database_url(raw)       # postgresql://… -> postgresql+asyncpg://…


def run_migrations_offline() -> None:
    """alembic upgrade head --sql: SQL миграции без подключения к базе."""
    context.configure(url=database_url(), target_metadata=target_metadata, literal_binds=True,
                      dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = create_async_engine(database_url(), poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
