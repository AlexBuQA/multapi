"""
Тесты чата (блок 4.1): общие фикстуры.

    pytest tests/chat/                    # JSON всегда, Postgres — если он запущен
    docker compose up -d postgres         # Postgres из compose.yaml на 127.0.0.1:5433

- repo — параметризованная фикстура: каждый тест контракта выполняется дважды, с
  JsonChatRepository (папка tmp_path) и с PostgresChatRepository.
- Postgres для тестов — отдельная база <имя из DATABASE_URL>_test (multapi_test) на том же
  сервере; её можно задать явно через TEST_DATABASE_URL. Фикстура создаёт базу, если её
  нет, применяет миграции Alembic (так проверяется и сама миграция) и очищает таблицы.
  Postgres не запущен — варианты [postgres] пропускаются с подсказкой, а не падают.
- Модель подменяет FakeLLM (chat_fakes.py): отвечает по тексту запроса и запоминает, что
  ей прислали.
- tiktoken при первом запуске скачивает словарь o200k_base (около 3,6 МБ) и кладёт в кеш.
"""
from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "tests", ROOT / "tests" / "chat"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Как в tests/unit/conftest.py: .env разработчика на тесты не влияет.
os.environ["LLM__OPENAI_API_KEY"] = "test-key"
os.environ["LLM__DEFAULT_MODEL"] = "test-model"
os.environ["LLM__BASE_URL"] = "http://llm.invalid/v1"
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = ""
os.environ["LOG_LEVEL"] = "CRITICAL"
os.environ["LLM__USE_SYSTEM_CERTS"] = "false"
os.environ["SECURITY__ENABLED"] = "true"
os.environ["RATE_LIMIT_PER_MIN"] = "0"
os.environ["LOG_FILE"] = os.devnull
os.environ["CHAT_REPOSITORY"] = "json"

from log_capture import quiet_logs  # noqa: E402

from app.chat.repositories.json_repo import JsonChatRepository  # noqa: E402
from app.core.config import Settings  # noqa: E402
from chat_fakes import (  # noqa: E402
    FakeLLM,
    database_url_for_tests,
    make_settings,
    migrate,
    prepare_database,
    truncate_tables,
)

quiet_logs()


@pytest.fixture(scope="session", autouse=True)
def tokenizer() -> None:
    """Словарь o200k_base для подсчёта токенов. Unit-тесты запускают lifespan сервиса при
    запрещённой сети: если словаря ещё нет в кеше, неудачная загрузка запомнилась бы на
    весь прогон. Здесь сеть есть — загрузка повторяется."""
    from app.chat import context

    context._encoding.cache_clear()
    context.preload_encoding()


@pytest.fixture
def chat_settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def json_repo(tmp_path: Path) -> JsonChatRepository:
    return JsonChatRepository(base_dir=tmp_path / "chats")


@pytest.fixture(scope="session")
def pg_url() -> str:
    """Тестовая база с применёнными миграциями. Postgres не запущен — skip с подсказкой."""
    url = database_url_for_tests()
    try:
        asyncio.run(prepare_database(url))
    except Exception as exc:  # noqa: BLE001 — нет сервера, неверный пароль, нет прав
        pytest.skip(f"Postgres недоступен ({type(exc).__name__}: {exc}). "
                    "Запустите docker compose up -d postgres или задайте TEST_DATABASE_URL")
    migrate(url)
    asyncio.run(truncate_tables(url))
    return url


@pytest.fixture
async def pg_sessions(pg_url: str) -> AsyncIterator[object]:
    """Фабрика сессий к тестовой базе; движок — на цикл событий этого теста."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(pg_url)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest.fixture
async def pg_repo(pg_sessions: object) -> AsyncIterator[object]:
    from app.chat.repositories.pg_repo import PostgresChatRepository

    async with pg_sessions() as session:  # type: ignore[operator]
        yield PostgresChatRepository(session=session)


@pytest.fixture(params=["json", "postgres"])
def repo(request: pytest.FixtureRequest) -> object:
    """Одна и та же проверка — против обеих реализаций ChatRepository."""
    return request.getfixturevalue("json_repo" if request.param == "json" else "pg_repo")


@pytest.fixture
async def clean_pg_repo(pg_url: str, pg_sessions: object) -> AsyncIterator[object]:
    """Postgres без данных других тестов — для сводок (stats, recent_users) блока 4.4."""
    from app.chat.repositories.pg_repo import PostgresChatRepository

    await truncate_tables(pg_url)
    async with pg_sessions() as session:  # type: ignore[operator]
        yield PostgresChatRepository(session=session)


@pytest.fixture(params=["json", "postgres"])
def clean_repo(request: pytest.FixtureRequest) -> object:
    """Пустое хранилище каждой реализации: сводки считают всё, что в нём есть."""
    return request.getfixturevalue("json_repo" if request.param == "json" else "clean_pg_repo")
