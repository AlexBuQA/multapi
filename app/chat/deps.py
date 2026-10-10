"""
Внедрение зависимостей чата (блок 4.1): какой репозиторий, какой клиент модели.

- get_repository() читает CHAT_REPOSITORY из настроек (pydantic-settings, блок 3.4):
  "json" -> JsonChatRepository(base_dir=CHAT_STORAGE_DIR),
  "postgres" -> PostgresChatRepository(session=...) с сессией на время запроса.
  Другое значение -> ValueError с понятным текстом. Схема Settings такое значение и так
  не пропустит (Literal), проверка здесь — на случай настроек, собранных в обход неё.
- get_chat_service() собирает ChatService из репозитория, LLMService и модерации (блок 4.4).

Долгоживущие объекты — движок SQLAlchemy и фабрика сессий — создаёт lifespan
(init_chat_storage в app/main.py) и кладёт в app.state; глобальных переменных-сервисов
на уровне модуля нет. Тесты подменяют зависимости через app.dependency_overrides.

Сессия Postgres живёт, пока идёт ответ: поток POST /chats/{id}/messages читает и пишет
историю после того, как обработчик вернул StreamingResponse. FastAPI 0.118+ закрывает
зависимости с yield после отправки ответа целиком — поэтому в requirements fastapi>=0.142.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import Depends, Request

from app.chat.domain import ChatStorageError
from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.repository import ChatRepository
from app.chat.context import preload_encoding
from app.chat.service import ChatLocks, ChatService
from app.core.config import Settings, async_database_url, get_settings
from app.deps.providers import get_llm_service
from app.moderation import ModerationService, build_moderation
from app.observability.logging import get_logger
from app.services.llm import LLMService

log = get_logger()

SettingsDep = Annotated[Settings, Depends(get_settings)]

# Таймаут подключения к Postgres, с: по умолчанию asyncpg ждёт 60 с, и недоступная база
# держала бы и старт сервиса, и каждый запрос к /chats.
DB_CONNECT_TIMEOUT = 5


async def init_chat_storage(state: Any, settings: Settings) -> None:
    """Lifespan:
    - замки чатов (ChatLocks) — общие для всех запросов процесса;
    - словарь токенизатора загружается в потоке заранее: первый запрос к /chats не ждёт
      скачивания и не блокирует цикл событий;
    - для CHAT_REPOSITORY=postgres — движок и фабрика сессий в app.state и проверка, что
      база доступна и миграция применена. База недоступна или адрес неверный — сервис всё
      равно стартует (/chat от неё не зависит), а в лог уходит chat_storage_unavailable
      с подсказкой; запросы к /chats получат 503."""
    state.chat_engine = state.chat_sessions = None
    state.chat_locks = ChatLocks()
    await asyncio.to_thread(preload_encoding)
    if settings.chat_repository != "postgres":
        log.info("chat_storage_ready", repository=settings.chat_repository, path=str(settings.chat_storage_dir))
        return
    from sqlalchemy import text
    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    try:
        url = make_url(async_database_url(settings.database_url.get_secret_value()))
        shown = url.render_as_string(hide_password=True)
        state.chat_engine = create_async_engine(url, pool_pre_ping=True,
                                                connect_args={"timeout": DB_CONNECT_TIMEOUT})
    except Exception as exc:  # noqa: BLE001 — неверный адрес, нет драйвера
        log.warning("chat_storage_unavailable", repository="postgres", error=repr(exc)[:300],
                    note="проверьте DATABASE_URL: postgresql+asyncpg://пользователь:пароль@хост:порт/база")
        return
    state.chat_sessions = async_sessionmaker(state.chat_engine, expire_on_commit=False)
    try:
        async with state.chat_engine.connect() as conn:
            await conn.execute(text("SELECT 1 FROM chats LIMIT 1"))
    except Exception as exc:  # noqa: BLE001 — нет соединения, нет таблиц, неверный пароль
        log.warning("chat_storage_unavailable", repository="postgres", database=shown, error=repr(exc)[:300],
                    note="docker compose up -d postgres, затем alembic upgrade head")
    else:
        log.info("chat_storage_ready", repository="postgres", database=shown)


async def close_chat_storage(state: Any) -> None:
    engine = getattr(state, "chat_engine", None)
    if engine is not None:
        await engine.dispose()


async def get_repository(request: Request, settings: SettingsDep) -> AsyncIterator[ChatRepository]:
    kind = settings.chat_repository
    if kind == "json":
        yield JsonChatRepository(base_dir=settings.chat_storage_dir)
    elif kind == "postgres":
        from app.chat.repositories.pg_repo import PostgresChatRepository

        sessions = getattr(request.app.state, "chat_sessions", None)
        if sessions is None:
            raise ChatStorageError("CHAT_REPOSITORY=postgres, но подключение к базе не создано при старте сервиса.")
        async with sessions() as session:
            yield PostgresChatRepository(session=session)
    else:
        raise ValueError(f"CHAT_REPOSITORY={kind!r}: ожидается json или postgres")


def get_llm_client(llm: Annotated[LLMService, Depends(get_llm_service)]) -> LLMService:
    """Клиент модели для чата — LLMService (блоки 3.4–3.8), адаптер вокруг AsyncOpenAI."""
    return llm


def get_audio_client(request: Request) -> Any | None:
    """Клиент Whisper (блок 4.3): AsyncOpenAI на AUDIO_BASE_URL, создаёт lifespan, если задан
    AUDIO_API_KEY. None — расшифровка голоса не настроена."""
    return getattr(request.app.state, "audio", None)


AudioClientDep = Annotated[Any | None, Depends(get_audio_client)]


def get_chat_locks(request: Request) -> ChatLocks | None:
    """Замки чатов процесса (создаёт lifespan); в тестах без lifespan — None."""
    return getattr(request.app.state, "chat_locks", None)


def get_moderation(request: Request, settings: SettingsDep) -> ModerationService:
    """Модерация (блок 4.4). Её собирает lifespan — шаблоны и клиент OpenAI Moderation, — а
    для настроек, подменённых в тестах, — заново по ним (только слой ключевых слов)."""
    state = request.app.state
    built = getattr(state, "moderation", None)
    if built is not None and getattr(state, "moderation_config", None) is settings.moderation:
        return built
    return build_moderation(settings)


ModerationDep = Annotated[ModerationService, Depends(get_moderation)]


def get_chat_service(
    repo: Annotated[ChatRepository, Depends(get_repository)],
    llm: Annotated[LLMService, Depends(get_llm_client)],
    settings: SettingsDep,
    locks: Annotated[ChatLocks | None, Depends(get_chat_locks)],
    moderation: ModerationDep,
) -> ChatService:
    return ChatService(repo, llm, settings, locks=locks, moderation=moderation)


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
