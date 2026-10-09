"""
Особенности каждого хранилища (блок 4.1) — то, чего нет в общем контракте:
- JSONL: файлы на диске, одна запись на строку, дописывание в конец, маркер soft_delete,
  битая строка не ломает чтение;
- Postgres: после очистки строки остаются с deleted_at, частичный индекс на месте,
  миграция Alembic совпадает с ORM-моделями.
"""
from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from app.chat.repositories.json_repo import JsonChatRepository
from chat_fakes import message
from log_capture import captured_logs, events


# ---------------------------------------------------------------- JSONL
async def test_json_layout_on_disk(json_repo: JsonChatRepository):
    chat = await json_repo.create_chat("test-1", "cli")
    folder = json_repo.base_dir / "chats" / str(chat.id)
    assert sorted(p.name for p in folder.iterdir()) == ["chat.json"]
    assert json.loads((folder / "chat.json").read_text(encoding="utf-8"))["owner_external_id"] == "test-1"

    await json_repo.append_message(chat.id, message(chat.id, "user", "Привет, меня зовут Аня"))
    await json_repo.append_message(chat.id, message(chat.id, "assistant", "Строка 1\nСтрока 2"))
    raw = (folder / "messages.jsonl").read_bytes()
    assert b"\r\n" not in raw                                   # \n на любой ОС
    lines = raw.decode("utf-8").splitlines()
    assert [json.loads(line)["role"] for line in lines] == ["user", "assistant"]   # одна запись — одна строка
    assert "Аня" in lines[0]                                    # кириллица без \u-экранирования


async def test_json_soft_delete_is_appended_marker(json_repo: JsonChatRepository):
    chat = await json_repo.create_chat("u", "cli")
    path = json_repo.base_dir / "chats" / str(chat.id) / "messages.jsonl"
    await json_repo.append_message(chat.id, message(chat.id, "user", "старое"))
    before = path.read_bytes()
    await json_repo.soft_delete_messages(chat.id)
    after = path.read_bytes()
    assert after.startswith(before)                             # только дописывание, файл не переписан
    marker = json.loads(after[len(before):])
    assert marker["type"] == "soft_delete" and marker["at"].startswith("20")
    assert '"type": "soft_delete"' in after.decode("utf-8")     # так его видно при grep по файлу


async def test_json_torn_write_does_not_swallow_next_record(json_repo: JsonChatRepository):
    """Процесс упал посреди записи: в конце файла обрывок без перевода строки. Следующие
    записи — маркер очистки и новое сообщение — не склеиваются с ним и не пропадают."""
    chat = await json_repo.create_chat("u", "cli")
    path = json_repo.base_dir / "chats" / str(chat.id) / "messages.jsonl"
    await json_repo.append_message(chat.id, message(chat.id, "user", "секрет до очистки"))
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write('{"id": "обрыв записи')                         # без \n — как при падении
    await json_repo.soft_delete_messages(chat.id)
    await json_repo.append_message(chat.id, message(chat.id, "user", "после очистки"))
    with captured_logs("INFO") as logs:
        contents = [m.content for m in await json_repo.list_messages(chat.id)]
    assert contents == ["после очистки"]                          # очистка сработала, секрет не вернулся
    assert events(logs, "chat_jsonl_line_skipped")[0]["line"] == 2
    assert path.read_text(encoding="utf-8").splitlines()[1] == '{"id": "обрыв записи'


async def test_json_owner_file_points_to_chat(json_repo: JsonChatRepository):
    chat, _ = await json_repo.get_or_create_chat("ivan@example.com/../x", "web")
    files = list((json_repo.base_dir / "owners").iterdir())
    assert len(files) == 1 and len(files[0].stem) == 64                 # sha256: любой owner — безопасное имя
    assert json.loads(files[0].read_text(encoding="utf-8")) == {"chat_id": str(chat.id)}


async def test_json_owner_file_to_deleted_chat_makes_new_chat(json_repo: JsonChatRepository):
    import shutil

    chat, _ = await json_repo.get_or_create_chat("u", "telegram")
    shutil.rmtree(json_repo.base_dir / "chats" / str(chat.id))
    again, created = await json_repo.get_or_create_chat("u", "telegram")
    assert created is True and again.id != chat.id


async def test_json_concurrent_get_or_create_makes_one_chat(tmp_path):
    """Репозиторий создаётся на каждый запрос (deps.py) — замок общий для процесса."""
    base = tmp_path / "chats"
    results = await asyncio.gather(*(JsonChatRepository(base).get_or_create_chat("u", "telegram") for _ in range(8)))
    assert len({chat.id for chat, _ in results}) == 1
    assert sum(created for _, created in results) == 1
    assert len(list((base / "chats").iterdir())) == 1


async def test_json_unknown_chat_creates_nothing(json_repo: JsonChatRepository):
    await json_repo.soft_delete_messages(uuid4())
    assert not (json_repo.base_dir / "chats").exists()


# ---------------------------------------------------------------- Postgres
async def test_pg_soft_delete_keeps_rows(pg_sessions):
    from sqlalchemy import text

    from app.chat.repositories.pg_repo import PostgresChatRepository

    async with pg_sessions() as session:
        repo = PostgresChatRepository(session=session)
        chat = await repo.create_chat("u", "cli")
        await repo.append_message(chat.id, message(chat.id, "user", "раз"))
        await repo.append_message(chat.id, message(chat.id, "assistant", "два"))
        await repo.soft_delete_messages(chat.id)
        await repo.append_message(chat.id, message(chat.id, "user", "три"))
        async with session.begin():
            rows = (await session.execute(
                text("SELECT content, deleted_at IS NOT NULL FROM chat_messages WHERE chat_id = :id ORDER BY seq"),
                {"id": chat.id})).all()
    assert [tuple(row) for row in rows] == [("раз", True), ("два", True), ("три", False)]


async def test_pg_media_refs_null_and_jsonb(pg_sessions):
    """Блок 4.3: без вложения — SQL NULL (а не JSON null), с вложением — объект JSONB."""
    from sqlalchemy import text

    from app.chat.domain import MediaRef
    from app.chat.repositories.pg_repo import PostgresChatRepository

    ref = MediaRef(kind="document", mime="application/pdf", size=3, filename="a.pdf",
                   part={"type": "text", "text": "[документ PDF]:\nтекст"})
    async with pg_sessions() as session:
        repo = PostgresChatRepository(session=session)
        chat = await repo.create_chat("u", "cli")
        await repo.append_message(chat.id, message(chat.id, "user", "без файла"))
        await repo.append_message(chat.id, message(chat.id, "user", "с файлом", media_refs=ref))
        async with session.begin():
            rows = (await session.execute(
                text("SELECT media_refs IS NULL, jsonb_typeof(media_refs), media_refs->'part'->>'type' "
                     "FROM chat_messages WHERE chat_id = :id ORDER BY seq"), {"id": chat.id})).all()
    assert [tuple(row) for row in rows] == [(True, None, None), (False, "object", "text")]


async def test_pg_concurrent_get_or_create_makes_one_chat(pg_sessions):
    """Восемь одновременных запросов из разных сессий — как из разных копий сервиса."""
    from sqlalchemy import func, select

    from app.chat.repositories.pg_models import ChatRow
    from app.chat.repositories.pg_repo import PostgresChatRepository

    owner = f"tg-{uuid4().hex[:12]}"
    sessions = [pg_sessions() for _ in range(8)]
    try:
        results = await asyncio.gather(*(PostgresChatRepository(s).get_or_create_chat(owner, "telegram")
                                         for s in sessions))
    finally:
        for session in sessions:
            await session.close()
    assert len({chat.id for chat, _ in results}) == 1
    assert sum(created for _, created in results) == 1
    async with pg_sessions() as session, session.begin():
        rows = await session.scalar(select(func.count()).select_from(ChatRow).where(ChatRow.owner_external_id == owner))
    assert rows == 1


async def test_pg_partial_index_and_migration_match_models(pg_sessions):
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from sqlalchemy import text

    from app.chat.repositories.pg_models import Base

    async with pg_sessions() as session:
        async with session.begin():
            indexdef = (await session.execute(text(
                "SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_chat_messages_chat_created'"))).scalar_one()
            conn = await session.connection()
            diff = await conn.run_sync(lambda sync: compare_metadata(MigrationContext.configure(sync), Base.metadata))
    assert "(chat_id, created_at DESC)" in indexdef and "WHERE (deleted_at IS NULL)" in indexdef
    async with pg_sessions() as session, session.begin():
        owner_index = (await session.execute(text(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_chats_owner_interface'"))).scalar_one()
    assert "(owner_external_id, interface)" in owner_index and "UNIQUE" not in owner_index
    assert diff == []           # alembic revision --autogenerate ничего бы не нашёл


# ---------------------------------------------------------------- адрес базы
def test_database_url_gets_async_driver():
    from app.core.config import async_database_url

    assert async_database_url("postgresql://u:p@h:5432/db") == "postgresql+asyncpg://u:p@h:5432/db"
    assert async_database_url("postgres://u:p@h/db") == "postgresql+asyncpg://u:p@h/db"
    assert async_database_url("postgresql+psycopg2://u@h/db") == "postgresql+asyncpg://u@h/db"
    assert async_database_url("postgresql+asyncpg://u@h/db") == "postgresql+asyncpg://u@h/db"


async def test_bad_database_url_does_not_stop_startup(tmp_path):
    from types import SimpleNamespace

    from app.chat.deps import init_chat_storage
    from chat_fakes import make_settings

    state = SimpleNamespace()
    settings = make_settings(tmp_path, chat_repository="postgres", database_url="mysql://u:p@h/db")
    with captured_logs("INFO") as logs:
        await init_chat_storage(state, settings)
    assert state.chat_sessions is None and events(logs, "chat_storage_unavailable")
