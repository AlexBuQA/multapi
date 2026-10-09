"""
Контракт ChatRepository (блок 4.1): один набор тестов на обе реализации.

Фикстура repo параметризована (tests/chat/conftest.py): каждая функция выполняется с
JsonChatRepository и с PostgresChatRepository — тест-функции одни и те же, отличается
только хранилище. Postgres не запущен — варианты [postgres] пропускаются.

    pytest tests/chat/test_repository_contract.py -v
"""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.chat.domain import ChatMessage, ChatNotFoundError, MediaRef
from app.chat.repository import ChatRepository
from chat_fakes import message


async def test_implements_protocol(repo):
    assert isinstance(repo, ChatRepository)          # структурная проверка: методы на месте


async def test_create_chat_and_read_it_back(repo):
    chat = await repo.create_chat("test-1", "cli", system_prompt="Отвечай кратко.")
    loaded = await repo.get_chat(chat.id)
    assert loaded == chat
    assert (loaded.owner_external_id, loaded.interface, loaded.system_prompt) == ("test-1", "cli", "Отвечай кратко.")
    assert loaded.created_at.tzinfo is not None


async def test_chat_without_system_prompt(repo):
    chat = await repo.create_chat("123456789", "telegram")
    assert (await repo.get_chat(chat.id)).system_prompt is None


async def test_messages_come_back_in_chronological_order(repo):
    chat = await repo.create_chat("u", "cli")
    sent = [message(chat.id, "user", "Привет, меня зовут Аня"),
            message(chat.id, "assistant", "Здравствуйте, Аня!", tokens=5),
            message(chat.id, "user", "Как меня зовут?")]
    for item in sent:
        assert await repo.append_message(chat.id, item) == item
    assert await repo.list_messages(chat.id) == sent


async def test_limit_returns_last_n_not_first(repo):
    chat = await repo.create_chat("u", "cli")
    for number in range(1, 6):
        await repo.append_message(chat.id, message(chat.id, "user", f"сообщение {number}"))
    assert [m.content for m in await repo.list_messages(chat.id, limit=2)] == ["сообщение 4", "сообщение 5"]
    assert len(await repo.list_messages(chat.id, limit=50)) == 5
    assert await repo.list_messages(chat.id, limit=0) == []


async def test_soft_delete_hides_history_but_new_messages_are_visible(repo):
    chat = await repo.create_chat("u", "cli")
    await repo.append_message(chat.id, message(chat.id, "user", "старое"))
    await repo.append_message(chat.id, message(chat.id, "assistant", "старый ответ"))
    await repo.soft_delete_messages(chat.id)
    assert await repo.list_messages(chat.id) == []
    new = message(chat.id, "user", "новое")
    await repo.append_message(chat.id, new)
    assert await repo.list_messages(chat.id) == [new]
    # Вторая очистка скрывает и то, что было после первой.
    await repo.soft_delete_messages(chat.id)
    assert await repo.list_messages(chat.id) == []
    assert await repo.get_chat(chat.id) is not None        # сам чат на месте


async def test_unknown_chat_is_none_and_empty_without_errors(repo):
    unknown = uuid4()
    assert await repo.get_chat(unknown) is None
    assert await repo.list_messages(unknown) == []
    await repo.soft_delete_messages(unknown)               # ничего не делает и не падает
    assert await repo.get_chat(unknown) is None


async def test_append_to_unknown_chat_raises(repo):
    unknown = uuid4()
    with pytest.raises(ChatNotFoundError):
        await repo.append_message(unknown, message(unknown, "user", "в никуда"))
    assert await repo.list_messages(unknown) == []


async def test_message_must_belong_to_the_chat(repo):
    chat = await repo.create_chat("u", "cli")
    with pytest.raises(ValueError):
        await repo.append_message(chat.id, message(uuid4(), "user", "чужое"))


async def test_chats_do_not_see_each_other(repo):
    first, second = await repo.create_chat("a", "web"), await repo.create_chat("b", "web")
    await repo.append_message(first.id, message(first.id, "user", "первый"))
    await repo.append_message(second.id, message(second.id, "user", "второй"))
    await repo.soft_delete_messages(first.id)
    assert await repo.list_messages(first.id) == []
    assert [m.content for m in await repo.list_messages(second.id)] == ["второй"]


async def test_fields_survive_round_trip(repo):
    chat = await repo.create_chat("ivan@example.com", "web")
    sent = ChatMessage(chat_id=chat.id, role="assistant", tokens=None,
                       content='Строка 1\nСтрока 2 — «кавычки», "двойные", \\слеш, эмодзи 👋, {"type": "soft_delete"}')
    await repo.append_message(chat.id, sent)
    [loaded] = await repo.list_messages(chat.id)
    assert loaded == sent and loaded.tokens is None and loaded.created_at.tzinfo is not None


async def test_equal_timestamps_keep_insertion_order(repo):
    """created_at задаёт приложение; на Windows до Python 3.13 у соседних сообщений он
    может совпасть. Порядок при этом — порядок записи (в Postgres — столбец seq)."""
    chat = await repo.create_chat("u", "cli")
    moment = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
    sent = [message(chat.id, "user" if i % 2 == 0 else "assistant", f"#{i}", created_at=moment) for i in range(6)]
    for item in sent:
        await repo.append_message(chat.id, item)
    assert [m.content for m in await repo.list_messages(chat.id)] == [f"#{i}" for i in range(6)]
    assert [m.content for m in await repo.list_messages(chat.id, limit=3)] == ["#3", "#4", "#5"]



# ---------------------------------------------------------------- get_or_create_chat (блок 4.2)
# Владельцы уникальны в каждом тесте: тестовая база Postgres очищается один раз на прогон.
def new_owner() -> str:
    return f"tg-{uuid4().hex[:12]}"


async def test_get_or_create_is_idempotent(repo):
    owner = new_owner()
    first, created = await repo.get_or_create_chat(owner, "telegram", system_prompt="Отвечай кратко.")
    again, created_again = await repo.get_or_create_chat(owner, "telegram", system_prompt="Другой промпт")
    assert created is True and created_again is False
    assert again.id == first.id
    assert again.system_prompt == "Отвечай кратко."                # промпт задаётся только при создании
    assert (await repo.get_chat(first.id)).owner_external_id == owner


async def test_get_or_create_separates_owner_and_interface(repo):
    owner = new_owner()
    telegram, _ = await repo.get_or_create_chat(owner, "telegram")
    web, _ = await repo.get_or_create_chat(owner, "web")
    other, _ = await repo.get_or_create_chat(owner + "-2", "telegram")
    assert len({telegram.id, web.id, other.id}) == 3


async def test_get_or_create_finds_earliest_chat_made_by_create_chat(repo):
    """Чаты, созданные до блока 4.2 (create_chat не идемпотентен), тоже находятся."""
    owner = new_owner()
    earliest = await repo.create_chat(owner, "cli")
    await repo.create_chat(owner, "cli")
    found, created = await repo.get_or_create_chat(owner, "cli")
    assert created is False and found.id == earliest.id
    assert (await repo.get_or_create_chat(owner, "cli"))[0].id == earliest.id


async def test_create_chat_always_makes_a_new_one(repo):
    owner = new_owner()
    assert (await repo.create_chat(owner, "cli")).id != (await repo.create_chat(owner, "cli")).id


async def test_media_refs_round_trip(repo):
    """Блок 4.3: вложение с готовым content-part сохраняется и читается обратно как было."""
    chat = await repo.create_chat("u", "telegram")
    image = MediaRef(kind="image", mime="image/png", size=4, filename="скрин.png",
                     part={"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBO"}})
    voice = MediaRef(kind="audio", mime="audio/ogg", size=10, filename=None,
                     part={"type": "text", "text": "[пользователь сказал голосом]:\nПривет"})
    sent = [message(chat.id, "user", "Что на скрине?", media_refs=image),
            message(chat.id, "user", "[голосовое сообщение]", media_refs=voice),
            message(chat.id, "assistant", "Ответ")]
    for item in sent:
        await repo.append_message(chat.id, item)
    loaded = await repo.list_messages(chat.id)
    assert loaded == sent
    assert loaded[0].media_refs == image and loaded[2].media_refs is None

