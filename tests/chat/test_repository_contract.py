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

from app.chat.domain import ChatMessage, ChatNotFoundError
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

