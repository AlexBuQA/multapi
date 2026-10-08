"""
Контракт хранилища истории (блок 4.1): ChatRepository через typing.Protocol.

Реализации (JsonChatRepository, PostgresChatRepository) от протокола не наследуются:
совместимость проверяется структурно — mypy / pyright и тест test_repository_contract.py,
который гоняет одни и те же сценарии против обеих. Фейковому репозиторию в тестах тоже
достаточно иметь эти методы.

Поведение, общее для реализаций:
- list_messages возвращает сообщения от старых к новым, как бы хранилище ни упорядочивало
  строки; limit — последние N, а не первые;
- сообщения до последнего soft_delete_messages в list_messages не попадают, но физически
  остаются в хранилище;
- get_chat и list_messages для неизвестного чата — None и пустой список, без исключения;
- append_message в неизвестный чат — ChatNotFoundError: сообщение без чата не сохраняется;
- soft_delete_messages для неизвестного чата ничего не делает;
- ошибка самого хранилища (нет соединения, нет таблиц, диск) — ChatStorageError.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from app.chat.domain import Chat, ChatMessage


@runtime_checkable
class ChatRepository(Protocol):
    async def create_chat(self, owner_external_id: str, interface: str,
                          system_prompt: str | None = None) -> Chat: ...

    async def get_chat(self, chat_id: UUID) -> Chat | None: ...

    async def append_message(self, chat_id: UUID, message: ChatMessage) -> ChatMessage: ...

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]: ...

    async def soft_delete_messages(self, chat_id: UUID) -> None: ...
