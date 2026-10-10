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
- get_or_create_chat (блок 4.2) идемпотентен по паре (owner_external_id, interface):
  повторный вызов возвращает тот же чат и created=False, системный промпт при этом не
  меняется. Если чатов с этой парой несколько (созданы create_chat до блока 4.2), —
  самый ранний. Одновременные вызовы с одной парой создают один чат;
- create_chat всегда создаёт новый чат;
- get_chat и list_messages для неизвестного чата — None и пустой список, без исключения;
- append_message в неизвестный чат — ChatNotFoundError: сообщение без чата не сохраняется;
- soft_delete_messages для неизвестного чата ничего не делает;
- ошибка самого хранилища (нет соединения, нет таблиц, диск) — ChatStorageError.

OpsRepository (блок 4.4) — то, что нужно production-обвязке; реализуют те же два класса:
- get_message — сообщение чата по id, в том числе скрытое /clear; чужого чата — None;
- add_feedback — оценка «up»/«down» ответа; одна на пару (owner_external_id, message_id):
  повтор не меняет первую и возвращает saved=False с ней;
- record_moderation — инцидент модерации (без текста, только отпечаток);
- enqueue_broadcast / claim_broadcast / finish_broadcast — очередь рассылки: отправитель
  своего интерфейса (бот — telegram) забирает самую раннюю pending (или зависшую в sending
  дольше stale_before) и отчитывается. Итог принимается только у рассылки в sending, иначе —
  BroadcastNotClaimedError;
- broadcast_recipients — owner_external_id всех чатов интерфейса, без повторов, в порядке
  первого чата;
- stats(since, top_n) — сводка с момента since: сообщения (и скрытые /clear — они были),
  активные клиенты, средняя задержка ответа, блокировки модерации, оценки, частые вопросы;
- recent_users(limit) — клиенты (owner_external_id + interface) с числом чатов и временем
  последнего сообщения, от недавних к давним.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

from app.chat.domain import (Broadcast, Chat, ChatMessage, ChatStats, FeedbackResult, FeedbackValue,
                             ModerationIncident, UserActivity)

# Частые вопросы (блок 4.4): нижний регистр, знаки препинания и лишние пробелы убраны. Те же
# шаблоны в SQL (regexp_replace в pg_repo.py, параметрами запроса) и в Python (json_repo.py):
# результаты совпадают. Пробелы — явным списком символов, а не \s: у Postgres \s зависит от
# локали базы (неразрывный пробел он не схлопывает, тонкий — схлопывает), у Python — нет.
QUESTION_PUNCTUATION = r"""[.,!?;:«»"'()\[\]{}…—–-]+"""
QUESTION_SPACES = "[ \t\n\r\f\v\u00a0\u1680\u2000-\u200b\u2028\u2029\u202f\u205f\u3000\ufeff]+"
_PUNCT = re.compile(QUESTION_PUNCTUATION)
_SPACES = re.compile(QUESTION_SPACES)


def normalize_question(text: str) -> str:
    return _SPACES.sub(" ", _PUNCT.sub(" ", text.lower())).strip(" ")       # btrim в SQL — только пробел


@runtime_checkable
class ChatRepository(Protocol):
    async def create_chat(self, owner_external_id: str, interface: str,
                          system_prompt: str | None = None) -> Chat: ...

    async def get_or_create_chat(self, owner_external_id: str, interface: str,
                                 system_prompt: str | None = None) -> tuple[Chat, bool]: ...

    async def get_chat(self, chat_id: UUID) -> Chat | None: ...

    async def append_message(self, chat_id: UUID, message: ChatMessage) -> ChatMessage: ...

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]: ...

    async def soft_delete_messages(self, chat_id: UUID) -> None: ...


@runtime_checkable
class OpsRepository(Protocol):
    async def get_message(self, chat_id: UUID, message_id: UUID) -> ChatMessage | None: ...

    async def add_feedback(self, chat_id: UUID, message_id: UUID, owner_external_id: str,
                           value: FeedbackValue) -> FeedbackResult: ...

    async def record_moderation(self, incident: ModerationIncident) -> None: ...

    async def enqueue_broadcast(self, message: str, interface: str) -> Broadcast: ...

    async def claim_broadcast(self, stale_before: datetime, interface: str = "telegram") -> Broadcast | None: ...

    async def finish_broadcast(self, broadcast_id: int, sent: int, failed: int) -> Broadcast | None: ...

    async def broadcast_recipients(self, interface: str) -> list[str]: ...

    async def stats(self, since: datetime, top_n: int = 5) -> ChatStats: ...

    async def recent_users(self, limit: int = 50) -> list[UserActivity]: ...
