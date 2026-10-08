"""
ChatService (блок 4.1): диалог, история которого живёт на сервере.

Конструктор получает репозиторий (ChatRepository) и клиента модели. Клиент — LLMService
из блоков 3.4–3.8, адаптер вокруг AsyncOpenAI: его stream() вызывает
chat.completions.create(..., stream=True) и уже умеет всё, что нужно ответу модели:
- защитный слой блока 3.8 — проверка входа и всей истории, канарейка, StreamGuard на
  потоке, маскирование персональных данных;
- семафор запросов к модели, перевод ошибок SDK в доменные (LLMError), span в Phoenix
  (session.id = id чата) и строка llm_request_completed в логе.
Поэтому ChatService занят только своим: история, контекст и бюджет токенов.

Перед бюджетом контекст проходит ту же проверку, что сделает LLMService
(screen_messages): отклонённый когда-то вопрос и отказ на него выбрасываются сразу и не
занимают бюджет, вытесняя из окна полезную историю. Персональные данные в сообщениях —
метками до модели (mask_message, блок 3.7); в истории они остаются как есть.

Ход диалога в одном чате идёт по очереди (ChatLocks): второй вопрос, присланный, пока
модель отвечает на первый, ждёт его ответа. Иначе история перемешалась бы — [user,
user, assistant, assistant], — и второй запрос ушёл бы модели без первого ответа. Замки
— в памяти процесса: для нескольких копий сервиса очередь не общая.

send_message(chat_id, текст):
1. Чат загружается — нет такого: ChatNotFoundError (404) до любой записи.
2. Вопрос сохраняется через append_message.
3. История — последние CHAT_CONTEXT_WINDOW сообщений (стратегия из context.py),
   перед ней системный промпт чата или CHAT_SYSTEM_PROMPT.
4. fit_to_budget режет старые сообщения, если запрос не помещается в
   CONTEXT_WINDOW - RESPONSE_TOKENS - SAFETY_MARGIN. В бюджете учтено и системное
   сообщение с канарейкой, которое LLMService добавит к запросу (блок 3.8).
5. Модель вызывается потоком, фрагменты отдаются по мере генерации.
6. Ответ сохраняется одним append_message(role="assistant"). Поток оборвался — клиент
   ушёл или провайдер упал — сохраняется накопленное, в лог пишется
   chat_stream_interrupted. Сохранение защищено от отмены (anyio.CancelScope(shield=True)):
   при обрыве соединения Starlette отменяет задачу ответа.

Если проверка входа (блок 3.8) отклонила вопрос, модель не вызывается: LLMService отдаёт
готовый отказ, и он сохраняется как ответ. В следующих запросах screen_messages выбросит
этот вопрос вместе с отказом из контекста. Если StreamGuard остановил ответ модели
посреди потока (метка, промпт, роль DAN), клиент получает отказ вместо придержанного
хвоста.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
import weakref
from collections.abc import AsyncIterator
from typing import Protocol
from uuid import UUID

import anyio

from app.chat.context import (
    MESSAGE_OVERHEAD,
    ContextStrategy,
    count_tokens,
    fit_to_budget,
    make_strategy,
    text_tokens,
)
from app.chat.domain import Chat, ChatInputError, ChatMessage, ChatNotFoundError, ChatStorageError
from app.chat.repository import ChatRepository
from app.core.config import Settings
from app.core.exceptions import LLMContentFiltered
from app.observability.logging import get_logger
from app.schemas.chat import ChatDelta, ChatRequest, Usage
from app.services.guardrails import mask_message
from app.services.security import refusal_for, screen_messages
from app.services.security.canary import canary_message, with_canary
from app.services.security.input_validator import validate_input

log = get_logger()


class LLMClient(Protocol):
    """Чем ChatService вызывает модель: LLMService или фейк в тестах."""

    def stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]: ...


class ChatLocks:
    """Замок на чат: вопрос и ответ одного чата не перемешиваются с соседним вопросом.
    Замок живёт, пока его кто-то держит или ждёт (WeakValueDictionary)."""

    def __init__(self) -> None:
        self._locks: weakref.WeakValueDictionary[UUID, asyncio.Lock] = weakref.WeakValueDictionary()

    def __call__(self, chat_id: UUID) -> asyncio.Lock:
        lock = self._locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[chat_id] = lock
        return lock


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


class ChatService:
    def __init__(self, repository: ChatRepository, llm_client: LLMClient, settings: Settings,
                 *, strategy: ContextStrategy | None = None, locks: ChatLocks | None = None) -> None:
        self.repo = repository
        self.llm = llm_client
        self.settings = settings
        self.strategy = strategy or make_strategy(settings.chat_context_strategy, settings.chat_context_window)
        self.locks = locks or ChatLocks()

    # ------------------------------------------------------------------ #
    def _check_system_prompt(self, system_prompt: str | None) -> None:
        if system_prompt and self.settings.security.enabled:
            # Системный промпт чата проверяется с каждым вопросом (screen_messages, блок 3.8);
            # не прошёл бы — каждый вопрос в этом чате получал бы отказ. Лучше сразу 422.
            verdict = validate_input(system_prompt, self.settings.security.max_input_chars)
            if not verdict.ok:
                raise ChatInputError("system_prompt", f"системный промпт не прошёл проверку входа: {verdict.reason}")

    async def create_chat(self, owner_external_id: str, interface: str, system_prompt: str | None = None) -> Chat:
        """Всегда новый чат."""
        self._check_system_prompt(system_prompt)
        chat = await self.repo.create_chat(owner_external_id, interface, system_prompt)
        log.info("chat_created", chat_id=str(chat.id), interface=interface,
                 own_system_prompt=system_prompt is not None, repository=self.settings.chat_repository)
        return chat

    async def get_or_create_chat(self, owner_external_id: str, interface: str,
                                 system_prompt: str | None = None) -> tuple[Chat, bool]:
        """Чат клиента: существующий для пары (owner_external_id, interface) или новый —
        POST /chats (блок 4.2). Второй элемент — создан ли чат сейчас. У существующего
        чата системный промпт не меняется."""
        self._check_system_prompt(system_prompt)
        chat, created = await self.repo.get_or_create_chat(owner_external_id, interface, system_prompt)
        log.info("chat_created" if created else "chat_reused", chat_id=str(chat.id), interface=interface,
                 own_system_prompt=chat.system_prompt is not None, repository=self.settings.chat_repository)
        return chat, created

    async def get_chat(self, chat_id: UUID) -> Chat:
        chat = await self.repo.get_chat(chat_id)
        if chat is None:
            raise ChatNotFoundError(chat_id)
        return chat

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        await self.get_chat(chat_id)
        return await self.repo.list_messages(chat_id, limit=limit)

    async def clear_history(self, chat_id: UUID) -> None:
        await self.get_chat(chat_id)
        async with self.locks(chat_id):          # не посреди чужого ответа
            await self.repo.soft_delete_messages(chat_id)
        log.info("chat_history_cleared", chat_id=str(chat_id))

    def system_prompt(self, chat: Chat) -> str:
        """Свой промпт чата или CHAT_SYSTEM_PROMPT. Системное сообщение есть всегда: без
        него LLMService подставил бы промпт ассистента /chat со статьями руководства
        (блок 3.7), а он отвечает только на вопросы о продукте — «Как меня зовут?» получил
        бы отказ."""
        if chat.system_prompt:
            return chat.system_prompt
        return self.settings.chat_system_prompt.replace("{product_name}", self.settings.support.product_name)

    def canary(self) -> str | None:
        """Канарейка, которую клиент модели добавит к запросу: у LLMService с включённым
        защитным слоем — app.state.canary, у фейков в тестах и без защиты — None."""
        return getattr(self.llm, "canary", None)

    def build_request(self, chat: Chat, history: list[ChatMessage]) -> tuple[ChatRequest, dict[str, int]]:
        """Запрос к модели по стратегии контекста и бюджету токенов + цифры для лога."""
        messages = self.strategy.build(self.system_prompt(chat), history)
        screened_out = 0
        if self.settings.security.enabled:
            screened = screen_messages(messages, self.settings.security.max_input_chars)
            if screened.verdict.ok:              # иначе LLMService сам ответит отказом
                screened_out = len(messages) - len(screened.messages)
                messages = screened.messages
        messages = [m if m["role"] == "system" else {**m, "content": mask_message(m["content"])} for m in messages]
        # LLMService добавит к запросу системное сообщение с канарейкой (блок 3.8): модель его
        # получает, значит, и бюджет его учитывает. prompt_tokens_est — оценка всего запроса,
        # её видно в chat_turn_finished рядом с prompt_tokens от модели.
        canary = self.canary()
        added = text_tokens(canary_message(canary)["content"]) + MESSAGE_OVERHEAD if canary else 0
        fitted = fit_to_budget(messages, self.settings.context_budget - added)
        stats = {"history_messages": len(history), "screened_out": screened_out, "context_messages": len(fitted),
                 "dropped_by_budget": len(messages) - len(fitted),
                 "prompt_tokens_est": count_tokens(with_canary(fitted, canary))}
        req = ChatRequest(messages=fitted, max_tokens=self.settings.response_tokens, session_id=str(chat.id))
        return req, stats

    # ------------------------------------------------------------------ #
    async def send_message(self, chat_id: UUID, user_content: str) -> AsyncIterator[str]:
        started = time.perf_counter()
        chat = await self.get_chat(chat_id)
        async with self.locks(chat_id):
            # aclosing: клиент ушёл — _turn закрывается здесь же и сохраняет ответ до того,
            # как замок отпустит следующий вопрос.
            async with contextlib.aclosing(self._turn(chat, user_content, started)) as chunks:
                async for chunk in chunks:
                    yield chunk

    async def _turn(self, chat: Chat, user_content: str, started: float) -> AsyncIterator[str]:
        chat_id = chat.id
        await self.repo.append_message(chat_id, ChatMessage(chat_id=chat_id, role="user", content=user_content,
                                                            tokens=text_tokens(user_content)))
        history = await self.repo.list_messages(chat_id, limit=self.strategy.history_limit)
        req, stats = self.build_request(chat, history)
        if stats["dropped_by_budget"]:
            log.info("chat_context_trimmed", chat_id=str(chat_id), budget=self.settings.context_budget, **stats)

        parts: list[str] = []
        usage: Usage | None = None
        outcome = "completed"
        try:
            # aclosing: если клиент ушёл, поток модели закрывается сразу (LLMService закроет
            # соединение с провайдером), а не когда сборщик мусора доберётся до генератора.
            async with contextlib.aclosing(self.llm.stream(req)) as deltas:
                async for delta in deltas:
                    if delta.content:
                        parts.append(delta.content)
                        yield delta.content
                    elif delta.usage is not None:
                        usage = delta.usage
        except LLMContentFiltered:
            # StreamGuard остановил ответ: придержанный хвост клиенту не ушёл, вместо него — отказ.
            outcome = "filtered"
            refusal = ("\n\n" if parts else "") + refusal_for("injection", self.settings.support.product_name)
            parts.append(refusal)
            yield refusal
        except Exception:
            # LLMError и всё непредвиденное: что успело прийти — сохраняется (finally).
            outcome = "failed"
            raise
        except (GeneratorExit, asyncio.CancelledError):
            outcome = "interrupted"
            raise
        finally:
            await self._save_answer(chat_id, "".join(parts), usage, outcome)
            log.info("chat_turn_finished", chat_id=str(chat_id), outcome=outcome, answer_chars=len("".join(parts)),
                     output_tokens=usage.completion_tokens if usage else None,
                     prompt_tokens=usage.prompt_tokens if usage else None,
                     latency_ms=_elapsed_ms(started), **stats)

    async def _save_answer(self, chat_id: UUID, text: str, usage: Usage | None, outcome: str) -> None:
        """Ответ — одним сообщением после потока. Обрыв — сохраняется то, что успело прийти."""
        if outcome not in {"completed", "filtered"}:
            log.warning("chat_stream_interrupted", chat_id=str(chat_id), reason=outcome, saved_chars=len(text))
        if not text:
            return
        tokens = usage.completion_tokens if usage is not None and usage.completion_tokens else text_tokens(text)
        message = ChatMessage(chat_id=chat_id, role="assistant", content=text, tokens=tokens)
        try:
            # Клиент ушёл — задачу ответа отменяют; без щита отменилась бы и запись.
            with anyio.CancelScope(shield=True):
                await self.repo.append_message(chat_id, message)
        except (ChatStorageError, ChatNotFoundError) as exc:
            # Клиент ответ уже получил; ошибку хранилища поднимать некуда — только в лог.
            log.error("chat_answer_not_saved", chat_id=str(chat_id), error=repr(exc)[:300])
