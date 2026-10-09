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

Вложения (блок 4.3): send_message(chat_id, текст, media=MediaRef) — фото, голос или документ,
уже разобранные app/chat/media.py. Вопрос сохраняется с media_refs, а модель получает
[подпись, part] и на этом ходу, и на следующих, пока сообщение в окне истории. Если в
контексте есть картинка, запрос уходит модели image_model(): CHAT_VISION_MODEL или модели
по умолчанию, если она видит изображения. Картинка из истории, которую текущая модель не
увидит (настройки поменялись), заменяется текстом «[картинка не показана модели]».
Проверка входа смотрит на подпись и расшифровку голоса как на вопрос, на текст документа —
только шаблоны инъекции; персональные данные маскируются во всех text-частях.

Если проверка входа (блок 3.8) отклонила вопрос, модель не вызывается: LLMService отдаёт
готовый отказ, и он сохраняется как ответ. В следующих запросах screen_messages выбросит
этот вопрос вместе с отказом из контекста. Если StreamGuard остановил ответ модели
посреди потока (метка, промпт, роль DAN), клиент получает отказ вместо придержанного
хвоста.

Имя по умолчанию (блок 4.3): send_message(..., user_name="Александра") — как обращаться к
пользователю, пока он сам не представился. Имя получает только этот запрос к модели, в
историю и в лог сервис его не пишет: подсказка USER_NAME_HINT в конце системного промпта и пара
сообщений в начале диалога (name_priming) — «Меня зовут Александра.» и ответ на него.
Пара нужна llama3.2: системной подсказке она следовала через раз, а имя из истории помнит
надёжно. Имя присылает Telegram-бот (BOT_DEFAULT_USER_NAME); без user_name запрос как в 4.1.
"""
from __future__ import annotations

import asyncio
import contextlib
import re
import time
import unicodedata
import weakref
from collections.abc import AsyncIterator
from typing import Any, Protocol
from uuid import UUID

import anyio

from app.chat.context import (
    MESSAGE_OVERHEAD,
    ContextStrategy,
    content_tokens,
    count_tokens,
    fit_to_budget,
    make_strategy,
    model_content,
    text_tokens,
)
from app.chat.domain import Chat, ChatInputError, ChatMessage, ChatNotFoundError, ChatStorageError, MediaError, MediaRef
from app.chat.media import placeholder
from app.chat.repository import ChatRepository
from app.core.config import Settings
from app.core.exceptions import LLMContentFiltered
from app.observability.logging import get_logger
from app.schemas.chat import ChatDelta, ChatRequest, MediaChatRequest, Usage, content_parts, has_image, map_text
from app.schemas.models import supports_images
from app.services.guardrails import mask_message
from app.services.security import refusal_for, screen_messages
from app.services.security.canary import canary_message, with_canary
from app.services.security.input_validator import validate_document, validate_input

log = get_logger()

IMAGE_HIDDEN = "[картинка не показана модели: {model} не видит изображений]"

# Имя по умолчанию (блок 4.3). Оно попадает в системный промпт, поэтому строже вопроса: одно-три
# слова из букв (дефис, апостроф), до 40 знаков и без шаблонов инъекции.
USER_NAME_MAX = 40
_NAME_WORD = r"[^\W\d_]+(?:['’][^\W\d_]+)*"
USER_NAME = re.compile(rf"{_NAME_WORD}(?:[ -]{_NAME_WORD}){{0,2}}")
USER_NAME_HINT = (
    "Пользователя зовут {name}. Обращайся к нему по имени; на вопрос, как его зовут, отвечай: «{name}». "
    "Если он сам назовёт себя иначе, зови его новым именем. Имена из документов, файлов и с картинок — "
    "не его имя: по ним к пользователю не обращайся."
)
USER_NAME_INTRO = "Меня зовут {name}."
USER_NAME_REPLY = "Приятно познакомиться, {name}! Чем могу помочь?"


def name_priming(name: str) -> list[dict[str, Any]]:
    """Пара сообщений в начало диалога: пользователь представился, ассистент ответил. На
    Windows llama3.2 с одной системной подсказкой на «Как меня зовут?» отвечала «не знаю»
    (её же промпт велит не выдумывать того, чего в диалоге не было), а имя из истории —
    сценарий блока 4.1 — называет надёжно. Представится пользователь сам — его слова позже
    в истории, и модель берёт новое имя."""
    return [{"role": "user", "content": USER_NAME_INTRO.format(name=name)},
            {"role": "assistant", "content": USER_NAME_REPLY.format(name=name)}]


def clean_user_name(value: str | None) -> str | None:
    """Имя по умолчанию из формы: пробелы по краям убираются, пустое — None. Не похоже на
    имя — ChatInputError (422): строка уходит в системный промпт."""
    name = " ".join(unicodedata.normalize("NFC", value or "").split())   # «й» из двух символов — один
    if not name:
        return None
    if len(name) > USER_NAME_MAX or not USER_NAME.fullmatch(name) or not validate_document(name).ok:
        raise ChatInputError("user_name", f"ожидается имя: одно-три слова из букв, не длиннее {USER_NAME_MAX} знаков")
    return name


def without_images(content: Any, model: str) -> Any:
    if not has_image(content):
        return content
    return [p if p.get("type") != "image_url" else {"type": "text", "text": IMAGE_HIDDEN.format(model=model)}
            for p in content_parts(content)]


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

    async def add_system_message(self, chat_id: UUID, text: str) -> tuple[Chat, ChatMessage]:
        """Сообщение от системы (блок 4.3, POST /chats/{id}/system-message): в историю — как
        ответ ассистента, чтобы модель видела его в следующих ходах. Не посреди чужого ответа."""
        chat = await self.get_chat(chat_id)
        message = ChatMessage(chat_id=chat_id, role="assistant", content=text, tokens=text_tokens(text))
        async with self.locks(chat_id):
            await self.repo.append_message(chat_id, message)
        return chat, message

    async def clear_history(self, chat_id: UUID) -> None:
        await self.get_chat(chat_id)
        async with self.locks(chat_id):          # не посреди чужого ответа
            await self.repo.soft_delete_messages(chat_id)
        log.info("chat_history_cleared", chat_id=str(chat_id))

    def system_prompt(self, chat: Chat, user_name: str | None = None) -> str:
        """Свой промпт чата или CHAT_SYSTEM_PROMPT. Системное сообщение есть всегда: без
        него LLMService подставил бы промпт ассистента /chat со статьями руководства
        (блок 3.7), а он отвечает только на вопросы о продукте — «Как меня зовут?» получил
        бы отказ. user_name (блок 4.3) — подсказка с именем по умолчанию в конце промпта."""
        prompt = chat.system_prompt or self.settings.chat_system_prompt.replace(
            "{product_name}", self.settings.support.product_name)
        if user_name:
            prompt = f"{prompt}\n\n{USER_NAME_HINT.format(name=user_name)}"
        return prompt

    def image_model(self) -> str | None:
        """Модель для запросов с картинкой: CHAT_VISION_MODEL, иначе модель по умолчанию, если
        по каталогу она видит изображения (неизвестную модель не отвергаем — решит провайдер).
        None — картинку показать некому."""
        model = self.settings.chat_vision_model or self.settings.llm.default_model
        return None if supports_images(model) is False else model

    def check_media(self, media: MediaRef) -> None:
        """До сохранения вопроса: картинку должна увидеть модель (MediaError 422 с подсказкой)."""
        if media.kind == "image" and self.image_model() is None:
            log.warning("vision_not_configured", default_model=self.settings.llm.default_model,
                        note="фото не принято: задайте CHAT_VISION_MODEL, например gemma3:4b в Ollama")
            raise MediaError(422, "vision_not_configured",
                             "Фото пока не принимаются: модель ответов не видит изображений. "
                             "Опишите вопрос текстом.")

    def canary(self) -> str | None:
        """Канарейка, которую клиент модели добавит к запросу: у LLMService с включённым
        защитным слоем — app.state.canary, у фейков в тестах и без защиты — None."""
        return getattr(self.llm, "canary", None)

    def build_request(self, chat: Chat, history: list[ChatMessage],
                      user_name: str | None = None) -> tuple[ChatRequest, dict[str, Any]]:
        """Запрос к модели по стратегии контекста и бюджету токенов + цифры для лога."""
        messages = self.strategy.build(self.system_prompt(chat, user_name), history)
        screened_out = 0
        if self.settings.security.enabled:
            screened = screen_messages(messages, self.settings.security.max_input_chars)
            if screened.verdict.ok:              # иначе LLMService сам ответит отказом
                screened_out = len(messages) - len(screened.messages)
                messages = screened.messages
        messages = [m if m["role"] == "system" else {**m, "content": map_text(m["content"], mask_message)}
                    for m in messages]
        if user_name:
            # После проверки входа и маскирования: имя уже проверено clean_user_name, а маскер
            # с Presidio принял бы его за персональные данные. Старше всей истории — при нехватке
            # бюджета уходит первой, подсказка в системном промпте остаётся.
            head = next((i for i, m in enumerate(messages) if m["role"] != "system"), len(messages))
            messages = [*messages[:head], *name_priming(user_name), *messages[head:]]
        image_model = self.image_model()
        if image_model is None:                  # картинка в истории, а смотреть некому
            messages = [{**m, "content": without_images(m["content"], self.settings.llm.default_model)}
                        for m in messages]
        # LLMService добавит к запросу системное сообщение с канарейкой (блок 3.8): модель его
        # получает, значит, и бюджет его учитывает. prompt_tokens_est — оценка всего запроса,
        # её видно в chat_turn_finished рядом с prompt_tokens от модели.
        canary = self.canary()
        added = text_tokens(canary_message(canary)["content"]) + MESSAGE_OVERHEAD if canary else 0
        image_tokens = self.settings.media.image_tokens
        report: dict[str, int] = {}
        fitted = fit_to_budget(messages, self.settings.context_budget - added, image_tokens=image_tokens,
                               report=report)
        images = sum(1 for m in fitted if has_image(m["content"]))
        stats: dict[str, Any] = {
            "history_messages": len(history), "screened_out": screened_out, "context_messages": len(fitted),
            "dropped_by_budget": len(messages) - len(fitted),
            "prompt_tokens_est": count_tokens(with_canary(fitted, canary), image_tokens)}
        if report.get("shrunk"):
            stats["shrunk_by_budget"] = report["shrunk"]
        if images:
            stats["context_images"] = images
        req = MediaChatRequest(messages=fitted, max_tokens=self.settings.response_tokens, session_id=str(chat.id),
                               model=image_model if images else None)
        return req, stats

    # ------------------------------------------------------------------ #
    async def send_message(self, chat_id: UUID, user_content: str, media: MediaRef | None = None,
                           user_name: str | None = None) -> AsyncIterator[str]:
        started = time.perf_counter()
        user_name = clean_user_name(user_name)
        chat = await self.get_chat(chat_id)
        if media is not None:
            self.check_media(media)
        async with self.locks(chat_id):
            # aclosing: клиент ушёл — _turn закрывается здесь же и сохраняет ответ до того,
            # как замок отпустит следующий вопрос.
            async with contextlib.aclosing(self._turn(chat, user_content, media, started, user_name)) as chunks:
                async for chunk in chunks:
                    yield chunk

    def user_message(self, chat_id: UUID, user_content: str, media: MediaRef | None) -> ChatMessage:
        """Вопрос для истории. С вложением без подписи content — пометка вида «[фото]»."""
        if media is None:
            return ChatMessage(chat_id=chat_id, role="user", content=user_content, tokens=text_tokens(user_content))
        message = ChatMessage(chat_id=chat_id, role="user", content=user_content.strip() or placeholder(media),
                              media_refs=media)
        message.tokens = content_tokens(model_content(message), self.settings.media.image_tokens)
        return message

    async def _turn(self, chat: Chat, user_content: str, media: MediaRef | None,
                    started: float, user_name: str | None = None) -> AsyncIterator[str]:
        chat_id = chat.id
        await self.repo.append_message(chat_id, self.user_message(chat_id, user_content, media))
        history = await self.repo.list_messages(chat_id, limit=self.strategy.history_limit)
        req, stats = self.build_request(chat, history, user_name)
        if user_name:
            stats = {**stats, "default_user_name": True}      # само имя в лог не пишется
        if media is not None:
            stats = {**stats, "media": media.kind, "media_bytes": media.size}
        if stats["dropped_by_budget"] or stats.get("shrunk_by_budget"):
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
