"""
Контекст для модели (блок 4.1): стратегия окна истории и бюджет токенов.

Стратегия — скользящее окно (sliding window): последние CHAT_CONTEXT_WINDOW сообщений
чата, перед ними системный промпт. Почему она, а не hybrid (сводка старой истории +
последние M), — в docs/chat.md. Значение hybrid в CHAT_CONTEXT_STRATEGY допустимо по
схеме настроек, но make_strategy() его отвергает: сервис не стартует с понятной ошибкой,
а не молча работает со скользящим окном.

Бюджет токенов: CONTEXT_WINDOW - RESPONSE_TOKENS - SAFETY_MARGIN. Если окно истории в него
не помещается, fit_to_budget() выбрасывает самые старые сообщения; системный промпт и
последний вопрос остаются всегда.

Подсчёт — count_tokens() через tiktoken, кодировка o200k_base (GPT-4o и новее): длина
каждого сообщения плюс 4 токена служебной разметки ChatML на сообщение и 2 на весь
запрос. Для других моделей (llama3.2 в Ollama) это оценка: у них свой токенизатор и своя
разметка — расхождение измеряет scripts/check_tokens.py. Запас на неточность —
SAFETY_MARGIN.

Словарь o200k_base tiktoken скачивает при первом вызове (около 3,6 МБ) и кладёт в кеш:
TIKTOKEN_CACHE_DIR, по умолчанию папка data-gym-cache во временном каталоге. Сервис
загружает его при старте, в потоке (preload_encoding). Если загрузка не удалась —
например, сеть проверяет HTTPS своим сертификатом, которому certifi не доверяет (см.
LLM__USE_SYSTEM_CERTS), — словарь скачивается ещё раз с проверкой по хранилищу
сертификатов ОС (truststore) и кладётся в тот же кеш. Не вышло и так — подсчёт идёт по
длине текста в байтах UTF-8 / 4 (для кириллицы это оценка с запасом), а в лог один раз
пишется tiktoken_unavailable. Неудача запоминается до перезапуска сервиса: повторять
загрузку на каждом запросе значило бы блокировать его ожиданием сети.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from app.chat.domain import ChatMessage
from app.observability.logging import get_logger

log = get_logger()

ENCODING_NAME = "o200k_base"
ENCODING_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
MESSAGE_OVERHEAD = 4     # <|im_start|>role ... <|im_end|> — на каждое сообщение
REPLY_OVERHEAD = 2       # начало ответа ассистента — на весь запрос
# Предел длины сообщения в ChatRequest (app/schemas/chat.py). Длиннее бывает только ответ
# модели при большом RESPONSE_TOKENS — такой ответ уходит в контекст обрезанным.
MAX_MESSAGE_CHARS = 32_000


def tiktoken_cache_file() -> Path:
    """Где tiktoken ищет скачанный словарь: имя файла — sha1 от адреса (tiktoken.load)."""
    folder = (os.environ.get("TIKTOKEN_CACHE_DIR") or os.environ.get("DATA_GYM_CACHE_DIR")
              or os.path.join(tempfile.gettempdir(), "data-gym-cache"))
    return Path(folder) / hashlib.sha1(ENCODING_URL.encode()).hexdigest()


def download_with_system_certs() -> None:
    """Скачивает словарь с проверкой HTTPS по хранилищу сертификатов ОС и кладёт в кеш
    tiktoken. Хеш содержимого tiktoken проверит сам при загрузке."""
    import ssl

    import httpx
    import truststore

    response = httpx.get(ENCODING_URL, verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
                         timeout=60, follow_redirects=True)
    response.raise_for_status()
    path = tiktoken_cache_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(response.content)
    os.replace(tmp, path)


@lru_cache(maxsize=1)
def _encoding() -> Any | None:
    try:
        import tiktoken
    except ImportError as exc:
        log.warning("tiktoken_unavailable", encoding=ENCODING_NAME, error=repr(exc)[:200],
                    note="токены считаются по длине текста: байты UTF-8 / 4")
        return None
    try:
        return tiktoken.get_encoding(ENCODING_NAME)
    except Exception as first:  # noqa: BLE001 — нет сети, сертификат, битый кеш
        try:
            download_with_system_certs()
            return tiktoken.get_encoding(ENCODING_NAME)
        except Exception as exc:  # noqa: BLE001 — нет truststore, нет сети
            log.warning("tiktoken_unavailable", encoding=ENCODING_NAME, error=repr(first)[:200],
                        retry_error=repr(exc)[:200], note="токены считаются по длине текста: байты UTF-8 / 4")
            return None


def preload_encoding() -> bool:
    """Загрузить словарь заранее (lifespan, в потоке). True — подсчёт точный."""
    ready = _encoding() is not None
    if ready:
        log.info("tiktoken_ready", encoding=ENCODING_NAME)
    return ready


def text_tokens(text: str) -> int:
    """Токены одного текста без служебной разметки."""
    encoding = _encoding()
    if encoding is not None:
        # disallowed_special=(): «<|endoftext|>» в тексте пользователя — обычные символы,
        # а не ошибка ValueError, как по умолчанию в tiktoken.
        return len(encoding.encode(text, disallowed_special=()))
    return -(-len(text.encode("utf-8")) // 4)


def count_tokens(messages: Iterable[Mapping[str, str]]) -> int:
    """Токены запроса к модели: сообщения + 4 на каждое + 2 на весь запрос."""
    return sum(text_tokens(m["content"]) + MESSAGE_OVERHEAD for m in messages) + REPLY_OVERHEAD


def fit_to_budget(messages: Sequence[Mapping[str, str]], budget: int) -> list[dict[str, str]]:
    """Режет историю с начала, пока запрос не уложится в budget. Системные сообщения в
    начале и последнее сообщение (вопрос пользователя) остаются всегда — даже если сами
    не помещаются: тогда запрос всё равно уходит, а ограничение контекста решает модель."""
    items = [dict(m) for m in messages]
    head_len = 0
    while head_len < len(items) and items[head_len]["role"] == "system":
        head_len += 1
    head, body = items[:head_len], items[head_len:]
    if len(body) <= 1:
        return items
    history, last = body[:-1], body[-1]
    sizes = [text_tokens(m["content"]) + MESSAGE_OVERHEAD for m in history]
    total = count_tokens([*head, last]) + sum(sizes)
    start = 0
    while start < len(history) and total > budget:
        total -= sizes[start]
        start += 1
    return [*head, *history[start:], last]


class ContextStrategy(Protocol):
    name: str

    @property
    def history_limit(self) -> int:
        """Сколько последних сообщений читать из хранилища."""

    def build(self, system_prompt: str | None, history: Sequence[ChatMessage]) -> list[dict[str, str]]:
        """Сообщения для модели: системный промпт и история в хронологическом порядке."""


@dataclass(frozen=True)
class SlidingWindow:
    """Последние window сообщений и системный промпт перед ними."""

    window: int = 10
    name: str = "sliding"

    @property
    def history_limit(self) -> int:
        return self.window

    def build(self, system_prompt: str | None, history: Sequence[ChatMessage]) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages += [{"role": m.role, "content": m.content[:MAX_MESSAGE_CHARS]}
                     for m in history[-self.window:] if m.content]
        return messages


def make_strategy(name: str, window: int) -> ContextStrategy:
    if name == "sliding":
        return SlidingWindow(window=window)
    if name == "hybrid":
        raise ValueError("CHAT_CONTEXT_STRATEGY=hybrid в блоке 4.1 не реализована: выбрано скользящее окно "
                         "(обоснование — docs/chat.md). Уберите переменную или задайте sliding.")
    raise ValueError(f"CHAT_CONTEXT_STRATEGY={name!r}: ожидается sliding")
