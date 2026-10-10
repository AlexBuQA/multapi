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

Вложения (блок 4.3). Сообщение с media_refs уходит модели списком content-part: подпись
пользователя и part вложения (SlidingWindow.build). Картинка в бюджете — MEDIA__IMAGE_TOKENS
(оценка: base64 картинки модель токенами текста не считает). Текст документа (до 30 000
символов, это 7–10 тыс. токенов) в окно llama3.2 целиком не помещается, а лишнее Ollama
молча отрезала бы с начала — вместе с системным промптом. Поэтому fit_to_budget не
выбрасывает такое сообщение, а укорачивает текст вложения до остатка бюджета с пометкой
«…не поместился в контекст»: последний вопрос — всегда, сообщение из истории — если
остаётся хотя бы MIN_SHRUNK_TOKENS. Так на уточняющий вопрос о документе модель видит его
начало. Документ в последнем вопросе оставляет до четверти бюджета прежней истории.

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
from app.schemas.chat import content_parts

log = get_logger()

ENCODING_NAME = "o200k_base"
ENCODING_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
MESSAGE_OVERHEAD = 4     # <|im_start|>role ... <|im_end|> — на каждое сообщение
REPLY_OVERHEAD = 2       # начало ответа ассистента — на весь запрос
# Предел длины сообщения в ChatRequest (app/schemas/chat.py). Длиннее бывает только ответ
# модели при большом RESPONSE_TOKENS — такой ответ уходит в контекст обрезанным.
MAX_MESSAGE_CHARS = 32_000
IMAGE_TOKENS = 800          # картинка в бюджете по умолчанию (MEDIA__IMAGE_TOKENS)
MIN_SHRUNK_TOKENS = 200     # меньше — укороченный документ из истории уже бесполезен
SHRUNK_NOTE = "\n[…текст вложения не поместился в контекст модели: показано {shown} из {total} символов]"


def tiktoken_cache_file(url: str = ENCODING_URL) -> Path:
    """Где tiktoken ищет скачанный словарь: имя файла — sha1 от адреса (tiktoken.load)."""
    folder = (os.environ.get("TIKTOKEN_CACHE_DIR") or os.environ.get("DATA_GYM_CACHE_DIR")
              or os.path.join(tempfile.gettempdir(), "data-gym-cache"))
    return Path(folder) / hashlib.sha1(url.encode()).hexdigest()


def download_with_system_certs(url: str = ENCODING_URL) -> None:
    """Скачивает словарь с проверкой HTTPS по хранилищу сертификатов ОС и кладёт в кеш
    tiktoken. Хеш содержимого tiktoken проверит сам при загрузке. url — другой словарь,
    например cl100k_base для подсчёта токенов эмбеддингов OpenAI (блок 5.1)."""
    import ssl

    import httpx
    import truststore

    response = httpx.get(url, verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
                         timeout=60, follow_redirects=True)
    response.raise_for_status()
    path = tiktoken_cache_file(url)
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


def content_tokens(content: str | Sequence[Any], image_tokens: int = IMAGE_TOKENS) -> int:
    """Токены содержимого сообщения: текст — tiktoken, картинка — оценка image_tokens."""
    if isinstance(content, str):
        return text_tokens(content)
    return sum(text_tokens(p["text"]) if p.get("type") == "text" else image_tokens for p in content_parts(content))


def message_tokens(message: Mapping[str, Any], image_tokens: int = IMAGE_TOKENS) -> int:
    return content_tokens(message["content"], image_tokens) + MESSAGE_OVERHEAD


def count_tokens(messages: Iterable[Mapping[str, Any]], image_tokens: int = IMAGE_TOKENS) -> int:
    """Токены запроса к модели: сообщения + 4 на каждое + 2 на весь запрос."""
    return sum(message_tokens(m, image_tokens) for m in messages) + REPLY_OVERHEAD


def truncate_tokens(text: str, limit: int) -> str:
    """Начало текста не длиннее limit токенов."""
    if limit <= 0:
        return ""
    encoding = _encoding()
    if encoding is not None:
        tokens = encoding.encode(text, disallowed_special=())
        return text if len(tokens) <= limit else encoding.decode(tokens[:limit])
    data = text.encode("utf-8")
    return text if len(data) <= limit * 4 else data[:limit * 4].decode("utf-8", errors="ignore")


def shrink(message: Mapping[str, Any], room: int, image_tokens: int = IMAGE_TOKENS) -> dict[str, Any] | None:
    """Сообщение, укороченное до room токенов за счёт текста вложений (part с пометкой media),
    самый длинный — первым. None — укорачивать нечего. Если и пустого текста вложения мало
    (длинная подпись, картинка), результат может остаться больше room."""
    content = message["content"]
    if isinstance(content, str):
        return None
    parts = content_parts(content)
    over = message_tokens(message, image_tokens) - room
    order = sorted((i for i, p in enumerate(parts) if p.get("type") == "text" and p.get("media")),
                   key=lambda i: -len(parts[i]["text"]))
    for i in order:
        if over <= 0:
            break
        text = parts[i]["text"]
        note_tokens = text_tokens(SHRUNK_NOTE.format(shown=len(text), total=len(text)))
        keep = text_tokens(text) - over - note_tokens
        head = truncate_tokens(text, keep)
        if len(head) < len(text):
            parts[i] = {**parts[i], "text": head + SHRUNK_NOTE.format(shown=len(head), total=len(text))}
        over = content_tokens(parts, image_tokens) + MESSAGE_OVERHEAD - room
    if not order:
        return None
    return {**message, "content": parts}


def fit_to_budget(messages: Sequence[Mapping[str, Any]], budget: int, *, image_tokens: int = IMAGE_TOKENS,
                  report: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """Режет историю с начала, пока запрос не уложится в budget. Системные сообщения в
    начале и последнее сообщение (вопрос пользователя) остаются всегда — даже если сами
    не помещаются: тогда запрос всё равно уходит, а ограничение контекста решает модель.
    Текст вложений (документ, расшифровка голоса) укорачивается, а не выбрасывается — см.
    docstring модуля. report["shrunk"] — сколько сообщений укорочено."""
    items = [dict(m) for m in messages]
    head_len = 0
    while head_len < len(items) and items[head_len]["role"] == "system":
        head_len += 1
    head, body = items[:head_len], items[head_len:]
    shrunk = 0
    if not body:
        return items
    total = count_tokens(head, image_tokens)
    last = body[-1]
    if total + message_tokens(last, image_tokens) > budget:
        # Документ в последнем вопросе не забирает весь бюджет: до четверти остаётся истории —
        # иначе модель забыла бы, что пользователь сказал перед тем, как прислать файл.
        reserve = min(sum(message_tokens(m, image_tokens) for m in body[:-1]), budget // 4)
        smaller = shrink(last, budget - total - reserve, image_tokens)
        if smaller is not None:
            last, shrunk = smaller, 1
    total += message_tokens(last, image_tokens)
    kept: list[dict[str, Any]] = []
    for message in reversed(body[:-1]):          # от новых к старым, пока помещается
        size = message_tokens(message, image_tokens)
        if total + size <= budget:
            kept.append(message)
            total += size
            continue
        room = budget - total
        smaller = shrink(message, room, image_tokens) if room >= MIN_SHRUNK_TOKENS else None
        if smaller is not None and total + message_tokens(smaller, image_tokens) <= budget:
            kept.append(smaller)
            shrunk += 1
        break
    if report is not None:
        report["shrunk"] = shrunk
    return [*head, *reversed(kept), last]


class ContextStrategy(Protocol):
    name: str

    @property
    def history_limit(self) -> int:
        """Сколько последних сообщений читать из хранилища."""

    def build(self, system_prompt: str | None, history: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        """Сообщения для модели: системный промпт и история в хронологическом порядке."""


def model_content(message: ChatMessage) -> str | list[dict[str, Any]]:
    """Содержимое сообщения для модели. С вложением — [подпись, part]; у text-части
    вложения — пометка media (audio/document) для защитного слоя и бюджета."""
    ref = message.media_refs
    if ref is None:
        return message.content[:MAX_MESSAGE_CHARS]
    part = dict(ref.part)
    if part.get("type") == "text":
        part["media"] = "audio" if ref.kind == "audio" else "document"
    return [{"type": "text", "text": message.content[:MAX_MESSAGE_CHARS]}, part]


@dataclass(frozen=True)
class SlidingWindow:
    """Последние window сообщений и системный промпт перед ними."""

    window: int = 10
    name: str = "sliding"

    @property
    def history_limit(self) -> int:
        return self.window

    def build(self, system_prompt: str | None, history: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages += [{"role": m.role, "content": model_content(m)}
                     for m in history[-self.window:] if m.content or m.media_refs]
        return messages


def make_strategy(name: str, window: int) -> ContextStrategy:
    if name == "sliding":
        return SlidingWindow(window=window)
    if name == "hybrid":
        raise ValueError("CHAT_CONTEXT_STRATEGY=hybrid в блоке 4.1 не реализована: выбрано скользящее окно "
                         "(обоснование — docs/chat.md). Уберите переменную или задайте sliding.")
    raise ValueError(f"CHAT_CONTEXT_STRATEGY={name!r}: ожидается sliding")
