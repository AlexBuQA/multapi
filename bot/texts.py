"""
Тексты бота (блок 4.2) — в одном месте, чтобы править формулировки, не трогая handlers.

user_message(exc) переводит ошибку обращения к chat-сервису в понятное сообщение: трассировка
пользователю не показывается никогда, она остаётся в логе бота.
"""
from __future__ import annotations

import httpx

from bot.services.backend_client import BackendStreamError, error_code

COMMANDS: list[tuple[str, str]] = [
    ("start", "начать и показать подсказку"),
    ("ask", "вопрос по теме: выбрать раздел, затем написать вопрос"),
    ("clear", "очистить историю диалога"),
    ("cancel", "отменить начатый сценарий /ask"),
    ("help", "список команд"),
]


def start_text(product_name: str) -> str:
    return (
        f"Здравствуйте! Я ассистент техподдержки «{product_name}».\n\n"
        "Напишите вопрос обычным сообщением — отвечу и запомню контекст разговора. "
        "Можно выбрать раздел через /ask.\n\n"
        "/clear — начать разговор заново, /help — все команды."
    )


def help_text() -> str:
    lines = [f"/{name} — {description}" for name, description in COMMANDS]
    return "Команды:\n" + "\n".join(lines) + "\n\nЛюбое другое сообщение — вопрос ассистенту."


HISTORY_CLEARED = "История очищена. Следующее сообщение начнёт разговор с чистого листа."
NOTHING_TO_CANCEL = "Отменять нечего: сценарий не запущен."
CANCELLED = "Отменено. Можно задать вопрос обычным сообщением или начать заново: /ask"
ASK_TOPIC = "Выберите раздел:"
ASK_QUESTION = "Раздел: {topic}. Напишите вопрос одним сообщением. Отмена — /cancel"
PICK_TOPIC = "Выберите раздел кнопкой выше или отмените: /cancel"
TOPIC_STALE = "Этот выбор уже неактуален. Начните заново: /ask"
UNKNOWN_TOPIC = "Такого раздела нет."
UNKNOWN_COMMAND = "Такой команды нет. Список команд — /help"
ONLY_TEXT = "Пока я понимаю только текст. Опишите вопрос словами."
EMPTY_ANSWER = "Ассистент не прислал ответа. Попробуйте переформулировать вопрос."
INTERRUPTED = "\n\n⚠️ Ответ прерван: {reason}"
UNEXPECTED = "Что-то пошло не так. Попробуйте ещё раз чуть позже."
ADMIN_ONLY = "Команда доступна только администраторам бота."

UNAVAILABLE = "Сервис ответов сейчас недоступен. Попробуйте через пару минут."
TIMEOUT = "Ассистент отвечает слишком долго. Попробуйте ещё раз или задайте вопрос короче."
RATE_LIMITED = "Слишком много сообщений подряд. Подождите минуту и повторите."
REJECTED = "Сообщение не принято: оно слишком длинное или содержит недопустимые символы."
STORAGE_DOWN = "История диалогов сейчас недоступна. Попробуйте через пару минут."
MODEL_DOWN = "Модель сейчас недоступна. Попробуйте через пару минут."
SERVICE_ERROR = "Сервис ответил ошибкой ({status}). Попробуйте позже."

STREAM_ERRORS = {
    "llm_timeout": TIMEOUT,
    "llm_unavailable": MODEL_DOWN,
    "llm_error": MODEL_DOWN,
    "chat_storage_unavailable": STORAGE_DOWN,
}


def user_message(exc: BaseException) -> str:
    if isinstance(exc, httpx.TimeoutException):          # раньше ConnectError: ConnectTimeout — тоже таймаут
        return TIMEOUT
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        code = error_code(exc)
        if status == 429:
            return RATE_LIMITED
        if status == 422:
            return REJECTED
        if status == 503 or code == "chat_storage_unavailable":
            return STORAGE_DOWN
        if status == 504:
            return TIMEOUT
        if status == 502:
            return MODEL_DOWN
        return SERVICE_ERROR.format(status=status)
    if isinstance(exc, httpx.HTTPError):                 # ConnectError, RemoteProtocolError, ...
        return UNAVAILABLE
    if isinstance(exc, BackendStreamError):
        return STREAM_ERRORS.get(exc.code, UNAVAILABLE)
    return UNEXPECTED
