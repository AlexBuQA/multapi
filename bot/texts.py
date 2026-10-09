"""
Тексты бота (блоки 4.2–4.3) — в одном месте, чтобы править формулировки, не трогая handlers.

user_message(exc) переводит ошибку обращения к chat-сервису в понятное сообщение: трассировка
пользователю не показывается никогда, она остаётся в логе бота.
- не подключиться к сервису (ConnectError, ConnectTimeout) — «Сервис недоступен»;
- сервис не ответил вовремя (ReadTimeout) — «Ответ занимает слишком долго»;
- 429 — «Слишком много запросов»; 5xx — «Внутренняя ошибка сервиса», а для известных
  кодов — точнее: модель или хранилище недоступны, модель не ответила вовремя;
- вложение не принято (коды media_*, audio_*, vision_*: файл велик, тип не тот, голос не
  настроен) — текст ошибки из ответа сервиса как есть: он написан для пользователя.
"""
from __future__ import annotations

import httpx

from bot.services.backend_client import BackendStreamError, error_body

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
        "Можно прислать фото или скриншот, голосовое сообщение, документ PDF или DOCX — "
        "подпись к файлу станет вопросом. Раздел можно выбрать через /ask.\n\n"
        "/clear — начать разговор заново, /help — все команды."
    )


def help_text() -> str:
    lines = [f"/{name} — {description}" for name, description in COMMANDS]
    return ("Команды:\n" + "\n".join(lines) + "\n\nЛюбое другое сообщение — вопрос ассистенту: текст, фото, "
            "голосовое, PDF или DOCX до 10 МБ.")


HISTORY_CLEARED = "История очищена. Следующее сообщение начнёт разговор с чистого листа."
NOTHING_TO_CANCEL = "Отменять нечего: сценарий не запущен."
CANCELLED = "Отменено. Можно задать вопрос обычным сообщением или начать заново: /ask"
ASK_TOPIC = "Выберите раздел:"
ASK_QUESTION = "Раздел: {topic}. Напишите вопрос одним сообщением — можно с фото или документом. Отмена — /cancel"
PICK_TOPIC = "Выберите раздел кнопкой выше или отмените: /cancel"
TOPIC_STALE = "Этот выбор уже неактуален. Начните заново: /ask"
UNKNOWN_TOPIC = "Такого раздела нет."
UNKNOWN_COMMAND = "Такой команды нет. Список команд — /help"
ONLY_TEXT = "Такое сообщение я не разберу. Понимаю текст, фото, голосовые сообщения и документы PDF и DOCX."

# Медиа (блок 4.3). Вопрос по умолчанию — если у файла нет подписи: сервису нужен текст
# вопроса (поле content), а модели — понятная задача.
PHOTO_PROMPT = "Посмотри на изображение и помоги разобраться: что на нём и что делать?"
VOICE_PROMPT = "Ответь на голосовое сообщение."
DOCUMENT_PROMPT = "Кратко перескажи документ и скажи, на какие вопросы он отвечает."
DOCUMENT_ONLY_PDF_DOCX = "Документы принимаю в форматах PDF и DOCX. Старый .doc сохраните как DOCX или PDF."
FILE_TOO_LARGE = "Файл больше {limit} МБ — пришлите поменьше."
PHOTO_TOO_LARGE = "Фото больше {limit} МБ — пришлите его сжатым, обычной отправкой фото."
DOWNLOAD_FAILED = "Не получилось скачать файл из Telegram. Попробуйте прислать его ещё раз."
EMPTY_ANSWER = "Ассистент не прислал ответа. Попробуйте переформулировать вопрос."
INTERRUPTED = "\n\n⚠️ Ответ прерван: {reason}"
UNEXPECTED = "Что-то пошло не так. Попробуйте ещё раз чуть позже."
ADMIN_ONLY = "Команда доступна только администраторам бота."

UNAVAILABLE = "Сервис недоступен, попробуйте позже."
TIMEOUT = "Ответ занимает слишком долго. Попробуйте ещё раз или задайте вопрос короче."
RATE_LIMITED = "Слишком много запросов, подождите минуту."
REJECTED = "Сообщение не принято: оно слишком длинное или содержит недопустимые символы."
STORAGE_DOWN = "История диалогов сейчас недоступна. Попробуйте через пару минут."
MODEL_DOWN = "Модель сейчас недоступна. Попробуйте через пару минут."
SERVICE_ERROR = "Внутренняя ошибка сервиса. Попробуйте позже."
CLIENT_ERROR = "Сервис не принял сообщение (ошибка {status})."

# Коды ошибок сервиса -> текст; известный код точнее HTTP-статуса.
CODE_TEXTS = {
    "llm_timeout": TIMEOUT,
    "llm_unavailable": MODEL_DOWN,
    "llm_error": MODEL_DOWN,
    "llm_rate_limit": RATE_LIMITED,
    "rate_limited": RATE_LIMITED,
    "chat_storage_unavailable": STORAGE_DOWN,
    "validation_error": REJECTED,
}
# Ошибки вложений: текст сервиса написан для пользователя — показываем как есть.
SERVICE_TEXT_PREFIXES = ("media_", "audio_", "vision_")


def user_message(exc: BaseException) -> str:
    if isinstance(exc, httpx.ConnectTimeout):             # не подключились — сервис недоступен, а не «долго»
        return UNAVAILABLE
    if isinstance(exc, httpx.TimeoutException):           # ReadTimeout, WriteTimeout, PoolTimeout
        return TIMEOUT
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        body = error_body(exc)
        code, message = str(body.get("code") or ""), str(body.get("message") or "")
        if code.startswith(SERVICE_TEXT_PREFIXES) and message:
            return message
        if code in CODE_TEXTS:
            return CODE_TEXTS[code]
        if status == 429:
            return RATE_LIMITED
        if status == 422:
            return REJECTED
        if status == 504:
            return TIMEOUT
        if status >= 500:
            return SERVICE_ERROR
        return CLIENT_ERROR.format(status=status)
    if isinstance(exc, httpx.HTTPError):                  # ConnectError, RemoteProtocolError, ...
        return UNAVAILABLE
    if isinstance(exc, BackendStreamError):
        return CODE_TEXTS.get(exc.code, SERVICE_ERROR)
    return UNEXPECTED
