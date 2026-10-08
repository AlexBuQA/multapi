"""
Маскирование персональных данных (блок 3.6): сырой текст пользователя в лог не попадает.

В лог-строку о вызове модели кладутся только prompt_hash (отпечаток для поиска
одинаковых запросов) и prompt_preview — первые 120 символов уже замаскированного текста.

Отличие от стартер-кода задания одно: PHONE_RU начинается с (?<!\\w), а не с \\b.
\\b перед «+» не срабатывает, если перед номером пробел (оба символа — не буквы), и
«тел +7 (999) 123-45-67» оставался бы в логе как есть. (?<!\\w) означает «перед номером
не буква и не цифра» — в том числе пробел, скобка или начало строки.
Шаблоны применяются по очереди: каждый следующий работает с текстом, где предыдущие уже
заменены.

Блок 3.8 добавил шаблоны предметной области «Личного кабинета»: API-ключ и JWT-токен
(раздел 5 руководства — ключи создают и по ошибке публикуют), расчётный счёт юрлица
(оплата по счёту, раздел 4.1) и СНИЛС. Они стоят первыми: в ключе и в 20-значном счёте
есть цифры, которые иначе приняли бы за ИНН или карту. Ещё три правки блока 3.8:
- EMAIL начинается с (?<![\\w.+-]), а не с \\b. С \\b строка «a.a.a.a…» без «@» проверялась
  с каждой позиции до конца — квадратичное время, 2,5 с на 32 тыс. символов (столько
  пускает схема запроса). Теперь попытка начинается только в начале слова — линейно;
- ИНН и расчётный счёт не берутся из ряда цифр, к которому примыкает дефис: последняя
  группа UUID («…-446655440000») — не ИНН, а session_id не должен ломаться в логе;
- prompt_preview маскирует только начало текста (4 окна превью), а не весь текст. Тот же redact_pii работает в трёх
местах: на входе модели (блок 3.7), на ответе модели (output_filter, блок 3.8) и в
каждой строке лога (redact_event — процессор structlog, блок 3.8).
"""
from __future__ import annotations

import hashlib
import re

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "API_KEY": re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}"),
    "JWT": re.compile(r"\beyJ[\w-]{5,}\.eyJ[\w-]{5,}\.[\w-]*"),
    "ACCOUNT": re.compile(r"(?<![\w-])\d{20}(?![\w-])"),
    "SNILS": re.compile(r"\b\d{3}-\d{3}-\d{3}[\s-]\d{2}\b"),
    "EMAIL": re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "PHONE_RU": re.compile(r"(?<!\w)(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b"),
    "CARD": re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"),
    "INN": re.compile(r"(?<![\w-])(?:\d{10}|\d{12})(?![\w-])"),
    "PASSPORT": re.compile(r"\b\d{2}\s?\d{2}\s?\d{6}\b"),
}

# Ключи строки лога, которые маскировать нельзя: случайный 12-символьный request_id
# иногда состоит из одних цифр и стал бы «[INN]» — строки запроса потеряли бы связь.
TECHNICAL_LOG_KEYS = frozenset({"timestamp", "level", "logger", "request_id", "trace_id", "prompt_hash"})

PREVIEW_CHARS = 120


def redact_pii(text: str) -> str:
    """Заменяет email, телефон, карту, ИНН и паспорт на [EMAIL], [PHONE_RU] и т. д."""
    for name, pattern in PII_PATTERNS.items():
        text = pattern.sub(f"[{name}]", text)
    return text


def _redact_value(value: object) -> object:
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_redact_value(item) for item in value)
    return value


def redact_event(logger: object, method_name: str, event_dict: dict) -> dict:
    """Процессор structlog (блок 3.8): персональные данные маскируются в каждой строке
    лога — и в своих полях (prompt_preview, answer_preview, user_id из тела запроса), и в
    сообщениях сторонних библиотек, и в трейсбеке. Поля, которые сервис маскирует сам,
    это не отменяет: процессор — страховка на случай, когда кто-то запишет сырой текст."""
    for key, value in event_dict.items():
        if key not in TECHNICAL_LOG_KEYS:
            event_dict[key] = _redact_value(value)
    return event_dict


def prompt_hash(text: str) -> str:
    """Отпечаток промпта: одинаковые запросы находятся по логу без хранения текста."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def prompt_preview(text: str) -> str:
    """Начало промпта для лога — только после маскирования. Маскируется окно в 4 превью,
    а не весь текст: длинное сообщение не замедляет запрос, а данные короче 360 символов,
    начатые в первых 120, видны в окне целиком."""
    return redact_pii(text[: PREVIEW_CHARS * 4])[:PREVIEW_CHARS]
