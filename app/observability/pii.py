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
"""
from __future__ import annotations

import hashlib
import re

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "PHONE_RU": re.compile(r"(?<!\w)(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b"),
    "CARD": re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"),
    "INN": re.compile(r"\b(?:\d{10}|\d{12})\b"),
    "PASSPORT": re.compile(r"\b\d{2}\s?\d{2}\s?\d{6}\b"),
}

PREVIEW_CHARS = 120


def redact_pii(text: str) -> str:
    """Заменяет email, телефон, карту, ИНН и паспорт на [EMAIL], [PHONE_RU] и т. д."""
    for name, pattern in PII_PATTERNS.items():
        text = pattern.sub(f"[{name}]", text)
    return text


def prompt_hash(text: str) -> str:
    """Отпечаток промпта: одинаковые запросы находятся по логу без хранения текста."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def prompt_preview(text: str) -> str:
    """Начало промпта для лога — только после маскирования."""
    return redact_pii(text)[:PREVIEW_CHARS]
