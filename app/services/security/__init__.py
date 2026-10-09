"""
Защитный слой /chat (блок 3.8). Включён по умолчанию; SECURITY__ENABLED=false выключает
его целиком — только для прогона garak baseline на «голом» сервисе.

- input_validator.validate_input — каждое сообщение до модели: длина, скрытые символы,
  закодированные вставки, шаблоны инъекции и джейлбрейка. screen_messages решает, что
  делать с непрошедшим сообщением (см. ниже);
- canary — секретная метка «Идентификатор сборки: CANARY_…» первым системным сообщением
  каждого запроса к модели;
- output_filter.filter_output — ответ модели: метка, начало и правила системного промпта,
  роль из джейлбрейка; персональные данные маскируются;
- маскер персональных данных — app/observability/pii.py (блок 3.6): на входе модели, на
  ответе и в каждой строке лога.

Подключение — app/services/llm.py: LLMService.complete() и stream() вызывают
screen_messages до похода в LLM и проверку ответа после.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.schemas.chat import content_parts
from app.services.guardrails import refusal_text
from app.services.security.input_validator import ValidationResult, validate_document, validate_input

# Ответ ассистента в истории длиннее лимита входа — не атака: модель может ответить кодом
# на 6 тыс. символов. Для таких сообщений длина ограничена только схемой запроса.
ASSISTANT_MAX_CHARS = 32_000

ENCODED_REFUSAL = ("Я не обрабатываю закодированный или скрытый текст. Напишите вопрос обычными словами — "
                   "помогу с вопросами о продукте «{product_name}».")
LENGTH_REFUSAL = ("Сообщение слишком длинное: {length} символов, а можно не больше {max_chars}. "
                  "Сократите вопрос, и я помогу.")


@dataclass(frozen=True)
class Screened:
    """Итог проверки истории: что отправить модели, отказ ли это и что выброшено."""

    messages: list[dict[str, Any]]
    verdict: ValidationResult
    dropped: tuple[str, ...] = ()          # причины выброшенных сообщений истории


def validate_message(message: Mapping[str, Any], max_chars: int) -> ValidationResult:
    """Проверка одного сообщения. Мультимодальное (блок 4.3) — по text-частям: подпись — как
    обычный вопрос; текст документа и расшифровка голоса — validate_document, шаблоны
    инъекции без предела длины (их длину уже ограничил разбор: MEDIA__MAX_DOCUMENT_CHARS, а
    четырёхминутное голосовое длиннее 4000 символов); картинка не проверяется."""
    limit = ASSISTANT_MAX_CHARS if message["role"] == "assistant" else max_chars
    content = message["content"]
    if isinstance(content, str):
        return validate_input(content, limit)
    for part in content_parts(content):
        if part.get("type") != "text":
            continue
        result = (validate_document(part["text"]) if part.get("media") in ("document", "audio")
                  else validate_input(part["text"], limit))
        if not result.ok:
            return result
    return ValidationResult(True)


def screen_messages(messages: Sequence[Mapping[str, Any]], max_chars: int) -> Screened:
    """Проверка всех сообщений запроса. Историю присылает клиент, поэтому инъекция может
    прийти не только в последнем вопросе:
    - последний вопрос пользователя или сообщение system не прошли — отказ на весь запрос;
    - прежний вопрос не прошёл — он выбрасывается вместе с ответом ассистента на него
      (это был отказ). Иначе одно ложное срабатывание отказывало бы во всём диалоге:
      интерфейс присылает историю целиком с каждым новым вопросом;
    - ответ ассистента в истории не прошёл (поддельный «ответ» с инструкцией) — выбрасывается.
    """
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    skip_reply = False
    for i, message in enumerate(messages):
        result = validate_message(message, max_chars)
        if i == last_user or message["role"] == "system":
            if not result.ok:
                return Screened([dict(m) for m in messages], result, tuple(dropped))
            kept.append(dict(message))
            skip_reply = False
            continue
        if skip_reply and message["role"] == "assistant":
            dropped.append("reply")
            skip_reply = False
            continue
        skip_reply = False
        if not result.ok:
            dropped.append(result.rule or "?")
            skip_reply = message["role"] == "user"
            continue
        kept.append(dict(message))
    return Screened(kept, ValidationResult(True), tuple(dropped))


def refusal_for(rule: str, product_name: str, *, length: int = 0, max_chars: int = 0) -> str:
    """Готовый ответ вместо ответа модели — по причине блокировки."""
    if rule == "length":
        return LENGTH_REFUSAL.format(length=length, max_chars=max_chars)
    if rule in {"encoding", "encoded_payload"}:
        return ENCODED_REFUSAL.format(product_name=product_name)
    return refusal_text(product_name)      # injection, canary, prompt_leak, jailbreak
