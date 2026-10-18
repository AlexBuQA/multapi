"""
Инструменты ReAct-агента (блок 6.2): те же три функции, что у наивного агента (app/tools/naive_tools.py,
не меняется — это baseline), но описания переписаны по чек-листу проектирования инструментов:

- имя — глагол: search_knowledge_base, get_current_time, send_telegram_message;
- description отвечает на четыре вопроса: что делает, когда вызывать, когда НЕ вызывать, что возвращает
  (и что делать с пустым результатом);
- строгая JSON Schema: "strict": true, все поля обязательны, additionalProperties: false, у строк — pattern.

Ollama ключ strict не применяет и шаблоны не проверяет, поэтому validate_args сверяет аргументы со
схемой сама: лишнее поле, пропущенное поле, не строка, не тот формат — ошибка для модели до вызова
функции, а не исключение внутри неё.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from app.tools.naive_tools import (
    get_current_time,
    search_knowledge_base,
    send_telegram_message,
)

DISPATCH: dict[str, Callable[..., str]] = {
    "search_knowledge_base": search_knowledge_base,
    "get_current_time": get_current_time,
    "send_telegram_message": send_telegram_message,
}


def _tool(name: str, what: str, when: str, when_not: str, returns: str,
          properties: dict[str, dict[str, str]]) -> dict[str, Any]:
    description = f"Что делает: {what} Когда вызывать: {when} Когда НЕ вызывать: {when_not} Что возвращает: {returns}"
    return {"type": "function", "function": {
        "name": name, "description": description, "strict": True,
        "parameters": {"type": "object", "properties": properties, "required": list(properties),
                       "additionalProperties": False}}}


TOOLS: list[dict[str, Any]] = [
    _tool("search_knowledge_base",
          what="ищет в базе знаний техподдержки «Личного кабинета» (аккаунт и вход, тарифы и оплата, возвраты, "
               "API, уведомления, мобильное приложение, интеграции, команда) один самый близкий фрагмент справки.",
          when="перед любым ответом о правилах, сроках, ценах, лимитах и настройках продукта; по одному вопросу "
               "за вызов.",
          when_not="для текущих даты и времени, для отправки сообщений, для данных конкретного клиента (баланс, "
                   "статус заявки, пароль) — их в базе нет; повторно с тем же запросом.",
          returns="строку «[файл, близость 0.71] текст фрагмента» или «В базе знаний ничего не найдено …» — тогда "
                  "данных нет: не выдумывай ответ и не ищи то же самое снова.",
          properties={"query": {"type": "string", "pattern": r"^[\s\S]{2,300}$",
                                "description": "Вопрос по-русски одной фразой, 2–300 символов, например «за сколько "
                                               "дней приходит письмо об автопродлении»"}}),
    _tool("get_current_time",
          what="возвращает текущие дату и время в часовом поясе IANA.",
          when="когда в ответе нужны сегодняшняя дата, текущее время или срок, отсчитанный от сегодняшнего дня.",
          when_not="если дата уже есть в задаче или в результате прошлого вызова; для времени на других планетах "
                   "и в выдуманных поясах.",
          returns="строку ISO 8601, например 2026-10-18T16:34:08+03:00; неизвестный пояс — ошибку.",
          properties={"timezone": {"type": "string", "pattern": r"^[A-Za-z_]+(/[A-Za-z0-9_+-]+){0,2}$",
                                   "description": "Часовой пояс IANA: Europe/Moscow, Asia/Yekaterinburg, UTC"}}),
    _tool("send_telegram_message",
          what="отправляет клиенту текстовое сообщение в Telegram-чат по chat_id. Действие необратимо: клиент "
               "сразу увидит текст.",
          when="только когда задача прямо просит написать клиенту, указан числовой chat_id и все факты для "
               "текста уже получены из базы знаний или других инструментов.",
          when_not="без chat_id (номер телефона, имя и email — не chat_id); для звонков, email и SMS; если факты "
                   "для текста не проверены инструментами.",
          returns="строку «Сообщение отправлено в <chat_id>».",
          properties={"chat_id": {"type": "string", "pattern": r"^-?[0-9]{1,20}$",
                                  "description": "Числовой идентификатор Telegram-чата клиента, например 12345"},
                      "text": {"type": "string", "pattern": r"^[\s\S]{1,4096}$",
                               "description": "Готовый текст сообщения клиенту по-русски, до 4096 символов"}}),
]

def schemas(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Имя инструмента → JSON Schema его аргументов."""
    return {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}


def validate_args(schema: dict[str, Any], args: Any) -> str | None:
    """Аргументы против строгой схемы инструмента. None — всё верно, иначе — что не так (для модели)."""
    if not isinstance(args, dict):
        return "аргументы должны быть объектом JSON"
    properties = schema.get("properties", {})
    extra = sorted(set(args) - set(properties))
    if extra and schema.get("additionalProperties") is False:
        return f"лишние поля {extra}; допустимы только {list(properties)}"
    missing = [field for field in schema.get("required", []) if field not in args]
    if missing:
        return f"нет обязательных полей {missing}"
    for field, rule in properties.items():
        if field not in args:
            continue
        value = args[field]
        if rule.get("type") == "string" and not isinstance(value, str):
            return f"поле {field} должно быть строкой, а пришло {type(value).__name__}"
        if "pattern" in rule and isinstance(value, str) and not re.search(rule["pattern"], value):
            return f"поле {field}={value[:60]!r} не подходит: {rule.get('description', rule['pattern'])}"
    return None
