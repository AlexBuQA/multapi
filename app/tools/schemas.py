"""
Описания инструментов (tools) в формате JSON Schema для Function Calling.

- description у tool — промпт, который модель читает на каждом запросе; он
  хранится в app/prompts/tools/<tool>.md и подтягивается в именованные константы;
- описания параметров — тоже именованные константы, а не строки внутри схемы;
- схемы проверяются через jsonschema при импорте модуля (битая схема — сразу
  ошибка), а аргументы, которые прислала модель, — перед вызовом обработчика.
"""
from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from app.prompts.loader import load_tool_description

# --------------------------------------------------------------------------- #
# Описания (промпты)
# --------------------------------------------------------------------------- #
SEARCH_KNOWLEDGE_BASE_DESCRIPTION = load_tool_description("search_knowledge_base")
CHECK_SERVICE_STATUS_DESCRIPTION = load_tool_description("check_service_status")

QUERY_DESCRIPTION = "Короткий поисковый запрос, 2–8 слов. Пример: сброс пароля."
PRODUCT_DESCRIPTION = (
    "Где возникла проблема: web — кабинет в браузере, mobile — приложение, api — API. "
    "Не указывай, если неясно."
)
COMPONENT_DESCRIPTION = "Компонент: auth, email, billing, api, mobile или all."

# Пример корректных аргументов — подсказка модели, если она прислала не то.
ARGUMENT_EXAMPLES: dict[str, str] = {
    "search_knowledge_base": '{"query": "сброс пароля"}',
    "check_service_status": '{"component": "email"}',
}

PRODUCTS = ("web", "mobile", "api")
COMPONENTS = ("all", "auth", "email", "billing", "api", "mobile")

# --------------------------------------------------------------------------- #
# Схемы параметров
# --------------------------------------------------------------------------- #
SEARCH_KNOWLEDGE_BASE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 2, "description": QUERY_DESCRIPTION},
        "product": {"type": "string", "enum": list(PRODUCTS), "description": PRODUCT_DESCRIPTION},
    },
    "required": ["query"],
    "additionalProperties": False,
}

CHECK_SERVICE_STATUS_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "component": {
            "type": "string", "enum": list(COMPONENTS), "description": COMPONENT_DESCRIPTION,
        },
    },
    "required": ["component"],
    "additionalProperties": False,
}

TOOL_PARAMETERS: dict[str, dict[str, Any]] = {
    "search_knowledge_base": SEARCH_KNOWLEDGE_BASE_PARAMETERS,
    "check_service_status": CHECK_SERVICE_STATUS_PARAMETERS,
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "search_knowledge_base": SEARCH_KNOWLEDGE_BASE_DESCRIPTION,
    "check_service_status": CHECK_SERVICE_STATUS_DESCRIPTION,
}


def _tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": TOOL_DESCRIPTIONS[name],
            "parameters": TOOL_PARAMETERS[name],
        },
    }


# Список tools в формате OpenAI Chat Completions (его же понимает Ollama).
TOOLS: list[dict[str, Any]] = [_tool(name) for name in TOOL_PARAMETERS]
TOOL_NAMES: tuple[str, ...] = tuple(TOOL_PARAMETERS)

# --------------------------------------------------------------------------- #
# Валидация
# --------------------------------------------------------------------------- #
def _build_validators() -> dict[str, Draft202012Validator]:
    validators = {}
    for name, schema in TOOL_PARAMETERS.items():
        Draft202012Validator.check_schema(schema)  # SchemaError, если схема некорректна
        validators[name] = Draft202012Validator(schema)
    return validators


_VALIDATORS = _build_validators()


def validate_arguments(name: str, arguments: dict[str, Any]) -> None:
    """Проверяет аргументы вызова по схеме; бросает jsonschema.ValidationError."""
    _VALIDATORS[name].validate(arguments)
