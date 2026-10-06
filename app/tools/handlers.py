"""
Обработчики инструментов — обычный Python, который делает реальную работу:
- search_knowledge_base — поиск по руководству пользователя (data/knowledge_base.json);
- check_service_status  — статус компонентов сервиса (data/service_status.json).

Файлы читаются при каждом вызове, поэтому правка данных сразу видна ассистенту.
DISPATCH — allowlist: модель может вызвать только перечисленные в нём функции
(никаких eval/getattr). execute_tool() никогда не бросает исключений: любая
проблема (неизвестный tool, битый JSON аргументов, нарушение схемы, исключение
внутри обработчика) возвращается модели как {"error": ..., "message": ...},
чтобы она могла поправиться или честно ответить.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

from jsonschema import ValidationError

from app.config import tool_settings
from app.tools.schemas import ARGUMENT_EXAMPLES, validate_arguments

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Поиск по базе знаний
# --------------------------------------------------------------------------- #
_STOPWORDS = {
    "и", "в", "во", "на", "с", "со", "к", "ко", "по", "за", "из", "у", "о", "об",
    "а", "но", "или", "не", "ни", "ли", "же", "бы", "то", "это", "как", "что",
    "где", "когда", "почему", "зачем", "мне", "меня", "мой", "моя", "мои", "я",
    "вы", "вас", "ваш", "у", "есть", "для", "от", "до", "при", "так", "уже",
    "можно", "нужно", "подскажите", "пожалуйста", "здравствуйте", "если",
}
_MIN_SCORE = 2
_TOP_K = 3


_ENDINGS = "аеиоуыэюяйь"


def _tokens(text: str) -> list[str]:
    text = text.lower().replace("ё", "е")
    return re.findall(r"\w+", text)


def _stem(token: str) -> str:
    # Грубый стемминг для русского: отбрасываем гласные окончания и берём
    # первые 5 символов («пароль», «пароля», «паролем» -> «парол»; «ключ», «ключа» -> «ключ»).
    return (token.rstrip(_ENDINGS) or token)[:5]


def _terms(text: str) -> set[str]:
    stems = (_stem(t) for t in _tokens(text) if t not in _STOPWORDS)
    return {s for s in stems if len(s) > 1}


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def search_knowledge_base(query: str, product: str | None = None) -> dict[str, Any]:
    """Ищет статьи руководства: совпадения в заголовке весят больше, чем в тексте."""
    kb = _load_json(tool_settings.knowledge_base_path)
    query_terms = _terms(query)
    scored = []
    for article in kb["articles"]:
        if product and article["product"] != product:
            continue
        score = (
            3 * len(query_terms & _terms(article["title"]))
            + 2 * len(query_terms & _terms(" ".join(article.get("keywords", []))))
            + len(query_terms & _terms(article["text"]))
        )
        if score >= _MIN_SCORE:
            scored.append((score, article))
    scored.sort(key=lambda item: (-item[0], item[1]["id"]))

    return {
        "source": kb.get("title", "Руководство пользователя"),
        "query": query,
        "product": product,
        "found": len(scored[:_TOP_K]),
        "articles": [
            {
                "id": a["id"],
                "section": a["section"],
                "title": a["title"],
                "score": score,
                "text": a["text"],
            }
            for score, a in scored[:_TOP_K]
        ],
    }


# --------------------------------------------------------------------------- #
# Статус сервиса
# --------------------------------------------------------------------------- #
STATUS_TEXT = {
    "operational": "работает штатно",
    "degraded": "работает с деградацией",
    "outage": "недоступен",
}


def check_service_status(component: str) -> dict[str, Any]:
    """Статус компонента (или всех сразу при component="all")."""
    data = _load_json(tool_settings.service_status_path)
    components = data["components"]
    names = list(components) if component == "all" else [component]

    items = []
    for name in names:
        info = components.get(name)
        if info is None:
            return {"error": "unknown_component", "message": f"Компонент {name!r} не найден."}
        item = {
            "component": name,
            "title": info["title"],
            "status": info["status"],
            "status_text": STATUS_TEXT.get(info["status"], info["status"]),
        }
        if info.get("incident"):
            item["incident"] = info["incident"]
            item["since"] = info.get("since")
        items.append(item)
    return {"updated_at": data.get("updated_at"), "components": items}


# --------------------------------------------------------------------------- #
# Диспетчер вызовов
# --------------------------------------------------------------------------- #
# Allowlist: имя инструмента -> реализация.
DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "search_knowledge_base": search_knowledge_base,
    "check_service_status": check_service_status,
}


def parse_arguments(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """Аргументы от модели: JSON-строка (OpenAI/Ollama) или уже dict."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("аргументы должны быть JSON-объектом")
    return value


def _is_schema_echo(value: Any) -> bool:
    """Модель прислала описание параметра из схемы ({"type": ..., "description": ...})."""
    return isinstance(value, dict) and bool({"type", "description"} & set(value))


def normalize_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Чинит типичные огрехи небольших моделей, не меняя смысла:
    - пустые необязательные поля (None, "") убираются;
    - «эхо схемы» со значением внутри ({"type": "string", "description": "...",
      "value": "сброс пароля"}) заменяется самим значением.
    """
    cleaned: dict[str, Any] = {}
    for key, value in arguments.items():
        if _is_schema_echo(value) and "value" in value:
            value = value["value"]
        if value is None or value == "":
            continue
        cleaned[key] = value
    return cleaned


def execute_tool(name: str, raw_arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    """Проверяет аргументы по JSON Schema и вызывает обработчик. Не бросает исключений."""
    handler = DISPATCH.get(name)
    if handler is None:
        return {
            "error": "unknown_tool",
            "message": f"Инструмент {name!r} недоступен. Доступные: {sorted(DISPATCH)}",
        }
    try:
        arguments = parse_arguments(raw_arguments)
    except ValueError as exc:  # json.JSONDecodeError — подкласс ValueError
        return {"error": "invalid_json", "message": f"Аргументы не разобраны: {exc}"}

    arguments = normalize_arguments(arguments)
    try:
        validate_arguments(name, arguments)
    except ValidationError as exc:
        field = exc.path[0] if exc.path else None
        if field is not None and _is_schema_echo(arguments.get(field)):
            message = (
                f"В аргументе {field!r} передано описание параметра из схемы, "
                f"а нужно само значение. Пример: {ARGUMENT_EXAMPLES[name]}"
            )
        else:
            message = f"{exc.message}. Пример: {ARGUMENT_EXAMPLES[name]}"
        return {"error": "invalid_arguments", "message": message}

    try:
        return handler(**arguments)
    except Exception as exc:  # noqa: BLE001 — ошибка инструмента уходит модели, а не роняет цикл
        logger.exception("Инструмент %s завершился ошибкой", name)
        return {"error": "tool_failed", "message": f"{type(exc).__name__}: {exc}"}
