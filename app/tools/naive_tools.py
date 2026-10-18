"""
Инструменты наивного агента (блок 6.1): три обычные Python-функции, их описания в JSON Schema для
Chat Completions (TOOLS) и allowlist DISPATCH — модель может вызвать только то, что в нём перечислено.

- search_knowledge_base — RAG блока 5.5 (app/services/rag.py): поиск по коллекции сервиса (kb_docs),
  первый фрагмент строкой; score-guard не пропустил ни одного кандидата — честное «ничего не найдено»;
- get_current_time — zoneinfo, без сети;
- send_telegram_message — заглушка: только print, Telegram API не вызывается.

Инструменты /chat из блока 3.1 (app/tools/handlers.py, schemas.py) — отдельные и не меняются.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

_RAG: list[Any] = []    # RAGService — один на процесс, подключается к коллекции при первом поиске


def knowledge_base() -> Any:
    """RAG сервиса: коллекция RAG__COLLECTION в Qdrant, эмбеддинги EMBEDDINGS__* — как у /rag/query."""
    if not _RAG:
        from app.core.config import get_settings
        from app.services.rag import RAGService, no_implicit_event_loop

        no_implicit_event_loop()           # CLI: LlamaIndex без неявного цикла asyncio (см. rag.py)
        _RAG.append(RAGService(get_settings()).build(create=False))
    return _RAG[0]


def search_knowledge_base(query: str) -> str:
    """Top-1 фрагмент базы знаний с файлом-источником и близостью."""
    found = knowledge_base().retrieve(query)
    if not found.nodes:
        return (f"В базе знаний ничего не найдено по запросу «{query}»: лучшая близость "
                f"{found.top_score:.2f} ниже порога {found.threshold}.")
    node = found.nodes[0]
    source = node.metadata.get("source") or node.metadata.get("file_name")
    return f"[{source}, близость {node.score or 0.0:.2f}] {node.node.get_content()}"


def get_current_time(timezone: str = "Europe/Moscow") -> str:
    """Текущие дата и время в часовом поясе IANA, ISO 8601."""
    return datetime.now(ZoneInfo(timezone)).isoformat()


def send_telegram_message(chat_id: str, text: str) -> str:
    """Заглушка отправки клиенту: печать в stdout вместо Telegram API."""
    print(f"[TELEGRAM → {chat_id}] {text}")
    return f"Сообщение отправлено в {chat_id}"


# Allowlist: имя инструмента → функция. Никаких eval/getattr: чего нет здесь, того модель не вызовет.
DISPATCH: dict[str, Callable[..., str]] = {
    "search_knowledge_base": search_knowledge_base,
    "get_current_time": get_current_time,
    "send_telegram_message": send_telegram_message,
}


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required,
                       "additionalProperties": False}}}


# Описания — то, по чему модель выбирает инструмент: что делает и когда вызывать.
TOOLS: list[dict[str, Any]] = [
    _tool("search_knowledge_base",
          "Ищет ответ в базе знаний техподдержки «Личного кабинета»: аккаунт и вход, тарифы и оплата, API, "
          "уведомления, мобильное приложение, интеграции, работа в команде. Возвращает один самый близкий "
          "фрагмент справки с файлом-источником или сообщение, что ничего не найдено. Вызывай перед ответом "
          "на любой вопрос о продукте, а не отвечай по памяти.",
          {"query": {"type": "string",
                     "description": "Вопрос или ключевые слова по-русски, например «как отключить автопродление»"}},
          ["query"]),
    _tool("get_current_time",
          "Возвращает текущие дату и время в указанном часовом поясе в формате ISO 8601. Вызывай, когда в "
          "ответе нужны сегодняшняя дата или время: «сейчас», «сегодня», расчёт сроков.",
          {"timezone": {"type": "string", "default": "Europe/Moscow",
                        "description": "Часовой пояс IANA, например Europe/Moscow или Asia/Yekaterinburg"}},
          []),
    _tool("send_telegram_message",
          "Отправляет клиенту текстовое сообщение в Telegram-чат по его chat_id. Вызывай, только когда задача "
          "прямо просит написать клиенту и текст уже готов: всё нужное найдено в базе знаний.",
          {"chat_id": {"type": "string", "description": "Идентификатор чата клиента в Telegram, например 12345"},
           "text": {"type": "string", "description": "Готовый текст сообщения клиенту"}},
          ["chat_id", "text"]),
]
