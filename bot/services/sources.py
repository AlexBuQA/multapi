"""
Источники ответа в Telegram (блок 5.5): подпись под ответом и память для оценок 👍/👎.

Сервис отвечает по базе знаний с цитатами [1], [2] и в событии sources присылает, что за ними
стоит. Бот:
1. выбирает, что показать (shown): источники, на которые модель сослалась (cited); не сослалась ни
   на один — первые MAX_SHOWN найденных. Отказ («По базе не нашёл…», confident=false) — без подписи;
2. дописывает к последнему сообщению ответа подпись (legend):
       Источники:
       [1] refunds_v2.docx
       [2] support-regulation_v2.3.pdf, стр. 2
   Модель сама заканчивает ответ строкой «Источники: [1], [2]» (формат выбран замером, docs/rag.md) —
   в последней правке её заменяет подпись: номера те же, но с именами файлов (without_sources_line).
   При отказе модель тоже дописывает эту строку — она убирается, подписи нет;
3. запоминает показанное по id ответа (ShownSources): кнопки 👍/👎 несут в callback_data только
   id ответа (лимит Telegram — 64 байта), а оценка уходит в сервис вместе со списком источников.

Память — в процессе бота, последние LIMIT ответов: после перезапуска бота оценка старого ответа
уходит без источников (сервис принимает и так).
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any
from uuid import UUID

from bot import texts

MAX_SHOWN = 3
LIMIT = 2000
# «Источники: [2][3].», «Источники: [1], [3]» — последняя строка ответа модели (app/services/rag_common.py).
SOURCES_LINE = re.compile(r"\s*Источники:\s*(?:\[\d{1,2}(?:\s*[,;]\s*\d{1,2})*\][\s,;]*)+\.?\s*$")


def shown(sources: list[dict[str, Any]], confident: bool | None) -> list[dict[str, Any]]:
    """Что показать под ответом: процитированные, иначе первые MAX_SHOWN; отказ — ничего."""
    if not confident or not sources:
        return []
    cited = [item for item in sources if item.get("cited")]
    return cited or sources[:MAX_SHOWN]


def where(item: dict[str, Any]) -> str:
    name = str(item.get("file_name") or item.get("source") or "без имени")
    page = item.get("page")
    return f"{name}, стр. {page}" if page else name


def without_sources_line(text: str) -> str:
    """Ответ без строки модели «Источники: [1], [3]» в конце: под ним будет подпись legend()."""
    return SOURCES_LINE.sub("", text)


def legend(items: list[dict[str, Any]]) -> str:
    """Подпись к ответу: пустая строка, если показывать нечего."""
    if not items:
        return ""
    lines = [f"[{item.get('id')}] {where(item)}" for item in items]
    return f"\n\n{texts.SOURCES_TITLE}\n" + "\n".join(lines)


class ShownSources:
    """id ответа в сервисе → источники, показанные под ним. Старые вытесняются (LRU)."""

    def __init__(self, limit: int = LIMIT) -> None:
        self.limit = limit
        self._items: OrderedDict[UUID, list[dict[str, Any]]] = OrderedDict()

    def remember(self, message_id: UUID, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        self._items[message_id] = items
        self._items.move_to_end(message_id)
        while len(self._items) > self.limit:
            self._items.popitem(last=False)

    def get(self, message_id: UUID) -> list[dict[str, Any]] | None:
        return self._items.get(message_id)

    def clear(self) -> None:
        self._items.clear()


SHOWN_SOURCES = ShownSources()
