"""
Кеширование ответов LLM (перенос ДЗ 2.4).

LLMCache — in-memory кеш:
- ключ = SHA-256 от (model + temperature + messages), без timestamp/request_id;
- TTL (по умолчанию 1 час) — просроченные записи не возвращаются;
- статистика hits / misses и метод stats() с hit rate в процентах.

Для мультимодальных запросов в messages входят и текст, и image_url-блоки,
поэтому одинаковая картинка с одинаковым вопросом отдаётся из кеша мгновенно.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class _Entry:
    value: str
    expires_at: float


class LLMCache:
    """Простой потокобезопасный-достаточный для CLI in-memory кеш с TTL."""

    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self._store: dict[str, _Entry] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _make_key(model: str, temperature: float, messages: list[dict[str, Any]]) -> str:
        """SHA-256 от стабильно сериализованных параметров запроса."""
        payload = {
            "model": model,
            "temperature": round(float(temperature), 4),
            "messages": messages,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self, model: str, messages: list[dict[str, Any]], temperature: float
    ) -> str | None:
        """Возвращает закешированный ответ или None (с учётом TTL)."""
        key = self._make_key(model, temperature, messages)
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        if entry.expires_at < time.time():
            # Просрочено — удаляем и считаем промахом.
            self._store.pop(key, None)
            self.misses += 1
            return None
        self.hits += 1
        return entry.value

    def set(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        value: str,
    ) -> None:
        key = self._make_key(model, temperature, messages)
        self._store[key] = _Entry(value=value, expires_at=time.time() + self.ttl)

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate_pct": round(hit_rate, 1),
            "size": len(self._store),
        }

    def clear(self) -> None:
        self._store.clear()
