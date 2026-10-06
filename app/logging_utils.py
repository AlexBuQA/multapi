"""
Структурированный лог прогонов ассистента: одна JSON-строка на событие.

Формат рассчитан на дальнейшую обработку (structlog в Б3.6, тесты в Б3.7):
    {"ts": "...", "level": "info", "event": "tool_call", "run_id": "...", ...}

События одного запроса (общий run_id):
    user_input -> llm_request/llm_response -> tool_call -> tool_result -> ...
    -> final_answer -> usage (prompt/completion/total_tokens)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOGGER_PREFIX = "tool_events"


class JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .astimezone()
            .isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_event_logger(path: Path) -> logging.Logger:
    """Логгер, пишущий JSON-строки в указанный файл (дописывает)."""
    path = Path(path)
    logger = logging.getLogger(f"{_LOGGER_PREFIX}:{path.resolve()}")
    if not logger.handlers:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(JsonLinesFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def close_event_logger(logger: logging.Logger) -> None:
    """Закрывает файл лога (нужно, например, перед удалением временной папки в тестах)."""
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(event, extra={"fields": fields})
