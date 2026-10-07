"""
Помощник тестов: перехват JSON-логов structlog (блок 3.6).

    with captured_logs() as records:
        ...                      # код, который пишет в лог
    records                     # список словарей — по одному на строку лога

Вне блока лог приглушён до CRITICAL, чтобы строки сервиса не засоряли вывод тестов.
Файл не начинается с test_, поэтому unittest и pytest не ищут в нём тестов.
"""
from __future__ import annotations

import contextlib
import io
import json
from collections.abc import Iterator

from app.observability.logging import setup_logging

QUIET_LEVEL = "CRITICAL"


def quiet_logs() -> None:
    setup_logging(QUIET_LEVEL)


@contextlib.contextmanager
def captured_logs(level: str = "DEBUG") -> Iterator[list[dict]]:
    buffer = io.StringIO()
    setup_logging(level, stream=buffer)
    records: list[dict] = []
    try:
        yield records
    finally:
        records.extend(json.loads(line) for line in buffer.getvalue().splitlines() if line.strip())
        quiet_logs()


def events(records: list[dict], name: str) -> list[dict]:
    return [record for record in records if record.get("event") == name]
