"""
JSON-логи через structlog (блок 3.6).

Каждая строка — один JSON-объект. request_id, method, path и user_id привязываются в
middleware через contextvars и автоматически попадают во все строки, записанные во время
обработки запроса: и в строку о HTTP-запросе, и в строку о вызове модели.

Сообщения стандартного logging (uvicorn, openai, opentelemetry) идут через тот же
рендерер, поэтому лог остаётся однородным JSON. Строки доступа uvicorn
(«GET /ready 200») выключены: их заменяет http_request из middleware — с request_id
и временем обработки.

Сырой промпт в лог не попадает: в строках о вызове модели только prompt_hash и
prompt_preview после redact_pii (app/observability/pii.py), а логгер openai, который на
уровне DEBUG печатает тело запроса, ограничен уровнем WARNING.
"""
from __future__ import annotations

import logging
import sys
from typing import TextIO

import structlog

LOGGER_NAME = "llm-service"


def _shared_processors() -> list:
    return [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]


def setup_logging(level: str = "INFO", stream: TextIO | None = None) -> None:
    """Настраивает structlog и стандартный logging. Повторный вызов перенастраивает
    вывод (тесты направляют лог в StringIO)."""
    stream = stream or sys.stdout
    level_no = logging.getLevelName(level.upper())
    if not isinstance(level_no, int):
        level_no = logging.INFO

    structlog.configure(
        processors=[
            *_shared_processors(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level_no),
        logger_factory=structlog.PrintLoggerFactory(file=stream),
        cache_logger_on_first_use=False,
    )

    # Стандартный logging — в тот же JSON, с теми же contextvars.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[*_shared_processors(), structlog.stdlib.add_logger_name],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
    )
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level_no)
    for name in ("uvicorn", "uvicorn.error", LOGGER_NAME):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    logging.getLogger("uvicorn.access").disabled = True
    # Сторонние клиенты не ниже WARNING даже при LOG_LEVEL=DEBUG:
    # - httpx/httpcore пишут строку на каждый запрос к модели — вызов и так описан
    #   строкой llm_request_completed;
    # - openai на DEBUG печатает тело запроса целиком («Request options: ... messages»),
    #   то есть сырой промпт с персональными данными в обход redact_pii.
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(max(level_no, logging.WARNING))


def get_logger() -> structlog.typing.FilteringBoundLogger:
    return structlog.get_logger(LOGGER_NAME)
