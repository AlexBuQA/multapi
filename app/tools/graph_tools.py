"""
Инструменты агента на LangGraph (блок 6.3): те же три функции, что у агентов блоков 6.1–6.2
(app/tools/naive_tools.py), оформленные через @tool из langchain_core.tools. Docstring — это и есть
description для модели: текст тот же, что у ReAct-агента блока 6.2 (app/tools/react_tools.py, четыре
вопроса), байт в байт — тест сверяет их, чтобы описания не разошлись. Переносы строк docstring
(перенос по ширине кода) _one_line склеивает в пробелы: иначе модель получила бы описание с «\n», не
таким, как в 6.2. Один и тот же список TOOLS идёт и в кастомный StateGraph, и в prebuilt-агента.

Схема аргументов — pydantic-модель со строгими правилами: extra="forbid" (в JSON Schema —
additionalProperties: false), шаблоны строк как у блока 6.2. Неверные аргументы — ValidationError
до вызова функции: кастомный граф и ToolNode prebuilt-агента возвращают её модели строкой.
Сбой внутри функции (Qdrant не отвечает, неизвестный часовой пояс) — тоже строка «Ошибка инструмента …»,
а не исключение: ToolNode prebuilt-агента ловит только ошибки аргументов, остальные уронили бы граф.
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field

from app.tools import naive_tools


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchArgs(_Strict):
    query: str = Field(pattern=r"^[\s\S]{2,300}$", description="Вопрос по-русски одной фразой, 2–300 символов, "
                       "например «за сколько дней приходит письмо об автопродлении»")


class TimeArgs(_Strict):
    timezone: str = Field(pattern=r"^[A-Za-z_]+(/[A-Za-z0-9_+-]+){0,2}$",
                          description="Часовой пояс IANA: Europe/Moscow, Asia/Yekaterinburg, UTC")


class SendArgs(_Strict):
    chat_id: str = Field(pattern=r"^-?[0-9]{1,20}$", description="Числовой идентификатор Telegram-чата клиента, "
                         "например 12345")
    text: str = Field(pattern=r"^[\s\S]{1,4096}$", description="Готовый текст сообщения клиенту по-русски, "
                      "до 4096 символов")


def _one_line(t: BaseTool) -> BaseTool:
    """Description одной строкой, как у 6.2. @tool берёт docstring как есть, с переносами строк кода, и в
    запросе к модели они уходят как «\\n»: модель видела бы не тот текст, что у агента 6.2."""
    t.description = " ".join(t.description.split())
    return t


def _safe(name: str, fn: Any, **kwargs: Any) -> str:
    try:
        return str(fn(**kwargs))
    except Exception as exc:  # noqa: BLE001 — сбой инструмента модель видит и может обойти
        return f"Ошибка инструмента {name}: {type(exc).__name__}: {exc}"


@_one_line
@tool(args_schema=SearchArgs)
def search_knowledge_base(query: str) -> str:
    """Что делает: ищет в базе знаний техподдержки «Личного кабинета» (аккаунт и вход, тарифы и оплата,
    возвраты, API, уведомления, мобильное приложение, интеграции, команда) один самый близкий фрагмент
    справки. Когда вызывать: перед любым ответом о правилах, сроках, ценах, лимитах и настройках
    продукта; по одному вопросу за вызов. Когда НЕ вызывать: для текущих даты и времени, для отправки
    сообщений, для данных конкретного клиента (баланс, статус заявки, пароль) — их в базе нет; повторно
    с тем же запросом. Что возвращает: строку «[файл, близость 0.71] текст фрагмента» или «В базе знаний
    ничего не найдено …» — тогда данных нет: не выдумывай ответ и не ищи то же самое снова."""
    return _safe("search_knowledge_base", naive_tools.search_knowledge_base, query=query)


@_one_line
@tool(args_schema=TimeArgs)
def get_current_time(timezone: str) -> str:
    """Что делает: возвращает текущие дату и время в часовом поясе IANA. Когда вызывать: когда в ответе
    нужны сегодняшняя дата, текущее время или срок, отсчитанный от сегодняшнего дня. Когда НЕ вызывать:
    если дата уже есть в задаче или в результате прошлого вызова; для времени на других планетах и в
    выдуманных поясах. Что возвращает: строку ISO 8601, например 2026-10-18T16:34:08+03:00; неизвестный
    пояс — ошибку."""
    return _safe("get_current_time", naive_tools.get_current_time, timezone=timezone)


@_one_line
@tool(args_schema=SendArgs)
def send_telegram_message(chat_id: str, text: str) -> str:
    """Что делает: отправляет клиенту текстовое сообщение в Telegram-чат по chat_id. Действие необратимо:
    клиент сразу увидит текст. Когда вызывать: только когда задача прямо просит написать клиенту, указан
    числовой chat_id и все факты для текста уже получены из базы знаний или других инструментов. Когда
    НЕ вызывать: без chat_id (номер телефона, имя и email — не chat_id); для звонков, email и SMS; если
    факты для текста не проверены инструментами. Что возвращает: строку «Сообщение отправлено в
    <chat_id>»."""
    return _safe("send_telegram_message", naive_tools.send_telegram_message, chat_id=chat_id, text=text)


TOOLS: list[BaseTool] = [search_knowledge_base, get_current_time, send_telegram_message]
