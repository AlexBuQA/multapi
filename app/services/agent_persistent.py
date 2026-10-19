"""
Персистентный агент с подтверждением опасного действия (блок 6.4).

ReAct-граф блока 6.3 (app/services/agent_graph.py — он не меняется и остаётся in-memory вариантом для
unit-тестов и бенчмарка) получает:
- checkpointer: состояние каждого thread сохраняется после каждого шага и переживает рестарт процесса.
  Backend — AGENT_CHECKPOINTER: memory (тесты), sqlite (разработка, по умолчанию), postgres (compose,
  та же база DATABASE_URL, что у чатов);
- human-in-the-loop на send_telegram_message — сообщение клиенту необратимо. Два узла и ребро между ними:
  prepare_send_message (идемпотентный: проверка аргументов и роли, черновик в state, побочных эффектов нет)
  → confirm_and_execute_send_message (interrupt() с превью; после Command(resume=True) — отправка,
  после Command(resume=False) — отказ, модель узнаёт об этом из ToolMessage);
- политику ролей из config["configurable"]["user_role"]: read-only — отправка запрещена, write-with-approve —
  только после подтверждения, full — без interrupt.

    START → call_model ─┬─ другие инструменты → execute_tool ─┬─ отправка в том же ответе → prepare_send_message
                        │                                     └─ иначе → call_model
                        ├─ только отправка → prepare_send_message → confirm_and_execute_send_message → call_model
                        └─ ответ или лимит → force_finish → END

Что до interrupt() и что после — ключевое для HIL. При resume LangGraph перезапускает прерванный узел с
начала, и всё, что стоит в нём до interrupt(), выполнится ещё раз. Поэтому:
- до interrupt: только чтение state и сборка превью (черновик готовит предыдущий узел);
- после interrupt: сам побочный эффект — отправка через sender с ключом идемпотентности request_id.
request_id детерминирован (thread_id + id вызова модели), а не uuid4: повтор того же шага — повтор
и ключа. TelegramOutbox по нему не отправляет одно сообщение дважды — например, при replay
прошлого чек-пойнта (scripts/time_travel_demo.py).

Жизненный цикл: agent_lifespan() открывает checkpointer, один раз вызывает setup() и отдаёт собранный граф;
FastAPI держит его в app.state.agent (app/main.py), POST /agent/stream (app/routers/agent.py) только
вызывает граф. thread_id — стабильный (для бота — tg-<chat_id>), uuid4() на запрос не генерируется:
иначе checkpointer не найдёт сохранённое состояние.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from app.observability.logging import get_logger
from app.services.agent_graph import (
    FORCE_FINISH,
    MAX_ITERATIONS,
    NOT_EXECUTED,
    PREVIEW,
    AgentState,
    _invoke,
    build_model,
)
from app.services.agent_react import SYSTEM_PROMPT
from app.tools import naive_tools
from app.tools.graph_tools import TOOLS, SendArgs

log = get_logger()

SEND = "send_telegram_message"
PREPARE = "prepare_send_message"
CONFIRM = "confirm_and_execute_send_message"
INTERRUPT_TYPE = "approve_send_message"

Role = Literal["read-only", "write-with-approve", "full"]
ROLES: tuple[Role, ...] = ("read-only", "write-with-approve", "full")
DEFAULT_ROLE: Role = "write-with-approve"

READ_ONLY_REFUSAL = ("Не выполнено: у роли read-only нет права отправлять сообщения клиентам. "
                     "Скажи пользователю, что сообщение не отправлено.")
ONE_PER_STEP = "Не выполнено: за одно подтверждение отправляется одно сообщение. Повтори вызов отдельно."
REJECTED = "Не отправлено: оператор отклонил отправку{comment}. Скажи пользователю, что сообщение не отправлено."

Sender = Callable[..., Awaitable[str]]          # await sender(chat_id=…, text=…, request_id=…) -> результат


class PersistentState(AgentState):
    """AgentState блока 6.3 и два поля отправки. draft — черновик сообщения от prepare_send_message:
    chat_id, text, request_id, tool_call_id (None — подтверждать нечего); sent — ушло ли сообщение в этом
    запуске. Всё сериализуемо: checkpointer пишет state целиком после каждого шага."""

    draft: dict[str, Any] | None
    sent: bool


def new_turn(messages: list[Any]) -> dict[str, Any]:
    """Вход графа на новое сообщение пользователя. История thread — в checkpointer, сюда только новое.
    Счётчик итераций и флаги отправки сбрасываются: лимит 6 — на запуск, а не на весь thread."""
    return {"messages": messages, "iteration_count": 0, "tool_results": [], "draft": None, "sent": False}


def thread_config(thread_id: str, *, user_role: Role = DEFAULT_ROLE, checkpoint_id: str | None = None,
                  recursion_limit: int = 50) -> RunnableConfig:
    """config запуска: стабильный thread_id (tg-<chat_id>, имя сценария — не uuid4 на запрос), роль
    пользователя и, для time travel, конкретный чек-пойнт."""
    configurable: dict[str, Any] = {"thread_id": thread_id, "user_role": user_role}
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable, "recursion_limit": recursion_limit}


def capped_role(requested: str | None, max_role: str) -> Role:
    """Роль запроса не выше AGENT_MAX_ROLE: из тела запроса можно только сузить права, не расширить."""
    want = requested or DEFAULT_ROLE
    return ROLES[min(ROLES.index(want), ROLES.index(max_role))]  # type: ignore[arg-type]


def request_id(thread_id: str, tool_call_id: str) -> str:
    """Ключ идемпотентности отправки: из thread_id и id вызова модели, без времени и случайности.
    Тот же шаг thread — тот же ключ; новый вызов модели — новый ключ."""
    return hashlib.sha256(f"{thread_id}\x1f{tool_call_id}".encode()).hexdigest()[:16]


class TelegramOutbox:
    """Отправка клиенту с ключом идемпотентности. Сообщение с уже отправленным request_id второй раз не
    уходит: replay прошлого чек-пойнта перезапускает узел отправки, а сообщение клиенту не задваивается.
    Журнал — в памяти процесса; для production его место — таблица с UNIQUE (request_id) в той же базе
    (раздел 8 отчёта). send — заглушка блока 6.1: печать в stdout, настоящий Telegram API не вызывается."""

    def __init__(self, send: Callable[[str, str], str] | None = None) -> None:
        self._send = send or naive_tools.send_telegram_message
        self._done: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def __call__(self, *, chat_id: str, text: str, request_id: str) -> str:
        async with self._lock:
            if request_id in self._done:
                log.warning("agent_send_duplicate", request_id=request_id, chat_id=chat_id)
                return f"Уже отправлено ранее (request_id {request_id}), повторно не отправляю: {self._done[request_id]}"
            result = await asyncio.to_thread(self._send, chat_id, text)
            self._done[request_id] = result
            return result


def _role(config: RunnableConfig) -> Role:
    role = (config.get("configurable") or {}).get("user_role") or DEFAULT_ROLE
    return role if role in ROLES else "read-only"          # незнакомая роль — самая узкая


def _last_ai(state: PersistentState) -> AIMessage | None:
    return next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)


def _sends(message: AIMessage | None) -> list[dict[str, Any]]:
    return [c for c in (message.tool_calls if message else []) if c["name"] == SEND]


def _decision(value: Any) -> tuple[bool, str]:
    """Ответ человека на interrupt: True/False или {"approved": bool, "comment": "…"}."""
    if isinstance(value, dict):
        return bool(value.get("approved")), str(value.get("comment") or "")[:200]
    return value is True, ""


def route_after_model(state: PersistentState) -> Literal["execute_tool", "prepare_send_message", "force_finish"]:
    """Как в 6.3, плюс ветка подтверждения: только чтение state. Сначала выполняются безопасные
    инструменты, отправка из того же ответа модели — после них (route_after_tools)."""
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "force_finish"
    last = state["messages"][-1]
    if not (isinstance(last, AIMessage) and last.tool_calls):
        return "force_finish"
    return "execute_tool" if any(c["name"] != SEND for c in last.tool_calls) else PREPARE


def route_after_tools(state: PersistentState) -> Literal["prepare_send_message", "call_model"]:
    return PREPARE if _sends(_last_ai(state)) else "call_model"


def build_agent(checkpointer: Any, *, model: BaseChatModel | None = None, tools: list[BaseTool] = TOOLS,
                sender: Sender | None = None, system_prompt: str = SYSTEM_PROMPT, name: str = "agent_persistent") -> Any:
    """Фабрика графа. checkpointer — InMemorySaver, AsyncSqliteSaver или AsyncPostgresSaver (уже после
    setup()); model — по умолчанию ChatOpenAI из настроек (как в 6.3); sender — внешний API отправки,
    по умолчанию TelegramOutbox над заглушкой. Инструмент send_telegram_message модель видит в списке, но
    сам он не вызывается никогда: отправку делает только confirm_and_execute_send_message."""
    model = model or build_model()
    bound = model.bind_tools(tools, strict=True)
    by_name = {t.name: t for t in tools if t.name != SEND}
    send_message = sender or TelegramOutbox()

    async def call_model(state: PersistentState) -> dict[str, Any]:
        response = await bound.ainvoke([SystemMessage(system_prompt), *state["messages"]])
        iteration = state["iteration_count"] + 1
        usage = response.usage_metadata or {}
        log.info("agent_model", iteration=iteration, tool_calls=[c["name"] for c in response.tool_calls],
                 input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))
        return {"messages": [response], "iteration_count": iteration}

    async def execute_tool(state: PersistentState) -> dict[str, Any]:
        """Безопасные инструменты (поиск, время). Вызовы отправки остаются ветке подтверждения."""
        messages, results = [], []
        for call in _last_ai(state).tool_calls:  # type: ignore[union-attr]
            if call["name"] == SEND:
                continue
            started = time.perf_counter()
            content = await _invoke(by_name, call)
            messages.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
            results.append({"iteration": state["iteration_count"], "name": call["name"], "args": call["args"],
                            "result": content[:PREVIEW], "ms": round((time.perf_counter() - started) * 1000, 1)})
        return {"messages": messages, "tool_results": results}

    async def prepare_send_message(state: PersistentState, config: RunnableConfig) -> dict[str, Any]:
        """Идемпотентная подготовка: проверка роли и аргументов, черновик в state. Побочных эффектов нет —
        повтор узла даёт тот же черновик (request_id — из thread_id и id вызова, не случайный)."""
        first, *rest = _sends(_last_ai(state))
        closing = [ToolMessage(content=ONE_PER_STEP, tool_call_id=c["id"], name=SEND) for c in rest]
        if _role(config) == "read-only":
            log.info("agent_send_refused", reason="read-only")
            return {"messages": [*closing, ToolMessage(content=READ_ONLY_REFUSAL, tool_call_id=first["id"], name=SEND)],
                    "draft": None}
        try:
            args = SendArgs.model_validate(first["args"])
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'аргументы'}: {e['msg']}" for e in exc.errors())
            error = f"Ошибка: аргументы {SEND} не по схеме — {problems}. Исправь и повтори вызов."
            return {"messages": [*closing, ToolMessage(content=error, tool_call_id=first["id"], name=SEND)],
                    "draft": None}
        thread_id = str((config.get("configurable") or {}).get("thread_id", ""))
        draft = {"tool_call_id": first["id"], "chat_id": args.chat_id, "text": args.text,
                 "request_id": request_id(thread_id, first["id"])}
        log.info("agent_send_prepared", chat_id=args.chat_id, chars=len(args.text), request_id=draft["request_id"])
        return {"messages": closing, "draft": draft}

    async def confirm_and_execute_send_message(state: PersistentState, config: RunnableConfig) -> dict[str, Any]:
        """До interrupt() — только чтение state: при resume узел выполняется с начала. После — отправка."""
        draft = state.get("draft")
        if not draft:                                    # подготовка отказала: подтверждать нечего
            return {}
        preview = {"chat_id": draft["chat_id"], "text": draft["text"], "request_id": draft["request_id"]}
        if _role(config) == "full":
            approved, comment = True, ""
            log.info("agent_send_auto_approved", request_id=draft["request_id"])
        else:
            approved, comment = _decision(interrupt({"type": INTERRUPT_TYPE, "preview": preview,
                                                     "question": "Отправить это сообщение клиенту?"}))
            if approved and _role(config) == "read-only":    # подтверждает тот, кто резюмирует: его роль
                approved, comment = False, "у роли read-only нет права подтверждать отправку"
        # ---- ниже — только после решения человека (или роли full) ----
        record = {"iteration": state["iteration_count"], "name": SEND, "args": {"chat_id": draft["chat_id"]},
                  "approved": approved}
        if not approved:
            log.info("agent_send_rejected", request_id=draft["request_id"])
            content = REJECTED.format(comment=f" ({comment})" if comment else "")
            return {"messages": [ToolMessage(content=content, tool_call_id=draft["tool_call_id"], name=SEND)],
                    "draft": None, "sent": False, "tool_results": [{**record, "result": content[:PREVIEW]}]}
        try:
            content = await send_message(chat_id=draft["chat_id"], text=draft["text"], request_id=draft["request_id"])
            sent = True
        except Exception as exc:  # noqa: BLE001 — сбой внешнего API модель видит и сообщает пользователю
            content, sent = f"Ошибка инструмента {SEND}: {type(exc).__name__}: {exc}", False
        log.info("agent_send_done", request_id=draft["request_id"], sent=sent)
        return {"messages": [ToolMessage(content=str(content), tool_call_id=draft["tool_call_id"], name=SEND)],
                "draft": None, "sent": sent, "tool_results": [{**record, "result": str(content)[:PREVIEW]}]}

    async def force_finish(state: PersistentState) -> dict[str, Any]:
        """Как в 6.3: ответ модели — как есть; при лимите — каждый вызов закрыт ToolMessage и явный ответ."""
        last = state["messages"][-1]
        if not (isinstance(last, AIMessage) and last.tool_calls):
            return {}
        log.warning("agent_force_finish", iteration=state["iteration_count"], pending=len(last.tool_calls))
        closing = [ToolMessage(content=NOT_EXECUTED, tool_call_id=c["id"], name=c["name"]) for c in last.tool_calls]
        return {"messages": [*closing, AIMessage(content=FORCE_FINISH, name="force_finish")], "draft": None}

    builder = StateGraph(PersistentState)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tool", execute_tool)
    builder.add_node(PREPARE, prepare_send_message)
    builder.add_node(CONFIRM, confirm_and_execute_send_message)
    builder.add_node("force_finish", force_finish)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", route_after_model,
                                  {"execute_tool": "execute_tool", PREPARE: PREPARE, "force_finish": "force_finish"})
    builder.add_conditional_edges("execute_tool", route_after_tools, {PREPARE: PREPARE, "call_model": "call_model"})
    builder.add_edge(PREPARE, CONFIRM)                   # подготовка и отправка — разные узлы, между ними ребро
    builder.add_edge(CONFIRM, "call_model")
    builder.add_edge("force_finish", END)
    return builder.compile(checkpointer=checkpointer, name=name)


# ---------------------------------------------------------------- checkpointer и жизненный цикл
@asynccontextmanager
async def open_checkpointer(cfg: Any) -> AsyncIterator[Any]:
    """Saver по AGENT_CHECKPOINTER. setup() — здесь и только здесь, один раз на процесс: у Postgres он
    создаёт таблицы checkpoints, checkpoint_writes, checkpoint_blobs и checkpoint_migrations (повторный
    вызов на каждый запрос — лишний раундтрип к базе), у SQLite — схему в файле."""
    backend = cfg.agent_checkpointer
    if backend == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        log.info("agent_checkpointer_ready", backend="memory")
        yield InMemorySaver()
        return
    if backend == "sqlite":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        path = cfg.agent_sqlite_path
        path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
            await saver.setup()
            log.info("agent_checkpointer_ready", backend="sqlite", path=path.name)
            yield saver
        return
    _require_selector_loop()
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    from app.core.config import psycopg_database_url

    # autocommit и dict_row — требования AsyncPostgresSaver; prepare_threshold=0 — без серверных prepared
    # statements (безопасно и за PgBouncer). Пул открывается здесь и закрывается при остановке сервиса.
    async with AsyncConnectionPool(psycopg_database_url(cfg.database_url.get_secret_value()), min_size=1,
                                   max_size=5, open=False,
                                   kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row}) as pool:
        saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
        await saver.setup()
        log.info("agent_checkpointer_ready", backend="postgres")
        yield saver


def _require_selector_loop() -> None:
    """psycopg в async-режиме не работает с ProactorEventLoop. На Windows uvicorn без --reload запускает
    именно его (с --reload и --workers — SelectorEventLoop), asyncio.run — тоже."""
    proactor = getattr(asyncio, "ProactorEventLoop", None)
    if sys.platform == "win32" and proactor is not None and isinstance(asyncio.get_running_loop(), proactor):
        raise RuntimeError("AGENT_CHECKPOINTER=postgres на Windows: psycopg не работает с ProactorEventLoop — его "
                           "uvicorn запускает без --reload. Варианты: uvicorn с --reload (SelectorEventLoop), сервис в "
                           "Docker (docker compose up -d --build) или AGENT_CHECKPOINTER=sqlite.")


@asynccontextmanager
async def agent_lifespan(cfg: Any = None, **build: Any) -> AsyncIterator[Any]:
    """Checkpointer по AGENT_CHECKPOINTER и собранный граф — на всё время жизни сервиса (lifespan FastAPI,
    скрипты). build — аргументы build_agent: model, tools, sender."""
    from app.core.config import get_settings

    cfg = cfg or get_settings()
    async with open_checkpointer(cfg) as saver:
        yield build_agent(saver, **build)


# ---------------------------------------------------------------- поток событий для SSE
def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def message_view(message: AnyMessage) -> dict[str, Any]:
    """Сообщение для события node: ответ модели — целиком, результат инструмента — первые 200 символов
    (поток показывает прогресс, полный фрагмент базы знаний клиенту SSE не нужен)."""
    text = _text(message.content)
    view: dict[str, Any] = {"type": message.type, "content": text}
    if isinstance(message, AIMessage) and message.tool_calls:
        view["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in message.tool_calls]
    if isinstance(message, ToolMessage):
        view["name"] = message.name
        view["content"] = text if len(text) <= PREVIEW else text[:PREVIEW] + "…"
    return view


def _update_view(update: dict[str, Any] | None) -> dict[str, Any]:
    return {key: [message_view(m) for m in value] if key == "messages" else value
            for key, value in (update or {}).items()}


def interrupt_view(interrupts: Any) -> list[dict[str, Any]]:
    return [{"id": item.id, "value": item.value} for item in interrupts]


async def stream_events(graph: Any, graph_input: Any, config: RunnableConfig) -> AsyncIterator[dict[str, Any]]:
    """graph.astream(stream_mode=["updates", "messages"]) в события для SSE:
    - node — узел закончил шаг (updates): что он записал в state, сообщения — в кратком виде;
    - token — кусок ответа модели (messages): текст по мере генерации;
    - interrupt — граф встал на подтверждение (__interrupt__ в updates): превью сообщения;
    - end — итог по aget_state: на каком узле стоит граф, ждёт ли решения, ушло ли сообщение, ответ."""
    async for mode, payload in graph.astream(graph_input, config, stream_mode=["updates", "messages"]):
        if mode == "messages":
            chunk, meta = payload
            # В режиме messages приходят и сообщения, которые узлы пишут в state (ToolMessage с результатом
            # поиска) — токены здесь только у ответа модели.
            text = _text(chunk.content) if isinstance(chunk, AIMessage) else ""
            if text:
                yield {"event": "token", "node": meta.get("langgraph_node"), "content": text}
            continue
        for node, update in payload.items():
            if node == "__interrupt__":
                yield {"event": "interrupt", "interrupts": interrupt_view(update)}
            else:
                yield {"event": "node", "node": node, "update": _update_view(update)}
    snapshot = await graph.aget_state(config)
    last = next((m for m in reversed(snapshot.values.get("messages", [])) if isinstance(m, AIMessage)), None)
    yield {"event": "end", "next": list(snapshot.next), "pending_approval": bool(snapshot.interrupts),
           "sent": snapshot.values.get("sent"),
           "answer": None if snapshot.interrupts or last is None else _text(last.content)}


def sse(event: dict[str, Any]) -> str:
    """Одно событие Server-Sent Events: data: {json}\\n\\n. default=str — на случай несериализуемого значения."""
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


def resume_command(decision: bool | dict[str, Any]) -> Command:
    """Command(resume=…) для подтверждения: True/False или {"approved": bool, "comment": "…"}."""
    return Command(resume=decision)
