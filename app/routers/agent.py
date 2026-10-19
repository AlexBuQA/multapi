"""
Персистентный агент по HTTP (блок 6.4): поток событий графа в Server-Sent Events.

    POST /agent/stream        {"thread_id": "demo-1", "input": {"messages": [{"role": "user", "content": "…"}]}}
                              {"thread_id": "demo-1", "resume": true}            — подтвердить отправку
                              {"thread_id": "demo-1", "resume": {"approved": false, "comment": "…"}} — отклонить
    GET  /agent/threads/{id}  состояние thread по checkpointer: на каком узле стоит граф, ждёт ли решения

Агент умеет отправлять сообщения клиентам, поэтому оба эндпоинта — под X-Admin-Token (ADMIN_TOKEN), как
/chats/admin/*. Роль запуска (user_role в config) — из тела запроса, но не выше AGENT_MAX_ROLE: из запроса
права можно только сузить. read-only не подтверждает отправку.

Поток: graph.astream(stream_mode=["updates", "messages"]) — updates показывает прогресс по узлам (что узел
записал в state), messages — токены модели. Пауза на подтверждение приходит событием interrupt (из
__interrupt__ в updates), последним идёт end с итогом по graph.aget_state. Формат — data: {json}\\n\\n.

Граф и checkpointer создаёт lifespan сервиса один раз (app.state.agent); эндпоинт только вызывает граф.
thread_id задаёт клиент и держит стабильным — по нему checkpointer находит историю thread.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Path, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.admin.deps import require_admin
from app.chat.deps import SettingsDep
from app.chat.domain import RequestError
from app.observability.logging import get_logger
from app.schemas.errors import ErrorResponse
from app.services.agent_persistent import (
    capped_role,
    interrupt_view,
    new_turn,
    resume_command,
    sse,
    stream_events,
    thread_config,
)
from app.services.security.input_validator import validate_input

log = get_logger()

THREAD_ID = r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$"
ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse, "description": "Нет или неверный X-Admin-Token"},
    403: {"model": ErrorResponse, "description": "Роль read-only не подтверждает отправку"},
    409: {"model": ErrorResponse, "description": "Thread ждёт решения (input) или решения не ждёт (resume)"},
    503: {"model": ErrorResponse, "description": "ADMIN_TOKEN не задан или агент не поднят"},
}

router = APIRouter(prefix="/agent", tags=["agent"], dependencies=[Depends(require_admin)], responses=ERRORS)


class AgentMessageIn(BaseModel):
    role: Literal["user"] = "user"          # system и assistant из запроса в историю thread не попадают
    content: str = Field(min_length=1, max_length=4000)


class AgentInput(BaseModel):
    messages: list[AgentMessageIn] = Field(min_length=1, max_length=10)


class ResumeDecision(BaseModel):
    approved: bool
    comment: str | None = Field(default=None, max_length=200)


class AgentStreamIn(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"thread_id": "demo-1", "input": {"messages": [{"role": "user", "content": (
            "Найди в базе знаний, за сколько дней до окончания периода приходит письмо об автопродлении, и "
            "отправь клиенту в Telegram-чат 12345 короткое напоминание об этом.")}]}},
        {"thread_id": "demo-1", "resume": True},
    ]})

    thread_id: str = Field(pattern=THREAD_ID, description="Стабильный id диалога: tg-<chat_id>, имя сценария")
    input: AgentInput | None = None
    resume: bool | ResumeDecision | None = None
    user_role: Literal["read-only", "write-with-approve", "full"] | None = None

    @model_validator(mode="after")
    def _input_or_resume(self) -> AgentStreamIn:
        if (self.input is None) == (self.resume is None):
            raise ValueError("нужно ровно одно поле: input (новое сообщение) или resume (решение по отправке)")
        return self


def get_agent(request: Request) -> Any:
    graph = getattr(request.app.state, "agent", None)
    if graph is None:
        raise RequestError(503, "agent_unavailable",
                           "Агент не поднят: checkpointer не открылся при старте сервиса (строка agent_unavailable в логе).")
    return graph


AgentDep = Annotated[Any, Depends(get_agent)]


@router.post(
    "/stream",
    summary="Агент потоком SSE",
    description="Новое сообщение (input) или решение по отправке (resume) → события data: {json}: node, token, "
                "interrupt (ждёт подтверждения), end. Отправка клиенту — только после resume: true.",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "Поток событий графа"}},
)
async def agent_stream(body: AgentStreamIn, graph: AgentDep, settings: SettingsDep) -> StreamingResponse:
    role = capped_role(body.user_role, settings.agent_max_role)
    config = thread_config(body.thread_id, user_role=role)
    snapshot = await graph.aget_state(config)
    pending = bool(snapshot.interrupts)
    if body.resume is not None:
        if not pending:
            raise RequestError(409, "nothing_to_resume", "Thread не ждёт подтверждения: resume не к чему применить.")
        decision = body.resume if isinstance(body.resume, bool) else body.resume.model_dump()
        approved = decision is True or (isinstance(decision, dict) and decision["approved"])
        if approved and role == "read-only":
            raise RequestError(403, "forbidden", "Роль read-only не может подтвердить отправку.")
        graph_input: Any = resume_command(decision)
    else:
        if pending:
            raise RequestError(409, "approval_pending",
                               "Thread ждёт подтверждения отправки: сначала resume (true или false), потом новое сообщение.")
        messages = body.input.messages  # type: ignore[union-attr]
        if settings.security.enabled:
            for message in messages:
                verdict = validate_input(message.content, settings.security.max_input_chars)
                if not verdict.ok:
                    log.warning("agent_input_rejected", rule=verdict.rule)
                    raise RequestError(400, "input_rejected", "Сообщение не прошло проверку безопасности.")
        graph_input = new_turn([m.model_dump() for m in messages])
    mode = "resume" if body.resume is not None else "input"
    log.info("agent_stream_started", thread_id=body.thread_id, user_role=role, mode=mode)

    async def events() -> AsyncIterator[str]:
        yield sse({"event": "start", "thread_id": body.thread_id, "user_role": role, "mode": mode})
        try:
            async for event in stream_events(graph, graph_input, config):
                yield sse(event)
        except Exception as exc:  # noqa: BLE001 — заголовки уже ушли: ошибку получает клиент событием
            log.warning("agent_stream_failed", thread_id=body.thread_id, error=type(exc).__name__)
            yield sse({"event": "error", "code": "agent_failed",
                       "message": f"Агент остановился с ошибкой: {type(exc).__name__}"})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/threads/{thread_id}", summary="Состояние thread по checkpointer")
async def thread_state(thread_id: Annotated[str, Path(pattern=THREAD_ID)], graph: AgentDep,
                       settings: SettingsDep) -> dict[str, Any]:
    snapshot = await graph.aget_state(thread_config(thread_id, user_role=capped_role(None, settings.agent_max_role)))
    values = snapshot.values or {}
    return {"thread_id": thread_id, "checkpoint_id": (snapshot.config or {}).get("configurable", {}).get("checkpoint_id"),
            "next": list(snapshot.next), "pending_approval": bool(snapshot.interrupts),
            "interrupts": interrupt_view(snapshot.interrupts), "sent": values.get("sent"),
            "draft": values.get("draft"), "messages": len(values.get("messages", []))}
