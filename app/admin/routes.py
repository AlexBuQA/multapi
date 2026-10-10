"""
Admin-эндпоинты (блок 4.4) — под общим префиксом /chats/admin, все с Depends(require_admin):
заголовок X-Admin-Token == ADMIN_TOKEN, иначе 401 (а без ADMIN_TOKEN в .env — 503).

    GET  /chats/admin/stats?hours=24&top=5   сводка за период (StatsOut)
    GET  /chats/admin/users?limit=50          последние клиенты: число чатов и last_seen_at
    POST /chats/admin/broadcast               {message, interface_filter} -> 202, рассылка в очередь
    POST /chats/admin/broadcast/claim?interface=telegram
                                              бот забирает рассылку своего интерфейса: 200 с
                                              получателями или 204
    POST /chats/admin/broadcast/{id}/result   бот отчитывается: {sent, failed}; только за
                                              забранную рассылку (sending), иначе 409

Сводка — ChatStats из хранилища (OpsRepository.stats):
- total_messages — сообщения за период, и скрытые /clear (они были);
- active_users — DAU: разные клиенты (owner_external_id + interface), задавшие вопрос;
- avg_latency_ms — среднее время от вопроса до записи ответа (в Postgres — LAG() по чату);
- moderation_block_rate — блокировки модерации / (принятые вопросы + отклонённые модерацией):
  доля вопросов, закончившихся блокировкой вопроса или ответа;
- feedback_up_ratio — 👍 / (👍 + 👎) за период; null — оценок не было;
- top_questions — частые вопросы: lower(regexp_replace(content, …)) + GROUP BY.

Рассылка — очередь, а не отправка из сервиса: Telegram-токен есть только у бота. Бот раз в
BROADCAST_POLL секунд вызывает claim, рассылает и сообщает result (bot/services/broadcast.py).
Рассылка, которую бот забрал и не закончил (упал), через STALE_AFTER снова отдаётся на claim.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Query, Response

from app.admin.deps import require_admin
from app.admin.schemas import BroadcastClaimOut, BroadcastIn, BroadcastOut, BroadcastResultIn, StatsOut, UserOut
from app.chat.deps import get_repository
from app.chat.domain import BroadcastNotClaimedError, RequestError, utc_now
from app.chat.repository import ChatRepository, OpsRepository
from app.observability.logging import get_logger
from app.schemas.errors import ErrorResponse

log = get_logger()

STALE_AFTER = timedelta(minutes=15)
AUTH = {401: {"model": ErrorResponse, "description": "Нет или неверный X-Admin-Token"},
        503: {"model": ErrorResponse, "description": "ADMIN_TOKEN не задан или хранилище недоступно"}}

router = APIRouter(prefix="/chats/admin", tags=["admin"], dependencies=[Depends(require_admin)], responses=AUTH)


def get_ops(repo: Annotated[ChatRepository, Depends(get_repository)]) -> OpsRepository:
    if not isinstance(repo, OpsRepository):
        raise RequestError(501, "not_supported", "Хранилище чатов не поддерживает admin API.")
    return repo


OpsDep = Annotated[OpsRepository, Depends(get_ops)]


@router.get("/stats", response_model=StatsOut, summary="Сводка: сообщения, DAU, задержка, модерация, оценки")
async def stats(ops: OpsDep, hours: Annotated[int, Query(ge=1, le=24 * 31)] = 24,
                top: Annotated[int, Query(ge=0, le=50)] = 5) -> StatsOut:
    data = await ops.stats(utc_now() - timedelta(hours=hours), top_n=top)
    return StatsOut.of(data, hours)


@router.get("/users", response_model=list[UserOut], summary="Последние клиенты")
async def users(ops: OpsDep, limit: Annotated[int, Query(ge=1, le=500)] = 50) -> list[UserOut]:
    return [UserOut(**u.model_dump()) for u in await ops.recent_users(limit=limit)]


@router.post("/broadcast", response_model=BroadcastOut, status_code=202, summary="Рассылка — в очередь бота")
async def broadcast(body: BroadcastIn, ops: OpsDep) -> BroadcastOut:
    item = await ops.enqueue_broadcast(body.message, body.interface_filter)
    recipients = len(await ops.broadcast_recipients(body.interface_filter))
    log.info("admin_broadcast_queued", broadcast_id=item.id, interface=item.interface, recipients=recipients,
             chars=len(body.message))
    return BroadcastOut.of(item, recipients)


@router.post("/broadcast/claim", response_model=BroadcastClaimOut, summary="Бот забирает рассылку",
             responses={204: {"description": "Очередь пуста"}})
async def claim(ops: OpsDep,
                interface: Annotated[str, Query(pattern=r"^[a-z][a-z0-9_-]*$", max_length=32)] = "telegram",
                ) -> BroadcastClaimOut | Response:
    """Рассылки другого интерфейса (например, web) остаются в очереди своему отправителю."""
    item = await ops.claim_broadcast(stale_before=utc_now() - STALE_AFTER, interface=interface)
    if item is None:
        return Response(status_code=204)
    recipients = await ops.broadcast_recipients(item.interface)
    structlog.contextvars.bind_contextvars(broadcast_id=item.id)
    log.info("admin_broadcast_claimed", broadcast_id=item.id, recipients=len(recipients))
    return BroadcastClaimOut(id=item.id, message=item.message, interface=item.interface, recipients=recipients)


@router.post("/broadcast/{broadcast_id}/result", response_model=BroadcastOut, summary="Бот отчитывается о рассылке",
             responses={404: {"model": ErrorResponse, "description": "Нет такой рассылки"},
                        409: {"model": ErrorResponse, "description": "Рассылку сейчас никто не отправляет"}})
async def result(broadcast_id: int, body: BroadcastResultIn, ops: OpsDep) -> BroadcastOut:
    try:
        item = await ops.finish_broadcast(broadcast_id, body.sent, body.failed)
    except BroadcastNotClaimedError as exc:
        raise RequestError(409, exc.code, f"Рассылка {broadcast_id} не отправляется (статус {exc.status}): "
                                          "итог принимается только за забранную рассылку.") from exc
    if item is None:
        raise RequestError(404, "broadcast_not_found", f"Рассылки {broadcast_id} нет.")
    log.info("admin_broadcast_finished", broadcast_id=item.id, status=item.status, sent=item.sent, failed=item.failed)
    return BroadcastOut.of(item)
