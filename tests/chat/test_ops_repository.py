"""
Контракт OpsRepository (блок 4.4): оценки, инциденты модерации, очередь рассылки, сводки.

Как и test_repository_contract.py, каждая проверка идёт против JSON и Postgres
(фикстуры repo и clean_repo в tests/chat/conftest.py). Postgres не запущен — варианты
[postgres] пропускаются.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app.chat.domain import BroadcastNotClaimedError, ModerationIncident
from app.chat.repository import OpsRepository, normalize_question
from chat_fakes import message

T0 = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)


def at(minutes: float = 0, seconds: float = 0) -> datetime:
    return T0 + timedelta(minutes=minutes, seconds=seconds)


async def test_implements_ops_protocol(repo):
    assert isinstance(repo, OpsRepository)


async def test_get_message_of_own_chat_even_after_clear(repo):
    chat = await repo.create_chat("u", "telegram")
    other = await repo.create_chat("v", "telegram")
    answer = message(chat.id, "assistant", "Ответ")
    await repo.append_message(chat.id, answer)
    await repo.soft_delete_messages(chat.id)
    assert await repo.get_message(chat.id, answer.id) == answer          # /clear скрыл, но ответ был
    assert await repo.get_message(other.id, answer.id) is None           # чужой чат — нет
    assert await repo.get_message(chat.id, message(chat.id, "user", "x").id) is None


async def test_feedback_is_unique_per_owner_and_message(repo):
    chat = await repo.create_chat("111", "telegram")
    answer = message(chat.id, "assistant", "Ответ")
    await repo.append_message(chat.id, answer)
    first = await repo.add_feedback(chat.id, answer.id, "111", "up")
    again = await repo.add_feedback(chat.id, answer.id, "111", "down")
    other = await repo.add_feedback(chat.id, answer.id, "222", "down")
    assert (first.saved, first.value) == (True, "up")
    assert (again.saved, again.value) == (False, "up")                   # первая оценка не меняется
    assert (other.saved, other.value) == (True, "down")


async def test_concurrent_votes_save_one(json_repo):
    chat = await json_repo.create_chat("111", "telegram")
    answer = message(chat.id, "assistant", "Ответ")
    await json_repo.append_message(chat.id, answer)
    results = await asyncio.gather(*(json_repo.add_feedback(chat.id, answer.id, "111", v)
                                     for v in ("up", "down", "up", "down", "up")))
    assert sum(r.saved for r in results) == 1
    assert len({r.value for r in results}) == 1                          # все видят одну и ту же оценку


async def test_concurrent_votes_save_one_postgres(pg_sessions):
    """Две сессии — как два запроса сервиса: решает уникальный индекс, а не проверка в коде."""
    from app.chat.repositories.pg_repo import PostgresChatRepository

    async with pg_sessions() as s1, pg_sessions() as s2:  # type: ignore[operator]
        first, second = PostgresChatRepository(s1), PostgresChatRepository(s2)
        chat = await first.create_chat("333", "telegram")
        answer = message(chat.id, "assistant", "Ответ")
        await first.append_message(chat.id, answer)
        results = await asyncio.gather(first.add_feedback(chat.id, answer.id, "333", "up"),
                                       second.add_feedback(chat.id, answer.id, "333", "down"))
    assert sorted(r.saved for r in results) == [False, True]
    assert results[0].value == results[1].value


async def test_broadcast_queue_flow(clean_repo):
    repo = clean_repo
    first = await repo.enqueue_broadcast("Плановые работы в 23:00", "telegram")
    second = await repo.enqueue_broadcast("Работы завершены", "telegram")
    assert (first.status, second.id > first.id) == ("pending", True)
    claimed = await repo.claim_broadcast(stale_before=datetime.now(UTC) - timedelta(minutes=15))
    assert (claimed.id, claimed.status, claimed.message) == (first.id, "sending", "Плановые работы в 23:00")
    assert claimed.claimed_at is not None
    assert (await repo.claim_broadcast(datetime.now(UTC) - timedelta(minutes=15))).id == second.id
    assert await repo.claim_broadcast(datetime.now(UTC) - timedelta(minutes=15)) is None
    # Бот упал посреди рассылки: «sending» старше stale_before забирается снова.
    assert (await repo.claim_broadcast(datetime.now(UTC) + timedelta(seconds=1))).id == first.id

    done = await repo.finish_broadcast(first.id, sent=3, failed=1)
    failed = await repo.finish_broadcast(second.id, sent=0, failed=2)
    assert (done.status, done.recipients, done.sent, done.failed) == ("sent", 4, 3, 1)
    assert done.finished_at is not None
    assert (failed.status, failed.recipients) == ("failed", 2)
    assert await repo.finish_broadcast(10_000, sent=1, failed=0) is None
    assert await repo.claim_broadcast(datetime.now(UTC) + timedelta(seconds=1)) is None   # готовые не берутся


async def test_broadcast_result_only_for_claimed(clean_repo):
    """Итог — только за рассылку в sending: незабранная и уже завершённая его не принимают."""
    repo = clean_repo
    item = await repo.enqueue_broadcast("Текст", "telegram")
    with pytest.raises(BroadcastNotClaimedError) as pending:
        await repo.finish_broadcast(item.id, sent=1, failed=0)
    assert pending.value.status == "pending"
    await repo.claim_broadcast(datetime.now(UTC) - timedelta(minutes=15))
    assert (await repo.finish_broadcast(item.id, sent=1, failed=0)).status == "sent"
    with pytest.raises(BroadcastNotClaimedError) as finished:
        await repo.finish_broadcast(item.id, sent=0, failed=1)            # опоздавший итог
    assert finished.value.status == "sent"


async def test_claim_is_per_interface(clean_repo):
    """Бот (telegram) не забирает рассылку для web: её отправит свой отправитель."""
    repo = clean_repo
    web = await repo.enqueue_broadcast("Для сайта", "web")
    tg = await repo.enqueue_broadcast("Для Telegram", "telegram")
    stale = datetime.now(UTC) - timedelta(minutes=15)
    assert (await repo.claim_broadcast(stale)).id == tg.id                 # по умолчанию — telegram
    assert await repo.claim_broadcast(stale) is None
    assert (await repo.claim_broadcast(stale, interface="web")).id == web.id


async def test_broadcast_recipients_are_owners_of_interface(clean_repo):
    repo = clean_repo
    await repo.create_chat("1001", "telegram")
    await repo.create_chat("cli-user", "cli")
    await repo.create_chat("1002", "telegram")
    await repo.create_chat("1001", "telegram")                           # второй чат того же клиента
    assert await repo.broadcast_recipients("telegram") == ["1001", "1002"]
    assert await repo.broadcast_recipients("web") == []


async def test_stats_over_period(clean_repo):
    repo = clean_repo
    since = at(0)
    alice = await repo.create_chat("1001", "telegram")
    bob = await repo.create_chat("1002", "telegram")
    rows = [
        message(alice.id, "user", "Старый вопрос", created_at=at(-120)),         # до периода
        message(alice.id, "assistant", "Старый ответ", created_at=at(-119)),
        message(alice.id, "user", "Как сбросить пароль?", created_at=at(1)),
        message(alice.id, "assistant", "Нажмите «Забыли пароль».", created_at=at(1, 2)),   # 2000 мс
        message(bob.id, "user", "как СБРОСИТЬ пароль!!", created_at=at(2)),
        message(bob.id, "assistant", "Через форму входа.", created_at=at(2, 4)),          # 4000 мс
        message(bob.id, "assistant", "Уведомление о работах", created_at=at(3)),          # без вопроса — не задержка
        message(bob.id, "user", "Где счёт?", created_at=at(4)),
    ]
    for row in rows:
        await repo.append_message(row.chat_id, row)
    await repo.soft_delete_messages(bob.id)                               # /clear не стирает статистику
    for direction, minutes in (("input", 5), ("output", 6), ("input", -200)):
        await repo.record_moderation(ModerationIncident(chat_id=alice.id, direction=direction, blocked_by="keywords",
                                                        categories=["violence"], text_hash="0" * 16,
                                                        created_at=at(minutes)))
    await repo.add_feedback(alice.id, rows[3].id, "1001", "up")
    await repo.add_feedback(bob.id, rows[5].id, "1002", "down")
    await repo.add_feedback(bob.id, rows[6].id, "1002", "up")

    stats = await repo.stats(since, top_n=2)
    assert (stats.total_messages, stats.user_messages, stats.active_users) == (6, 3, 2)
    assert stats.avg_latency_ms == 3000.0
    assert (stats.moderation_blocks, stats.moderation_input_blocks) == (2, 1)
    assert stats.moderation_block_rate == round(2 / (3 + 1), 4)
    assert (stats.feedback_up, stats.feedback_down, stats.feedback_up_ratio) == (2, 1, round(2 / 3, 4))
    assert [(q.question, q.count) for q in stats.top_questions] == [("как сбросить пароль", 2), ("где счёт", 1)]


async def test_stats_of_empty_storage(clean_repo):
    stats = await clean_repo.stats(at(0))
    assert (stats.total_messages, stats.active_users, stats.avg_latency_ms) == (0, 0, None)
    assert (stats.moderation_block_rate, stats.feedback_up_ratio, stats.top_questions) == (0.0, None, [])


async def test_recent_users(clean_repo):
    repo = clean_repo
    alice = await repo.create_chat("1001", "telegram")
    bob = await repo.create_chat("1002", "telegram")
    await repo.create_chat("1001", "cli")
    await repo.append_message(alice.id, message(alice.id, "user", "раньше", created_at=datetime.now(UTC) - timedelta(hours=1)))
    await repo.append_message(bob.id, message(bob.id, "user", "позже", created_at=datetime.now(UTC) + timedelta(hours=1)))
    users = await repo.recent_users(limit=2)
    assert [(u.owner_external_id, u.interface, u.chats) for u in users] == [("1002", "telegram", 1), ("1001", "cli", 1)]
    assert users[0].last_seen_at > users[1].last_seen_at


@pytest.mark.parametrize("raw, normalized", [
    ("Как СБРОСИТЬ пароль?!", "как сбросить пароль"),
    ("  «Вход»  —  (ЛК)… ", "вход лк"),
    ("???", ""),
])
def test_normalize_question(raw, normalized):
    assert normalize_question(raw) == normalized


async def test_top_questions_normalized_the_same_in_sql_and_python(clean_repo):
    """Postgres (regexp_replace) и JSON (normalize_question) дают одну и ту же строку."""
    repo = clean_repo
    chat = await repo.create_chat("u-norm", "telegram")
    raw = "  Ёлка\xa0и ЁЖ?!  Как\tсбросить\n«пароль»…  "
    await repo.append_message(chat.id, message(chat.id, "user", raw, created_at=at()))
    stats = await repo.stats(at(-1))
    assert [q.question for q in stats.top_questions] == [normalize_question(raw)]
