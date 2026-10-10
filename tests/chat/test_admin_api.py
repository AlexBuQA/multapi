"""
Admin API и оценки ответов (блок 4.4) через приложение целиком: JSON-хранилище, поддельная модель.

Сами сводки (stats, recent_users) и очередь рассылки на обоих хранилищах проверяет
test_ops_repository.py; здесь — HTTP: X-Admin-Token, коды ответов, формат, порядок роутеров.
"""
from __future__ import annotations

import json
from uuid import uuid4

import httpx
import pytest
from chat_fakes import FakeLLM, make_settings
from log_capture import captured_logs
from log_capture import events as log_events
from pydantic import SecretStr

from app.chat.deps import get_llm_client
from app.core.config import get_settings
from app.main import app

ADMIN = "admin-token-0123456789abcdef"
HEADERS = {"X-Admin-Token": ADMIN}


@pytest.fixture
def admin_app(tmp_path):
    settings = make_settings(tmp_path, admin_token=SecretStr(ADMIN))
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_llm_client] = lambda: FakeLLM()
    yield settings
    app.dependency_overrides.clear()


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def new_chat(http: httpx.AsyncClient, owner: str | None = None, interface: str = "telegram") -> str:
    response = await http.post("/chats", json={"owner_external_id": owner or f"{uuid4().int % 10**9}",
                                               "interface": interface})
    return response.json()["chat_id"]


async def ask(http: httpx.AsyncClient, chat_id: str, text: str) -> dict:
    """Вопрос; возвращает последнее событие потока (done с message_id) или JSON ошибки."""
    response = await http.post(f"/chats/{chat_id}/messages", data={"content": text})
    if response.status_code != 200:
        return response.json()
    return json.loads(response.text.rstrip().rsplit("data: ", 1)[1])


# ---------------------------------------------------------------- доступ
@pytest.mark.parametrize("path, method", [("/chats/admin/stats", "GET"), ("/chats/admin/users", "GET"),
                                          ("/chats/admin/broadcast", "POST"), ("/chats/admin/broadcast/claim", "POST"),
                                          ("/chats/admin/broadcast/1/result", "POST")])
async def test_every_admin_endpoint_needs_token(admin_app, path, method):
    async with client() as http:
        missing = await http.request(method, path, json={})
        wrong = await http.request(method, path, json={}, headers={"X-Admin-Token": "x" * 28})
    assert (missing.status_code, wrong.status_code) == (401, 401)       # 401, а не 422: до проверки полей тела не дошло
    assert missing.json()["error"]["code"] == "unauthorized"


async def test_admin_api_is_off_without_admin_token(tmp_path):
    app.dependency_overrides[get_settings] = lambda: make_settings(tmp_path)
    try:
        async with client() as http:
            response = await http.get("/chats/admin/stats", headers=HEADERS)
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503 and response.json()["error"]["code"] == "admin_token_not_configured"


async def test_admin_routes_win_over_chat_routes(admin_app):
    """GET /chats/admin/stats не уходит в GET /chats/{chat_id} (там был бы 422 «admin — не UUID»)."""
    async with client() as http:
        response = await http.get("/chats/admin/stats", headers=HEADERS)
        chat = await http.get(f"/chats/{uuid4()}")
    assert response.status_code == 200 and chat.status_code == 404


# ---------------------------------------------------------------- stats и users
async def test_stats_after_real_dialogs(admin_app):
    async with client() as http:
        alice, bob = await new_chat(http, "1001"), await new_chat(http, "1002")
        answer = await ask(http, alice, "Как сбросить пароль?")
        await ask(http, bob, "как сбросить ПАРОЛЬ!")
        blocked = await ask(http, bob, "Я тебя убью")
        vote = await http.post(f"/chats/{alice}/messages/{answer['message_id']}/feedback", json={"value": "up"})
        stats = await http.get("/chats/admin/stats", headers=HEADERS, params={"top": 3})
        users = await http.get("/chats/admin/users", headers=HEADERS, params={"limit": 1})
    assert blocked["detail"]["code"] == "moderation_blocked" and vote.status_code == 200
    body = stats.json()
    assert (body["period_hours"], body["total_messages"], body["active_users"]) == (24, 4, 2)
    assert body["avg_latency_ms"] is not None and body["avg_latency_ms"] >= 0
    assert (body["moderation_blocks"], body["moderation_block_rate"]) == (1, round(1 / 3, 4))
    assert (body["feedback_votes"], body["feedback_up_ratio"]) == (1, 1.0)
    assert body["top_questions"] == [{"question": "как сбросить пароль", "count": 2}]
    assert [u["owner_external_id"] for u in users.json()] == ["1002"]       # последний писавший
    assert set(users.json()[0]) == {"owner_external_id", "interface", "chats", "last_seen_at"}


async def test_stats_query_limits(admin_app):
    async with client() as http:
        too_long = await http.get("/chats/admin/stats", headers=HEADERS, params={"hours": 0})
        too_many = await http.get("/chats/admin/users", headers=HEADERS, params={"limit": 501})
    assert (too_long.status_code, too_many.status_code) == (422, 422)


# ---------------------------------------------------------------- рассылка
async def test_broadcast_queue_over_http(admin_app):
    async with client() as http:
        await new_chat(http, "1001")
        await new_chat(http, "1002")
        await new_chat(http, "web-user", interface="web")
        queued = await http.post("/chats/admin/broadcast", headers=HEADERS,
                                 json={"message": "  Плановые работы в 23:00  ", "interface_filter": "telegram"})
        claimed = await http.post("/chats/admin/broadcast/claim", headers=HEADERS)
        empty = await http.post("/chats/admin/broadcast/claim", headers=HEADERS)
        done = await http.post(f"/chats/admin/broadcast/{claimed.json()['id']}/result", headers=HEADERS,
                               json={"sent": 2, "failed": 0})
        unknown = await http.post("/chats/admin/broadcast/999/result", headers=HEADERS, json={"sent": 1, "failed": 0})
        late = await http.post(f"/chats/admin/broadcast/{claimed.json()['id']}/result", headers=HEADERS,
                               json={"sent": 0, "failed": 2})
        web = await http.post("/chats/admin/broadcast", headers=HEADERS, json={"message": "Сайт", "interface_filter": "web"})
        tg_claim = await http.post("/chats/admin/broadcast/claim", headers=HEADERS)
        web_claim = await http.post("/chats/admin/broadcast/claim", headers=HEADERS, params={"interface": "web"})
    assert queued.status_code == 202
    assert {k: queued.json()[k] for k in ("status", "interface", "recipients")} == \
        {"status": "pending", "interface": "telegram", "recipients": 2}
    assert claimed.json() == {"id": queued.json()["id"], "message": "Плановые работы в 23:00", "interface": "telegram",
                              "recipients": ["1001", "1002"]}
    assert empty.status_code == 204 and empty.content == b""
    assert (done.json()["status"], done.json()["sent"], done.json()["recipients"]) == ("sent", 2, 2)
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "broadcast_not_found"
    assert late.status_code == 409 and late.json()["error"]["code"] == "broadcast_not_sending"
    assert tg_claim.status_code == 204                          # рассылку для web бот не забирает
    assert web_claim.json()["id"] == web.json()["id"] and web_claim.json()["recipients"] == ["web-user"]


@pytest.mark.parametrize("body", [{"message": "   "}, {"message": ""}, {"message": "я" * 4001},
                                  {"message": "x", "interface_filter": "Telegram!"}, {}])
async def test_broadcast_validation(admin_app, body):
    async with client() as http:
        response = await http.post("/chats/admin/broadcast", headers=HEADERS, json=body)
    assert response.status_code == 422


# ---------------------------------------------------------------- оценки
async def test_feedback_flow(admin_app):
    async with client() as http:
        chat_id = await new_chat(http, "1001")
        done = await ask(http, chat_id, "Меня зовут Аня")
        url = f"/chats/{chat_id}/messages/{done['message_id']}/feedback"
        first = await http.post(url, json={"value": "up"})
        again = await http.post(url, json={"value": "down"})
        history = (await http.get(f"/chats/{chat_id}/messages")).json()
        question = await http.post(f"/chats/{chat_id}/messages/{history[0]['id']}/feedback", json={"value": "up"})
        missing = await http.post(f"/chats/{chat_id}/messages/{uuid4()}/feedback", json={"value": "up"})
        no_chat = await http.post(f"/chats/{uuid4()}/messages/{done['message_id']}/feedback", json={"value": "up"})
        bad_value = await http.post(url, json={"value": "like"})
    assert first.json() == {"message_id": done["message_id"], "value": "up", "saved": True}
    assert again.json() == {"message_id": done["message_id"], "value": "up", "saved": False}
    assert (question.status_code, question.json()["error"]["code"]) == (422, "not_an_answer")
    assert (missing.status_code, missing.json()["error"]["code"]) == (404, "message_not_found")
    assert (no_chat.status_code, no_chat.json()["error"]["code"]) == (404, "chat_not_found")
    assert bad_value.status_code == 422


async def test_feedback_from_other_chat_is_404(admin_app):
    """Ответ чужого чата оценить нельзя: сообщение ищется в чате из URL."""
    async with client() as http:
        mine, other = await new_chat(http, "1001"), await new_chat(http, "1002")
        done = await ask(http, mine, "Привет")
        response = await http.post(f"/chats/{other}/messages/{done['message_id']}/feedback", json={"value": "down"})
    assert response.status_code == 404


async def test_empty_queue_polls_stay_out_of_info_log(admin_app):
    """Бот опрашивает очередь каждые 5 с: пустой ответ — только на DEBUG, ошибка — в лог."""
    async with client() as http:
        with captured_logs("INFO") as logs:
            empty = await http.post("/chats/admin/broadcast/claim", headers=HEADERS)
            denied = await http.post("/chats/admin/broadcast/claim")
    assert (empty.status_code, denied.status_code) == (204, 401)
    assert [e["status"] for e in log_events(logs, "http_request")] == [401]
