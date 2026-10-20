"""
Веб-интерфейс демонстрации (итоговая аттестация): GET /ui — страница app/ui/index.html.

    pytest tests/test_ui.py

Без браузера: страница отдаётся без ключа, порог близости RAG подставлен из настроек, ключ администратора
в разметку не попадает, а каждый адрес API, в который ходит скрипт страницы, есть у сервиса — переименуют
эндпоинт, и тест покажет, что страница сломается.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Как tests/test_service.py: настройки до импорта app.main, .env разработчика на тесты не влияет.
os.environ["LLM__OPENAI_API_KEY"] = "test-key"
os.environ["LLM__DEFAULT_MODEL"] = "test-model"
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"
os.environ["RATE_LIMIT_PER_MIN"] = "0"
os.environ["LOG_FILE"] = os.devnull

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import Settings, get_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.routers.ui import PAGE, THRESHOLD_MARK, render_page  # noqa: E402

ADMIN = "ui-test-admin-token-0123456789"

# Что вызывает скрипт страницы: (метод, путь в OpenAPI, как путь записан в index.html).
API = [
    ("get", "/health", '"/health"'),
    ("post", "/chats", '"/chats"'),
    ("post", "/chats/{chat_id}/messages", "`/chats/${chatId}/messages`"),
    ("delete", "/chats/{chat_id}/messages", "`/chats/${chatId}/messages`"),
    ("post", "/chats/{chat_id}/messages/{message_id}/feedback", "`/chats/${chatId}/messages/${turn.messageId}/feedback`"),
    ("post", "/agent/stream", '"/agent/stream"'),
    ("get", "/agent/threads/{thread_id}", "`/agent/threads/${encodeURIComponent(id)}`"),
    ("post", "/documents/upload", '"/documents/upload"'),
    ("get", "/documents/status", "`/documents/status?file=${encodeURIComponent(acc.file)}`"),
]


@pytest.fixture
def client():
    cfg = Settings(llm={"openai_api_key": "k", "default_model": "test-model"}, admin_token=ADMIN,  # type: ignore[arg-type]
                   rag={"score_threshold": 0.42}, _env_file=None)
    app.dependency_overrides[get_settings] = lambda: cfg
    yield TestClient(app)      # без with: lifespan (Redis, Postgres, агент) странице не нужен
    app.dependency_overrides.pop(get_settings, None)


def test_page_is_served_without_a_key_with_the_threshold_filled_in(client):
    response = client.get("/ui")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    html = response.text
    assert THRESHOLD_MARK not in html and 'Number("0.420")' in html
    for tab in ("Вопросы по документам", "Поручение агенту", "Пополнить базу"):
        assert f">{tab}</button>" in html
    assert '<span class="by">by Alexandra Bujor</span>' in html          # авторство — в шапке
    assert "© 2026 Produced by Alexandra Bujor. Все права защищены." in html   # и в подвале


def test_admin_token_stays_out_of_the_page(client):
    html = client.get("/ui").text
    assert ADMIN not in html
    assert "localStorage" not in html            # ключ — только в sessionStorage вкладки
    assert "X-Admin-Token" in html               # и уходит заголовком, как в API


def test_page_has_no_external_scripts():
    html = PAGE.read_text(encoding="utf-8")
    assert not re.search(r"<script[^>]+src=", html)
    assert html.count("<script>") == 1


@pytest.mark.parametrize(("method", "path", "in_page"), API, ids=[f"{m} {p}" for m, p, _ in API])
def test_every_endpoint_the_page_calls_exists(method, path, in_page):
    assert method in app.openapi()["paths"].get(path, {}), f"нет {method.upper()} {path}"
    assert in_page in PAGE.read_text(encoding="utf-8")


def test_page_stays_out_of_the_api_docs():
    assert "/ui" not in app.openapi()["paths"]


def test_render_page_formats_the_threshold():
    assert 'Number("0.500")' in render_page(0.5)
    assert 'Number("0.375")' in render_page(0.375)
