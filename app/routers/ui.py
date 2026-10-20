"""
Веб-интерфейс для демонстрации (итоговая аттестация): GET /ui — одна страница без сборки и зависимостей.

    GET /ui   app/ui/index.html: вопросы по документам (POST /chats, /chats/{id}/messages — SSE, источники,
              👍/👎, файл в чате), поручение агенту (POST /agent/stream, GET /agent/threads/{id} — пауза перед
              отправкой и решение кнопками) и пополнение базы (POST /documents/upload, GET /documents/status).

Страница ходит в API того же сервиса (тот же origin — CORS не нужен). Агент и загрузка — под X-Admin-Token,
как в API: ключ администратора вводится на странице, живёт в sessionStorage вкладки и в разметку не попадает.
В страницу подставляется только порог близости RAG — для отметки «порог» на шкале близости источников.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.deps.providers import SettingsDep

PAGE = Path(__file__).resolve().parent.parent / "ui" / "index.html"
THRESHOLD_MARK = "__SCORE_THRESHOLD__"

router = APIRouter(tags=["ui"])


def render_page(score_threshold: float) -> str:
    return PAGE.read_text(encoding="utf-8").replace(THRESHOLD_MARK, f"{score_threshold:.3f}")


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def ui_page(settings: SettingsDep) -> HTMLResponse:
    return HTMLResponse(render_page(settings.rag.score_threshold), headers={"Cache-Control": "no-store"})
