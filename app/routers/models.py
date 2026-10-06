"""GET /models — статический каталог моделей с ценами (блок 3.4)."""
from __future__ import annotations

from fastapi import APIRouter

from app.schemas.models import MODEL_CATALOG, ModelInfo

router = APIRouter(prefix="/models", tags=["models"])


@router.get(
    "",
    response_model=list[ModelInfo],
    summary="Каталог моделей и цен",
    description=(
        "Статический список: модели OpenAI со справочными ценами за 1 млн токенов "
        "и локальные модели Ollama проекта. Провайдер не вызывается."
    ),
    responses={200: {"description": "Список моделей"}},
)
async def list_models() -> list[ModelInfo]:
    return MODEL_CATALOG
