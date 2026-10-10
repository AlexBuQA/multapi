"""
Модерация (блок 4.4): ModerationService — вопрос до вызова модели, ответ после.

    app/moderation/service.py          ModerationService, ModerationResult, ModerationBlocked, log_incident
    app/moderation/keywords.py         слой 1 — регулярные выражения из moderation_keywords.yaml
    app/moderation/openai_layer.py     слой 2 — OpenAI Moderation (omni-moderation-latest), необязательный
    app/moderation/moderation_keywords.yaml   шаблоны по категориям

Подключение — app/chat/service.py (ChatService) и app/main.py (lifespan, ответ 403).
"""
from app.moderation.service import (
    ALLOWED,
    OUTPUT_REFUSAL,
    ModerationBlocked,
    ModerationResult,
    ModerationService,
    build_moderation,
    log_incident,
    text_hash,
)

__all__ = ["ALLOWED", "OUTPUT_REFUSAL", "ModerationBlocked", "ModerationResult", "ModerationService",
           "build_moderation", "log_incident", "text_hash"]
