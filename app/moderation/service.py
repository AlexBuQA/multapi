"""
ModerationService (блок 4.4): модерация вопроса до вызова модели и ответа после.

    result = await moderation.check_input(text)     # ModerationResult(allowed, categories, reasons, blocked_by)
    result = await moderation.check_output(text)
    result = moderation.check_keywords(text)        # только слой 1 — для фрагментов потока

Слои, от дешёвого к дорогому; первый, кто заблокировал, — blocked_by:
1. keywords — регулярные выражения из moderation_keywords.yaml (app/moderation/keywords.py);
2. openai — OpenAI Moderation, omni-moderation-latest, свои пороги по категориям
   (app/moderation/openai_layer.py). Необязательный: без ключа его нет.

Чем модерация отличается от защитного слоя блока 3.8 (app/services/security/): тот ловит
атаки на модель — инъекцию, джейлбрейк, утечку промпта — и отвечает готовым отказом. Модерация
ловит запрещённые темы — насилие, самоповреждение, оружие, наркотики, взлом чужого — и
вопрос не принимает вовсе: 403 moderation_blocked. Проверки независимы: сначала модерация,
потом защитный слой.

Инцидент — строка moderation_blocked в логе (log_incident): направление, слой, категории,
sha256 текста (первые 16 знаков) и маскированный текст — персональные данные заменены
метками redact_pii блока 3.6. Сырого текста в логе нет. Счётчики для /chats/admin/stats —
в хранилище (ChatService записывает ModerationIncident).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from app.moderation.keywords import KeywordModerator
from app.moderation.openai_layer import OpenAIModerator
from app.observability.logging import get_logger
from app.observability.pii import redact_pii

if TYPE_CHECKING:
    from app.core.config import Settings

log = get_logger()

Direction = Literal["input", "output"]
OUTPUT_REFUSAL = "Не могу показать ответ — он мог нарушить правила."
MASKED_CHARS = 300          # сколько маскированного текста класть в лог инцидента


@dataclass(frozen=True)
class ModerationResult:
    allowed: bool
    categories: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)      # «keywords:violence#2», «openai:self_harm»
    blocked_by: str = ""                                   # keywords | openai; пусто — пропущен


ALLOWED = ModerationResult(True)


class ModerationBlocked(Exception):
    """Вопрос не прошёл модерацию: в HTTP — 403 {"detail": {"code": "moderation_blocked", ...}}."""

    code = "moderation_blocked"

    def __init__(self, result: ModerationResult) -> None:
        self.result = result
        self.categories = list(result.categories)
        self.message = "Сообщение не прошло модерацию. Переформулируйте вопрос."
        super().__init__(self.message)


def text_hash(text: str) -> str:
    """Отпечаток текста для поиска повторов: sha256, первые 16 шестнадцатеричных знаков."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def log_incident(direction: Direction, result: ModerationResult, text: str, **context: Any) -> None:
    log.warning("moderation_blocked", direction=direction, blocked_by=result.blocked_by,
                categories=result.categories, reasons=result.reasons, text_hash=text_hash(text),
                text_masked=redact_pii(text)[:MASKED_CHARS], **context)


class ModerationService:
    def __init__(self, keywords: KeywordModerator | None = None, openai: OpenAIModerator | None = None, *,
                 enabled: bool = True, fail_closed: bool = False) -> None:
        self.keywords = keywords
        self.openai = openai
        self.enabled = enabled
        self.fail_closed = fail_closed

    def check_keywords(self, content: str) -> ModerationResult:
        if not self.enabled or self.keywords is None or not content.strip():
            return ALLOWED
        matches = self.keywords.find(content)
        if not matches:
            return ALLOWED
        return ModerationResult(False, [m.category for m in matches], [f"keywords:{m.rule}" for m in matches],
                                "keywords")

    async def check_input(self, content: str) -> ModerationResult:
        return await self._check(content, "input")

    async def check_output(self, content: str) -> ModerationResult:
        return await self._check(content, "output")

    async def _check(self, content: str, direction: Direction) -> ModerationResult:
        result = self.check_keywords(content)
        if not result.allowed or not self.enabled or self.openai is None or not content.strip():
            return result
        try:
            categories = await self.openai.categories(content)
        except Exception as exc:  # noqa: BLE001 — сеть, ключ, 429: решает fail_closed
            log.warning("moderation_api_failed", direction=direction, error=type(exc).__name__,
                        detail=str(exc)[:200], fail_closed=self.fail_closed)
            if self.fail_closed:
                return ModerationResult(False, ["moderation_unavailable"], ["openai:unavailable"], "openai")
            return ALLOWED
        if categories:
            return ModerationResult(False, categories, [f"openai:{c}" for c in categories], "openai")
        return ALLOWED


def build_moderation(settings: Settings, openai_client: Any = None) -> ModerationService:
    """Сервис модерации по настройкам MODERATION__*. Ошибка в файле шаблонов — ошибка старта."""
    cfg = settings.moderation
    keywords = KeywordModerator.from_file(cfg.keywords_file) if cfg.enabled else None
    openai = (OpenAIModerator(openai_client, model=cfg.openai_model, thresholds=cfg.thresholds)
              if cfg.enabled and openai_client is not None else None)
    return ModerationService(keywords, openai, enabled=cfg.enabled, fail_closed=cfg.fail_closed)
