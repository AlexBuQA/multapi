"""
ModerationService (блок 4.4) без сервиса и без сети: слой ключевых слов из
moderation_keywords.yaml, слой OpenAI Moderation на подменённом клиенте, лог инцидента.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from openai.types import ModerationCreateResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # tests/: log_capture.py
from log_capture import captured_logs  # noqa: E402

from app.moderation import ModerationService, log_incident, text_hash
from app.moderation.keywords import DEFAULT_KEYWORDS_FILE, KeywordModerator, normalize_for_match
from app.moderation.openai_layer import MAX_INPUT_CHARS, OpenAIModerator, category_name

KEYWORDS = KeywordModerator.from_file(DEFAULT_KEYWORDS_FILE)

BLOCKED = [
    ("Я тебя убью", "violence"),
    ("убью тебя, если не вернёте деньги", "violence"),
    ("I'll kill you", "violence"),
    ("Хочу покончить с собой", "self_harm"),
    ("думаю о СУИЦИДЕ", "self_harm"),
    ("Не хочу жить.", "self_harm"),                  # проверка на Windows: так и не было поймано
    ("Больше не хочу жить", "self_harm"),
    ("жить не хочется", "self_harm"),
    ("как сделать бомбу дома", "weapons"),
    ("где купить мефедрон", "drugs"),
    ("Помогите взломать чужой аккаунт", "hacking"),
    ("как украсть пароль от почты коллеги", "hacking"),
    ("Уб​ью тебя", "violence"),                 # невидимый символ внутри слова
    ("покончить\n  с   собой", "self_harm"),         # переводы строк и пробелы
]
# Похожие, но обычные вопросы поддержки — блокироваться не должны.
ALLOWED = [
    "Как убить зависший процесс?",
    "Как убить процесс в диспетчере задач",
    "Мой аккаунт взломали, что делать?",
    "Меня пытались взломать — как защитить аккаунт?",
    "Как добавить закладку в личном кабинете?",
    "Как распознать фишинговое письмо?",
    "Удалить аккаунт навсегда",
    "Бомба, а не обновление! Спасибо",
    "Пароль украли? Нет, я его забыла",
    "Не хочу жить без тёмной темы в приложении",
]


@pytest.mark.parametrize("text, category", BLOCKED)
def test_keywords_block(text, category):
    result = ModerationService(KEYWORDS).check_keywords(text)
    assert not result.allowed and category in result.categories
    assert result.blocked_by == "keywords"
    assert all(re.fullmatch(r"keywords:\w+#\d+", reason) for reason in result.reasons)   # без текста


@pytest.mark.parametrize("text", ALLOWED)
def test_keywords_do_not_block_support_questions(text):
    assert ModerationService(KEYWORDS).check_keywords(text).allowed


def test_normalization():
    assert normalize_for_match("  ЁЛКА​  и\nЁж ") == "елка и еж"


@pytest.mark.parametrize("content, error", [
    ("categories: {violence: ['(']}", "неверный шаблон"),
    ("categories: {}", "ожидается раздел categories"),
    ("categories: {violence: 'убью'}", "список строк"),
    ("- просто список", "ожидается раздел categories"),
])
def test_broken_keywords_file_fails_at_start(tmp_path, content, error):
    path = tmp_path / "keywords.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        KeywordModerator.from_file(path)


# ---------------------------------------------------------------- OpenAI Moderation
CATEGORIES = ["harassment", "harassment/threatening", "hate", "hate/threatening", "illicit", "illicit/violent",
              "self-harm", "self-harm/instructions", "self-harm/intent", "sexual", "sexual/minors", "violence",
              "violence/graphic"]


def moderation_response(flagged: dict[str, bool] | None = None, scores: dict[str, float] | None = None):
    flagged, scores = flagged or {}, scores or {}
    return ModerationCreateResponse.model_validate({"id": "modr-1", "model": "omni-moderation-latest", "results": [{
        "flagged": any(flagged.values()),
        "categories": {c: flagged.get(c, False) for c in CATEGORIES},
        "category_scores": {c: scores.get(c, 0.0) for c in CATEGORIES},
        "category_applied_input_types": {c: ["text"] for c in CATEGORIES},
    }]})


def fake_client(response=None, error: Exception | None = None):
    create = AsyncMock(return_value=response, side_effect=error)
    return SimpleNamespace(moderations=SimpleNamespace(create=create)), create


async def test_openai_flagged_categories():
    client, create = fake_client(moderation_response({"self-harm": True, "self-harm/intent": True},
                                                     {"self-harm": 0.9, "self-harm/intent": 0.8}))
    moderator = OpenAIModerator(client)
    assert await moderator.categories("x" * (MAX_INPUT_CHARS + 10)) == ["self_harm", "self_harm_intent"]
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "omni-moderation-latest" and len(kwargs["input"]) == MAX_INPUT_CHARS


async def test_openai_custom_thresholds_both_ways():
    """Свой порог строже (violence 0.3 при score 0.4) и мягче (harassment 0.95 при флаге OpenAI)."""
    client, _ = fake_client(moderation_response({"harassment": True}, {"violence": 0.4, "harassment": 0.7}))
    moderator = OpenAIModerator(client, thresholds={"violence": 0.3, "harassment": 0.95})
    assert await moderator.categories("текст") == ["violence"]


def test_category_names():
    assert [category_name(c) for c in ("self-harm/intent", "violence", "hate/threatening")] == \
        ["self_harm_intent", "violence", "hate_threatening"]


async def test_keywords_go_first_and_openai_is_not_called():
    client, create = fake_client(moderation_response())
    service = ModerationService(KEYWORDS, OpenAIModerator(client))
    result = await service.check_input("Я тебя убью")
    assert (result.allowed, result.blocked_by) == (False, "keywords")
    create.assert_not_called()


async def test_openai_layer_blocks_what_keywords_miss():
    client, _ = fake_client(moderation_response({"violence": True}, {"violence": 0.8}))
    result = await ModerationService(KEYWORDS, OpenAIModerator(client)).check_output("тонкая угроза")
    assert (result.allowed, result.blocked_by, result.categories, result.reasons) == \
        (False, "openai", ["violence"], ["openai:violence"])


def api_error() -> Exception:
    request = httpx.Request("POST", "https://api.openai.com/v1/moderations")
    return openai.APIConnectionError(request=request)


async def test_openai_error_is_fail_open_by_default():
    client, _ = fake_client(error=api_error())
    with captured_logs("INFO") as logs:
        result = await ModerationService(KEYWORDS, OpenAIModerator(client)).check_input("Как сменить пароль?")
    assert result.allowed
    assert [e["event"] for e in logs] == ["moderation_api_failed"] and logs[0]["fail_closed"] is False


async def test_openai_error_fail_closed():
    client, _ = fake_client(error=api_error())
    result = await ModerationService(KEYWORDS, OpenAIModerator(client), fail_closed=True).check_input("вопрос")
    assert (result.allowed, result.categories) == (False, ["moderation_unavailable"])


async def test_disabled_and_empty_text_pass():
    assert (await ModerationService(KEYWORDS, enabled=False).check_input("Я тебя убью")).allowed
    assert (await ModerationService(KEYWORDS).check_input("   ")).allowed
    assert (await ModerationService(None).check_input("Я тебя убью")).allowed


def test_incident_log_has_hash_and_masked_text_only():
    raw = "Я тебя убью, пиши на ivan@example.com или +7 999 123-45-67"
    result = ModerationService(KEYWORDS).check_keywords(raw)
    with captured_logs("INFO") as logs:
        log_incident("input", result, raw, chat_id="c-1")
    event = logs[0]
    assert event["event"] == "moderation_blocked" and event["level"] == "warning"
    assert (event["direction"], event["blocked_by"], event["categories"]) == ("input", "keywords", ["violence"])
    assert event["text_hash"] == text_hash(raw) and re.fullmatch(r"[0-9a-f]{16}", event["text_hash"])
    assert "ivan@example.com" not in str(event) and "123-45-67" not in str(event)
    assert "[EMAIL]" in event["text_masked"] and "[PHONE_RU]" in event["text_masked"]


def test_demo_keywords_file_for_manual_check():
    """samples/moderation_demo.yaml (README, проверка на Windows): вопрос про вход проходит, ответ с «паролем» — нет."""
    demo = ModerationService(KeywordModerator.from_file(Path(__file__).resolve().parents[3] / "samples" /
                                                        "moderation_demo.yaml"))
    assert demo.check_keywords("Не могу войти в личный кабинет, что делать?").allowed
    assert demo.check_keywords("Нажмите «Забыли пароль?» на странице входа").categories == ["demo"]
