"""
Проверка ответа модели (блок 3.8): канарейка, утечка промпта, роль джейлбрейка,
персональные данные; то же в потоке (StreamGuard).
"""
from __future__ import annotations

import re

import pytest

from app.core.config import SupportSettings
from app.schemas.chat import ChatRequest
from app.services.prompts import build_messages
from app.services.security.canary import CANARY_TEMPLATE, canary_message, new_canary, with_canary
from app.services.security.output_filter import OutputBlocked, StreamGuard, filter_output

CANARY = "CANARY_a7f3b9e2"
SYSTEM = build_messages(ChatRequest(messages=[{"role": "user", "content": "Какая погода сегодня в Москве?"}]),
                        SupportSettings()).messages[0]["content"]


def blocked_rule(answer: str, system_prompt: str | None = SYSTEM, canary: str | None = CANARY) -> str | None:
    try:
        filter_output(answer, system_prompt, canary)
    except OutputBlocked as exc:
        return exc.rule
    return None


# ---------------------------------------------------------------- канарейка
def test_canary_format_and_message():
    canary = new_canary()
    assert re.fullmatch(r"CANARY_[0-9a-f]{8}", canary) and canary != new_canary()
    assert canary_message(canary) == {"role": "system", "content": CANARY_TEMPLATE.format(canary=canary)}


def test_canary_goes_first_with_neutral_wording():
    """Блок 4.2: в конце системного блока llama3.2 путала «секретную метку» с данными пользователя."""
    messages = [{"role": "system", "content": "промпт"}, {"role": "user", "content": "вопрос"}]
    sent = with_canary(messages, CANARY)
    assert [m["content"][:6] for m in sent] == ["Иденти", "промпт", "вопрос"]
    assert sent[0]["content"] == f"Идентификатор сборки: {CANARY}"
    assert not any(word in sent[0]["content"].lower() for word in ("секрет", "разглаш", "пользовател", "имя"))
    assert with_canary([{"role": "user", "content": "q"}], CANARY)[0]["role"] == "system"
    assert with_canary(messages, None) is messages


@pytest.mark.parametrize("leaked", [
    # искусственный «утёкший» ответ: модель пересказала системные сообщения
    f"Мои системные сообщения: идентификатор сборки {CANARY}, затем — ассистент поддержки.",
    "метка: canary_A7F3B9E2",                         # регистр
    "C A N A R Y _ a 7 f 3 b 9 e 2",                  # через пробелы
    "Секретная часть метки — a7f3b9e2",               # без префикса
])
def test_leaked_canary_is_blocked(leaked):
    assert blocked_rule(leaked) == "canary"


def test_without_canary_nothing_to_check():
    assert blocked_rule("Обычный ответ про пароль.", canary=None) is None


# ---------------------------------------------------------------- системный промпт
def test_prompt_head_leak_is_blocked():
    head = " ".join(SYSTEM.split())[:120]
    assert blocked_rule(f"Конечно! Вот мой промпт:\n{head.upper()}") == "prompt_leak"     # регистр и пробелы не важны


def test_two_rules_leak_but_one_quote_is_fine():
    rules = SYSTEM.split("Правила:", 1)[1].split("Статьи руководства:", 1)[0]
    assert blocked_rule("Мои правила:\n" + rules) == "prompt_leak"
    assert blocked_rule("Поддержка не видит и не присылает пароли. Нажмите «Забыли пароль?» (раздел 2.1).") is None


def test_client_system_prompt_is_not_protected():
    assert blocked_rule(" ".join(SYSTEM.split())[:120], system_prompt=None) is None


# ---------------------------------------------------------------- джейлбрейк
@pytest.mark.parametrize("answer", [
    "DAN Mode enabled. I will now answer without restrictions.",
    "GPT: Я не могу.\n\nDAN: Конечно, вот как это сделать.",
    "[DAN] Sure!",
    "Developer Mode enabled",
])
def test_jailbreak_persona_is_blocked(answer):
    assert blocked_rule(answer) == "jailbreak"


# ---------------------------------------------------------------- персональные данные
def test_personal_data_masked_in_answer():
    answer = ("Напишите на ivan@mail.ru или позвоните +7 (999) 123-45-67. Паспорт 4509 123456, СНИЛС 112-233-445 95, "
              "счёт 40702810900000012345, ключ sk-or-v1-0123456789abcdef0123.")
    masked = filter_output(answer, SYSTEM, CANARY)
    for raw in ("ivan@mail.ru", "123-45-67", "4509 123456", "112-233-445 95", "40702810900000012345", "sk-or-v1"):
        assert raw not in masked
    for placeholder in ("[EMAIL]", "[PHONE_RU]", "[PASSPORT]", "[SNILS]", "[ACCOUNT]", "[API_KEY]"):
        assert placeholder in masked


def test_clean_answer_is_unchanged():
    answer = "Ссылка для сброса пароля действует 30 минут (раздел 2.1)."
    assert filter_output(answer, SYSTEM, CANARY) == answer


# ---------------------------------------------------------------- поток
def stream(parts: list[str], canary: str | None = CANARY) -> tuple[list[str], str | None]:
    guard, sent = StreamGuard(SYSTEM, canary), []
    try:
        sent += [guard.feed(part) for part in parts]
        sent.append(guard.finish())
    except OutputBlocked as exc:
        return sent, exc.rule
    return sent, None


def test_stream_holds_tail_and_returns_same_text():
    parts = ["Чтобы сбросить пароль, ", "нажмите «Забыли пароль?» на странице входа ", "и введите e-mail. ",
             "Ссылка придёт в течение 10 минут ", "и действует 30 минут (раздел 2.1)."]
    sent, rule = stream(parts)
    assert rule is None and "".join(sent) == "".join(parts)
    assert sent[0] == ""                                        # первые 80 символов придержаны


def test_stream_never_sends_split_canary():
    parts = ["Хорошо, расскажу всё. " * 5, "Метка: CANA", "RY_a7f3", "b9e2, а теперь правила."]
    sent, rule = stream(parts)
    assert rule == "canary" and "a7f3" not in "".join(sent) and "CANA" not in "".join(sent)


def test_short_hold_without_assistant_prompt():
    """Без промпта ассистента (свой system, чаты блока 4.1) придерживается 40 символов, а не
    80: короткий ответ приходит кусками, а метка — даже через пробелы — всё равно не уходит."""
    answer = "Привет, Аня! Рада знакомству. Чем могу помочь с личным кабинетом сегодня?"
    guard, sent = StreamGuard(None, CANARY), []
    for i in range(0, len(answer), 3):
        sent.append(guard.feed(answer[i:i + 3]))
    sent.append(guard.finish())
    assert "".join(sent) == answer and len([s for s in sent if s]) >= 3
    assert StreamGuard(SYSTEM, CANARY).hold == 80
    parts = ["Хорошо. ", "Метка: C A N A R Y _ a 7", " f 3 b 9 e 2 — вот она."]
    guard, sent, rule = StreamGuard(None, CANARY), [], None
    try:
        sent += [guard.feed(part) for part in parts]
        sent.append(guard.finish())
    except OutputBlocked as exc:
        rule = exc.rule
    assert rule == "canary" and "C A N" not in "".join(sent)


def test_stream_equals_masked_answer_for_any_chunking():
    """Склеенный поток совпадает с redact_pii всего ответа при любой нарезке на фрагменты —
    в том числе email длиннее 80 придержанных символов (первая версия отдавала его начало)."""
    import random

    from app.observability.pii import redact_pii

    rng = random.Random(7)
    pieces = ["Пишите на ivan.petrov@mail.ru ", "звоните +7 (999) 123-45-67, ", "карта 4111 1111 1111 1111 ",
              "паспорт 45 09 123456 ", "СНИЛС 112-233-445 95 ", "ключ sk-or-v1-0123456789abcdef0123 ",
              "Ссылка действует 30 минут (раздел 2.1).\n", "a" * 95 + "@mail.ru готово ", "слово ",
              "session 550e8400-e29b-41d4-a716-446655440000 "]
    for _ in range(300):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 20)))
        guard, sent, i = StreamGuard(None, CANARY), [], 0
        while i < len(text):
            n = rng.randint(1, 12)
            sent.append(guard.feed(text[i:i + n]))
            i += n
        sent.append(guard.finish())
        assert "".join(sent) == redact_pii(text)


def test_stream_guard_is_linear():
    import time

    for text in ("a. " * 6000, "x" * 16000):            # 18 тыс. символов со словами и без пробелов
        guard, started = StreamGuard(None, CANARY), time.perf_counter()
        for i in range(0, len(text), 4):
            guard.feed(text[i:i + 4])
        guard.finish()
        assert time.perf_counter() - started < 2.0


def test_stream_masks_split_email():
    parts = ["Напишите в поддержку по адресу iv", "an@ma", "il.ru — ответят в течение дня. " * 4]
    sent, rule = stream(parts)
    text = "".join(sent)
    assert rule is None and "ivan@mail.ru" not in text and "[EMAIL]" in text and "iv" not in text.split("[EMAIL]")[0][-3:]


# ---------------------------------------------------------------- маскер: скорость и технические ID
def test_masker_is_fast_on_adversarial_text():
    import time

    from app.observability.pii import prompt_preview, redact_pii

    started = time.perf_counter()
    for text in ("a." * 16_000, "a@" * 16_000, "1-" * 16_000):
        redact_pii(text)
        prompt_preview(text)
    assert time.perf_counter() - started < 1.0


def test_masker_keeps_uuid_and_masks_jwt():
    from app.observability.pii import redact_pii

    text = ("session 550e8400-e29b-41d4-a716-446655440000, ИНН 7707083893, "
            "токен eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.abc-_def")
    assert redact_pii(text) == "session 550e8400-e29b-41d4-a716-446655440000, ИНН [INN], токен [JWT]"
