"""Pydantic-схемы запроса: границы сообщений и маскирование PII в repr."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest, Message

PII = "Мой email ivan@mail.ru, тел +7 (999) 123-45-67, карта 4111 1111 1111 1111"


@pytest.mark.parametrize("content", ["", "x" * 32_001], ids=["empty", "too-long"])
def test_message_length_bounds(content):
    with pytest.raises(ValidationError) as error:
        Message(role="user", content=content)
    assert error.value.errors()[0]["loc"] == ("content",)


def test_message_at_limit_is_valid():
    assert len(Message(role="user", content="x" * 32_000).content) == 32_000


@pytest.mark.parametrize("messages", [
    [],
    [{"role": "user", "content": "hi"}] * 51,
    [{"role": "assistant", "content": "Здравствуйте!"}],
    [{"role": "robot", "content": "hi"}],
], ids=["no-messages", "too-many", "starts-with-assistant", "unknown-role"])
def test_invalid_dialogs_rejected(messages):
    with pytest.raises(ValidationError):
        ChatRequest(messages=messages)


def test_repr_and_str_mask_pii_but_payload_keeps_it():
    req = ChatRequest(messages=[{"role": "user", "content": PII}], user_id="u-42")
    for text in (repr(req), str(req), repr(req.messages[0]), str(req.messages[0])):
        for fragment in ("ivan@mail.ru", "123-45-67", "4111"):
            assert fragment not in text
        assert "[EMAIL]" in text and "[PHONE_RU]" in text and "[CARD]" in text
    assert req.model_dump()["messages"][0]["content"] == PII   # модели уходит исходный текст


def test_fstring_and_logging_of_request_are_masked():
    req = ChatRequest(messages=[{"role": "user", "content": PII}])
    assert "4111" not in f"{req!r} {req}"          # так модель попадает в лог и трейсбек
    assert "user_id=None" in repr(req)              # остальные поля видны как обычно
