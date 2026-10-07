"""Разбор ответов модели: JSON (в ```-блоке и без), битый JSON, tool_calls, вердикт судьи."""
from __future__ import annotations

import json

import pytest

from app.core.llm_output import parse_json_object
from app.llm.client import tool_calls_from_text
from app.tools.handlers import parse_arguments
from judge import parse_verdict

VERDICT = {"reasoning": "1. ... 2. ... 3. ...", "scores": {"relevance": 5, "correctness": 4, "completeness": 3},
           "explanation": "Верно, но неполно."}


@pytest.mark.parametrize("text", [
    json.dumps(VERDICT, ensure_ascii=False),
    "```json\n" + json.dumps(VERDICT, ensure_ascii=False) + "\n```",
    "Вот оценка:\n```\n" + json.dumps(VERDICT, ensure_ascii=False) + "\n```\nГотово.",
    "Оценка: " + json.dumps(VERDICT, ensure_ascii=False) + " — конец.",
], ids=["plain", "json-fence", "fence-with-prose", "prose-without-fence"])
def test_json_object_with_and_without_markdown_fence(text):
    data = parse_json_object(text)
    assert data == VERDICT
    assert list(data)[0] == "reasoning"       # порядок ключей сохраняется


@pytest.mark.parametrize("text", ['{"reasoning": "не закрыто', "```json\n{oops}\n```", "[1, 2, 3]", "", None],
                         ids=["truncated", "broken-in-fence", "array", "empty", "none"])
def test_malformed_json_raises_value_error(text):
    with pytest.raises(ValueError):
        parse_json_object(text)


def test_tool_calls_from_text_in_fence():
    text = ('```json\n[{"name": "search_knowledge_base", "parameters": {"query": "сброс пароля"}},'
            ' {"name": "check_service_status", "arguments": {"component": "email"}}]\n```')
    calls = tool_calls_from_text(text, step=2)
    assert [c.function.name for c in calls] == ["search_knowledge_base", "check_service_status"]
    assert [json.loads(c.function.arguments) for c in calls] == [{"query": "сброс пароля"}, {"component": "email"}]
    assert calls[0].id == "call_text_2_0"


def test_tool_calls_broken_json_and_unknown_tool():
    broken = tool_calls_from_text('{"name": "search_knowledge_base", "parameters": {"query": ', step=1)
    assert broken[0].function.name == "search_knowledge_base"     # исходный текст уйдёт модели как ошибка
    assert tool_calls_from_text('{"name": "rm_rf", "parameters": {}}', step=1) == []
    assert tool_calls_from_text("Просто текст ответа.", step=1) == []


@pytest.mark.parametrize("raw", ['{"query": ', "[1, 2]", '"строка"'])
def test_tool_arguments_must_be_json_object(raw):
    with pytest.raises(ValueError):
        parse_arguments(raw)


def test_judge_verdict_reasoning_first_and_flat_scores():
    verdict, reasoning_first = parse_verdict(json.dumps(VERDICT, ensure_ascii=False))
    assert reasoning_first and verdict.scores.correctness == 4
    flat = {"relevance": 5, "reasoning": "...", "correctness": "4", "completeness": 3}
    verdict, reasoning_first = parse_verdict(json.dumps(flat))
    assert not reasoning_first and verdict.scores.correctness == 4   # "4" -> 4


@pytest.mark.parametrize("scores", [{"relevance": 6, "correctness": 4, "completeness": 3},
                                    {"relevance": 5, "correctness": 0, "completeness": 3},
                                    {"relevance": 5, "correctness": 4.5, "completeness": 3},
                                    {"relevance": 5, "correctness": 4}])
def test_judge_verdict_out_of_scale_raises(scores):
    with pytest.raises(ValueError):
        parse_verdict(json.dumps({"reasoning": "...", "scores": scores}))
