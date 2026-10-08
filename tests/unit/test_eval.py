"""
Eval-блок без моделей: golden dataset, поиск статей, судья, артефакт прогона и пороги.

Медленный прогон с настоящими моделями — eval/run_evaluation.py; здесь проверяется,
что он устроен правильно: судья вызывается с temperature=0 и json_object, рассуждение
идёт раньше оценок, файл прогона — валидный JSON с нужной структурой, а
check_thresholds.py завершается с кодом 1 при нарушенном пороге.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

import check_thresholds
import run_evaluation
from app.core.config import SupportSettings
from app.services.knowledge import load_knowledge_base, search_articles
from conftest import ROOT, fake_completion
from judge import JUDGE_PROMPT_VERSION, JUDGE_SYSTEM_PROMPT, NO_SOURCES, build_judge_messages, judge_answer

GOLDEN_PATH = ROOT / "eval" / "golden_dataset.json"
GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
ITEMS = GOLDEN["items"]
FIELDS = {"id", "question", "expected_answer", "expected_keywords", "category", "difficulty", "source"}


# ---------------------------------------------------------------- golden dataset
def test_golden_dataset_contract():
    # Каждая правка эталонов — новая версия с записью в changelog: прогоны на разных
    # версиях golden между собой не сравнивают.
    assert isinstance(GOLDEN["version"], int) and GOLDEN["version"] >= 1
    assert [c["version"] for c in GOLDEN["changelog"]] == list(range(1, GOLDEN["version"] + 1))
    assert len(ITEMS) >= 20
    ids = [item["id"] for item in ITEMS]
    assert len(set(ids)) == len(ids) and all(re.fullmatch(r"faq_\d{3}", i) for i in ids)
    for item in ITEMS:
        assert FIELDS <= set(item), item["id"]
        assert item["difficulty"] in {"easy", "medium", "hard"}
        assert re.fullmatch(r"synthetic|support_ticket|phoenix_trace|bugfix_\w+", item["source"]), item["id"]
        assert item["question"].strip() and item["expected_answer"].strip()
    assert len(Counter(item["category"] for item in ITEMS)) >= 3
    assert sum(item["difficulty"] == "hard" for item in ITEMS) >= 3


@pytest.mark.parametrize("item", ITEMS, ids=[item["id"] for item in ITEMS])
def test_retrieval_finds_expected_articles(item):
    """Поиск по руководству находит нужные статьи — без этого ответ модели не по теме."""
    kb = load_knowledge_base(SupportSettings().knowledge_base_path)
    found = {article["id"] for _, article in search_articles(item["question"], kb)}
    assert set(item["expected_articles"]) <= found
    if not item["expected_articles"]:
        assert not found


# ---------------------------------------------------------------- проверки без LLM
def test_keyword_recall_and_forbidden_words_normalization():
    answer = "Оплата по счёту зачисляется за 1–3 рабочих дня. Зайдите в «Биллинг»."
    recall, missing = run_evaluation.keyword_report(answer, ["1-3|от одного до трех", "счет", "тариф"])
    assert (recall, missing) == (0.667, ["тариф"])          # тире и ё/е не важны
    assert run_evaluation.forbidden_hits("Цена — 990 ₽ в месяц", ["₽", "руб/мес"]) == ["₽"]
    assert run_evaluation.keyword_report(answer, []) == (None, [])


def test_aggregates():
    def item(i, correctness, **extra):
        return {"id": f"faq_{i:03d}", "category": "support" if i % 2 else "billing",
                "scores": {"relevance": 5, "correctness": correctness, "completeness": 4},
                "keyword_recall": 1.0, "reasoning_first": True, "must_not_contain_hits": [], "error": None, **extra}

    items = [item(1, 5), item(2, 4), item(3, 2, must_not_contain_hits=["₽"]),
             {"id": "faq_004", "category": "api", "scores": None, "error": "chat_502: llm_unavailable"}]
    agg = run_evaluation.aggregate(items)
    assert (agg["correctness_avg"], agg["min_correctness"], agg["relevance_avg"]) == (3.67, 2, 5.0)
    assert (agg["items_total"], agg["items_scored"], agg["errors"], agg["must_not_contain_violations"]) == (4, 3, 1, 1)
    assert agg["correctness_by_category"] == {"billing": 4.0, "support": 3.5}
    assert agg["worst_items"][0] == {"id": "faq_003", "correctness": 2}


# ---------------------------------------------------------------- судья
def test_judge_prompt_requires_reasoning_before_scores():
    instruction = JUDGE_SYSTEM_PROMPT
    assert instruction.index('сначала "reasoning"') < instruction.index('затем "scores"')
    example_line = next(line for line in instruction.splitlines() if line.startswith('{"reasoning"'))
    example = json.loads(example_line)                      # пример в промпте — валидный JSON
    assert list(example) == ["reasoning", "scores", "explanation"]
    assert set(example["scores"]) == {"relevance", "correctness", "completeness"}


def test_judge_sees_manual_excerpts():
    """Факт из статьи руководства сверх эталона — не выдумка: судья видит статьи,
    которые сервис подставил модели (смоук-прогон: «требования раздела 2.2» из KB-001
    стоили ответу correctness 3, пока судья сверял только с эталоном)."""
    article = "[раздел 2.1] Восстановление (сброс) пароля\nНовый пароль должен соответствовать требованиям раздела 2.2."
    user = build_judge_messages(ITEMS[2], "ответ", sources=[article])[1]["content"]
    assert article in user and user.index("Выдержки из руководства") < user.index("Ответ ассистента")
    assert NO_SOURCES in build_judge_messages(ITEMS[2], "ответ")[1]["content"]
    assert "есть в выдержках — верно" in JUDGE_SYSTEM_PROMPT


def test_judge_scale_describes_every_score():
    """Судья geval_v2 ставил только 5, 3 и 1 — в шкале были описаны лишь они.
    Теперь у каждого критерия описан каждый балл от 5 до 1."""
    for name in ("relevance", "correctness", "completeness"):
        line = next(line for line in JUDGE_SYSTEM_PROMPT.splitlines() if line.startswith(f"- {name} —"))
        assert all(f"{score} — " in line for score in range(1, 6)), name
    assert "отказ ответить на вопрос о продукте" in JUDGE_SYSTEM_PROMPT


async def test_judge_called_with_temperature_zero_and_json_mode(mocker):
    reply = '```json\n{"reasoning": "1. ... 2. ... 3. ...", "scores": {"relevance": 5, "correctness": 4, "completeness": 3}, "explanation": "ок"}\n```'
    client = mocker.Mock()
    client.chat.completions.create = mocker.AsyncMock(return_value=fake_completion(reply))
    result = await judge_answer(client, "qwen3:4b-instruct", ITEMS[0], "Ссылка действует {30} минут.")
    assert result.verdict.scores.correctness == 4 and result.reasoning_first and result.attempts == 1
    kwargs = client.chat.completions.create.await_args.kwargs
    assert (kwargs["temperature"], kwargs["response_format"], kwargs["model"]) == (0, {"type": "json_object"}, "qwen3:4b-instruct")
    assert "Ссылка действует {30} минут." in kwargs["messages"][1]["content"]   # скобки в ответе не ломают шаблон


async def test_judge_retries_once_on_broken_json(mocker):
    good = '{"reasoning": "...", "scores": {"relevance": 4, "correctness": 4, "completeness": 4}, "explanation": "ок"}'
    client = mocker.Mock()
    client.chat.completions.create = mocker.AsyncMock(side_effect=[fake_completion("не JSON"), fake_completion(good)])
    result = await judge_answer(client, "judge", ITEMS[0], "ответ")
    assert result.verdict is not None and result.attempts == 2
    client.chat.completions.create = mocker.AsyncMock(return_value=fake_completion('{"reasoning": "...", "scores": {"relevance": 9}}'))
    failed = await judge_answer(client, "judge", ITEMS[0], "ответ")
    assert failed.verdict is None and failed.error.startswith("judge_format")


# ---------------------------------------------------------------- прогон целиком на моках
def test_run_evaluation_writes_valid_run_file(mocker, tmp_path):
    production = mocker.Mock()
    production.chat.completions.create = mocker.AsyncMock(return_value=fake_completion("Ссылка действует 30 минут (раздел 2.1)."))
    production.close = mocker.AsyncMock()
    mocker.patch("app.main.AsyncOpenAI", return_value=production)     # модель под тестом — в lifespan сервиса
    mocker.patch("app.main.setup_tracing", return_value=None)
    verdict = '{"reasoning": "...", "scores": {"relevance": 5, "correctness": 5, "completeness": 4}, "explanation": "верно"}'
    judge_client = mocker.Mock()
    judge_client.chat.completions.create = mocker.AsyncMock(return_value=fake_completion(verdict))
    judge_client.close = mocker.AsyncMock()
    mocker.patch("run_evaluation.make_judge_client", return_value=judge_client)

    out = tmp_path / "runs" / "2026-10-07.json"
    assert run_evaluation.main(["--ids", "faq_001,faq_015,faq_023", "--judge", "judge-model", "--out", str(out)]) == 0

    run = json.loads(out.read_text(encoding="utf-8"))
    assert {"run_id", "timestamp", "model_under_test", "judge_model", "golden_version", "items", "aggregates"} <= set(run)
    assert (run["judge_model"], run["golden_version"], len(run["items"])) == ("judge-model", GOLDEN["version"], 3)
    assert run["judge_prompt_version"] == JUDGE_PROMPT_VERSION
    for item in run["items"]:
        assert {"id", "question", "answer", "scores", "reasoning", "explanation"} <= set(item)
        assert set(item["scores"]) == {"relevance", "correctness", "completeness"}
    assert {"relevance_avg", "correctness_avg", "completeness_avg", "min_correctness"} <= set(run["aggregates"])
    assert run["aggregates"]["correctness_avg"] == 5.0
    sent = production.chat.completions.create.await_args_list
    # faq_023 — просьба показать системный промпт: отвечает проверка, модель не вызывается.
    assert len(sent) == 2 and all(call.kwargs["temperature"] == 0 for call in sent)
    by_id = {item["id"]: item for item in run["items"]}
    assert (by_id["faq_023"]["response_model"], by_id["faq_023"]["finish_reason"]) == ("guardrail", "content_filter")
    assert by_id["faq_023"]["answer"].startswith("Я не могу показать свои инструкции")
    assert by_id["faq_001"]["response_model"] == "test-model"

    # Судья получил те же статьи, что модель в системном промпте.
    assert "KB-001" in by_id["faq_001"]["source_articles"] and by_id["faq_023"]["source_articles"] == []
    judged = [call.kwargs["messages"][1]["content"] for call in judge_client.chat.completions.create.await_args_list]
    assert "[раздел 2.1] Восстановление (сброс) пароля" in judged[0]
    assert "[раздел 2.1] Восстановление (сброс) пароля" in sent[0].kwargs["messages"][1]["content"]  # [0] — канарейка
    assert NO_SOURCES in judged[2]


# ---------------------------------------------------------------- пороги
def write_run(runs: Path, name: str, timestamp: str, **aggregates) -> Path:
    runs.mkdir(parents=True, exist_ok=True)
    base = {"correctness_avg": 4.3, "min_correctness": 3, "must_not_contain_violations": 0, "errors": 0}
    path = runs / name
    path.write_text(json.dumps({"run_id": name, "timestamp": timestamp, "items": [],
                                "aggregates": {**base, **aggregates}}), encoding="utf-8")
    return path


def test_check_thresholds_pass_and_fail(tmp_path, capsys):
    runs = tmp_path / "runs"
    write_run(runs, "2026-10-06.json", "2026-10-06T10:00:00Z", correctness_avg=4.6)
    write_run(runs, "2026-10-07.json", "2026-10-07T10:00:00Z")
    thresholds = ROOT / "eval" / "thresholds.yaml"
    assert check_thresholds.main(["--runs-dir", str(runs), "--thresholds", str(thresholds)]) == 0
    assert "2026-10-07.json" in capsys.readouterr().out          # берётся последний прогон

    write_run(runs, "2026-10-08.json", "2026-10-08T10:00:00Z", correctness_avg=3.4, min_correctness=1, errors=2)
    assert check_thresholds.main(["--runs-dir", str(runs), "--thresholds", str(thresholds)]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] correctness_avg = 3.4, порог >= 4.0" in out
    assert "[FAIL] min_correctness = 1, порог >= 2.0" in out
    assert "[FAIL] errors = 2, порог <= 0" in out
    assert "Нельзя релизить: нарушено порогов — 3." in out


def test_check_thresholds_without_runs(tmp_path, capsys):
    assert check_thresholds.main(["--runs-dir", str(tmp_path)]) == 2
    assert "нет прогонов" in capsys.readouterr().out
