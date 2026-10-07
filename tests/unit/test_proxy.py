"""
Прокси до внешнего провайдера, судья из .env и судья на OpenRouter (блок 3.7).

Прокси учебной группы нужен, чтобы ходить к api.openai.com. Он применяется только к
внешним адресам: запросы к локальной Ollama на чужой сервер уходить не должны. Логин и
пароль из адреса прокси в лог не попадают.
"""
from __future__ import annotations

import json
import os

import httpx
import pytest

import run_evaluation
from app.core.config import get_settings, is_local_url, proxy_display, proxy_for
from app.main import app
from conftest import fake_completion
from judge import token_limit_param
from log_capture import captured_logs, events

PROXY = "http://student:secret-pass@proxy.example:8888"


@pytest.fixture
def fresh_settings(monkeypatch):
    """Настройки перечитываются из окружения; после теста — прежние значения и кеш."""
    for name in ("LLM__BASE_URL", "LLM__OPENAI_API_KEY", "LLM__PROXY_URL", "LLM__USE_SYSTEM_CERTS"):
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])     # вернётся после теста, даже если код его поменяет
        else:
            monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


# ---------------------------------------------------------------- какие адреса идут через прокси
@pytest.mark.parametrize(("base_url", "local"), [
    ("http://localhost:11434/v1", True),
    ("http://127.0.0.1:11434/v1", True),
    ("http://[::1]:11434/v1", True),
    ("http://host.docker.internal:11434/v1", True),
    ("http://ollama:11434/v1", True),                 # имя сервиса в compose
    ("http://192.168.1.10:11434/v1", True),
    ("https://api.openai.com/v1", False),
    ("https://openrouter.ai/api/v1", False),
    (None, False),                                     # не задан — api.openai.com
])
def test_proxy_only_for_external_hosts(base_url, local):
    assert is_local_url(base_url) is local
    assert proxy_for(base_url, PROXY) == (None if local else PROXY)
    assert proxy_for(base_url, None) is None


def test_proxy_display_hides_credentials():
    assert proxy_display(PROXY) == "http://proxy.example:8888"


# ---------------------------------------------------------------- клиент сервиса
async def run_lifespan(mocker):
    fake = mocker.Mock()
    fake.close = mocker.AsyncMock()
    openai_cls = mocker.patch("app.main.AsyncOpenAI", return_value=fake)
    http_cls = mocker.patch("app.main.DefaultAsyncHttpxClient", return_value="proxied-http-client")
    mocker.patch("app.main.setup_tracing", return_value=None)
    with captured_logs("INFO") as logs:
        async with app.router.lifespan_context(app):
            pass
    return openai_cls, http_cls, logs


async def test_service_uses_proxy_for_openai(mocker, fresh_settings):
    fresh_settings.setenv("LLM__BASE_URL", "https://api.openai.com/v1")
    fresh_settings.setenv("LLM__PROXY_URL", PROXY)
    openai_cls, http_cls, logs = await run_lifespan(mocker)
    http_cls.assert_called_once_with(proxy=PROXY)
    assert openai_cls.call_args.kwargs["http_client"] == "proxied-http-client"
    line = events(logs, "llm_proxy_enabled")[0]
    assert line["proxy"] == "http://proxy.example:8888"
    assert "secret-pass" not in json.dumps(logs, ensure_ascii=False, default=str)


async def test_service_uses_system_certs_for_openrouter(mocker, fresh_settings):
    import ssl

    fresh_settings.setenv("LLM__BASE_URL", "https://openrouter.ai/api/v1")
    fresh_settings.setenv("LLM__USE_SYSTEM_CERTS", "true")
    openai_cls, http_cls, logs = await run_lifespan(mocker)
    assert isinstance(http_cls.call_args.kwargs["verify"], ssl.SSLContext)
    assert openai_cls.call_args.kwargs["default_headers"]["X-Title"] == "multapi"


async def test_service_calls_local_ollama_directly(mocker, fresh_settings):
    fresh_settings.setenv("LLM__BASE_URL", "http://localhost:11434/v1")
    fresh_settings.setenv("LLM__PROXY_URL", PROXY)
    openai_cls, http_cls, logs = await run_lifespan(mocker)
    http_cls.assert_not_called()
    assert openai_cls.call_args.kwargs["http_client"] is None
    assert not events(logs, "llm_proxy_enabled")


# ---------------------------------------------------------------- судья и eval
def test_env_value_reads_environment_then_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("EVAL_JUDGE_MODEL=gpt-5.2   # судья\nEVAL_JUDGE_BASE_URL=\n", encoding="utf-8")
    monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("EVAL_JUDGE_BASE_URL", raising=False)
    assert run_evaluation.env_value("EVAL_JUDGE_MODEL", env_file) == "gpt-5.2"
    assert run_evaluation.env_value("EVAL_JUDGE_BASE_URL", env_file) is None      # пусто — как нет
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "gpt-4.1")
    assert run_evaluation.env_value("EVAL_JUDGE_MODEL", env_file) == "gpt-4.1"   # окружение важнее .env


@pytest.mark.parametrize(("model", "param"), [
    ("gpt-5.2", "max_completion_tokens"),
    ("gpt-5-mini", "max_completion_tokens"),
    ("openai/gpt-5.2", "max_completion_tokens"),
    ("o4-mini", "max_completion_tokens"),
    ("gpt-4.1", "max_tokens"),
    ("qwen3:4b-instruct", "max_tokens"),
])
def test_judge_token_limit_parameter(model, param):
    assert token_limit_param(model) == param


def test_judge_client_gets_proxy(mocker):
    http_cls = mocker.patch("run_evaluation.DefaultAsyncHttpxClient", return_value=httpx.AsyncClient())
    client = run_evaluation.make_judge_client("https://api.openai.com/v1", "sk-test", 60, PROXY)
    http_cls.assert_called_once_with(proxy=PROXY)
    assert client.base_url.host == "api.openai.com"
    run_evaluation.make_judge_client("http://localhost:11434/v1", "ollama", 60, None)
    http_cls.assert_called_once()                       # без прокси — клиент SDK по умолчанию


def test_external_model_and_judge_through_proxy(mocker, fresh_settings, tmp_path):
    """--model-base-url и судья OpenAI: обоим клиентам — прокси и ключ EVAL_JUDGE_API_KEY."""
    fresh_settings.setenv("LLM__PROXY_URL", PROXY)
    fresh_settings.setenv("EVAL_JUDGE_API_KEY", "sk-test")
    production = mocker.Mock()
    production.chat.completions.create = mocker.AsyncMock(return_value=fake_completion("Ссылка действует 30 минут."))
    production.close = mocker.AsyncMock()
    openai_cls = mocker.patch("app.main.AsyncOpenAI", return_value=production)
    http_cls = mocker.patch("app.main.DefaultAsyncHttpxClient", return_value="proxied-http-client")
    mocker.patch("app.main.setup_tracing", return_value=None)
    verdict = '{"reasoning": "...", "scores": {"relevance": 5, "correctness": 5, "completeness": 5}, "explanation": "ок"}'
    judge = mocker.Mock()
    judge.chat.completions.create = mocker.AsyncMock(return_value=fake_completion(verdict))
    judge.close = mocker.AsyncMock()
    make_judge = mocker.patch("run_evaluation.make_judge_client", return_value=judge)

    out = tmp_path / "run.json"
    assert run_evaluation.main(["--ids", "faq_001", "--model", "gpt-4.1-mini",
                                "--model-base-url", "https://api.openai.com/v1",
                                "--judge", "gpt-5.2", "--judge-base-url", "https://api.openai.com/v1",
                                "--out", str(out)]) == 0

    kwargs = openai_cls.call_args.kwargs
    assert (kwargs["base_url"], kwargs["api_key"], kwargs["http_client"]) == (
        "https://api.openai.com/v1", "sk-test", "proxied-http-client")
    http_cls.assert_called_once_with(proxy=PROXY)
    assert make_judge.call_args.args == ("https://api.openai.com/v1", "sk-test", 600.0, PROXY)
    assert "max_completion_tokens" in judge.chat.completions.create.await_args.kwargs     # gpt-5.2
    run = json.loads(out.read_text(encoding="utf-8"))
    assert (run["model_under_test"], run["judge_model"]) == ("gpt-4.1-mini", "gpt-5.2")


# ---------------------------------------------------------------- судья на OpenRouter
async def test_judge_reasoning_effort_sent_only_when_set(mocker):
    from judge import judge_answer

    verdict = '{"reasoning": "...", "scores": {"relevance": 5, "correctness": 5, "completeness": 5}, "explanation": "ок"}'
    client = mocker.Mock()
    client.chat.completions.create = mocker.AsyncMock(return_value=fake_completion(verdict))
    item = {"question": "q", "expected_answer": "a", "expected_keywords": []}
    await judge_answer(client, "google/gemma-4-31b-it:free", item, "ответ", reasoning="none")
    assert client.chat.completions.create.await_args.kwargs["extra_body"] == {"reasoning": {"effort": "none"}}
    await judge_answer(client, "qwen3:4b-instruct", item, "ответ")
    assert "extra_body" not in client.chat.completions.create.await_args.kwargs


async def test_judge_empty_answer_at_token_limit_is_explained(mocker):
    """У OpenRouter max_tokens общий для скрытых рассуждений и ответа: модель может
    потратить его целиком и вернуть пустой content с finish_reason=length."""
    from judge import judge_answer

    empty = fake_completion("")
    empty.choices[0].finish_reason = "length"
    client = mocker.Mock()
    client.chat.completions.create = mocker.AsyncMock(return_value=empty)
    result = await judge_answer(client, "google/gemma-4-31b-it:free",
                                {"question": "q", "expected_answer": "a"}, "ответ")
    assert result.verdict is None and result.error.startswith("judge_length")
    assert "EVAL_JUDGE_REASONING=none" in result.error
    client.chat.completions.create.assert_awaited_once()          # повтор не поможет — не тратим лимит


@pytest.mark.parametrize(("base_url", "sent"), [
    # require_parameters: только провайдеры, которые выполнят temperature=0 и json_object.
    ("https://openrouter.ai/api/v1", {"reasoning": {"effort": "none"}, "provider": {"require_parameters": True}}),
    ("https://api.openai.com/v1", None),                          # OpenAI ответил бы 400
])
def test_reasoning_setting_only_for_openrouter(mocker, fresh_settings, tmp_path, base_url, sent):
    fresh_settings.setenv("EVAL_JUDGE_REASONING", "none")
    fresh_settings.setenv("EVAL_JUDGE_API_KEY", "sk-or-test")
    production = mocker.Mock()
    production.chat.completions.create = mocker.AsyncMock(return_value=fake_completion("Ссылка действует 30 минут."))
    production.close = mocker.AsyncMock()
    mocker.patch("app.main.AsyncOpenAI", return_value=production)
    mocker.patch("app.main.setup_tracing", return_value=None)
    verdict = '{"reasoning": "...", "scores": {"relevance": 5, "correctness": 5, "completeness": 5}, "explanation": "ок"}'
    judge = mocker.Mock()
    judge.chat.completions.create = mocker.AsyncMock(return_value=fake_completion(verdict))
    judge.close = mocker.AsyncMock()
    mocker.patch("run_evaluation.make_judge_client", return_value=judge)

    out = tmp_path / "run.json"
    assert run_evaluation.main(["--ids", "faq_001", "--judge", "google/gemma-4-31b-it:free",
                                "--judge-base-url", base_url, "--out", str(out)]) == 0
    assert judge.chat.completions.create.await_args.kwargs.get("extra_body") == sent
    run = json.loads(out.read_text(encoding="utf-8"))
    assert run["params"]["judge_reasoning_effort"] == (sent and "none")
    assert run["params"]["judge_provider_require_parameters"] is bool(sent)


def test_openrouter_attribution_headers(mocker, fresh_settings):
    """Заголовки атрибуции — только для OpenRouter: в статистике учебного ключа видно приложение."""
    from app.core.config import provider_headers

    assert provider_headers("https://openrouter.ai/api/v1", "multapi eval") == {
        "HTTP-Referer": "https://github.com/AlexBuQA/multapi", "X-Title": "multapi eval",
        "X-OpenRouter-Title": "multapi eval"}
    assert provider_headers("https://api.openai.com/v1", "x") is None
    assert provider_headers("http://localhost:11434/v1", "x") is None
    judge = run_evaluation.make_judge_client("https://openrouter.ai/api/v1", "sk-or-test", 60)
    assert judge.default_headers["X-Title"] == "multapi eval"
    assert "X-Title" not in run_evaluation.make_judge_client("http://localhost:11434/v1", "ollama", 60).default_headers


def test_judge_reasoning_off_overrides_env(mocker, fresh_settings, tmp_path):
    """--judge-reasoning off: судья с обязательными рассуждениями (nemotron) не получает
    effort=none из .env (пустую строку PowerShell 5.1 в программу не передаёт)."""
    fresh_settings.setenv("EVAL_JUDGE_REASONING", "none")
    fresh_settings.setenv("EVAL_JUDGE_API_KEY", "sk-or-test")
    production = mocker.Mock()
    production.chat.completions.create = mocker.AsyncMock(return_value=fake_completion("Ссылка действует 30 минут."))
    production.close = mocker.AsyncMock()
    mocker.patch("app.main.AsyncOpenAI", return_value=production)
    mocker.patch("app.main.setup_tracing", return_value=None)
    verdict = '{"reasoning": "...", "scores": {"relevance": 5, "correctness": 5, "completeness": 5}, "explanation": "ок"}'
    judge = mocker.Mock()
    judge.chat.completions.create = mocker.AsyncMock(return_value=fake_completion(verdict))
    judge.close = mocker.AsyncMock()
    mocker.patch("run_evaluation.make_judge_client", return_value=judge)
    out = tmp_path / "run.json"
    assert run_evaluation.main(["--ids", "faq_001", "--judge", "nvidia/nemotron-3-super-120b-a12b:free",
                                "--judge-base-url", "https://openrouter.ai/api/v1", "--judge-reasoning", "off",
                                "--judge-max-tokens", "4000", "--out", str(out)]) == 0
    kwargs = judge.chat.completions.create.await_args.kwargs
    assert kwargs["extra_body"] == {"provider": {"require_parameters": True}} and kwargs["max_tokens"] == 4000


def test_judge_max_tokens_from_flag_env_or_default(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    monkeypatch.delenv("EVAL_JUDGE_MAX_TOKENS", raising=False)
    env_file.write_text("EVAL_JUDGE_MAX_TOKENS=4000   # nemotron\n", encoding="utf-8")
    assert run_evaluation.judge_max_tokens(None, env_file) == 4000
    assert run_evaluation.judge_max_tokens(800, env_file) == 800                 # флаг важнее .env
    env_file.write_text("EVAL_JUDGE_MAX_TOKENS=\n", encoding="utf-8")
    assert run_evaluation.judge_max_tokens(None, env_file) == 1200               # пусто — по умолчанию
    env_file.write_text("EVAL_JUDGE_MAX_TOKENS=4k\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="целое число"):
        run_evaluation.judge_max_tokens(None, env_file)
    with pytest.raises(SystemExit, match="больше нуля"):
        run_evaluation.judge_max_tokens(0, env_file)


def test_env_example_judge_setup_matches_run_4(mocker, fresh_settings, tmp_path):
    """Учебная связка из .env.example (плюс ключ из .env) вызывает судью так же, как флаги
    полного прогона 4: nemotron на OpenRouter, без reasoning.effort, max_tokens 4000,
    provider.require_parameters."""
    root = run_evaluation.ROOT
    env_file = tmp_path / ".env"
    env_file.write_text((root / ".env.example").read_text(encoding="utf-8")
                        + "\nEVAL_JUDGE_API_KEY=sk-or-test\n", encoding="utf-8")
    for name in ("EVAL_JUDGE_MODEL", "EVAL_JUDGE_BASE_URL", "EVAL_JUDGE_API_KEY",
                 "EVAL_JUDGE_REASONING", "EVAL_JUDGE_MAX_TOKENS"):
        fresh_settings.delenv(name, raising=False)
    fresh_settings.setattr(run_evaluation, "ENV_FILE", env_file)     # .env разработчика не читается
    production = mocker.Mock()
    production.chat.completions.create = mocker.AsyncMock(return_value=fake_completion("Ссылка действует 30 минут."))
    production.close = mocker.AsyncMock()
    mocker.patch("app.main.AsyncOpenAI", return_value=production)
    mocker.patch("app.main.setup_tracing", return_value=None)
    verdict = '{"reasoning": "...", "scores": {"relevance": 5, "correctness": 5, "completeness": 5}, "explanation": "ок"}'
    judge = mocker.Mock()
    judge.chat.completions.create = mocker.AsyncMock(return_value=fake_completion(verdict))
    judge.close = mocker.AsyncMock()
    make_judge = mocker.patch("run_evaluation.make_judge_client", return_value=judge)

    out = tmp_path / "run.json"
    assert run_evaluation.main(["--ids", "faq_001", "--out", str(out)]) == 0
    assert make_judge.call_args.args[:2] == ("https://openrouter.ai/api/v1", "sk-or-test")
    kwargs = judge.chat.completions.create.await_args.kwargs
    assert kwargs["model"] == "nvidia/nemotron-3-super-120b-a12b:free"
    assert kwargs["max_tokens"] == 4000
    assert kwargs["extra_body"] == {"provider": {"require_parameters": True}}
    params = json.loads(out.read_text(encoding="utf-8"))["params"]
    assert (params["judge_max_tokens"], params["judge_reasoning_effort"]) == (4000, None)
