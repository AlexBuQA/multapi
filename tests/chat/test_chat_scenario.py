"""scripts/chat_scenario.py против приложения целиком (JSON-хранилище, FakeLLM) — без сети."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import httpx
import pytest

from app.chat.deps import get_llm_client
from app.core.config import get_settings
from app.main import app
from chat_fakes import FakeLLM, make_settings

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("chat_scenario", ROOT / "scripts" / "chat_scenario.py")
chat_scenario = importlib.util.module_from_spec(_spec)
sys.modules["chat_scenario"] = chat_scenario   # dataclass ищет модуль в sys.modules
_spec.loader.exec_module(chat_scenario)


@pytest.fixture
def use_llm(tmp_path):
    """Подменяет модель приложения; возвращает функцию «llm -> AsyncClient к приложению»."""
    settings = make_settings(tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings

    def client(llm: FakeLLM) -> httpx.AsyncClient:
        app.dependency_overrides[get_llm_client] = lambda: llm
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    yield client
    app.dependency_overrides.clear()


async def test_model_with_memory_passes_every_check(use_llm, capsys):
    async with use_llm(FakeLLM()) as http:
        assert await chat_scenario.run(http, runs=2) == 0
    out = capsys.readouterr().out
    assert "< Вас зовут Аня." in out and "< Вы не называли своего имени." in out
    assert "2/2  второй ответ называет Аню" in out and "2/2  после очистки имени нет" in out
    assert "2/2  ответы без отказа защитного слоя" in out


async def test_model_that_invents_a_name_fails(use_llm, capsys):
    async with use_llm(FakeLLM(reply=lambda req: "Вас зовут Иван.")) as http:
        assert await chat_scenario.run(http, runs=1) == 1
    out = capsys.readouterr().out
    assert "[!]  второй ответ называет Аню" in out
    assert "1/1  после очистки имени нет" in out


async def test_model_that_forgets_fails_recall(use_llm, capsys):
    """Как llama3.2 с промптом «если не знаешь имени — так и скажи»: имя было в истории."""
    async with use_llm(FakeLLM(reply=lambda req: "Я не знаю, как вас зовут.")) as http:
        assert await chat_scenario.run(http, runs=1) == 1
    assert "0/1  второй ответ называет Аню" in capsys.readouterr().out


async def test_multiline_chunks_are_joined_with_newline(use_llm):
    async with use_llm(FakeLLM(reply=lambda req: "Первая строка\n\nВторая, Аня", chunk=100)) as http:
        result = await chat_scenario.run_once(http)
    assert result.dialog[1][1] == "Первая строка\n\nВторая, Аня"


async def test_service_down_is_exit_2(capsys):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse), base_url="http://test") as http:
        assert await chat_scenario.run(http, runs=3) == 2
    assert "запустите uvicorn" in capsys.readouterr().out


async def test_guard_refusal_is_counted(use_llm, capsys):
    """Как в блоке 4.2: llama3.2 начала пересказывать канарейку, StreamGuard заменил ответ отказом."""
    from app.services.guardrails import refusal_text

    def reply(req) -> str:
        asked = [m.content for m in req.messages if m.role == "user"]
        if asked == [chat_scenario.GREETING]:
            return "Привет, Аня!"
        return refusal_text("Личный кабинет") if len(asked) > 1 else "Вы не называли своего имени."

    async with use_llm(FakeLLM(reply=reply)) as http:
        assert await chat_scenario.run(http, runs=1) == 1
    out = capsys.readouterr().out
    assert "[!]  ответы без отказа защитного слоя" in out and "0/1  ответы без отказа защитного слоя" in out
    assert "1/1  после очистки имени нет" in out


def test_refusal_mark_is_the_guard_refusal():
    from app.services.guardrails import REFUSAL_TEMPLATE

    assert chat_scenario.REFUSAL_MARK == "Я не могу показать свои инструкции или действовать в обход них"
    assert REFUSAL_TEMPLATE.startswith(chat_scenario.REFUSAL_MARK)


def test_plural_runs():
    assert [chat_scenario.plural_runs(n) for n in (1, 2, 5, 11, 21, 22)] == [
        "прогон", "прогона", "прогонов", "прогонов", "прогон", "прогона"]


@pytest.mark.parametrize("text, found", [
    ("Вас зовут Аня.", True), ("Привет, Аня!", True), ("Рада помочь Ане", True), ("Аню я помню", True),
    ("Я не знаю вашего имени.", False), ("Анна", False), ("Анялиз", False),
])
def test_name_in_any_case(text, found):
    assert bool(chat_scenario.NAME.search(text)) is found
