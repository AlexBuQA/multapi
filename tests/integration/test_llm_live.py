"""
Живая проверка с настоящей моделью (маркер llm, блок 3.7).

По умолчанию не запускается: в pyproject.toml addopts = "-m 'not llm'". Запуск:

    pytest -m llm -v

Адрес и модель берутся из .env (LLM__BASE_URL, LLM__DEFAULT_MODEL). Если модель
недоступна (Ollama не запущен), тест пропускается. Одного вопроса достаточно, чтобы
убедиться, что сервис с настоящей моделью отвечает по руководству; качество по всему
golden dataset — дело eval/run_evaluation.py.
"""
from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.llm

ENV = dotenv_values(ROOT / ".env")
BASE_URL = ENV.get("LLM__BASE_URL") or "http://localhost:11434/v1"
MODEL = ENV.get("LLM__DEFAULT_MODEL") or "llama3.2"


def reachable(url: str) -> bool:
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=1):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not reachable(BASE_URL), reason=f"модель недоступна: {BASE_URL}")
async def test_service_answers_from_manual_with_real_model():
    from openai import AsyncOpenAI

    from app.core.config import Settings
    from app.schemas.chat import ChatRequest
    from app.services.llm import LLMService

    settings = Settings(llm={"openai_api_key": ENV.get("LLM__OPENAI_API_KEY") or "ollama", "base_url": BASE_URL,
                             "default_model": MODEL, "request_timeout": 300}, _env_file=None)
    client = AsyncOpenAI(api_key=settings.llm.openai_api_key.get_secret_value(), base_url=BASE_URL, timeout=300)
    try:
        service = LLMService(client, None, settings, limiter=asyncio.Semaphore(1))
        response = await service.complete(ChatRequest(
            messages=[{"role": "user", "content": "Сколько времени действует ссылка для сброса пароля?"}],
            temperature=0, max_tokens=200))
    finally:
        await client.close()
    assert "30" in response.content, response.content          # срок из раздела 2.1 руководства
