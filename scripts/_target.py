"""
Выбор цели для скриптов блока 3.3 — вызывается ДО импорта src.config,
потому что настройки провайдера читаются из окружения при импорте.

  mock   — локальный мок OpenAI API с фиксированной задержкой (модель облачного провайдера);
  ollama — реальный локальный Ollama из .env (SUPPORT_PRIMARY_MODEL).
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def configure_target(target: str, latency: float = 1.0) -> str:
    """Готовит окружение и возвращает описание цели для отчёта."""
    # Цепочка из одного провайдера: замер не должен смешиваться с fallback.
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["LLM_FALLBACK_ORDER"] = "ollama"
    if target == "mock":
        from scripts.mock_llm_server import start_in_background

        os.environ["OPENAI_BASE_URL"] = start_in_background(latency)
        os.environ["OPENAI_API_KEY"] = "mock"
        os.environ["SUPPORT_PRIMARY_MODEL"] = "mock-model"
        return f"мок OpenAI API, задержка {latency:g} с на ответ"
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT, ".env"))
    return (f"локальный Ollama, модель {os.getenv('SUPPORT_PRIMARY_MODEL', 'llama3.2')}, "
            f"{os.getenv('OPENAI_BASE_URL', 'http://localhost:11434/v1')}")
