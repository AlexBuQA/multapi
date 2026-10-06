"""
Настройки блока 3 (Function Calling) через pydantic-settings.

Значения читаются из переменных окружения и файла .env в корне проекта.
Подключение к LLM (провайдер, base_url, ключи, retry/fallback) по-прежнему
берётся из src/config.py — здесь только то, что относится к ассистенту с tools.
Внешние API с ключами этим инструментам не нужны: оба читают локальные файлы.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import settings as llm_settings

ROOT = Path(__file__).resolve().parent.parent


class ToolSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Модель для Function Calling (должна поддерживать tools). Пусто -> SUPPORT_PRIMARY_MODEL.
    support_tools_model: str = ""
    # Подставляется в system prompt (app/prompts/system_<версия>.j2).
    support_product_name: str = "Личный кабинет"
    system_prompt_version: str = "v1"
    answer_max_sentences: int = 3

    # Цикл tool_call: максимум раундов с вызовом инструментов, затем — финальный ответ.
    tools_max_rounds: int = 3
    tools_temperature: float = 0.0
    tools_max_tokens: int = 400

    # Данные, с которыми работают инструменты, и JSON-лог прогонов.
    knowledge_base_path: Path = ROOT / "data" / "knowledge_base.json"
    service_status_path: Path = ROOT / "data" / "service_status.json"
    tool_log_path: Path = ROOT / "logs" / "tool_calls.jsonl"

    @property
    def tools_model(self) -> str:
        return self.support_tools_model or llm_settings.primary().chat_model


tool_settings = ToolSettings()
