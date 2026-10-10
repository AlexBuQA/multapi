"""
Блок 4.4, проверка на Windows: docker compose (env_file) читает строку .env
«MODERATION__THRESHOLDS=   # пусто => …» как значение «# пусто => …». Сервис в контейнере падал
на SettingsError без подсказки — теперь get_settings называет переменные и формат КЛЮЧ="".
"""
from __future__ import annotations

import pytest

from app.core.config import commented_values, get_settings


def test_commented_values():
    env = {"MODERATION__THRESHOLDS": "# пусто => решение OpenAI", "EMPTY": "", "PASSWORD": "#no-space-is-a-value",
           "WITH_COMMENT": "value # docker compose такой комментарий отрезает"}
    assert commented_values(env) == ["MODERATION__THRESHOLDS"]


def test_get_settings_names_the_variable(monkeypatch):
    monkeypatch.setenv("MODERATION__THRESHOLDS", '# пусто => свои пороги JSON: {"violence": 0.5}')
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match=r'MODERATION__THRESHOLDS.*КЛЮЧ=""'):
            get_settings()
    finally:
        get_settings.cache_clear()
