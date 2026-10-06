"""
Загрузка промптов из файлов.

- System prompt ассистента — Jinja2-шаблон app/prompts/system_<версия>.j2.
  Версионирование — через имя файла (system_v1.j2 -> system_v2.j2) и git-историю.
- Описания инструментов (поле description у tool) — app/prompts/tools/<tool>.md:
  это тоже промпт, модель читает его на каждом запросе.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import StrictUndefined, Template

PROMPTS_DIR = Path(__file__).parent
TOOL_PROMPTS_DIR = PROMPTS_DIR / "tools"


@lru_cache(maxsize=16)
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_system_prompt(version: str = "v1", **context: object) -> str:
    """Рендерит system prompt нужной версии; незаданная переменная шаблона — ошибка."""
    text = _read_text(PROMPTS_DIR / f"system_{version}.j2")
    return Template(text, undefined=StrictUndefined).render(**context).strip()


def load_tool_description(tool_name: str) -> str:
    """Описание tool из app/prompts/tools/<tool_name>.md (без хвостовых пробелов)."""
    text = _read_text(TOOL_PROMPTS_DIR / f"{tool_name}.md").strip()
    if not text:
        raise ValueError(f"Пустое описание инструмента: {tool_name}")
    return text
