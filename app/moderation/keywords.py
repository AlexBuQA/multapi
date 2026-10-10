"""
Слой 1 модерации (блок 4.4): ключевые слова и регулярные выражения из moderation_keywords.yaml.

Самый дешёвый слой — микросекунды, без сети. Поэтому он же проверяет ответ модели по ходу
потока, на каждом фрагменте (ChatService), а OpenAI Moderation — только ответ целиком.

Текст перед проверкой нормализуется (normalize_for_match): NFKC и без невидимых символов —
та же функция, что у защитного слоя блока 3.8, чтобы «уб​ью» не проходило; затем нижний
регистр, «ё» -> «е» и пробелы, схлопнутые в один. Шаблоны в YAML пишутся под этот вид.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.services.security.input_validator import normalize

DEFAULT_KEYWORDS_FILE = Path(__file__).with_name("moderation_keywords.yaml")
_SPACES = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    return _SPACES.sub(" ", normalize(text).lower().replace("ё", "е")).strip()


@dataclass(frozen=True)
class KeywordMatch:
    category: str
    rule: str          # «violence#2» — категория и номер шаблона в YAML, без самого текста


class KeywordModerator:
    """Шаблоны по категориям. Порядок категорий и шаблонов — как в YAML."""

    def __init__(self, patterns: dict[str, list[re.Pattern[str]]]) -> None:
        self.patterns = patterns

    @classmethod
    def from_file(cls, path: Path | str = DEFAULT_KEYWORDS_FILE) -> KeywordModerator:
        """Файл с ошибкой — ошибка старта сервиса с понятным текстом, а не 500 на запросе."""
        path = Path(path)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"Не удалось прочитать файл модерации {path}: {exc}") from exc
        categories = data.get("categories") if isinstance(data, dict) else None
        if not isinstance(categories, dict) or not categories:
            raise ValueError(f"{path}: ожидается раздел categories: {{категория: [шаблоны]}}")
        patterns: dict[str, list[re.Pattern[str]]] = {}
        for category, items in categories.items():
            if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
                raise ValueError(f"{path}: categories.{category} — список строк-шаблонов")
            try:
                patterns[str(category)] = [re.compile(item) for item in items]
            except re.error as exc:
                raise ValueError(f"{path}: categories.{category}: неверный шаблон — {exc}") from exc
        return cls(patterns)

    def find(self, text: str) -> list[KeywordMatch]:
        """Первое совпадение в каждой категории: категорий в ответе — без повторов."""
        normalized = normalize_for_match(text)
        found: list[KeywordMatch] = []
        for category, patterns in self.patterns.items():
            for number, pattern in enumerate(patterns, start=1):
                if pattern.search(normalized):
                    found.append(KeywordMatch(category, f"{category}#{number}"))
                    break
        return found
