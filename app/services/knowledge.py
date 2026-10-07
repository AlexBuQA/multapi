"""
Поиск по руководству пользователя (data/knowledge_base.json).

Общий для ассистента с инструментами (блок 3.1, инструмент search_knowledge_base) и
HTTP-сервиса (блок 3.7: статьи подставляются в системный промпт /chat). Модуль без
внешних зависимостей — он нужен и в Docker-образе сервиса.

Поиск простой и предсказуемый: грубый стемминг по первым 5 символам слова, совпадения
в заголовке весят 3, в ключевых словах — 2, в тексте — 1. Статьи с весом меньше
MIN_SCORE не возвращаются. Файл читается при каждом вызове: правка руководства сразу
видна без перезапуска.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

STOPWORDS = {
    "и", "в", "во", "на", "с", "со", "к", "ко", "по", "за", "из", "у", "о", "об",
    "а", "но", "или", "не", "ни", "ли", "же", "бы", "то", "это", "как", "что",
    "где", "когда", "почему", "зачем", "мне", "меня", "мой", "моя", "мои", "я",
    "вы", "вас", "ваш", "есть", "для", "от", "до", "при", "так", "уже",
    "можно", "нужно", "подскажите", "пожалуйста", "здравствуйте", "если",
}
MIN_SCORE = 2
TOP_K = 3
_ENDINGS = "аеиоуыэюяйь"


def tokens(text: str) -> list[str]:
    text = text.lower().replace("ё", "е")
    return re.findall(r"\w+", text)


def stem(token: str) -> str:
    # Грубый стемминг для русского: отбрасываем гласные окончания и берём первые
    # 5 символов («пароль», «пароля», «паролем» -> «парол»; «ключ», «ключа» -> «ключ»).
    return (token.rstrip(_ENDINGS) or token)[:5]


def terms(text: str) -> set[str]:
    stems = (stem(t) for t in tokens(text) if t not in STOPWORDS)
    return {s for s in stems if len(s) > 1}


def load_knowledge_base(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_article(query_terms: set[str], article: dict[str, Any]) -> int:
    return (
        3 * len(query_terms & terms(article["title"]))
        + 2 * len(query_terms & terms(" ".join(article.get("keywords", []))))
        + len(query_terms & terms(article["text"]))
    )


def search_articles(
    query: str,
    kb: dict[str, Any],
    *,
    product: str | None = None,
    top_k: int = TOP_K,
    min_score: int = MIN_SCORE,
) -> list[tuple[int, dict[str, Any]]]:
    """Статьи по убыванию веса: [(вес, статья), ...], не больше top_k."""
    query_terms = terms(query)
    scored = []
    for article in kb["articles"]:
        if product and article["product"] != product:
            continue
        score = score_article(query_terms, article)
        if score >= min_score:
            scored.append((score, article))
    scored.sort(key=lambda item: (-item[0], item[1]["id"]))
    return scored[:top_k]
