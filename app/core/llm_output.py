"""
Разбор структурированного ответа модели (блок 3.7).

parse_json_object() достаёт JSON-объект из текста ответа. Даже с
response_format={"type": "json_object"} небольшие модели иногда заворачивают JSON в
```json ... ``` или добавляют фразу до или после. Поэтому по порядку:
1) содержимое первого блока ```...```, если он есть;
2) весь текст;
3) фрагмент от первой «{» до последней «}».
Битый JSON, не объект (список, число) и пустой ответ — ValueError
(json.JSONDecodeError — его подкласс): вызывающий код решает, повторить запрос или
записать ошибку. Порядок ключей сохраняется — по нему eval проверяет, что судья
написал рассуждение раньше оценок.
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def parse_json_object(text: str | None) -> dict[str, Any]:
    if text is None or not text.strip():
        raise ValueError("пустой ответ модели: JSON-объекта нет")
    candidate = text.strip()
    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise
        value = json.loads(candidate[start:end + 1])   # не разобралось — JSONDecodeError
    if not isinstance(value, dict):
        raise ValueError(f"ожидался JSON-объект, получен {type(value).__name__}")
    return value
