"""
Подставные судья и эмбеддинги для тестов оценки RAGAS (блок 5.6): без сети и без модели.

FakeJudge — InstructorBaseRagasLLM: по имени схемы ответа возвращает правдоподобную структуру и
запоминает запросы. Правила простые и предсказуемые: утверждение «поддержано» контекстом, если его
первое слово есть в контексте; фрагмент «полезен», если в нём есть число из эталона; ответ со
ссылкой — если в нём есть «[N]», «Источники», «согласно» или имя файла. FakeEmbeddings — векторы
из триграмм символов.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM

DIM = 64
# «Ссылка на источник» по-простому: [N]/[doc_id], «Источники», «согласно …», имя файла.
CITED = re.compile(r"\[[\w.-]+\]|Источники|[Сс]огласно|\b[\w-]+\.(?:md|html|pdf|docx)\b")


def _field(prompt: str, name: str) -> str:
    """Значение поля из JSON-входа промпта RAGAS: "name": "..."."""
    match = re.search(rf'"{name}":\s*"((?:[^"\\]|\\.)*)"', prompt)
    return match.group(1).encode().decode("unicode_escape") if match else ""


class FakeJudge(InstructorBaseRagasLLM):
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_on = fail_on

    def generate(self, prompt: str, response_model: Any) -> Any:  # pragma: no cover — только async
        raise NotImplementedError

    async def agenerate(self, prompt: str, response_model: Any) -> Any:
        name = response_model.__name__
        self.calls.append((name, prompt))
        if name == self.fail_on:
            raise ValueError(f"судья ответил не по схеме {name}")
        if name == "StatementGeneratorOutput":
            answer = prompt.rsplit('"answer":', 1)[-1]
            sentences = [s.strip(' ."\n}') for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip(' ."\n}')) > 3]
            return response_model(statements=sentences or [])
        if name == "NLIStatementOutput":
            context = prompt.lower()
            statements = re.findall(r'"([^"]{4,})"', prompt.rsplit('"statements":', 1)[-1])
            return response_model(statements=[
                {"statement": s, "reason": "проверено", "verdict": int(context.count(s.split()[0].lower()) > 1)}
                for s in statements])
        if name == "AnswerRelevanceOutput":
            response = prompt.rsplit('"response":', 1)[-1]
            refused = "не нашёл" in response or "не нашел" in response
            return response_model(question="Каков срок возврата оплаты?", noncommittal=int(refused))
        if name == "ContextPrecisionOutput":
            reference = prompt.rsplit('"answer":', 1)[-1]
            context = prompt.rsplit('"context":', 1)[-1].split('"answer":')[0]
            numbers = set(re.findall(r"\d+", reference))
            return response_model(reason="есть число из эталона", verdict=int(bool(numbers & set(re.findall(r"\d+", context)))))
        if name == "ContextRecallOutput":
            return response_model(classifications=[{"statement": "Срок возврата — 14 дней.", "reason": "в контексте",
                                                    "attributed": int("14" in prompt.rsplit('"context":', 1)[-1])}])
        if name == "CitationVerdict":
            answer = prompt.rsplit("<ответ>", 1)[-1].split("</ответ>")[0]
            found = CITED.search(answer)
            return response_model(found=found.group(0) if found else "", has_citation="yes" if found else "no")
        raise AssertionError(f"неизвестная схема {name}")


class FakeEmbeddings(BaseRagasEmbedding):
    def __init__(self) -> None:
        self.texts: list[str] = []

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        low = text.lower()
        for i in range(len(low) - 2):
            vec[int(hashlib.md5(low[i:i + 3].encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        self.texts.append(text)
        return self._vector(text)

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_text(text)

    def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

    async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return self.embed_texts(texts)


# ---------------------------------------------------------------- генератор golden dataset
def _schema_title(prompt: str) -> str:
    """Название схемы ответа из промпта RAGAS: «…as specified in JSON Schema:\\n{…}»."""
    import json

    start = prompt.find("JSON Schema:\n")
    if start < 0:
        return ""
    schema, _ = json.JSONDecoder().raw_decode(prompt[start + len("JSON Schema:\n"):])
    return schema.get("title", "")


def _input(prompt: str) -> dict[str, Any]:
    import json

    tail = prompt.rsplit("\ninput: ", 1)[-1].rsplit("\nOutput:", 1)[0]
    try:
        return json.loads(tail)
    except ValueError:
        return {}


def _words(text: str, n: int) -> list[str]:
    seen: list[str] = []
    for word in re.findall(r"[А-ЯЁA-Z][а-яёa-z]{4,}", text):
        if word not in seen:
            seen.append(word)
    return seen[:n] or ["Личный кабинет"]


def generator_reply(prompt: str) -> str:
    """Ответ «модели» на промпт TestsetGenerator: по названию схемы — правдоподобный JSON."""
    import json

    title, data = _schema_title(prompt), _input(prompt)
    text = data.get("text") or data.get("context") or ""
    if title == "StringIO":
        return json.dumps({"text": " ".join(str(text).split())[:200] or "Краткое содержание."}, ensure_ascii=False)
    if title == "ThemesAndConcepts":
        return json.dumps({"output": _words(str(text), 3)}, ensure_ascii=False)
    if title == "NEROutput":
        return json.dumps({"entities": _words(str(text), 4)}, ensure_ascii=False)
    if title in ("Keyphrases", "Headlines"):
        key = "keyphrases" if title == "Keyphrases" else "headlines"
        return json.dumps({key: _words(str(text), 3)}, ensure_ascii=False)
    if title == "PersonaThemesMapping":
        themes = data.get("themes") or ["Личный кабинет"]
        return json.dumps({"mapping": {p["name"]: themes[:2] for p in data.get("personas", [])}}, ensure_ascii=False)
    if title == "ConceptCombinations":
        lists = data.get("lists_of_concepts") or [["A"], ["B"]]
        pairs = [[a, b] for a in lists[0][:2] for b in (lists[1][:1] if len(lists) > 1 else lists[0][-1:])]
        return json.dumps({"combinations": pairs[:2]}, ensure_ascii=False)
    if title == "GeneratedQueryAnswer":
        term = data.get("term") or ", ".join(data.get("themes") or []) or "сервис"
        context = str(data.get("context") or "")
        sentence = re.split(r"(?<=[.!?])\s", " ".join(context.split()))[0][:200] if context else "Нет данных."
        return json.dumps({"query": f"Что известно про {term}?", "answer": sentence}, ensure_ascii=False)
    raise AssertionError(f"неизвестная схема {title!r}")


class FakeGeneratorJudge(InstructorBaseRagasLLM):
    """Модель генератора golden dataset через llm_factory-интерфейс (agenerate с response_model), как у
    судьи метрик: ответы — generator_reply. plain_text=True — модель отвечает обычным текстом, как
    qwen3:4b-instruct без JSON-режима: каждый вызов падает."""

    is_async = True

    def __init__(self, *, plain_text: bool = False) -> None:
        self.prompts: list[str] = []
        self.plain_text = plain_text

    def generate(self, prompt: str, response_model: Any) -> Any:  # pragma: no cover — только async
        raise NotImplementedError

    async def agenerate(self, prompt: str, response_model: Any) -> Any:
        self.prompts.append(prompt)
        if self.plain_text:
            raise ValueError("Invalid json output: Краткое содержание обычным текстом.")
        return response_model.model_validate_json(generator_reply(prompt))


def make_generator_judge(*, plain_text: bool = False) -> FakeGeneratorJudge:
    return FakeGeneratorJudge(plain_text=plain_text)
