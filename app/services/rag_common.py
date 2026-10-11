"""
Общее для двух версий RAG (блоки 5.3–5.5): промпт с цитатами, ответ «не знаю», источники и форма
ответа.

Вынесено отдельно, чтобы версия на LlamaIndex (rag.py), версия без фреймворка (rag_baremetal.py)
и чат (app/chat/service.py) отличались только конвейером: модель видит одни и те же инструкции, а
вызывающий код получает один и тот же словарь (блок 5.5):

    {"answer": "Вернуть оплату можно в течение 14 дней после списания [1].",
     "top_score": 0.71, "confident": true,
     "sources": [{"id": 1, "file_name": "refunds_v2.docx", "source": "billing/refunds_v2.docx",
                  "category": "billing", "page": null, "score": 0.71, "snippet": "…", "cited": true}, …]}

Цитаты — нумерованный контекст и правило в промпте (как в reference-сервисе курса): фрагменты
уходят модели как «[1] (refunds_v2.docx) текст», модель ставит [1], [2] после утверждений; id
источника — тот же номер. cited — сослалась ли модель на источник в ответе.

Отказ — два слоя: код (score-guard: лучший фрагмент дальше RAG__SCORE_THRESHOLD — модель не
вызывается) и промпт (нет ответа в контексте — ровно NO_ANSWER). В обоих случаях confident=false и
sources пустой: показывать как источник то, по чему ответа нет, — путать пользователя.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.core.config import Settings

NO_ANSWER = "По базе не нашёл, могу эскалировать."
SNIPPET_CHARS = 200         # сколько символов фрагмента показывать в sources
DEFAULT_QUESTION = "Каков срок возврата оплаты?"

RULES = (
    "Отвечай только по фрагментам базы знаний из контекста: они пронумерованы [1], [2] и так далее. "
    "В конце каждого предложения ответа ставь номер фрагмента, на котором оно основано, в квадратных "
    "скобках: [1] или [1][3]. "
    f"Если в контексте нет ответа на вопрос, ответь ровно «{NO_ANSWER}» и ничего не добавляй. "
    "Не придумывай цены, сроки, названия и шаги, которых нет во фрагментах."
)
# Образцов готовых фраз в правилах нет: «Пример формата: «Первое утверждение [1]. Второе утверждение
# [2][3].»» llama3.2 в боте вставила в ответ дословно — «Первое утверждение [1]. Возврат оплаты можно в
# течение 14 дней…» (docs/rag.md, «Подводные камни блока 5.5»). Образец остался только у строки
# «Источники: [1], [3]» — её модель и должна повторить.
# Формат цитат выбран замером на Windows (scripts/citation_experiment.py, docs/rag.md, «Цитаты
# (Windows)»): llama3.2 (3B) с одним правилом RULES ставила [N] в 3 ответах из 5, а со строкой
# «Источники: …» в конце — в 5 из 5 и не перестала отказывать на вопросы без ответа в базе. Пример
# диалога тоже давал 5 из 5, но модель начинала отвечать на вопросы, ответа на которые в базе нет.
# Замер шёл ещё с образцом фраз в RULES; без него вариант sources перемерен — там же.
SOURCES_RULE = ("В конце ответа отдельной строкой напиши «Источники:» и номера фрагментов, на которых "
                "основан ответ, например «Источники: [1], [3]».")
ANSWER_FORMAT = f"последняя строка — «Источники: [номера фрагментов]»; ответа во фрагментах нет — «{NO_ANSWER}»"
# Прежний формат — номер после каждого предложения; остался для сравнения в citation_experiment.py.
INLINE_FORMAT = f"номер фрагмента — в конце каждого предложения, например [1]; ответа во фрагментах нет — «{NO_ANSWER}»"


def system_prompt(product_name: str, *, sources_line: bool = True) -> str:
    """Инструкции модели для /rag/query: только по контексту, с цитатами, без ответа — NO_ANSWER."""
    base = f"Ты — ассистент техподдержки продукта «{product_name}». Отвечай по-русски, коротко и по делу. {RULES}"
    return f"{base} {SOURCES_RULE}" if sources_line else base


def answer_template(answer_format: str) -> str:
    """Сообщение с контекстом и вопросом; последняя строка — напоминание о формате ответа: маленькие
    модели лучше выполняют то, что стоит прямо перед ответом. {context_str} и {query_str} — имена
    переменных шаблонов LlamaIndex; bare-metal подставляет их сам."""
    return ("Контекст — фрагменты базы знаний:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Вопрос: {query_str}\n"
            f"Ответ ({answer_format}):")


USER_TEMPLATE = answer_template(ANSWER_FORMAT)
# Контекст в чате (app/chat/service.py) — отдельным системным сообщением перед историей.
CHAT_CONTEXT = ("Контекст для ответа на последний вопрос — фрагменты базы знаний:\n\n{context_str}\n\n"
                f"В ответе: {ANSWER_FORMAT}")
CHAT_RULES = f"Вопросы о продукте — по базе знаний. {RULES} {SOURCES_RULE}"

_CITATION = re.compile(r"\[(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\]")
_NUMBER = re.compile(r"\d{1,2}")
_SOURCES_LINE = re.compile(r"\s*Источники:\s*(?:" + _CITATION.pattern + r"[\s,;]*)+\.?\s*$")


def where(file_name: str | None, page: int | None) -> str:
    """Подпись фрагмента: «refunds_v2.docx» или «support-regulation_v2.3.pdf, стр. 2»."""
    name = file_name or "без имени"
    return f"{name}, стр. {page}" if page else name


def numbered_context(fragments: Sequence[tuple[str, str | None, int | None]]) -> str:
    """[(текст, файл, страница), …] → «[1] (файл)\\nтекст» через пустую строку."""
    return "\n\n".join(f"[{number}] ({where(file_name, page)})\n{text.strip()}"
                       for number, (text, file_name, page) in enumerate(fragments, start=1))


def cited_numbers(answer: str, count: int) -> list[int]:
    """Номера источников, на которые сослалась модель: [1], [2][3], [1, 3]. Чужие номера (их не было
    в контексте) отбрасываются."""
    found = {int(n) for group in _CITATION.findall(answer) for n in _NUMBER.findall(group)}
    return sorted(n for n in found if 1 <= n <= count)


def strip_citations(text: str) -> str:
    """Ответ без [1], [2] и без строки «Источники: [1], [3]» в конце — для истории чата: в следующем
    ходу у фрагментов другие номера."""
    text = _SOURCES_LINE.sub("", text)
    text = re.sub(r"[ \t]*" + _CITATION.pattern, "", text)
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", text).strip()


def source(number: int, text: str, *, file_name: str | None, page: int | None, score: float | None,
           path: str | None = None, category: str | None = None, cited: bool = False) -> dict[str, Any]:
    """Источник ответа: id — номер цитаты [id] в тексте ответа."""
    return {"id": number, "file_name": file_name, "source": path, "category": category, "page": page,
            "score": round(float(score or 0.0), 3), "snippet": " ".join(text.split())[:SNIPPET_CHARS],
            "cited": cited}


def mark_cited(sources: list[dict[str, Any]], answer: str) -> list[dict[str, Any]]:
    cited = set(cited_numbers(answer, len(sources)))
    return [{**item, "cited": item["id"] in cited} for item in sources]


def result(answer: str, top_score: float, sources: list[dict[str, Any]], *, confident: bool) -> dict[str, Any]:
    return {"answer": answer, "top_score": round(float(top_score), 3), "confident": confident, "sources": sources}


def refusal(top_score: float) -> dict[str, Any]:
    """Ответ score-guard: модель не вызывалась."""
    return result(NO_ANSWER, top_score, [], confident=False)


def is_no_answer(answer: str) -> bool:
    """Модель (или порог) отказалась отвечать: в ответе — фраза NO_ANSWER (ё и е не различаются)."""
    def norm(text: str) -> str:
        return text.lower().replace("ё", "е")

    return norm(NO_ANSWER.split(",")[0]) in norm(answer)


def final_answer(answer: str, top_score: float, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Ответ модели → словарь контракта. Модель сама отказалась по промпту — confident=false и
    без источников, как у score-guard; иначе источники с пометкой cited."""
    answer = answer.strip()
    if not answer or is_no_answer(answer):
        return result(NO_ANSWER, top_score, [], confident=False)
    return result(answer, top_score, mark_cited(sources, answer), confident=True)


# ---------------------------------------------------------------- ошибки (общие для обеих версий)
class RAGError(RuntimeError):
    """RAG не готов или сломан: текст — что случилось и что сделать."""


class RAGUnavailable(RAGError):
    """Нет связи с Qdrant."""


class RAGNotIndexed(RAGError):
    """Коллекции нет или она пуста: корпус ещё не проиндексирован."""


class RAGLLMError(RAGError):
    """Модель ответа не ответила."""


def llm_error(exc: Exception, cfg: Settings) -> Exception:
    """Ошибки модели — понятным текстом; прочие исключения — как есть."""
    import openai

    model = cfg.rag.llm_model or cfg.llm.default_model
    where_ = cfg.llm.base_url or "https://api.openai.com/v1"
    if isinstance(exc, openai.APIConnectionError):     # в том числе таймаут
        return RAGLLMError(f"Модель {model} ({where_}) не ответила: {type(exc).__name__}. Запущена ли Ollama; для CPU "
                           "увеличьте LLM__REQUEST_TIMEOUT")
    if isinstance(exc, openai.APIStatusError):
        return RAGLLMError(f"Модель {model} ({where_}) ответила {exc.status_code}: {str(exc)[:200]}")
    return exc


def qdrant_error(exc: Exception, url: str | None) -> RAGError:
    """Ошибка qdrant-client — понятным текстом."""
    from qdrant_client.http.exceptions import ResponseHandlingException

    from app.services.vector_store import connection_error

    if isinstance(exc, ResponseHandlingException):
        return RAGUnavailable(f"Нет связи с Qdrant ({url}): {connection_error(exc)}. Запущен ли он — "
                              "docker compose up -d qdrant")
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return RAGError(f"Qdrant не принял ключ ({status}): QDRANT_API_KEY в .env")
    return RAGError(f"Qdrant ответил ошибкой: {exc}")
