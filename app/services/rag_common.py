"""
Общее для двух версий RAG (блок 5.3): промпт, ответ «не знаю», порог близости и форма ответа.

Вынесено отдельно, чтобы версия на LlamaIndex (rag.py) и версия без фреймворка
(rag_baremetal.py) отличались только конвейером: модель видит одни и те же инструкции, а
вызывающий код получает один и тот же словарь:

    {"answer": str, "top_score": float, "sources": [{"text": str, "source": str, "score": float}, …]}
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.core.config import Settings

NO_ANSWER = "В базе знаний ответа нет."
SOURCE_CHARS = 300          # сколько символов фрагмента показывать в sources
DEFAULT_QUESTION = "Сколько стоит тариф «Команда» в месяц?"


def system_prompt(product_name: str) -> str:
    """Инструкции модели. Отвечать только по фрагментам, а без ответа во фрагментах — ровно
    фразой NO_ANSWER: по ней видно, что сработал отказ, а не выдумка."""
    return (f"Ты — ассистент техподдержки продукта «{product_name}». Отвечай по-русски, коротко и по делу, "
            "только по фрагментам базы знаний из сообщения пользователя. Если во фрагментах нет ответа на "
            f"вопрос, ответь ровно «{NO_ANSWER}» и ничего не добавляй. Не придумывай цены, сроки, названия и "
            "шаги, которых нет во фрагментах.")


# {context_str} и {query_str} — имена переменных шаблона LlamaIndex; bare-metal подставляет их сам.
USER_TEMPLATE = (
    "Фрагменты базы знаний:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Вопрос: {query_str}\n"
    "Ответ:"
)
# Уточнение ответа, когда фрагменты не влезают в один запрос (режим compact у LlamaIndex).
REFINE_TEMPLATE = (
    "Вопрос: {query_str}\n"
    "Черновик ответа: {existing_answer}\n"
    "Ещё фрагменты базы знаний:\n"
    "---------------------\n"
    "{context_msg}\n"
    "---------------------\n"
    "Уточни черновик по новым фрагментам, если они помогают; иначе повтори черновик без изменений.\n"
    "Ответ:"
)


def source(text: str, file_name: str | None, score: float | None) -> dict[str, Any]:
    return {"text": text[:SOURCE_CHARS], "source": file_name, "score": round(float(score or 0.0), 3)}


def result(answer: str, top_score: float, sources: list[dict[str, Any]]) -> dict[str, Any]:
    return {"answer": answer, "top_score": round(float(top_score), 3), "sources": sources}


def is_no_answer(answer: str) -> bool:
    """Модель (или порог) отказалась отвечать: в ответе — фраза NO_ANSWER."""
    return NO_ANSWER.rstrip(".").lower() in answer.lower()


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
    where = cfg.llm.base_url or "https://api.openai.com/v1"
    if isinstance(exc, openai.APIConnectionError):     # в том числе таймаут
        return RAGLLMError(f"Модель {model} ({where}) не ответила: {type(exc).__name__}. Запущена ли Ollama; для CPU "
                           "увеличьте LLM__REQUEST_TIMEOUT")
    if isinstance(exc, openai.APIStatusError):
        return RAGLLMError(f"Модель {model} ({where}) ответила {exc.status_code}: {str(exc)[:200]}")
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
