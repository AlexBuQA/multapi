"""
POST /rag/query — ответ по базе знаний через RAG на LlamaIndex (блок 5.3).

Отдельно от /chat: там обычный ответ модели (блок 3.4) и история чатов (блок 4.1). Индекс
(RAGService) создаёт lifespan один раз на процесс; эндпоинт только задаёт вопрос. RAGService
синхронный (LlamaIndex + qdrant-client + OpenAI SDK), поэтому вызов — в потоке, а одновременных
обращений к модели не больше LLM__MAX_CONCURRENCY, как у /chat.

Вопрос проходит ту же проверку входа, что и /chat (блок 3.8): длина, скрытые символы,
закодированные вставки, шаблоны инъекций.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter

from app.chat.domain import RequestError
from app.deps.providers import LimiterDep, RagDep, SettingsDep
from app.observability.logging import get_logger
from app.schemas.rag import RagAnswer, RagQuery
from app.services import embeddings
from app.services.rag_common import RAGError, RAGLLMError, RAGNotIndexed, RAGUnavailable
from app.services.security.input_validator import validate_input

router = APIRouter(prefix="/rag", tags=["rag"])
log = get_logger()


@router.post(
    "/query",
    response_model=RagAnswer,
    summary="Ответ по базе знаний (RAG)",
    description="Вопрос → эмбеддинг → top-k фрагментов из Qdrant → ответ модели только по ним. Если лучший "
                "фрагмент ниже порога RAG__MIN_SCORE, модель не вызывается: ответ «В базе знаний ответа нет.».",
    responses={400: {"description": "Вопрос не прошёл проверку безопасности"},
               502: {"description": "Модель ответа не ответила"},
               503: {"description": "RAG не настроен, корпус не проиндексирован, нет Qdrant или модели эмбеддингов"}},
)
async def rag_query(body: RagQuery, rag: RagDep, limiter: LimiterDep, settings: SettingsDep) -> RagAnswer:
    question = body.question.strip()
    if not question:
        raise RequestError(422, "empty_question", "Пустой вопрос.")
    if settings.security.enabled:
        verdict = validate_input(question, settings.security.max_input_chars)
        if not verdict.ok:
            log.warning("rag_input_rejected", rule=verdict.rule)
            raise RequestError(400, "input_rejected", "Вопрос не прошёл проверку безопасности. Переформулируйте его.")

    def run() -> dict:
        rag.ensure_ready()
        return rag.answer(question)

    try:
        async with limiter:
            result = await asyncio.to_thread(run)
    except RAGNotIndexed as exc:
        raise RequestError(503, "rag_not_indexed", str(exc)) from exc
    except RAGUnavailable as exc:
        raise RequestError(503, "vector_store_unavailable", str(exc)) from exc
    except RAGLLMError as exc:
        raise RequestError(502, "llm_unavailable", str(exc)) from exc
    except embeddings.EmbeddingError as exc:
        raise RequestError(503, "embeddings_unavailable", str(exc)) from exc
    except RAGError as exc:
        raise RequestError(503, "rag_error", str(exc)) from exc
    log.info("rag_answered", top_score=result["top_score"], sources=[s["source"] for s in result["sources"]],
             answered=result["top_score"] >= settings.rag.min_score)
    return RagAnswer(**result)
