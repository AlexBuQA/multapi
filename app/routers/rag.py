"""
POST /rag/query — ответ по базе знаний через RAG на LlamaIndex (блоки 5.3–5.5): синхронно, с цитатами
[1], [2], источниками {id, file_name, page, score, snippet} и флагом confident.

Отдельно от /chat: там обычный ответ модели (блок 3.4) и история чатов (блок 4.1). Индекс
(RAGService) создаёт lifespan один раз на процесс; эндпоинт только задаёт вопрос. RAGService
синхронный (LlamaIndex + qdrant-client + OpenAI SDK), поэтому вызов — в потоке, а одновременных
обращений к модели не больше LLM__MAX_CONCURRENCY, как у /chat.

Вопрос проходит ту же проверку входа, что и /chat (блок 3.8): длина, скрытые символы,
закодированные вставки, шаблоны инъекций.

Блок 5.6: вопрос и ответ — на корневой span запроса, как у /chat (блок 3.6): по ним Phoenix заполняет
колонки input/output в списке трейсов, и трейсы разных вопросов видно без раскрытия. Только
маскированный текст длиной, как в логе; полный — в спанах LlamaIndex под RAGService.answer.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter
from opentelemetry import trace

from app.chat.domain import RequestError
from app.deps.providers import LimiterDep, RagDep, SettingsDep
from app.observability.logging import get_logger
from app.observability.pii import PREVIEW_CHARS, redact_pii
from app.schemas.rag import RagAnswer, RagQuery
from app.services import embeddings
from app.services.rag_common import RAGError, RAGLLMError, RAGNotIndexed, RAGUnavailable
from app.services.reranker import RerankError
from app.services.security.input_validator import validate_input

router = APIRouter(prefix="/rag", tags=["rag"])
log = get_logger()


@router.post(
    "/query",
    response_model=RagAnswer,
    summary="Ответ по базе знаний (RAG)",
    description="Вопрос → эмбеддинг → 10 кандидатов из Qdrant → score-guard → (re-ranker) → 5 фрагментов → ответ "
                "модели только по ним, с цитатами [1], [2]. Если лучший фрагмент ниже порога RAG__SCORE_THRESHOLD, "
                "модель не вызывается: ответ «По базе не нашёл, могу эскалировать.», confident=false.",
    responses={400: {"description": "Вопрос не прошёл проверку безопасности"},
               502: {"description": "Модель ответа не ответила"},
               503: {"description": "RAG не настроен, корпус не проиндексирован, нет Qdrant, модели эмбеддингов "
                                    "или re-ranker"}},
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
    except RerankError as exc:
        raise RequestError(503, "rerank_unavailable", str(exc)) from exc
    except RAGError as exc:
        raise RequestError(503, "rag_error", str(exc)) from exc
    log.info("rag_answered", top_score=result["top_score"], confident=result["confident"],
             sources=[s["source"] for s in result["sources"]],
             cited=[s["id"] for s in result["sources"] if s["cited"]])
    describe_root(question, result["answer"])
    return RagAnswer(**result)


def describe_root(question: str, answer: str) -> None:
    """Маскированные вопрос и ответ — на span HTTP-запроса (его создаёт FastAPI, если трейсинг включён)."""
    root = trace.get_current_span()
    if not root.is_recording():
        return
    root.set_attribute("input.value", redact_pii(question)[:PREVIEW_CHARS])
    root.set_attribute("output.value", redact_pii(answer)[:PREVIEW_CHARS])
