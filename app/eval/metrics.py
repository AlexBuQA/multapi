"""
Метрики RAGAS 0.4 для оценки RAG (блок 5.6) и оценка одной строки golden dataset.

RAGAS 0.4 — collections-API: каждая метрика — объект с корутиной ascore(...), аргументы только
именованные (у разных метрик разный набор полей), результат — MetricResult с .value. Судья —
llm_factory (InstructorLLM: JSON по схеме ответа), эмбеддинги для AnswerRelevancy — ragas.embeddings.
OpenAIEmbeddings. Кастомная метрика has_citation — декоратором @discrete_metric (AspectCritic в 0.4
убран).

Судья — OpenAI-совместимый провайдер из RAGAS__* (app/core/config.py, RagasSettings): по умолчанию
qwen3:4b-instruct в Ollama, или OpenRouter — тогда с прокси LLM__PROXY_URL, сертификатами ОС
(LLM__USE_SYSTEM_CERTS) и заголовками атрибуции, как у судьи блока 3.7. Эмбеддинги — bge-m3, та же
модель, что у поиска (EMBEDDINGS__*). Ни судья, ни его эмбеддинги — не production-LLM сервиса.

Ответ сервиса заканчивается строкой «Источники: [2][3]» — это разметка, а не утверждение о продукте.
Faithfulness и AnswerRelevancy получают ответ без неё и без [N] (strip_citations): иначе судья счёл бы
«Источники: [2][3]» утверждением, которого нет в контексте, и занизил faithfulness каждому ответу со
ссылкой. has_citation смотрит ответ как есть.

Пустые поля не роняют прогон. Ответ пустой — все метрики NaN. Score-guard отказал, фрагментов нет —
faithfulness NaN (проверять не по чему), context_precision и context_recall — 0.0: вопрос из golden
dataset, ответ на него в базе есть, а поиск не нашёл ничего. Нет reference — context_precision и
context_recall NaN. Судья ответил не по схеме или упал — NaN у этой метрики и текст ошибки в errors.

Модуль — только для офлайн-оценки (группа eval: pip install -r requirements-eval.txt); сервис его не
импортирует.
"""
from __future__ import annotations

import math
import os
import re
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

# Телеметрия RAGAS (POST на t.explodinggradients.com при каждом вызове судьи) — выключена до импорта.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

import httpx  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from ragas.cache import DiskCacheBackend  # noqa: E402
from ragas.embeddings import OpenAIEmbeddings  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics import discrete_metric  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from app.core.config import (  # noqa: E402
    http_client_options,
    is_local_url,
    is_openrouter,
    provider_headers,
    proxy_for,
)
from app.services.rag_common import strip_citations  # noqa: E402

if TYPE_CHECKING:
    from app.core.config import Settings


def _ragas_user_id_quietly() -> None:
    """RAGAS собирает событие телеметрии и при RAGAS_DO_NOT_TRACK (отправка выключена, а объект события —
    нет): первое событие читает ~/.local/share/ragas/uuid.json через json.load(open(...)), не закрывая
    файл, — ResourceWarning (pytest -W error падал). Первый вызов — здесь, без предупреждения; дальше
    значение из lru_cache, файл больше не открывается."""
    try:
        from ragas._analytics import get_userid
    except ImportError:                       # модуль внутренний — в другой версии его может не быть
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        get_userid()


_ragas_user_id_quietly()

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "has_citation")
JUDGE_TITLE = "multapi-ragas"


# ---------------------------------------------------------------- судья и эмбеддинги
def judge_endpoint(cfg: Settings) -> tuple[str | None, str]:
    """Адрес и ключ судьи: RAGAS__JUDGE_* или, если пусто, LLM__* (Ollama сервиса)."""
    base_url = cfg.ragas.judge_base_url or cfg.llm.base_url
    key = cfg.ragas.judge_api_key or cfg.llm.openai_api_key
    return base_url, key.get_secret_value()


def _http_client(cfg: Settings, base_url: str | None) -> httpx.AsyncClient | None:
    proxy = proxy_for(base_url, cfg.llm.proxy_url)
    options = http_client_options(proxy, cfg.llm.use_system_certs and not is_local_url(base_url))
    return httpx.AsyncClient(**options) if options else None


def judge_client(cfg: Settings) -> AsyncOpenAI:
    base_url, key = judge_endpoint(cfg)
    return AsyncOpenAI(api_key=key, base_url=base_url, timeout=cfg.ragas.judge_timeout, max_retries=3,
                       default_headers=provider_headers(base_url, JUDGE_TITLE),
                       http_client=_http_client(cfg, base_url))


def judge_cache(cfg: Settings) -> DiskCacheBackend | None:
    """Кеш ответов судьи. Ключ RAGAS — текст запроса без имени модели, поэтому у каждого судьи своя
    папка: иначе после смены судьи прогон взял бы оценки прежнего."""
    if not cfg.ragas.cache_enabled:
        return None
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", cfg.ragas.judge_model).strip("_") or "judge"
    path = cfg.ragas.cache_dir / slug
    path.mkdir(parents=True, exist_ok=True)
    return DiskCacheBackend(cache_dir=str(path))


def build_judge(cfg: Settings, *, client: Any = None, cache: Any = None) -> Any:
    """Судья RAGAS через llm_factory: OpenAI-совместимый клиент, JSON по схеме ответа. У OpenRouter —
    provider.require_parameters: запрос не уйдёт провайдеру, который молча проигнорирует JSON-режим."""
    base_url, _ = judge_endpoint(cfg)
    extra: dict[str, Any] = {}
    if is_openrouter(base_url):
        extra["extra_body"] = {"provider": {"require_parameters": True}}
    return llm_factory(cfg.ragas.judge_model, provider="openai", client=client or judge_client(cfg),
                       cache=cache, max_tokens=cfg.ragas.judge_max_tokens, **extra)


def build_embeddings(cfg: Settings, *, client: Any = None) -> OpenAIEmbeddings:
    """Эмбеддинги для AnswerRelevancy: по умолчанию модель и адрес поиска (bge-m3 в Ollama)."""
    model = cfg.ragas.embeddings_model or cfg.embeddings.model
    base_url = cfg.ragas.embeddings_base_url or cfg.embeddings.base_url or cfg.llm.base_url
    key = (cfg.embeddings.api_key or cfg.llm.openai_api_key).get_secret_value()
    if client is None:
        client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=cfg.embeddings.timeout, max_retries=3,
                             default_headers=provider_headers(base_url, JUDGE_TITLE),
                             http_client=_http_client(cfg, base_url))
    return OpenAIEmbeddings(client=client, model=model)


def judge_info(cfg: Settings) -> dict[str, Any]:
    """Кто и чем оценивал — в JSON прогона (без ключей и логина прокси)."""
    base_url, _ = judge_endpoint(cfg)
    host = urlsplit(base_url or "https://api.openai.com/v1").hostname
    return {"judge_model": cfg.ragas.judge_model, "judge_host": host,
            "judge_max_tokens": cfg.ragas.judge_max_tokens,
            "judge_embeddings": cfg.ragas.embeddings_model or cfg.embeddings.model}


# ---------------------------------------------------------------- метрики
class CitationVerdict(BaseModel):
    has_citation: Literal["yes", "no"]


CITATION_PROMPT = (
    "Содержит ли ответ ссылку на источник: маркер вида '[1]'/'[doc_id]', имя файла, или фразу "
    "'согласно …'? Ответь yes или no.\n\nОтвет:\n{response}"
)


def make_has_citation(judge: Any) -> Any:
    """Метрика «ответ ссылается на источник» (критерий 3 чекпоинта 5) через @discrete_metric: судья
    отвечает структурой CitationVerdict, значение — yes / no."""

    @discrete_metric(name="has_citation", allowed_values=["yes", "no"])
    async def has_citation(response: str) -> str:
        verdict = await judge.agenerate(CITATION_PROMPT.replace("{response}", response), response_model=CitationVerdict)
        return verdict.has_citation

    return has_citation


@dataclass
class RagasMetrics:
    """Четыре collections-метрики RAGAS и has_citation на одном судье."""

    faithfulness: Faithfulness
    answer_relevancy: AnswerRelevancy
    context_precision: ContextPrecision
    context_recall: ContextRecall
    has_citation: Any


def build_metrics(judge: Any, embeddings: Any) -> RagasMetrics:
    """strictness=1 у AnswerRelevancy: по умолчанию судья трижды восстанавливает вопрос по ответу, чтобы
    усреднить разброс, но llm_factory ставит temperature 0.01 — три вызова дают один и тот же вопрос
    (а с кешем второй и третий — копии первого). На CPU это два лишних вызова судьи на строку."""
    return RagasMetrics(faithfulness=Faithfulness(llm=judge),
                        answer_relevancy=AnswerRelevancy(llm=judge, embeddings=embeddings, strictness=1),
                        context_precision=ContextPrecision(llm=judge),
                        context_recall=ContextRecall(llm=judge),
                        has_citation=make_has_citation(judge))


def _value(name: str, value: Any) -> float:
    if name == "has_citation":
        return 1.0 if str(value).strip().lower() == "yes" else 0.0
    return float(value) if value is not None else math.nan


async def score_row(metrics: RagasMetrics, *, question: str, reference: str, response: str,
                    contexts: list[str]) -> tuple[dict[str, float], dict[str, str]]:
    """Пять метрик одной строки: (оценки, ошибки по метрикам). has_citation — 1.0 (yes) или 0.0 (no),
    чтобы среднее по прогону было долей ответов со ссылкой."""
    scores: dict[str, float] = {}
    errors: dict[str, str] = {}

    async def measure(name: str, call: Callable[[], Awaitable[Any]]) -> None:
        try:
            result = await call()
        except Exception as exc:  # noqa: BLE001 — сбой судьи на одной метрике не роняет прогон
            scores[name] = math.nan
            errors[name] = f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"
            return
        if result.value is None:
            # @discrete_metric сам ловит исключение судьи и отдаёт value=None с текстом ошибки в reason:
            # без этой проверки сбой судьи посчитался бы ответом «no» (has_citation = 0).
            scores[name] = math.nan
            errors[name] = f"нет значения: {' '.join(str(result.reason or '').split())[:300]}"
            return
        scores[name] = _value(name, result.value)
        if math.isnan(scores[name]):
            errors[name] = "RAGAS вернул NaN (например, судья не выделил в ответе ни одного утверждения)"

    def skip(name: str, value: float, why: str) -> None:
        scores[name] = value
        errors[name] = why

    if not (response or "").strip():
        for name in METRICS:
            skip(name, math.nan, "пустой ответ")
        return scores, errors
    text = strip_citations(response) or response          # содержание ответа без разметки цитат
    if contexts:
        await measure("faithfulness", lambda: metrics.faithfulness.ascore(
            user_input=question, response=text, retrieved_contexts=contexts))
    else:
        skip("faithfulness", math.nan, "нет фрагментов: отказ score-guard")
    await measure("answer_relevancy", lambda: metrics.answer_relevancy.ascore(user_input=question, response=text))
    if not (reference or "").strip():
        skip("context_precision", math.nan, "нет reference")
        skip("context_recall", math.nan, "нет reference")
    elif not contexts:
        skip("context_precision", 0.0, "ничего не найдено: отказ score-guard")
        skip("context_recall", 0.0, "ничего не найдено: отказ score-guard")
    else:
        await measure("context_precision", lambda: metrics.context_precision.ascore(
            user_input=question, reference=reference, retrieved_contexts=contexts))
        await measure("context_recall", lambda: metrics.context_recall.ascore(
            user_input=question, retrieved_contexts=contexts, reference=reference))
    await measure("has_citation", lambda: metrics.has_citation.ascore(response=response))
    return scores, errors


async def eval_row(rag: Any, row: dict[str, Any], metrics: RagasMetrics) -> dict[str, Any]:
    """Одна строка golden dataset целиком: ответ RAG (RAGService.evaluate_inputs, в потоке — сервис
    синхронный) и пять метрик. Строка результата — поля golden, ответ, фрагменты, оценки и ошибки.

    scripts/run_eval.py делает то же в два прохода (сначала все ответы, потом судья: на CPU модель ответа
    и судья не помещаются в память вместе), а эта функция — для одного вопроса: verify_eval.py --live."""
    import asyncio

    result = await asyncio.to_thread(rag.evaluate_inputs, row["user_input"])
    scores, errors = await score_row(metrics, question=row["user_input"], reference=row.get("reference") or "",
                                     response=result["answer"], contexts=result["retrieved_contexts"])
    return {**row, "response": result["answer"], "retrieved_contexts": result["retrieved_contexts"],
            "latency_ms": result["latency_ms"], **scores, "errors": errors}
