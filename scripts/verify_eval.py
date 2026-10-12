"""
Проверка установки оценки и трейсинга (блок 5.6): версии пакетов, импорты RAGAS 0.4, модуля метрик,
LlamaIndexInstrumentor и phoenix.evals; с --live — один вызов судьи и эмбеддингов вживую.

    pip install -r requirements.txt -r requirements-eval.txt -r requirements-tracing.txt
    python scripts/verify_eval.py            # только импорты и сборка судьи из .env — без запросов
    python scripts/verify_eval.py --live     # плюс faithfulness и has_citation на одном примере

Группы зависимостей (pyproject.toml): eval — ragas, langchain-community, anthropic, pandas, arize-phoenix,
rapidfuzz, graphql-core<3.3; tracing — openinference-instrumentation-llama-index и OpenTelemetry. В production-образ идёт
только tracing (Dockerfile, UV_EXTRAS), eval — на машине, где запускают scripts/run_eval.py.

--live проверяет то, что импорт проверить не может: судья RAGAS__JUDGE_MODEL отвечает JSON по схеме
(llm_factory, режим instructor), эмбеддинги для AnswerRelevancy доступны. На CPU (qwen3:4b-instruct)
первый вызов — до минуты: Ollama загружает модель. Кеш судьи не используется — запрос уходит всегда.

Код выхода: 0 — всё на месте (и --live прошёл), 1 — нет пакета группы eval, импорт упал или судья не
ответил.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

PACKAGES = {
    "eval": ("ragas", "langchain-community", "anthropic", "pandas", "arize-phoenix", "arize-phoenix-evals",
             "arize-phoenix-client", "rapidfuzz", "graphql-core"),
    "tracing": ("openinference-instrumentation-llama-index", "opentelemetry-sdk", "opentelemetry-exporter-otlp"),
    "сервис": ("llama-index-core", "openai"),
}
# Пример из материалов курса: ответ подтверждается контекстом и ссылается на источник.
QUESTION = "Сколько дней на возврат?"
ANSWER = "Возврат возможен в течение 14 дней с момента покупки [1]."
CONTEXTS = ["Возврат товара возможен в течение 14 дней с момента покупки."]


def versions() -> list[str]:
    """Пакеты, которых нет (из групп eval и сервиса); tracing — только предупреждение."""
    missing = []
    for group, names in PACKAGES.items():
        print(f"=== {group} ===")
        for name in names:
            try:
                print(f"  {name}=={version(name)}")
            except PackageNotFoundError:
                print(f"  {name}: НЕ УСТАНОВЛЕН")
                if group != "tracing":
                    missing.append(name)
    return missing


def check(title: str, probe) -> bool:
    try:
        detail = probe()
    except Exception as exc:  # noqa: BLE001 — любой сбой импорта или сборки — в отчёт
        print(f"[{title}] ОШИБКА: {type(exc).__name__}: {' '.join(str(exc).split())[:300]}")
        return False
    print(f"[{title}] OK{': ' + detail if detail else ''}")
    return True


def ragas_imports() -> str:
    import rapidfuzz  # noqa: F401  — OverlapScoreBuilder генератора golden dataset
    from ragas.cache import DiskCacheBackend  # noqa: F401
    from ragas.embeddings import OpenAIEmbeddings  # noqa: F401
    from ragas.llms import llm_factory  # noqa: F401
    from ragas.metrics import discrete_metric  # noqa: F401
    from ragas.metrics.collections import (  # noqa: F401
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    from ragas.testset import TestsetGenerator  # noqa: F401
    from ragas.testset.transforms import default_transforms  # noqa: F401

    return "collections-метрики, llm_factory, discrete_metric, TestsetGenerator"


def metrics_module() -> str:
    from app.eval.metrics import (  # noqa: F401
        build_embeddings,
        build_judge,
        build_metrics,
        eval_row,
        make_has_citation,
        score_row,
    )

    return "build_judge, build_metrics, make_has_citation, eval_row"


def judge_from_env() -> str:
    """Судья и эмбеддинги из .env собираются (без запросов): адрес, прокси, сертификаты, ключ."""
    from app.core.config import get_settings
    from app.eval.metrics import (
        build_embeddings,
        build_judge,
        build_metrics,
        judge_info,
    )

    cfg = get_settings()
    build_metrics(build_judge(cfg), build_embeddings(cfg))
    info = judge_info(cfg)
    return f"судья {info['judge_model']} @ {info['judge_host']}, эмбеддинги {info['judge_embeddings']}"


def tracing_imports() -> str:
    from openinference.instrumentation.llama_index import (
        LlamaIndexInstrumentor,  # noqa: F401
    )

    from app.observability.tracing import llama_index_instrumentor

    if llama_index_instrumentor() is None:
        raise ImportError("app.observability.tracing не нашёл LlamaIndexInstrumentor")
    return "LlamaIndexInstrumentor"


def phoenix_imports() -> str:
    from phoenix.client import Client  # noqa: F401
    from phoenix.evals import LLM, evaluate_dataframe  # noqa: F401
    from phoenix.evals.metrics.hallucination import HallucinationEvaluator  # noqa: F401

    return "phoenix.client.Client, phoenix.evals.HallucinationEvaluator"


async def live() -> bool:
    """Faithfulness и has_citation на одном примере, плюс эмбеддинг — вживую, без кеша судьи."""
    from app.core.config import get_settings
    from app.eval.metrics import build_embeddings, build_judge, build_metrics
    from app.services.rag_common import strip_citations

    cfg = get_settings()
    embeddings = build_embeddings(cfg)
    metrics = build_metrics(build_judge(cfg, cache=None), embeddings)
    ok = True
    for title, call in (                       # faithfulness — без [1], как в run_eval (score_row)
        ("faithfulness", lambda: metrics.faithfulness.ascore(user_input=QUESTION, response=strip_citations(ANSWER),
                                                             retrieved_contexts=CONTEXTS)),
        ("has_citation", lambda: metrics.has_citation.ascore(response=ANSWER)),
    ):
        started = time.perf_counter()
        try:
            result = await call()
        except Exception as exc:  # noqa: BLE001
            print(f"[live {title}] ОШИБКА: {type(exc).__name__}: {' '.join(str(exc).split())[:300]}")
            ok = False
            continue
        if result.value is None:              # @discrete_metric ловит сбой судьи сам: value=None, текст в reason
            print(f"[live {title}] ОШИБКА: {' '.join(str(result.reason or '').split())[:300]}")
            ok = False
            continue
        print(f"[live {title}] OK: {result.value} ({time.perf_counter() - started:.1f} с; ожидается "
              f"{'1.0' if title == 'faithfulness' else 'yes'})")
    try:
        vector = await embeddings.aembed_text(QUESTION)
        print(f"[live эмбеддинги] OK: размерность {len(vector)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[live эмбеддинги] ОШИБКА: {type(exc).__name__}: {' '.join(str(exc).split())[:300]}")
        ok = False
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Проверка eval- и tracing-зависимостей (блок 5.6)")
    parser.add_argument("--live", action="store_true", help="один вызов судьи и эмбеддингов вживую")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):     # вывод в файл на Windows (cp1251): «₽» и др. — не падать
        sys.stdout.reconfigure(errors="replace")

    missing = versions()
    print()
    ok = not missing
    ok &= check("ragas", ragas_imports)
    ok &= check("app.eval.metrics", metrics_module)
    ok &= check("судья из .env", judge_from_env)
    ok &= check("phoenix", phoenix_imports)
    if not check("tracing", tracing_imports):
        print("  трейсинг LlamaIndex при локальном uvicorn не включится: pip install -r requirements-tracing.txt "
              "(в Docker-образе пакет есть)")
    if args.live and ok:
        ok &= asyncio.run(live())
    print("\nВсё на месте." if ok else "\nЕсть проблемы — см. выше.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
