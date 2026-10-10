"""
Сравнение моделей эмбеддингов на данных проекта (блок 5.1).

    python scripts/embeddings_benchmark.py                     # кандидаты по умолчанию
    python scripts/embeddings_benchmark.py --models ollama:bge-m3,st:intfloat/multilingual-e5-small
    python scripts/embeddings_benchmark.py --models openrouter:openai/text-embedding-3-small

Модель задаётся как «вид:имя»:
- baseline:trigrams    — не модель, а точка отсчёта: вектор из триграмм символов (совпадение слов).
                         Модель эмбеддингов, которая не обходит его, смысл не понимает;
- ollama:<модель>      — Ollama на --ollama-url (http://localhost:11434/v1), сначала ollama pull <модель>;
- st:<модель HF>       — sentence-transformers в процессе (pip install -r requirements-embeddings.txt),
                         веса скачиваются с Hugging Face при первом запуске;
- openrouter:<модель>  — OpenRouter, ключ из переменной --api-key-env (по умолчанию EVAL_JUDGE_API_KEY
                         из .env), прокси — LLM__PROXY_URL; модели OpenAI там платные (центы);
- openai:<модель>      — api.openai.com или --base-url, ключ из --api-key-env;
- env                  — модель из .env (EMBEDDINGS__*), как у embed_texts().

Что меряется (кеш выключен: меряется модель, а не кеш):
1. Пары из tests/eval/mini_benchmark.json: близость вопроса к нужному фрагменту (rel) и к
   похожему, но не тому (irr). «Пары» — у скольких rel > irr; «отрыв» — средняя rel − irr.
2. Поиск по всей базе data/help_center.jsonl (56 документов): на каком месте нужный фрагмент
   среди всех — hit@1, hit@3, MRR (среднее 1/место).
3. Время: индексация базы (embed_documents всех документов, после разогрева модели) и
   один запрос embed_query — медиана.
4. Для моделей с префиксами (E5 и др.) — что будет, если префиксы забыть: вопрос без
   «query: » при базе с «passage: » (частая ошибка) и без префиксов везде. Близость нужной
   пары, отрыв от ложной, пары и MRR — smoke-тест из задания.

Модель, которую не удалось запустить (нет пакета, Ollama не запущена, нет ключа), пропускается
с причиной — остальные считаются. Итог — таблица Markdown и JSON (--output, по умолчанию
docs/embeddings_benchmark.json): его читает scripts/indexing_cost.py.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCHMARK = ROOT / "tests" / "eval" / "mini_benchmark.json"
CORPUS = ROOT / "data" / "help_center.jsonl"
OUTPUT = ROOT / "docs" / "embeddings_benchmark.json"
DEFAULT_MODELS = ("baseline:trigrams,ollama:bge-m3,st:intfloat/multilingual-e5-small,"
                  "st:intfloat/multilingual-e5-base")
OPENROUTER_URL = "https://openrouter.ai/api/v1"
WARMUP = "Разогрев модели перед замером"
TRIGRAM_DIM = 4096


@dataclass
class Options:
    ollama_url: str = "http://localhost:11434/v1"
    base_url: str | None = None
    api_key_env: str = "EVAL_JUDGE_API_KEY"
    timeout: float = 120.0
    device: str = "cpu"


def load_items(path: Path = BENCHMARK) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus(path: Path = CORPUS) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TrigramBackend:
    """Лексический бейзлайн: мешок триграмм символов (хеш в TRIGRAM_DIM корзин). Близость —
    доля общих кусочков слов, смысла нет: «вернуть аккаунт» и «восстановить учётную запись»
    для него разные тексты. Хеш md5, а не hash(): у hash() строк своя соль в каждом процессе."""

    provider = "baseline"
    namespace = "lexical"
    model = "char-trigrams"
    dimensions = None
    local = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        vectors = []
        for text in texts:
            vector = [0.0] * TRIGRAM_DIM
            padded = f"  {text.lower()}  "
            for start in range(len(padded) - 2):
                digest = hashlib.md5(padded[start:start + 3].encode("utf-8")).digest()
                vector[int.from_bytes(digest[:4], "little") % TRIGRAM_DIM] += 1.0
            vectors.append(vector)
        return vectors

    def close(self) -> None:
        pass


def _secret(name: str) -> str | None:
    """Ключ из окружения или .env — в вывод и JSON не попадает."""
    value = os.environ.get(name)
    if value:
        return value
    env_file = ROOT / ".env"
    if env_file.exists():
        from dotenv import dotenv_values

        return dotenv_values(env_file).get(name) or None
    return None


def _llm_network() -> tuple[Any, bool]:
    """Прокси и проверка HTTPS из .env сервиса; без .env — напрямую и certifi."""
    try:
        from app.core.config import get_settings

        cfg = get_settings()
        return cfg.llm.proxy_url, cfg.llm.use_system_certs
    except Exception:  # noqa: BLE001 — нет .env или ключа LLM: бенчмарку они не обязательны
        return None, False


def make_service(spec: str, options: Options):
    """EmbeddingService без кеша по строке «вид:модель»."""
    from app.core.config import proxy_for
    from app.services import embeddings as emb

    kind, _, model = spec.partition(":")
    if kind == "env":
        return emb.build_service(use_cache=False)
    if kind == "baseline":
        return emb.EmbeddingService(TrigramBackend(), cache=None)
    if not model:
        raise ValueError(f"{spec}: нужна модель после двоеточия, например ollama:bge-m3")
    if kind == "ollama":
        backend = emb.OpenAICompatibleBackend(model=model, base_url=options.ollama_url, api_key="ollama",
                                              timeout=options.timeout)
    elif kind in ("openrouter", "openai"):
        base_url = OPENROUTER_URL if kind == "openrouter" else options.base_url
        key = _secret(options.api_key_env)
        if not key:
            raise emb.EmbeddingError(f"нет ключа: задайте {options.api_key_env} в .env или в окружении")
        proxy_url, system_certs = _llm_network()
        backend = emb.OpenAICompatibleBackend(model=model, base_url=base_url, api_key=key, timeout=options.timeout,
                                              proxy=proxy_for(base_url, proxy_url), use_system_certs=system_certs)
    elif kind == "st":
        _, system_certs = _llm_network()
        backend = emb.SentenceTransformerBackend(model, device=options.device, use_system_certs=system_certs)
    else:
        raise ValueError(f"{spec}: неизвестный вид {kind!r} — baseline, ollama, st, openrouter, openai или env")
    return emb.EmbeddingService(backend, cache=None)


def rank_of(target: int, query: list[float], doc_vectors: list[list[float]]) -> int:
    """Место документа target среди всех по близости к запросу (1 — первый)."""
    from app.services.embeddings import cosine

    scores = [cosine(query, vector) for vector in doc_vectors]
    return 1 + sum(score > scores[target] for score in scores)


def score(queries: list[list[float]], doc_vectors: list[list[float]], items: list[dict[str, str]],
          index: dict[str, int]) -> dict[str, Any]:
    """Метрики пар и поиска для готовых векторов."""
    from app.services.embeddings import cosine

    pairs, ranks = [], []
    for item, query in zip(items, queries, strict=True):
        relevant, irrelevant = index[item["relevant"]], index[item["irrelevant"]]
        rel, irr = cosine(query, doc_vectors[relevant]), cosine(query, doc_vectors[irrelevant])
        rank = rank_of(relevant, query, doc_vectors)
        pairs.append({"query": item["query"], "rel": round(rel, 4), "irr": round(irr, 4),
                      "margin": round(rel - irr, 4), "rank": rank})
        ranks.append(rank)
    margins = [pair["margin"] for pair in pairs]
    return {
        "pair_accuracy": sum(margin > 0 for margin in margins) / len(pairs),
        "mean_margin": round(statistics.fmean(margins), 4),
        "min_margin": round(min(margins), 4),
        "mean_rel": round(statistics.fmean(pair["rel"] for pair in pairs), 4),
        "mean_irr": round(statistics.fmean(pair["irr"] for pair in pairs), 4),
        "hit_at_1": sum(rank == 1 for rank in ranks) / len(ranks),
        "hit_at_3": sum(rank <= 3 for rank in ranks) / len(ranks),
        "mrr": round(statistics.fmean(1 / rank for rank in ranks), 4),
        "pairs": pairs,
    }


def evaluate(service, items: list[dict[str, str]], corpus: list[dict[str, str]]) -> dict[str, Any]:
    texts = [doc["text"] for doc in corpus]
    extra = [text for item in items for text in (item["relevant"], item["irrelevant"]) if text not in texts]
    texts += list(dict.fromkeys(extra))       # фрагменты пар, которых нет в базе (в нашей — все есть)
    index = {text: number for number, text in enumerate(texts)}

    service.embed_query(WARMUP)               # загрузка модели (Ollama, torch) — не в замере
    started = time.perf_counter()
    doc_vectors = service.embed_documents(texts)
    index_s = time.perf_counter() - started
    queries, latencies = [], []
    for item in items:
        started = time.perf_counter()
        queries.append(service.embed_query(item["query"]))
        latencies.append((time.perf_counter() - started) * 1000)

    result = {
        "model": service.label,
        "provider": service.backend.provider,
        "namespace": service.backend.namespace,
        "dim": service.dimension,
        "prefixes": {"query": service.prefixes.query, "document": service.prefixes.document,
                     "family": service.prefixes.family},
        "batch_size": service.batch_size,
        "docs": len(texts),
        "index_s": round(index_s, 3),
        "docs_per_s": round(len(texts) / index_s, 1) if index_s else None,
        "query_ms_median": round(statistics.median(latencies), 1),
        **score(queries, doc_vectors, items, index),
    }
    if service.prefixes.query or service.prefixes.document:
        raw_queries = service.embed_texts([item["query"] for item in items])
        # Частая ошибка: база проиндексирована с «passage: », а вопрос при поиске — без «query: ».
        result["query_without_prefix"] = ablation(score(raw_queries, doc_vectors, items, index))
        # Префиксов нет нигде — и у базы, и у вопроса.
        result["without_prefixes"] = ablation(score(raw_queries, service.embed_texts(texts), items, index))
    return result


def ablation(scored: dict[str, Any]) -> dict[str, Any]:
    """Метрики варианта без префиксов: без списка пар, но с близостью нужной пары по вопросам."""
    summary = {key: value for key, value in scored.items() if key != "pairs"}
    summary["rel_by_pair"] = [pair["rel"] for pair in scored["pairs"]]
    return summary


def percent(value: float) -> str:
    return f"{value * 100:.0f}%"


def markdown(results: list[dict[str, Any]], skipped: list[dict[str, str]]) -> str:
    lines = ["| Модель | Размерность | Пары rel > irr | Отрыв rel − irr | hit@1 | hit@3 | MRR | "
             "Индексация, с | Документов/с | Запрос, мс |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        pairs = round(r["pair_accuracy"] * len(r["pairs"]))
        lines.append(f"| {r['model']} | {r['dim']} | {pairs}/{len(r['pairs'])} | {r['mean_margin']:+.3f} "
                     f"(мин. {r['min_margin']:+.3f}) | {percent(r['hit_at_1'])} | {percent(r['hit_at_3'])} | "
                     f"{r['mrr']:.3f} | {r['index_s']:.1f} | {r['docs_per_s']} | {r['query_ms_median']:.0f} |")
    with_prefixes = [r for r in results if "without_prefixes" in r]
    if with_prefixes:
        lines += ["", "Префиксы (smoke-тест для моделей с префиксами): как меняется поиск, если их забыть:", "",
                  "| Модель | Как кодировали | Близость нужной пары | Пар, где она ниже, чем с префиксами | "
                  "Отрыв rel − irr | Пары rel > irr | MRR |",
                  "|---|---|---|---|---|---|---|"]
        for r in with_prefixes:
            n = len(r["pairs"])
            lines.append(f"| {r['model']} | с префиксами (как надо) | {r['mean_rel']:.3f} | — | "
                         f"{r['mean_margin']:+.3f} | {round(r['pair_accuracy'] * n)}/{n} | {r['mrr']:.3f} |")
            for key, title in (("query_without_prefix", "вопрос без префикса, база с префиксом"),
                               ("without_prefixes", "без префиксов везде")):
                variant = r.get(key)
                if variant is None:
                    continue
                lower = sum(new < pair["rel"] for new, pair in zip(variant["rel_by_pair"], r["pairs"], strict=True))
                lines.append(f"| | {title} | {variant['mean_rel']:.3f} ({variant['mean_rel'] - r['mean_rel']:+.3f}) | "
                             f"{lower}/{n} | {variant['mean_margin']:+.3f} | "
                             f"{round(variant['pair_accuracy'] * n)}/{n} | {variant['mrr']:.3f} |")
    if skipped:
        lines += ["", "Не запустились:"] + [f"- {s['spec']}: {s['reason']}" for s in skipped]
    return "\n".join(lines)


def details(result: dict[str, Any]) -> str:
    lines = [f"\n{result['model']} — по вопросам (rel / irr / место нужного фрагмента):"]
    for pair in result["pairs"]:
        mark = "ok " if pair["margin"] > 0 else "ОШ "
        lines.append(f"  {mark}{pair['rel']:.3f} / {pair['irr']:.3f} / {pair['rank']:>2}  {pair['query']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сравнение моделей эмбеддингов на данных проекта (блок 5.1)")
    parser.add_argument("--models", default=DEFAULT_MODELS, help=f"через запятую (по умолчанию {DEFAULT_MODELS})")
    parser.add_argument("--ollama-url", default=Options.ollama_url)
    parser.add_argument("--base-url", default=None, help="адрес для openai:<модель> (пусто — api.openai.com)")
    parser.add_argument("--api-key-env", default=Options.api_key_env,
                        help="переменная с ключом для openrouter:/openai: (значение не печатается)")
    parser.add_argument("--device", default="cpu", help="для st: cpu или cuda")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="JSON с результатами; «-» — не сохранять")
    parser.add_argument("--details", action="store_true", help="близости по каждому вопросу")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from app.observability.logging import setup_logging
    from app.services.embeddings import EmbeddingError

    setup_logging("WARNING", stream=sys.stdout)       # строки embeddings_request на каждый батч не нужны
    options = Options(ollama_url=args.ollama_url, base_url=args.base_url, api_key_env=args.api_key_env,
                      device=args.device)
    items, corpus = load_items(), load_corpus()
    results, skipped = [], []
    for spec in [s.strip() for s in args.models.split(",") if s.strip()]:
        print(f"→ {spec} …", flush=True)
        service = None
        try:
            service = make_service(spec, options)
            result = evaluate(service, items, corpus)
        except Exception as exc:  # noqa: BLE001 — нет torch, модель не скачалась, нет связи: следующая модель
            text = str(exc).strip().splitlines()
            reason = (text[0] if isinstance(exc, EmbeddingError) and text
                      else f"{type(exc).__name__}: {text[0] if text else ''}")[:300]
            print(f"  пропущена: {reason}", flush=True)
            skipped.append({"spec": spec, "reason": reason})
            continue
        finally:
            if service is not None:
                service.close()
        result["spec"] = spec
        results.append(result)
        print(f"  готово: {result['dim']} чисел, пары {result['pair_accuracy'] * 100:.0f}%, MRR {result['mrr']:.3f}, "
              f"индексация {result['index_s']:.1f} с", flush=True)

    print()
    print(markdown(results, skipped))
    if args.details:
        for result in results:
            print(details(result))
    if str(args.output) != "-" and results:
        report = {
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "machine": {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
                        "cpu_count": os.cpu_count(), "python": platform.python_version()},
            "benchmark": str(BENCHMARK.relative_to(ROOT)).replace("\\", "/"),
            "corpus": str(CORPUS.relative_to(ROOT)).replace("\\", "/"),
            "results": results,
            "skipped": skipped,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON: {args.output}")
    return 0 if results else 2


if __name__ == "__main__":
    sys.exit(main())
