"""
Эмбеддинги из командной строки (блок 5.1): проверка кеша и батчинга app/services/embeddings.py.

    python scripts/embeddings_cli.py "тот же текст"     # embed_texts два раза в одном процессе
    python scripts/embeddings_cli.py --query "Как сменить пароль?"   # embed_query (префикс модели)
    python scripts/embeddings_cli.py --benchmark        # пары из tests/eval/mini_benchmark.json
    python scripts/embeddings_cli.py --corpus           # все документы data/help_center.jsonl
    python scripts/embeddings_cli.py --cache-info       # что лежит в файле кеша, по моделям
    python scripts/embeddings_cli.py --clear-cache      # очистить кеш

Модель, адрес, батч и кеш — из .env (EMBEDDINGS__*), как у сервиса. Вызывается именно
публичный embed_texts / embed_query / embed_documents модуля.

Критерий задания — повторный запуск не обращается к модели. Первый запуск:
«Вызов 1: … запросов к модели 1, 812.4 мс», в том же процессе «Вызов 2: из кеша 1 (память),
запросов к модели 0». Второй запуск той же команды: «Вызов 1: из кеша 1 (диск), запросов к
модели 0» и время в единицах миллисекунд. Каждое обращение к модели — строка
embeddings_request в логе (её видно и без -v: уровень INFO).

Смена модели: поменяйте EMBEDDINGS__MODEL в .env и запустите ту же команду — у новой модели
свои ключи кеша, будет запрос; --cache-info покажет записи обеих моделей.

Код выхода: 0 — готово, 2 — ошибка настроек или модели (текст — что сделать).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCHMARK = ROOT / "tests" / "eval" / "mini_benchmark.json"
CORPUS = ROOT / "data" / "help_center.jsonl"


def load_benchmark(path: Path = BENCHMARK) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus(path: Path = CORPUS) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def plural_texts(count: int) -> str:
    if count % 10 == 1 and count % 100 != 11:
        return f"{count} текст"
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return f"{count} текста"
    return f"{count} текстов"


class Meter:
    """Разница счётчиков сервиса за один вызов и его время."""

    def __init__(self, service) -> None:
        self.service = service

    def __call__(self, title: str, function, *args):
        before = _snapshot(self.service.stats)
        started = time.perf_counter()
        result = function(*args)
        elapsed = (time.perf_counter() - started) * 1000
        after = _snapshot(self.service.stats)
        d = {name: after[name] - before[name] for name in after}
        print(f"{title}: {plural_texts(d['texts'])}, из кеша {d['memory_hits'] + d['disk_hits']} "
              f"(память {d['memory_hits']}, диск {d['disk_hits']}), запросов к модели {d['requests']}, "
              f"{elapsed:.1f} мс", flush=True)
        return result


def _snapshot(stats) -> dict[str, int]:
    return {name: getattr(stats, name) for name in ("texts", "memory_hits", "disk_hits", "requests")}


def describe_vector(vector: list[float]) -> str:
    norm = sum(x * x for x in vector) ** 0.5
    head = ", ".join(f"{x:.4f}" for x in vector[:4])
    return f"Вектор: {len(vector)} чисел, длина {norm:.6f}, начало [{head}, …]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Эмбеддинги модели из .env и проверка кеша (блок 5.1)")
    parser.add_argument("texts", nargs="*", help="тексты для embed_texts")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--query", help="один вопрос через embed_query (с префиксом модели)")
    mode.add_argument("--benchmark", action="store_true", help="пары из tests/eval/mini_benchmark.json")
    mode.add_argument("--corpus", action="store_true", help="все документы data/help_center.jsonl")
    mode.add_argument("--cache-info", action="store_true", help="записи в файле кеша по моделям")
    mode.add_argument("--clear-cache", action="store_true", help="очистить кеш")
    parser.add_argument("--repeat", type=int, default=2, help="сколько раз вызвать в этом процессе (по умолчанию 2)")
    parser.add_argument("-v", "--verbose", action="store_true", help="лог уровня DEBUG")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if not (args.texts or args.query or args.benchmark or args.corpus or args.cache_info or args.clear_cache):
        parser.error("укажите текст или режим: --query, --benchmark, --corpus, --cache-info, --clear-cache")

    from pydantic import ValidationError

    from app.core.config import get_settings
    from app.observability.logging import setup_logging
    from app.services import embeddings

    setup_logging("DEBUG" if args.verbose else "INFO", stream=sys.stdout)
    try:
        get_settings()
        service = embeddings.default_service()
    except (ValidationError, ValueError, embeddings.EmbeddingError) as exc:
        print(f"Ошибка настроек: {exc}", file=sys.stderr)
        return 2

    cache = service.cache
    if args.clear_cache or args.cache_info:
        if cache is None or not cache.persistent:
            print("Кеш на диске выключен (EMBEDDINGS__CACHE_ENABLED=false) или файл не открылся.")
            return 0
        if args.clear_cache:
            print(f"Удалено записей: {cache.clear()} ({cache.path})")
            return 0
        rows = cache.summary()
        print(f"Кеш: {cache.path}")
        for model, dim, count in rows:
            print(f"  {model}: векторов {count}, по {dim} чисел")
        if not rows:
            print("  пусто")
        return 0

    prefixes = service.prefixes
    shown = f"«{prefixes.query}» / «{prefixes.document}»" if prefixes.query or prefixes.document else "нет"
    print(f"Модель: {service.label} · префиксы: {shown} ({prefixes.family}) · батч {service.batch_size}")
    print(f"Кеш: {cache.path if cache is not None and cache.persistent else 'только память' if cache else 'выключен'}")
    meter = Meter(service)
    try:
        for number in range(1, max(1, args.repeat) + 1):
            title = f"Вызов {number}" + (" (тот же процесс)" if number > 1 else "")
            if args.query:
                result = [meter(title + ", embed_query", embeddings.embed_query, args.query)]
            elif args.benchmark:
                result = run_benchmark(meter, title, embeddings)
            elif args.corpus:
                docs = load_corpus()
                result = meter(title + f", embed_documents ({len(docs)} документов)", embeddings.embed_documents,
                               [doc["text"] for doc in docs])
            else:
                result = meter(title + ", embed_texts", embeddings.embed_texts, args.texts)
    except embeddings.EmbeddingError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    if result and not args.benchmark:
        print(describe_vector(result[0]))
    total = service.stats
    print(f"Итого: {plural_texts(total.texts)}, из кеша {total.cache_hits}, запросов к модели {total.requests}, "
          f"посчитано моделью {total.embedded}, повторов после ошибок {total.retries}")
    return 0


def run_benchmark(meter: Meter, title: str, embeddings) -> list[list[float]]:
    items = load_benchmark()
    queries = meter(title + ", embed_query × " + str(len(items)),
                    lambda: [embeddings.embed_query(item["query"]) for item in items])
    docs = meter(title + ", embed_documents", embeddings.embed_documents,
                 [text for item in items for text in (item["relevant"], item["irrelevant"])])
    right = 0
    for number, query in enumerate(queries):
        relevant = embeddings.cosine(query, docs[2 * number])
        irrelevant = embeddings.cosine(query, docs[2 * number + 1])
        right += relevant > irrelevant
    print(f"  нужный фрагмент ближе ложного: {right} из {len(items)}")
    return queries


if __name__ == "__main__":
    sys.exit(main())
