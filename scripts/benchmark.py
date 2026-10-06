"""
Бенчмарк sync vs async (блок 3.3).

Один и тот же набор из N промптов («Объясни одним абзацем концепцию №i: …»):
  (а) старый синхронный RobustLLMClient — последовательно, запрос за запросом;
  (б) AsyncLLMClient.batch_chat при concurrency 1, 5 и 10.
Для каждого режима — общее время и ускорение относительно sync. Затем сравнение
batch_chat и batch_chat_strict, когда на 3-м запросе подставлена невалидная модель.

Кеш асинхронного клиента выключен, и на каждый уровень создаётся новый клиент:
иначе повторный прогон тех же промптов отвечал бы из кеша и «ускорение» было бы ложным.

Запуск из корня проекта:
    python scripts/benchmark.py                          # мок-провайдер, задержка 1 с, 20 запросов
    python scripts/benchmark.py --target ollama --n 6 --max-tokens 60
Результаты дописываются в scripts/benchmark_results.md (раздел для каждой цели).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _target import ROOT, configure_target  # noqa: E402

CONCEPTS = [
    "event loop", "корутина", "семафор", "гонка данных", "кеширование", "retry с backoff",
    "circuit breaker", "rate limit", "идемпотентность", "очередь сообщений", "стриминг SSE",
    "fallback", "таймаут", "балансировка нагрузки", "векторный поиск", "эмбеддинг",
    "токенизация", "контекстное окно", "температура генерации", "function calling",
]
RESULTS_FILE = os.path.join(ROOT, "scripts", "benchmark_results.md")


def make_prompts(n: int) -> list[str]:
    return [f"Объясни одним абзацем концепцию №{i + 1}: {CONCEPTS[i % len(CONCEPTS)]}."
            for i in range(n)]


def quiet_sync_console() -> None:
    """Синхронный клиент пишет INFO о каждом запросе в консоль — для бенчмарка это шум.
    В консоли оставляем только предупреждения; полный лог по-прежнему в logs/app.log."""
    import logging

    import src.robust_client  # noqa: F401 — создаёт логгер "client"

    for handler in logging.getLogger("client").handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.WARNING)


def warm_up() -> None:
    """Один короткий запрос до замеров: Ollama загружает модель в память при первом
    обращении (до минуты на CPU), и это время не должно попасть в замер sync."""
    from src.robust_client import RobustLLMClient

    RobustLLMClient().chat([{"role": "user", "content": "Привет"}], max_tokens=1,
                           label="benchmark-warmup")


def run_sync(prompts: list[str], max_tokens: int | None) -> tuple[float, int]:
    from src.robust_client import RobustLLMClient

    client = RobustLLMClient()
    started = time.perf_counter()
    errors = 0
    for prompt in prompts:
        answer = client.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens,
                             label="benchmark-sync")
        errors += answer == RobustLLMClient.USER_FACING_FAILURE
    return time.perf_counter() - started, errors


async def run_async(prompts: list[str], levels: list[int], max_tokens: int | None
                    ) -> list[tuple[int, float, int]]:
    from app.services.llm_client import AsyncLLMClient

    rows = []
    for concurrency in levels:
        async with AsyncLLMClient(concurrency=concurrency, use_cache=False) as client:
            started = time.perf_counter()
            results = await client.batch_chat(prompts, concurrency=concurrency,
                                              max_tokens=max_tokens)
            elapsed = time.perf_counter() - started
        rows.append((concurrency, elapsed, sum(isinstance(r, BaseException) for r in results)))
    return rows


async def compare_strict(prompts: list[str], concurrency: int, max_tokens: int | None
                         ) -> list[str]:
    """batch_chat против batch_chat_strict: на 3-м запросе невалидная модель."""
    from app.services.llm_client import AsyncLLMClient
    from scripts.mock_llm_server import INVALID_MODEL

    models = [None] * len(prompts)
    models[2] = INVALID_MODEL
    lines = []

    async with AsyncLLMClient(concurrency=concurrency, use_cache=False) as client:
        started = time.perf_counter()
        results = await client.batch_chat(prompts, concurrency=concurrency, models=models,
                                          max_tokens=max_tokens)
        elapsed = time.perf_counter() - started
        failed = [i + 1 for i, r in enumerate(results) if isinstance(r, BaseException)]
        error = next(r for r in results if isinstance(r, BaseException))
        lines.append(
            f"| `batch_chat` (gather, return_exceptions) | {elapsed:.1f} с | "
            f"{len(results) - len(failed)} из {len(results)} ответов получены; ошибка на позиции "
            f"{failed} — `{type(error).__name__}` | батч не упал |"
        )

    async with AsyncLLMClient(concurrency=concurrency, use_cache=False) as client:
        started = time.perf_counter()
        try:
            await client.batch_chat_strict(prompts, concurrency=concurrency, models=models,
                                           max_tokens=max_tokens)
            lines.append("| `batch_chat_strict` (TaskGroup) | — | ошибка не возникла | — |")
        except* Exception as group:
            elapsed = time.perf_counter() - started
            lines.append(
                f"| `batch_chat_strict` (TaskGroup) | {elapsed:.1f} с | ни одного ответа: "
                f"`ExceptionGroup` (ошибок: {len(group.exceptions)}, "
                f"`{type(group.exceptions[0]).__name__}`), остальные задачи отменены | "
                f"поймано через `except*` |"
            )
    return lines


def save_section(target: str, text: str) -> None:
    content = ""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            content = f.read()
    if not content:
        content = ("# Результаты бенчмарка sync vs async (блок 3.3)\n\n"
                   "Файл обновляет `scripts/benchmark.py`: у каждой цели свой раздел.\n")
    begin, end = f"<!-- target:{target} -->", f"<!-- /target:{target} -->"
    block = f"{begin}\n{text}\n{end}"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
    content = pattern.sub(lambda _: block, content) if pattern.search(content) \
        else content.rstrip() + "\n\n" + block + "\n"
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк sync vs async (блок 3.3)")
    parser.add_argument("--target", choices=["mock", "ollama"], default="mock")
    parser.add_argument("--n", type=int, default=20, help="число промптов")
    parser.add_argument("--latency", type=float, default=1.0, help="задержка мока, с")
    parser.add_argument("--max-tokens", type=int, default=None, help="лимит ответа")
    parser.add_argument("--levels", default="1,5,10", help="уровни concurrency")
    parser.add_argument("--strict-n", type=int, default=10,
                        help="сколько промптов в сравнении batch_chat и batch_chat_strict")
    args = parser.parse_args()

    description = configure_target(args.target, args.latency)
    prompts = make_prompts(args.n)
    levels = [int(x) for x in args.levels.split(",")]
    print(f"Цель: {description}\nЗапросов: {args.n}\n")

    quiet_sync_console()
    if args.target == "ollama":
        print("Прогрев: загружаем модель в память…")
        warm_up()
    sync_time, sync_errors = run_sync(prompts, args.max_tokens)
    print(f"sync, последовательно      : {sync_time:6.1f} с  (ошибок: {sync_errors})")
    if sync_errors == len(prompts):
        sys.exit("\nВсе запросы завершились ошибкой — сравнивать нечего. Проверьте, что Ollama "
                 "запущен, модель SUPPORT_PRIMARY_MODEL скачана, а в .env задан OPENAI_API_KEY.")
    rows = asyncio.run(run_async(prompts, levels, args.max_tokens))
    for concurrency, elapsed, errors in rows:
        print(f"async, concurrency={concurrency:<2}      : {elapsed:6.1f} с  "
              f"(ускорение ×{sync_time / elapsed:.1f}, ошибок: {errors})")

    strict_prompts = prompts[: max(3, min(args.strict_n, args.n))]
    strict_lines = asyncio.run(compare_strict(strict_prompts, max(levels), args.max_tokens))
    print("\nНевалидная модель на 3-м запросе:")
    for line in strict_lines:
        print("  " + line.strip("| ").replace(" | ", " — "))

    table = [
        f"## Цель: {args.target}\n",
        f"{description}; запросов: {args.n}; дата: {datetime.now():%Y-%m-%d %H:%M}.\n",
        "| Режим | Время | Ускорение к sync | Ошибок |",
        "|-------|-------|------------------|--------|",
        f"| sync, последовательно | {sync_time:.1f} с | ×1.0 | {sync_errors} |",
        *[f"| async `batch_chat`, concurrency={c} | {t:.1f} с | ×{sync_time / t:.1f} | {e} |"
          for c, t, e in rows],
        "",
        f"Невалидная модель на 3-м запросе из {len(strict_prompts)} (concurrency={max(levels)}):\n",
        "| Метод | Время | Результат | Поведение |",
        "|-------|-------|-----------|-----------|",
        *strict_lines,
    ]
    save_section(args.target, "\n".join(table))
    print(f"\nРезультаты сохранены: {os.path.relpath(RESULTS_FILE, ROOT)}")


if __name__ == "__main__":
    main()
