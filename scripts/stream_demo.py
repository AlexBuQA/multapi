"""
Демо stream_chat (блок 3.3): печатает фрагменты ответа по мере прихода и отметки
time.perf_counter() на первом и последнем yield — TTFT должен быть меньше общего времени.

Запуск из корня проекта:
    python scripts/stream_demo.py                                   # мок-провайдер
    python scripts/stream_demo.py --target ollama "Что такое event loop?"
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _target import configure_target  # noqa: E402


async def run(prompt: str, max_tokens: int | None) -> None:
    from app.services.llm_client import AsyncLLMClient

    async with AsyncLLMClient(use_cache=False) as client:
        started = time.perf_counter()
        first = last = None
        async for delta in client.stream_chat(prompt, max_tokens=max_tokens):
            now = time.perf_counter()
            first = first or now
            last = now
            print(delta, end="", flush=True)
        done = time.perf_counter()

    print("\n")
    print(f"perf_counter: старт {started:.3f} | первый yield {first:.3f} | "
          f"последний yield {last:.3f}")
    print(f"TTFT (до первого фрагмента): {first - started:.2f} с")
    print(f"До последнего фрагмента    : {last - started:.2f} с")
    print(f"Всего, вместе с закрытием  : {done - started:.2f} с")
    print("TTFT < общего времени:", "да" if first - started < done - started else "НЕТ")
    print("Usage и время записаны в logs/llm_calls.jsonl (событие llm.stream).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Демо stream_chat")
    parser.add_argument("prompt", nargs="?", default="Что такое event loop? Ответь в двух предложениях.")
    parser.add_argument("--target", choices=["mock", "ollama"], default="mock")
    parser.add_argument("--latency", type=float, default=2.0, help="задержка мока, с")
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()
    print(f"Цель: {configure_target(args.target, args.latency)}\n")
    asyncio.run(run(args.prompt, args.max_tokens))


if __name__ == "__main__":
    main()
