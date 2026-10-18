"""
Наивный агент (блок 6.1): for-цикл вокруг Chat Completions — модель отвечает текстом или просит вызвать
инструменты (tool_calls), их результаты уходят обратно в messages как role="tool". Без агентных библиотек:
цикл — openai SDK и stdlib; клиент — как у сервиса (build_openai: LLM__*, прокси, сертификаты); инструменты
и allowlist — app/tools/naive_tools.py. Запуск: python -m app.services.agent_naive "<задача>" --trace
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any

from openai import OpenAIError
from pydantic import ValidationError

from app.core.config import get_settings
from app.observability.logging import setup_logging
from app.services.rag_baremetal import build_openai
from app.tools.naive_tools import DISPATCH, TOOLS

log = logging.getLogger(__name__)
DEFAULT_MODEL = "gpt-5.4-mini"   # канон задания; с Ollama — AGENT__MODEL=qwen3:4b-instruct
PREVIEW = 200                    # tool_result в trace — первые 200 символов


def call_tool(name: str, raw_args: str | None) -> str:
    """Только инструменты из DISPATCH; любой сбой — строка-ошибка для модели, цикл идёт дальше."""
    if name not in DISPATCH:
        return f"Ошибка: инструмента {name!r} нет. Доступны только: {', '.join(DISPATCH)}."
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError as exc:
        return f"Ошибка: аргументы {name} — не JSON ({exc.msg}). Повтори вызов с объектом JSON."
    try:
        return str(DISPATCH[name](**args))
    except Exception as exc:  # noqa: BLE001 — сбой инструмента модель видит и может обойти
        return f"Ошибка инструмента {name}: {type(exc).__name__}: {exc}"


def run_agent(task: str, max_steps: int = 6, *, client: Any = None, model: str | None = None) -> dict[str, Any]:
    """{"answer", "steps", "trace"}; не уложился в max_steps — answer=None и "error"."""
    if client is None:
        client, model = build_openai(get_settings()), model or get_settings().agent.model
    messages: list[Any] = [{"role": "user", "content": task}]
    trace: list[dict[str, Any]] = []
    for step in range(max_steps):                  # step в trace — номер итерации с 0, как в range
        started = time.perf_counter()
        response = client.chat.completions.create(model=model or DEFAULT_MODEL, messages=messages, tools=TOOLS)
        message = response.choices[0].message
        messages.append(message)
        for call in message.tool_calls or []:
            name, raw_args = call.function.name, call.function.arguments
            result = call_tool(name, raw_args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            trace.append(entry(step, name, raw_args, result, response.usage, started))
            log.info("step=%d tool=%s args=%s -> %s", step, name, raw_args, result[:80])
        if not message.tool_calls:
            trace.append(entry(step, None, None, None, response.usage, started))
            log.info("step=%d final answer", step)
            return {"answer": message.content, "steps": step + 1, "trace": trace}
    log.warning("max_steps=%d: no final answer", max_steps)
    return {"answer": None, "steps": max_steps, "trace": trace, "error": f"max_steps: {max_steps} шагов без ответа"}


def entry(step: int, name: str | None, raw_args: str | None, result: str | None, usage: Any, started: float) -> dict:
    """Запись trace: шаг, инструмент, исходные JSON-аргументы, результат, токены шага, время от начала шага."""
    return {"step": step, "tool_name": name, "tool_args": raw_args, "tool_result": result and result[:PREVIEW],
            "llm_input_tokens": usage and usage.prompt_tokens, "llm_output_tokens": usage and usage.completion_tokens,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Наивный агент на Chat Completions (блок 6.1)")
    parser.add_argument("task", help="задача для агента")
    parser.add_argument("--max-steps", type=int, help="лимит шагов; пусто — AGENT__MAX_STEPS (6)")
    parser.add_argument("--model", help="модель; пусто — AGENT__MODEL или gpt-5.4-mini")
    parser.add_argument("--trace", action="store_true", help="напечатать trace по шагам")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # консоль Windows (cp1251) и «→» заглушки Telegram
    setup_logging("INFO", stream=sys.stderr)      # JSON-лог сервиса в stderr; stdout — ответ и trace
    try:
        result = run_agent(args.task, args.max_steps or get_settings().agent.max_steps, model=args.model)
    except (ValidationError, OpenAIError) as exc:   # нет ключа в .env, модель не запущена, ошибка провайдера
        print(f"Остановка: {type(exc).__name__}: {exc}")
        return 2
    print(f"Остановка: {result['error']}" if "error" in result else result["answer"])
    if args.trace:
        print("\n--- trace ---", *(json.dumps(item, ensure_ascii=False) for item in result["trace"]), sep="\n")
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
