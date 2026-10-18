"""
ReAct-агент с лимитами и self-reflection (блок 6.2) — рядом с наивным baseline (agent_naive.py не меняется).

Цикл Thought → Action → Observation на нативном tool calling Chat Completions (tools, tool_choice="auto"):
формат держит сам API, системный промпт задаёт поведение — пояснение перед действием, ровно один
инструмент за шаг, остановка, когда данных достаточно, запрет выдумывать и честный отказ.

Предохранители:
- max_iterations (10, допустимо 8–20) → ответ «Превышен лимит итераций»;
- timeout_per_iteration_sec на итерацию целиком (модель + инструмент + критик) → «Timeout». Время
  проверяется до каждого вызова, у вызова модели — свой таймаут на остаток (без повторов SDK), инструмент
  ждут не дольше остатка: зависший шаг не держит цикл;
- критик (Reflexion-light): после каждого observation — отдельный вызов модели с вердиктом «OK» или
  «REVISE: <причина>». REVISE — системная приписка с критикой к следующей итерации и премиум-модель на
  этот шаг; ревизий не больше max_revisions (2) за запуск, счётчик между итерациями не сбрасывается;
- аргументы сверяются со строгой схемой инструмента (app/tools/react_tools.py) до вызова функции.

Токены каждого вызова (и критика) складываются в usage — он возвращается с ответом и пишется в лог.

    python -m app.services.agent_react "Сколько запросов в минуту можно делать к API на тарифе «Команда»?" --trace
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass
from typing import Any

from openai import APITimeoutError, OpenAIError
from pydantic import ValidationError

from app.core.config import get_settings
from app.observability.logging import get_logger, setup_logging
from app.services.rag_baremetal import build_openai
from app.tools.react_tools import DISPATCH, TOOLS, schemas, validate_args

log = get_logger()

DEFAULT_MODEL = "gpt-5.4-mini"          # канон задания: и шаг ReAct, и критик
ITERATION_LIMIT = "Превышен лимит итераций"
TIMEOUT = "Timeout"
PREVIEW = 200                           # observation и рассуждение в trace — первые 200 символов
ITERATIONS_RANGE = (8, 20)

SYSTEM_PROMPT = (
    "Ты — агент техподдержки сервиса «Личный кабинет». Действуй как агент: самостоятельно выбирай "
    "инструменты и их порядок, при необходимости разбивая задачу на подзадачи. "
    "На каждом шаге сначала одним предложением поясни, что и зачем делаешь, затем вызови ровно один "
    "инструмент и опирайся на его результат. "
    "Как только данных достаточно — дай финальный ответ без вызова инструментов. "
    "Не выдумывай данные: используй только то, что вернули инструменты; если поиск ничего не нашёл — так и скажи. "
    "Если доступными инструментами задачу решить нельзя — нет нужного инструмента или данных — сразу прямо "
    "сообщи об этом, не вызывая инструментов. "
    "Пиши клиенту, только когда задача это просит, указан chat_id и все факты для текста проверены инструментами. "
    "Отвечай по-русски."
)

CRITIC_PROMPT = (
    "Ты — критик шага агента техподдержки «Личного кабинета». Тебе дают задачу пользователя, план агента на "
    "этом шаге, вызов инструмента, его результат и прошлые шаги. Ответь одной строкой: «OK», если с этим "
    "результатом можно идти дальше, или «REVISE: <причина одним предложением>», если шаг надо исправить: "
    "инструмент вернул ошибку; результат не по теме задачи; аргументы не соответствуют задаче (не тот вопрос, "
    "chat_id или часовой пояс); в тексте клиенту есть факты, которых нет в результатах инструментов. "
    "«Ничего не найдено» — не повод для REVISE: агент должен честно сообщить, что данных нет."
)
SKIPPED = "Не выполнено: за шаг вызывается ровно один инструмент. Если он нужен — вызови его на следующем шаге."
VERDICT = re.compile(r"\b(REVISE|OK)\b\W*(.*)", re.IGNORECASE | re.DOTALL)


class IterationTimeout(Exception):
    """Итерация не уложилась в timeout_per_iteration_sec."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, usage: Any) -> dict[str, int]:
        """Токены одного вызова модели — в сумму; возвращает их же для записи trace."""
        prompt = (usage and usage.prompt_tokens) or 0
        completion = (usage and usage.completion_tokens) or 0
        total = (usage and usage.total_tokens) or prompt + completion
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        self.calls += 1
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def call_tool(name: str, raw_args: str | None, dispatch: dict[str, Any], tool_schemas: dict[str, Any]) -> str:
    """Инструмент из allowlist с проверкой аргументов по схеме; любой сбой — строка-ошибка для модели."""
    if name not in dispatch:
        return f"Ошибка: инструмента {name!r} нет. Доступны только: {', '.join(dispatch)}."
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError as exc:
        return f"Ошибка: аргументы {name} — не JSON ({exc.msg}). Повтори вызов с объектом JSON."
    problem = validate_args(tool_schemas[name], args)
    if problem:
        return f"Ошибка: аргументы {name} не по схеме — {problem}. Исправь и повтори вызов."
    try:
        return str(dispatch[name](**args))
    except Exception as exc:  # noqa: BLE001 — сбой инструмента модель видит и может обойти
        return f"Ошибка инструмента {name}: {type(exc).__name__}: {exc}"


def parse_verdict(text: str) -> tuple[str, str]:
    """«OK» / «REVISE: причина» → (вердикт, причина). Первое из слов OK и REVISE в ответе критика решает;
    ни того, ни другого — «UNPARSED»: ревизии нет, как при OK, а сам ответ — в причине для trace."""
    match = VERDICT.search(text)
    if not match:
        return "UNPARSED", text.strip()[:PREVIEW]
    verdict = match.group(1).upper()
    if verdict == "OK":
        return "OK", ""
    lines = match.group(2).strip().splitlines()
    return "REVISE", (lines[0].strip() if lines else "") or "без причины"


def _remaining(deadline: float) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise IterationTimeout
    return left


def _complete(client: Any, deadline: float, **kwargs: Any) -> Any:
    """Вызов модели с таймаутом на остаток итерации и без повторов SDK: повтор съел бы лимит молча."""
    try:
        return client.with_options(timeout=_remaining(deadline), max_retries=0).chat.completions.create(**kwargs)
    except APITimeoutError as exc:
        raise IterationTimeout from exc


def _run_tool(deadline: float, *args: Any) -> str:
    """Инструмент в отдельном потоке: ждём не дольше остатка итерации. Поток зависшего инструмента
    остаётся в фоне — Python не умеет снять его принудительно, но цикл агента не ждёт."""
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(call_tool, *args).result(timeout=_remaining(deadline))
    except FutureTimeout as exc:
        raise IterationTimeout from exc
    finally:
        pool.shutdown(wait=False)


def _critic_input(question: str, thought: str | None, name: str, raw_args: str | None, observation: str,
                  history: list[dict[str, Any]]) -> str:
    past = "\n".join(f"{n}. {e['tool_name']}({e['tool_args']}) → {e['observation'][:300]}"
                     for n, e in enumerate(history[-4:], start=1)) or "—"
    return (f"Задача: {question}\nПлан агента: {thought or 'агент не написал'}\nВызов: {name}({raw_args})\n"
            f"Результат: {observation[:1500]}\nПрошлые шаги:\n{past}")


def run_react_with_reflection(
    question: str,
    tools: list[dict[str, Any]] = TOOLS,
    tool_dispatch: dict[str, Any] = DISPATCH,
    *,
    max_iterations: int = 10,
    timeout_per_iteration_sec: float = 10.0,
    max_revisions: int = 2,
    model_main: str = DEFAULT_MODEL,
    model_premium: str | None = None,       # канон — "gpt-5.4" на шаг после REVISE; пусто — model_main
    model_critic: str | None = None,        # пусто — model_main
    client: Any = None,
) -> dict[str, Any]:
    """{"answer", "steps", "revisions", "usage", "trace"}; лимит итераций или таймаут — answer с явным
    сообщением и "error": "max_iterations" / "timeout"."""
    if not ITERATIONS_RANGE[0] <= max_iterations <= ITERATIONS_RANGE[1]:
        raise ValueError(f"max_iterations={max_iterations}: допустимо {ITERATIONS_RANGE[0]}–{ITERATIONS_RANGE[1]}")
    if not 0 < timeout_per_iteration_sec <= 600 or not 0 <= max_revisions <= 2:
        raise ValueError("timeout_per_iteration_sec — от 0 до 600 с, max_revisions — 0–2")
    client = client or build_openai(get_settings())
    tool_schemas = schemas(tools)
    messages: list[Any] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    usage, critic_usage = Usage(), Usage()
    trace: list[dict[str, Any]] = []
    revisions, premium_next = 0, False

    def finish(answer: str | None, steps: int, error: str | None = None) -> dict[str, Any]:
        total = Usage(usage.prompt_tokens + critic_usage.prompt_tokens,
                      usage.completion_tokens + critic_usage.completion_tokens,
                      usage.total_tokens + critic_usage.total_tokens, usage.calls + critic_usage.calls)
        result = {"answer": answer, "steps": steps, "revisions": revisions,
                  "usage": {**asdict(total), "agent": asdict(usage), "critic": asdict(critic_usage)}, "trace": trace}
        if error:
            result["error"] = error
        log.info("react_usage", steps=steps, revisions=revisions, stopped=error, **result["usage"])
        return result

    for iteration in range(1, max_iterations + 1):
        started = time.monotonic()
        deadline = started + timeout_per_iteration_sec
        model = (model_premium or model_main) if premium_next else model_main
        entry: dict[str, Any] = {"iteration": iteration, "model": model, "thought": None, "tool_name": None,
                                 "tool_args": None, "observation": None, "skipped_calls": [], "critic_verdict": None,
                                 "critic_reason": None, "revision_applied": False}
        try:
            response = _complete(client, deadline, model=model, messages=messages, tools=tools,
                                 tool_choice="auto", parallel_tool_calls=False)
            llm_ms = (time.monotonic() - started) * 1000
            entry.update(usage.add(response.usage), llm_ms=round(llm_ms, 1))
            message = response.choices[0].message
            messages.append(message)
            entry["thought"] = (message.content or "").strip()[:PREVIEW] or None
            calls = message.tool_calls or []
            if not calls:                                    # данных достаточно — финальный ответ
                entry.update(final=True, latency_ms=round(llm_ms, 1))
                trace.append(entry)
                log.info("react_step", **entry)
                return finish(message.content, iteration)
            call, extra = calls[0], calls[1:]                # ровно один инструмент за шаг
            entry.update(tool_name=call.function.name, tool_args=call.function.arguments,
                         skipped_calls=[c.function.name for c in extra])
            tool_started = time.monotonic()
            observation = _run_tool(deadline, call.function.name, call.function.arguments, tool_dispatch, tool_schemas)
            entry.update(observation=observation[:PREVIEW], tool_ms=round((time.monotonic() - tool_started) * 1000, 1))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})
            for skipped in extra:                            # на каждый tool_call API ждёт ответ role="tool"
                messages.append({"role": "tool", "tool_call_id": skipped.id, "content": SKIPPED})

            critic_started = time.monotonic()
            critic = _complete(client, deadline, model=model_critic or model_main, messages=[
                {"role": "system", "content": CRITIC_PROMPT},
                {"role": "user", "content": _critic_input(question, entry["thought"], call.function.name,
                                                          call.function.arguments, observation, trace)}])
            verdict, reason = parse_verdict(critic.choices[0].message.content or "")
            entry.update(critic_verdict=verdict, critic_reason=reason or None,
                         critic_ms=round((time.monotonic() - critic_started) * 1000, 1),
                         critic_tokens=critic_usage.add(critic.usage)["total_tokens"])
            premium_next = False
            if verdict == "REVISE" and revisions < max_revisions:
                revisions += 1
                premium_next = True
                entry["revision_applied"] = True
                messages.append({"role": "system", "content": f"Критика шага {iteration}: {reason} Учти её: исправь "
                                 "следующий шаг или честно сообщи, что задачу решить нельзя."})
            elif verdict == "REVISE":
                entry["critic_reason"] = f"{reason} (лимит ревизий {max_revisions} исчерпан — без приписки)"
            if time.monotonic() > deadline:
                raise IterationTimeout
        except IterationTimeout:
            entry.update(error="timeout", latency_ms=round((time.monotonic() - started) * 1000, 1))
            trace.append(entry)
            log.warning("react_timeout", iteration=iteration, timeout_sec=timeout_per_iteration_sec)
            return finish(TIMEOUT, iteration, "timeout")
        entry["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
        trace.append(entry)
        log.info("react_step", **entry)
    log.warning("react_iteration_limit", max_iterations=max_iterations)
    return finish(ITERATION_LIMIT, max_iterations, "max_iterations")


def run_react(task: str, *, client: Any = None, model: str | None = None, max_iterations: int | None = None,
              timeout_per_iteration_sec: float | None = None) -> dict[str, Any]:
    """run_react_with_reflection с настройками AGENT__* (модели, лимиты) — для CLI и сравнения агентов."""
    cfg = get_settings().agent
    main_model = model or cfg.model or DEFAULT_MODEL
    return run_react_with_reflection(
        task, max_iterations=max_iterations or cfg.max_iterations,
        timeout_per_iteration_sec=timeout_per_iteration_sec or cfg.timeout_per_iteration_sec,
        max_revisions=cfg.max_revisions, model_main=main_model, model_premium=cfg.premium_model or main_model,
        model_critic=cfg.critic_model or main_model, client=client)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ReAct-агент с лимитами и self-reflection (блок 6.2)")
    parser.add_argument("task", help="задача для агента")
    parser.add_argument("--max-iterations", type=int, help="лимит итераций 8–20; пусто — AGENT__MAX_ITERATIONS (10)")
    parser.add_argument("--timeout", type=float, help="секунд на итерацию; пусто — AGENT__TIMEOUT_PER_ITERATION_SEC")
    parser.add_argument("--model", help="модель; пусто — AGENT__MODEL или gpt-5.4-mini")
    parser.add_argument("--trace", action="store_true", help="напечатать trace по итерациям")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # консоль Windows (cp1251) и «→» заглушки Telegram
    setup_logging("INFO", stream=sys.stderr)      # JSON-лог в stderr: react_step на итерацию, react_usage
    try:
        result = run_react(args.task, model=args.model, max_iterations=args.max_iterations,
                           timeout_per_iteration_sec=args.timeout)
    except (ValidationError, ValueError, OpenAIError) as exc:   # настройки, лимиты, провайдер
        print(f"Остановка: {type(exc).__name__}: {exc}")
        return 2
    print(f"Остановка: {result['answer']}" if "error" in result else result["answer"])
    u = result["usage"]
    print(f"\nИтераций: {result['steps']}, ревизий: {result['revisions']}, токенов: {u['total_tokens']} "
          f"(prompt {u['prompt_tokens']}, completion {u['completion_tokens']}; агент {u['agent']['total_tokens']}, "
          f"критик {u['critic']['total_tokens']}), вызовов модели: {u['calls']}")
    if args.trace:
        print("\n--- trace ---", *(json.dumps(item, ensure_ascii=False) for item in result["trace"]), sep="\n")
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
