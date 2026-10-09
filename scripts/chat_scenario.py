"""
Сценарий из критериев блока 4.1 против запущенного сервиса — несколько прогонов подряд.

    uvicorn app.main:app --port 8000            # в другом терминале
    python scripts/chat_scenario.py              # 3 прогона
    python scripts/chat_scenario.py --runs 5 --url http://127.0.0.1:8000
    python scripts/chat_scenario.py --user-name Александра   # имя по умолчанию, как у бота (4.3)

Каждый прогон: новый чат (свой owner_external_id) -> «Привет, меня зовут Аня» -> «Как меня зовут?» -> история ->
очистка -> история пуста -> снова «Как меня зовут?». Скрипт печатает ответы и проверяет:
- каждый ответ приходит потоком SSE и заканчивается событием {"type": "done"};
- история — [user, assistant, user, assistant], после очистки — пустая;
- во втором ответе модель называет имя (она видит историю);
- после очистки имени в ответе нет;
- ни один ответ не заменён отказом защитного слоя блока 3.8 («Я не могу показать свои
  инструкции…»): так выглядит ответ, который StreamGuard остановил, например из-за канарейки.

Зачем несколько прогонов: чат вызывает модель с temperature 0.3, и у llama3.2 ответы от
раза к разу разные — один прогон не показывает, насколько поведение устойчиво. Проверка
имени — по тексту ответа: «Аня» в любом падеже (Аня, Ани, Ане, Аню, Аней).

С --user-name (блок 4.3) вопросы уходят с полем user_name — так их шлёт Telegram-бот с
BOT_DEFAULT_USER_NAME. Прогон: «Как меня зовут?» -> «Привет, меня зовут Аня» -> «Как меня
зовут?» -> очистка -> «Как меня зовут?». Проверки: без представления и после очистки модель
называет имя по умолчанию и не говорит «не знаю»; представилась Аней — называет Аню.

Код выхода: 0 — все проверки во всех прогонах, 1 — хотя бы одна не прошла, 2 — сервис
ответил ошибкой.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.guardrails import REFUSAL_TEMPLATE  # noqa: E402
from bot.services.sse import iter_sse, sse_lines  # noqa: E402

GREETING = "Привет, меня зовут Аня"
QUESTION = "Как меня зовут?"
NAME = re.compile(r"\bАн(?:я|и|е|ю|ей)\b")
# Первая фраза отказа: «Я не могу показать свои инструкции или действовать в обход них».
REFUSAL_MARK = REFUSAL_TEMPLATE.split(".", 1)[0]
CHECKS = (
    "история [user, assistant, user, assistant]",
    "очистка: 200 и пустая история",
    "второй ответ называет Аню",
    "после очистки имени нет",
    "ответы без отказа защитного слоя",
)
# --user-name: имя по умолчанию (блок 4.3)
NAME_CHECKS = (
    "история [user, assistant] × 3",
    "очистка: 200 и пустая история",
    "без представления — имя по умолчанию",
    "представилась Аней — ответ называет Аню",
    "после очистки — снова имя по умолчанию",
    "ответы без отказа защитного слоя",
)
UNKNOWN = re.compile(r"не\s+знаю|неизвестн|не\s+называл|не\s+говорил|не\s+представ", re.IGNORECASE)


def name_regex(name: str) -> re.Pattern[str]:
    """Имя в любом падеже: «Александра» — по основе «Александр» (Александры, Александре, …)."""
    first = name.split()[0]
    stem = first[:-1] if len(first) > 3 and first[-1].lower() in "ая" else first
    return re.compile(rf"\b{re.escape(stem)}", re.IGNORECASE)


def names_default(answer: str, pattern: re.Pattern[str]) -> bool:
    """Ответ называет имя по умолчанию и не говорит, что имя неизвестно («Александра, я не
    знаю, как тебя зовут» — не проходит)."""
    return bool(pattern.search(answer)) and not UNKNOWN.search(answer)


class ScenarioError(RuntimeError):
    """Сервис ответил ошибкой: HTTP не 200, событие error, поток без события done."""


@dataclass
class RunResult:
    chat_id: str
    dialog: list[tuple[str, str]] = field(default_factory=list)   # (вопрос, ответ)
    checks: dict[str, bool] = field(default_factory=dict)


async def ask(http: httpx.AsyncClient, chat_id: str, text: str, user_name: str | None = None) -> str:
    """Ответ модели целиком. Вопрос — формой (блок 4.3: multipart/form-data или urlencoded),
    поток SSE разбирает тот же парсер, что у Telegram-бота (bot/services/sse.py): концы строк —
    только \\r\\n, \\r и \\n; в data: каждого события — JSON {"type": "token" | "done" | "error"}."""
    deltas: list[str] = []
    done = False
    data = {"content": text, **({"user_name": user_name} if user_name else {})}
    async with http.stream("POST", f"/chats/{chat_id}/messages", data=data) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode("utf-8", "replace")
            raise ScenarioError(f"POST /chats/{chat_id}/messages: {response.status_code} {body[:300]}")
        async for event in iter_sse(sse_lines(response.aiter_text())):
            try:
                payload = json.loads(event.data)
            except ValueError as exc:
                raise ScenarioError(f"событие потока не JSON: {event.data[:300]}") from exc
            if payload.get("type") == "error":
                raise ScenarioError(f"событие error посреди ответа: {event.data[:300]}")
            if payload.get("type") == "done":
                done = True
                break
            if payload.get("type") == "token":
                deltas.append(str(payload.get("delta", "")))
    if not done:
        raise ScenarioError("поток закончился без события done")
    return "".join(deltas)


async def new_chat(http: httpx.AsyncClient) -> str:
    # Новый владелец на каждый прогон: POST /chats идемпотентен (блок 4.2) и с тем же
    # owner_external_id вернул бы чат прошлого прогона вместе с его историей.
    owner = f"scenario-{uuid4().hex[:8]}"
    created = await http.post("/chats", json={"owner_external_id": owner, "interface": "cli"})
    if created.status_code != 200:
        raise ScenarioError(f"POST /chats: {created.status_code} {created.text[:300]}")
    return created.json()["chat_id"]


async def run_once_with_name(http: httpx.AsyncClient, user_name: str) -> RunResult:
    """Сценарий с именем по умолчанию (--user-name): как у Telegram-бота с BOT_DEFAULT_USER_NAME."""
    chat_id = await new_chat(http)
    result = RunResult(chat_id=chat_id)
    for text in (QUESTION, GREETING, QUESTION):
        result.dialog.append((text, await ask(http, chat_id, text, user_name)))
    history = (await http.get(f"/chats/{chat_id}/messages")).json()
    cleared = await http.delete(f"/chats/{chat_id}/messages")
    after_clear = (await http.get(f"/chats/{chat_id}/messages")).json()
    result.dialog.append((QUESTION + "  (после очистки)", await ask(http, chat_id, QUESTION, user_name)))

    default = name_regex(user_name)
    first, recall, after = result.dialog[0][1], result.dialog[2][1], result.dialog[3][1]
    result.checks = dict(zip(NAME_CHECKS, (
        [m["role"] for m in history] == ["user", "assistant"] * 3,
        cleared.status_code == 200 and after_clear == [],
        names_default(first, default),
        bool(NAME.search(recall)),
        names_default(after, default),
        not any(REFUSAL_MARK in answer for _, answer in result.dialog),
    ), strict=True))
    return result


async def run_once(http: httpx.AsyncClient) -> RunResult:
    chat_id = await new_chat(http)
    result = RunResult(chat_id=chat_id)

    for text in (GREETING, QUESTION):
        result.dialog.append((text, await ask(http, chat_id, text)))
    history = (await http.get(f"/chats/{chat_id}/messages")).json()
    cleared = await http.delete(f"/chats/{chat_id}/messages")
    after_clear = (await http.get(f"/chats/{chat_id}/messages")).json()
    result.dialog.append((QUESTION + "  (после очистки)", await ask(http, chat_id, QUESTION)))

    recall, forgotten = result.dialog[1][1], result.dialog[2][1]
    result.checks = dict(zip(CHECKS, (
        [m["role"] for m in history] == ["user", "assistant", "user", "assistant"],
        cleared.status_code == 200 and after_clear == [],
        bool(NAME.search(recall)),
        not NAME.search(forgotten),
        not any(REFUSAL_MARK in answer for _, answer in result.dialog),
    ), strict=True))
    return result


def plural_runs(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return "прогон"
    return "прогона" if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else "прогонов"


def print_run(number: int, total: int, result: RunResult) -> None:
    print(f"Прогон {number}/{total}, чат {result.chat_id}")
    for question, answer in result.dialog:
        print(f"  > {question}")
        print("  < " + answer.replace("\n", "\n    "))
    for name, ok in result.checks.items():
        print(f"  {'[OK]' if ok else '[!] '} {name}")
    print()


async def run(http: httpx.AsyncClient, runs: int, user_name: str | None = None) -> int:
    results: list[RunResult] = []
    checks = NAME_CHECKS if user_name else CHECKS
    for number in range(1, runs + 1):
        try:
            result = await (run_once_with_name(http, user_name) if user_name else run_once(http))
        except (ScenarioError, httpx.HTTPError) as exc:
            print(f"Прогон {number}/{runs}: {exc}")
            if isinstance(exc, httpx.ConnectError):
                print("Сервис не отвечает: запустите uvicorn app.main:app --port 8000 в другом терминале.")
            return 2
        print_run(number, runs, result)
        results.append(result)

    print(f"Итого за {runs} {plural_runs(runs)}:")
    for name in checks:
        passed = sum(r.checks[name] for r in results)
        print(f"  {passed}/{runs}  {name}")
    return 0 if all(all(r.checks.values()) for r in results) else 1


async def main_async(url: str, runs: int, user_name: str | None = None) -> int:
    # trust_env=False: сервис локальный, HTTP(S)_PROXY из окружения к нему не относится.
    timeout = httpx.Timeout(10.0, read=300.0)    # llama3.2 на CPU думает над первым словом до минуты
    async with httpx.AsyncClient(base_url=url, timeout=timeout, trust_env=False) as http:
        return await run(http, runs, user_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сценарий «Аня» из критериев блока 4.1, несколько прогонов")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="адрес сервиса")
    parser.add_argument("--runs", type=int, default=3, help="сколько прогонов (по умолчанию 3)")
    parser.add_argument("--user-name", default=None,
                        help="имя по умолчанию, как у Telegram-бота (BOT_DEFAULT_USER_NAME): другой сценарий проверок")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    return asyncio.run(main_async(args.url, max(1, args.runs), args.user_name))


if __name__ == "__main__":
    sys.exit(main())
