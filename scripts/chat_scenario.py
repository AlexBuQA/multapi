"""
Сценарий из критериев блока 4.1 против запущенного сервиса — несколько прогонов подряд.

    uvicorn app.main:app --port 8000            # в другом терминале
    python scripts/chat_scenario.py              # 3 прогона
    python scripts/chat_scenario.py --runs 5 --url http://127.0.0.1:8000

Каждый прогон: новый чат -> «Привет, меня зовут Аня» -> «Как меня зовут?» -> история ->
очистка -> история пуста -> снова «Как меня зовут?». Скрипт печатает ответы и проверяет:
- каждый ответ приходит потоком SSE и заканчивается [DONE];
- история — [user, assistant, user, assistant], после очистки — пустая;
- во втором ответе модель называет имя (она видит историю);
- после очистки имени в ответе нет.

Зачем несколько прогонов: чат вызывает модель с temperature 0.3, и у llama3.2 ответы от
раза к разу разные — один прогон не показывает, насколько поведение устойчиво. Проверка
имени — по тексту ответа: «Аня» в любом падеже (Аня, Ани, Ане, Аню, Аней).

Код выхода: 0 — все проверки во всех прогонах, 1 — хотя бы одна не прошла, 2 — сервис
ответил ошибкой.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass, field

import httpx

GREETING = "Привет, меня зовут Аня"
QUESTION = "Как меня зовут?"
NAME = re.compile(r"\bАн(?:я|и|е|ю|ей)\b")
CHECKS = (
    "история [user, assistant, user, assistant]",
    "очистка: 200 и пустая история",
    "второй ответ называет Аню",
    "после очистки имени нет",
)


class ScenarioError(RuntimeError):
    """Сервис ответил ошибкой: HTTP не 200, событие error, поток без [DONE]."""


@dataclass
class RunResult:
    chat_id: str
    dialog: list[tuple[str, str]] = field(default_factory=list)   # (вопрос, ответ)
    checks: dict[str, bool] = field(default_factory=dict)


async def ask(http: httpx.AsyncClient, chat_id: str, text: str) -> str:
    """Ответ модели целиком: события SSE склеены, строки data: одного события — через \\n."""
    events: list[str] = []
    lines: list[str] = []
    kind = "message"
    done = False

    def finish_event() -> None:
        nonlocal kind, done
        data = "\n".join(lines)
        lines.clear()
        if kind == "error":
            raise ScenarioError(f"событие error посреди ответа: {data[:300]}")
        if data == "[DONE]":
            done = True
        else:
            events.append(data)
        kind = "message"

    async with http.stream("POST", f"/chats/{chat_id}/messages", json={"content": text}) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode("utf-8", "replace")
            raise ScenarioError(f"POST /chats/{chat_id}/messages: {response.status_code} {body[:300]}")
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                kind = line[len("event:"):].strip()
            elif line.startswith("data:"):
                lines.append(line[len("data:"):].removeprefix(" "))
            elif not line and lines:
                finish_event()
    if lines:
        finish_event()
    if not done:
        raise ScenarioError("поток закончился без data: [DONE]")
    return "".join(events)


async def run_once(http: httpx.AsyncClient) -> RunResult:
    created = await http.post("/chats", json={"owner_external_id": "scenario", "interface": "cli"})
    if created.status_code != 200:
        raise ScenarioError(f"POST /chats: {created.status_code} {created.text[:300]}")
    chat_id = created.json()["chat_id"]
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


async def run(http: httpx.AsyncClient, runs: int) -> int:
    results: list[RunResult] = []
    for number in range(1, runs + 1):
        try:
            result = await run_once(http)
        except (ScenarioError, httpx.HTTPError) as exc:
            print(f"Прогон {number}/{runs}: {exc}")
            if isinstance(exc, httpx.ConnectError):
                print("Сервис не отвечает: запустите uvicorn app.main:app --port 8000 в другом терминале.")
            return 2
        print_run(number, runs, result)
        results.append(result)

    print(f"Итого за {runs} {plural_runs(runs)}:")
    for name in CHECKS:
        passed = sum(r.checks[name] for r in results)
        print(f"  {passed}/{runs}  {name}")
    return 0 if all(all(r.checks.values()) for r in results) else 1


async def main_async(url: str, runs: int) -> int:
    # trust_env=False: сервис локальный, HTTP(S)_PROXY из окружения к нему не относится.
    timeout = httpx.Timeout(10.0, read=300.0)    # llama3.2 на CPU думает над первым словом до минуты
    async with httpx.AsyncClient(base_url=url, timeout=timeout, trust_env=False) as http:
        return await run(http, runs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сценарий «Аня» из критериев блока 4.1, несколько прогонов")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="адрес сервиса")
    parser.add_argument("--runs", type=int, default=3, help="сколько прогонов (по умолчанию 3)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    return asyncio.run(main_async(args.url, max(1, args.runs)))


if __name__ == "__main__":
    sys.exit(main())
