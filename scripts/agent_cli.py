"""
Консольный клиент агента для демонстрации: POST /agent/stream (блок 6.4) в читаемом виде и подтверждение
отправки клиенту вопросом «Отправить? [y/n]» вместо второго curl с resume.

    python scripts/agent_cli.py --example                          # задача из examples/requests/agent_stream_input.json
    python scripts/agent_cli.py "Найди срок возврата оплаты и отправь клиенту в чат 12345 напоминание"
    python scripts/agent_cli.py --thread demo-video --example      # свой thread: его можно продолжить после рестарта
    python scripts/agent_cli.py --thread demo-video --resume       # thread ждёт подтверждения — спросить решение
    python scripts/agent_cli.py --example --yes                    # без вопроса: сразу одобрить (или --no — отклонить)

Что печатает: шаги графа — вызов инструмента и его результат, черновик сообщения, паузу с превью, решение
оператора, итог (sent и ответ модели). Сервис — http://127.0.0.1:8000 (--url), заголовок X-Admin-Token —
из переменной окружения ADMIN_TOKEN или из .env; сам токен не печатается. send_telegram_message в проекте —
заглушка: «отправка» — строка [TELEGRAM → chat_id] в логе сервиса (docker compose logs app).
Сразу после docker compose restart app скрипт сам ждёт готовности сервиса (GET /health, до --wait секунд).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "requests" / "agent_stream_input.json"
PREVIEW = 160


def admin_token(env_file: Path = ROOT / ".env") -> str | None:
    """ADMIN_TOKEN из окружения или .env — значение только передаётся в заголовок, в вывод не попадает."""
    token = os.environ.get("ADMIN_TOKEN")
    if not token and env_file.exists():
        from dotenv import dotenv_values

        token = dotenv_values(env_file).get("ADMIN_TOKEN")
    return (token or "").strip() or None


def example_task(path: Path = EXAMPLE) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["input"]["messages"][0]["content"]


def short(text: Any, limit: int = PREVIEW) -> str:
    one = " ".join(str(text).split())
    return one if len(one) <= limit else one[: limit - 1] + "…"


def sse_events(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Строки text/event-stream → события: data: {json}, пустая строка — граница события."""
    for line in lines:
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload:
                yield json.loads(payload)


class Renderer:
    """Печать событий агента. Токены ответа модели — подряд в одной строке, остальное — строкой на шаг."""

    def __init__(self, out: Callable[[str], None] = print) -> None:
        self.out = out
        self.streaming = False
        self.interrupt: dict[str, Any] | None = None
        self.end: dict[str, Any] | None = None

    def _close_tokens(self) -> None:
        if self.streaming:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.streaming = False

    def __call__(self, event: dict[str, Any]) -> None:
        kind = event.get("event")
        if kind == "token":
            if not self.streaming:
                sys.stdout.write("  [модель] ")
                self.streaming = True
            sys.stdout.write(str(event.get("content", "")))
            sys.stdout.flush()
            return
        self._close_tokens()
        if kind == "start":
            self.out(f"thread {event.get('thread_id')} · роль {event.get('user_role')} · {event.get('mode')}")
        elif kind == "node":
            self._node(event.get("node", ""), event.get("update") or {})
        elif kind == "interrupt":
            self.interrupt = (event.get("interrupts") or [{}])[0].get("value") or {}
            self.out(pause_text(self.interrupt))
        elif kind == "end":
            self.end = event
        elif kind == "error":
            self.out(f"  [ошибка] {event.get('message')}")

    def _node(self, node: str, update: dict[str, Any]) -> None:
        for message in update.get("messages") or []:
            for call in message.get("tool_calls") or []:
                args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in call["args"].items())
                self.out(f"  [{node}] вызов {call['name']}({short(args, 120)})")
            if message.get("type") == "tool":
                self.out(f"  [{node}] {message.get('name')} → {short(message.get('content'))}")
        if node == "prepare_send_message" and update.get("draft"):
            self.out(f"  [{node}] черновик готов, отправка ждёт подтверждения")


def pause_text(value: dict[str, Any]) -> str:
    preview = value.get("preview") or {}
    return ("\n  ┌─ ПАУЗА: агент хочет написать клиенту ─────────────────\n"
            f"  │ чат: {preview.get('chat_id')}\n"
            f"  │ текст: {preview.get('text')}\n"
            "  └───────────────────────────────────────────────────────")


def ask(question: str, read: Callable[[str], str] = input) -> bool:
    while True:
        answer = read(f"  {question} [y/n] ").strip().lower()
        if answer in {"y", "yes", "д", "да"}:
            return True
        if answer in {"n", "no", "н", "нет"}:
            return False


def stream(client: httpx.Client, body: dict[str, Any], render: Renderer) -> None:
    with client.stream("POST", "/agent/stream", json=body) as response:
        if response.status_code != 200:
            response.read()
            raise SystemExit(f"Сервис ответил {response.status_code}: {error_text(response)}")
        for event in sse_events(response.iter_lines()):
            render(event)
    render({"event": "flush"})


def error_text(response: httpx.Response) -> str:
    try:
        error = response.json().get("error") or {}
        return f"{error.get('code')}: {error.get('message')}"
    except ValueError:
        return response.text[:200]


def decide(render: Renderer, preset: bool | None, read: Callable[[str], str]) -> bool:
    question = (render.interrupt or {}).get("question") or "Отправить это сообщение клиенту?"
    if preset is not None:
        print(f"  {question} → {'да' if preset else 'нет'} (--{'yes' if preset else 'no'})")
        return preset
    return ask(question, read)


def summary(render: Renderer, started: float) -> str:
    end = render.end or {}
    sent = "да" if end.get("sent") else "нет"
    lines = [f"\nИтог за {time.perf_counter() - started:.0f} с: отправлено клиенту — {sent}"]
    if end.get("pending_approval"):
        lines.append("Thread ждёт решения: python scripts/agent_cli.py --thread <id> --resume")
    if end.get("answer"):
        lines.append(f"Ответ агента: {end['answer']}")
    return "\n".join(lines)


def wait_ready(client: httpx.Client, limit: float, *, sleep: Callable[[float], None] = time.sleep,
               clock: Callable[[], float] = time.monotonic) -> bool:
    """Ждёт GET /health = 200 до limit секунд: сразу после docker compose restart app сервис ещё стартует
    и рвёт соединение (RemoteProtocolError) — вместо traceback скрипт ждёт и пишет, чего ждёт."""
    deadline = clock() + limit
    waiting = False
    while True:
        try:
            if client.get("/health", timeout=5.0).status_code == 200:
                if waiting:
                    print("  сервис готов")
                return True
        except httpx.TransportError:
            pass
        if clock() >= deadline:
            return False
        if not waiting:
            print(f"  Сервис ещё не готов (перезапускается?) — жду до {limit:.0f} с…")
            waiting = True
        sleep(2.0)


def run(args: argparse.Namespace, client: httpx.Client, read: Callable[[str], str] = input) -> int:
    render = Renderer()
    if not wait_ready(client, args.wait):
        print(f"Сервис {args.url} не ответил за {args.wait:.0f} с: docker compose ps app, "
              "docker compose logs app --tail 30", file=sys.stderr)
        return 2
    started = time.perf_counter()
    if args.resume:
        state = client.get(f"/agent/threads/{args.thread}")
        if state.status_code != 200:
            raise SystemExit(f"Сервис ответил {state.status_code}: {error_text(state)}")
        view = state.json()
        if not view.get("pending_approval"):
            if view.get("sent"):
                print(f"thread {args.thread} уже завершён: сообщение отправлено (sent=True), "
                      "повторной отправки не будет")
            else:
                print(f"thread {args.thread} не ждёт подтверждения: отправки не было (sent=False)")
            return 1
        render.interrupt = (view.get("interrupts") or [{}])[0].get("value") or {}
        print(f"thread {args.thread}: checkpoint {view.get('checkpoint_id')} ждёт решения")
        print(pause_text(render.interrupt))
    else:
        task = example_task() if args.example else " ".join(args.task)
        if not task:
            raise SystemExit("Нужна задача: текст в кавычках или --example")
        print(f"Задача: {task}\n")
        body: dict[str, Any] = {"thread_id": args.thread, "input": {"messages": [{"role": "user", "content": task}]}}
        if args.role:
            body["user_role"] = args.role
        stream(client, body, render)
    while render.end is None or render.end.get("pending_approval"):
        if render.interrupt is None:
            break
        try:
            approved = decide(render, args.decision, read)
        except (EOFError, KeyboardInterrupt):
            print(f"\n  Решение отложено: thread {args.thread} ждёт подтверждения и переживёт перезапуск сервиса.\n"
                  f"  Продолжить: python scripts/agent_cli.py --thread {args.thread} --resume")
            return 0
        render.interrupt, render.end = None, None
        resume: dict[str, Any] = {"thread_id": args.thread, "resume": approved}
        if args.role:
            resume["user_role"] = args.role
        stream(client, resume, render)
    print(summary(render, started))
    return 0


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Агент блока 6.4 в консоли: шаги, пауза перед отправкой, решение")
    p.add_argument("task", nargs="*", help="задача агенту")
    p.add_argument("--example", action="store_true", help="задача из examples/requests/agent_stream_input.json")
    p.add_argument("--thread", default=f"cli-{time.strftime('%Y%m%d-%H%M%S')}", help="thread_id (по умолчанию новый)")
    p.add_argument("--resume", action="store_true", help="продолжить thread, который ждёт подтверждения")
    p.add_argument("--role", choices=["read-only", "write-with-approve", "full"], help="роль запуска")
    p.add_argument("--url", default="http://127.0.0.1:8000", help="адрес сервиса")
    p.add_argument("--timeout", type=float, default=900.0, help="ожидание потока, с (Ollama на CPU — минуты)")
    p.add_argument("--wait", type=float, default=90.0, help="ждать готовности сервиса (/health) до, с")
    decision = p.add_mutually_exclusive_group()
    decision.add_argument("--yes", dest="decision", action="store_const", const=True, help="одобрить без вопроса")
    decision.add_argument("--no", dest="decision", action="store_const", const=False, help="отклонить без вопроса")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")       # консоль Windows (cp1251) и рамка паузы
    args = parse(argv)
    token = admin_token()
    if not token:
        print("ADMIN_TOKEN не задан: впишите его в .env (тот же, что у сервиса)", file=sys.stderr)
        return 2
    timeout = httpx.Timeout(10.0, read=args.timeout)
    with httpx.Client(base_url=args.url, headers={"X-Admin-Token": token}, timeout=timeout) as client:
        try:
            return run(args, client)
        except httpx.TransportError as exc:
            print(f"Связь с сервисом {args.url} оборвалась ({type(exc).__name__}): docker compose ps app.\n"
                  f"Если thread ждал подтверждения, он сохранён: --thread {args.thread} --resume", file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
