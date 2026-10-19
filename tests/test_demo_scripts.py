"""
Скрипты демонстрации (итоговая аттестация, docs/demo.md): scripts/preflight.py — готов ли стенд,
scripts/agent_cli.py — агент в консоли с подтверждением отправки. Без сети: httpx.MockTransport и
подставные ответы docker compose.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import agent_cli  # noqa: E402
import preflight  # noqa: E402


def response(status: int = 200, payload: Any = None, text: str = "") -> httpx.Response:
    request = httpx.Request("GET", "http://test")
    if payload is not None:
        return httpx.Response(status, json=payload, request=request)
    return httpx.Response(status, text=text, request=request)


# ---------------------------------------------------------------- preflight
def test_env_check_names_missing_variables_without_values(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text('LLM__OPENAI_API_KEY=ollama\nQDRANT_API_KEY="very-secret-value"\nADMIN_TOKEN=\nBOT_TOKEN=\n',
                   encoding="utf-8")
    checks = preflight.check_env(env)
    assert not checks[0].ok and checks[0].detail == "пусто: ADMIN_TOKEN, INTERNAL_TOKEN"
    assert checks[1].name == "BOT_TOKEN" and not checks[1].critical
    assert all("very-secret-value" not in c.line() for c in checks)
    assert not preflight.check_env(tmp_path / "нет")[0].ok


def test_ollama_models_present_and_missing() -> None:
    tags = {"models": [{"name": "llama3.2:latest"}, {"name": "qwen3:4b-instruct"}, {"name": "bge-m3:latest"}]}
    seen: list[str] = []

    def get(url: str, **kwargs: Any) -> httpx.Response:
        seen.append(url)
        return response(payload=tags)

    checks = preflight.check_ollama(get, "http://localhost:11434/v1", {
        "ответа": "llama3.2", "агента": "qwen3:4b-instruct", "эмбеддингов": "bge-m3", "vision": "gemma3:4b"})
    assert seen == ["http://localhost:11434/api/tags"]
    assert [c.ok for c in checks] == [True, True, True, True, False]
    assert checks[-1].critical is False and "ollama pull gemma3:4b" in checks[-1].detail
    missing = preflight.check_ollama(get, None, {"агента": "qwen3:8b"})
    assert not missing[1].ok and missing[1].critical


def test_ollama_down() -> None:
    def get(url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("refused")

    [check] = preflight.check_ollama(get, "http://127.0.0.1:11434/v1", {"ответа": "llama3.2"})
    assert not check.ok and "запустите Ollama" in check.detail


def test_compose_ps_both_json_formats() -> None:
    rows = [{"Service": "app", "State": "running", "Health": "healthy"}]

    def run(stdout: str) -> Any:
        return lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=stdout, stderr="")

    assert preflight.compose_ps(run("\n".join(json.dumps(r) for r in rows))) == rows
    assert preflight.compose_ps(run(json.dumps(rows))) == rows
    assert preflight.compose_ps(run("")) == []
    failing = lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="no daemon")  # noqa: E731
    with pytest.raises(RuntimeError, match="no daemon"):
        preflight.compose_ps(failing)


def test_containers_states() -> None:
    rows = [{"Service": s, "State": "running", "Health": ""} for s in ("redis", "postgres", "qdrant", "phoenix")]
    rows += [{"Service": "app", "State": "running", "Health": "starting"},
             {"Service": "migrate", "State": "exited", "ExitCode": 0}]
    checks = {c.name: c for c in preflight.check_containers(rows)}
    assert not checks["контейнер app"].ok and checks["контейнер app"].critical       # ещё не healthy
    assert not checks["контейнер bot"].ok and not checks["контейнер bot"].critical   # бот не обязателен
    assert checks["миграции"].ok and checks["контейнер qdrant"].ok


def test_qdrant_points_and_missing_collection() -> None:
    def get(points: int | None, status: int = 200) -> Any:
        return lambda url, **k: response(status, {"result": {"points_count": points}} if status == 200 else {})

    assert preflight.check_qdrant(get(102), "http://127.0.0.1:6333", "k", "kb_docs").ok
    assert not preflight.check_qdrant(get(10), "http://127.0.0.1:6333", "k", "kb_docs").ok
    assert "ingest" in preflight.check_qdrant(get(None, 404), "http://127.0.0.1:6333", "k", "kb_docs").detail
    assert not preflight.check_qdrant(get(None), None, None, "kb_docs").ok


def test_ask_rag_reports_time_and_sources() -> None:
    body = {"answer": "30 минут [1]", "confident": True, "sources": [{"file_name": "account/password-reset.html"}]}
    check = preflight.ask_rag(lambda url, **k: response(payload=body))
    assert check.ok and "account/password-reset.html" in check.detail


def test_collect_and_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(preflight, "check_env", lambda: [preflight.Check(".env", True, "ok")])
    cfg = SimpleNamespace(qdrant_api_key=None, qdrant_url="http://q", chat_vision_model=None,
                          llm=SimpleNamespace(base_url="http://localhost:11434/v1", default_model="llama3.2"),
                          rag=SimpleNamespace(llm_model=None, collection="kb_docs"),
                          agent=SimpleNamespace(model="qwen3:4b-instruct"), embeddings=SimpleNamespace(model="bge-m3"))

    def get(url: str, **kwargs: Any) -> httpx.Response:
        if url.endswith("/api/tags"):
            return response(payload={"models": [{"name": n} for n in ("llama3.2", "qwen3:4b-instruct", "bge-m3")]})
        if "/collections/" in url:
            return response(payload={"result": {"points_count": 103}})
        return response(text="ok")

    rows = [{"Service": s, "State": "running", "Health": "healthy" if s == "app" else ""} for s in preflight.SERVICES]
    checks = preflight.collect(cfg, ask=False, get=get, ps=lambda: rows)
    assert all(c.ok for c in checks), [c.line() for c in checks if not c.ok]
    broken = preflight.collect(cfg, ask=False, get=get, ps=lambda: (_ for _ in ()).throw(OSError("docker нет")))
    assert any(c.name == "Docker" and not c.ok for c in broken)


# ---------------------------------------------------------------- agent_cli
INTERRUPT = {"type": "approve_send_message", "question": "Отправить это сообщение клиенту?",
             "preview": {"chat_id": "12345", "text": "Письмо приходит за 3 дня.", "request_id": "r1"}}
FIRST = [
    {"event": "start", "thread_id": "t1", "user_role": "write-with-approve", "mode": "input"},
    {"event": "node", "node": "call_model", "update": {"messages": [
        {"type": "ai", "content": "",
         "tool_calls": [{"name": "search_knowledge_base", "args": {"query": "автопродление"}}]}]}},
    {"event": "node", "node": "execute_tool", "update": {"messages": [
        {"type": "tool", "name": "search_knowledge_base", "content": "[billing/auto-renewal.docx, близость 0.73] …"}]}},
    {"event": "node", "node": "prepare_send_message", "update": {"draft": {"chat_id": "12345"}, "messages": []}},
    {"event": "interrupt", "interrupts": [{"id": "i1", "value": INTERRUPT}]},
    {"event": "end", "next": ["confirm_and_execute_send_message"], "pending_approval": True, "sent": False,
     "answer": None},
]
SECOND = [
    {"event": "start", "thread_id": "t1", "user_role": "write-with-approve", "mode": "resume"},
    {"event": "token", "node": "call_model", "content": "Напоминание "},
    {"event": "token", "node": "call_model", "content": "отправлено."},
    {"event": "end", "next": [], "pending_approval": False, "sent": True, "answer": "Напоминание отправлено."},
]


def sse_body(events: list[dict[str, Any]]) -> str:
    return "".join(f"data: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events)


def fake_service(bodies: list[dict[str, Any]], thread: dict[str, Any] | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Admin-Token"] == "tok"
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "GET":
            return httpx.Response(200, json=thread)
        body = json.loads(request.content)
        bodies.append(body)
        events = SECOND if "resume" in body else FIRST
        return httpx.Response(200, text=sse_body(events), headers={"content-type": "text/event-stream"})

    return httpx.Client(base_url="http://svc", headers={"X-Admin-Token": "tok"}, transport=httpx.MockTransport(handler))


def test_sse_events_skips_blank_and_comment_lines() -> None:
    lines = ["data: {\"event\": \"start\"}", "", ": ping", "data: {\"event\": \"end\"}"]
    assert [e["event"] for e in agent_cli.sse_events(lines)] == ["start", "end"]


def test_run_asks_and_resumes(capsys: pytest.CaptureFixture[str]) -> None:
    bodies: list[dict[str, Any]] = []
    args = agent_cli.parse(["--thread", "t1", "--example"])
    answers = iter(["может быть", "да"])
    assert agent_cli.run(args, fake_service(bodies), read=lambda prompt: next(answers)) == 0
    assert bodies[0]["input"]["messages"][0]["content"] == agent_cli.example_task()
    assert bodies[1] == {"thread_id": "t1", "resume": True}
    out = capsys.readouterr().out
    assert "вызов search_knowledge_base(query=\"автопродление\")" in out
    assert "search_knowledge_base → [billing/auto-renewal.docx" in out
    assert "ПАУЗА" in out and "текст: Письмо приходит за 3 дня." in out
    assert "[модель] Напоминание отправлено." in out and "отправлено клиенту — да" in out


def test_run_with_no_flag_rejects_without_question() -> None:
    bodies: list[dict[str, Any]] = []
    args = agent_cli.parse(["--thread", "t1", "--no", "Найди срок и отправь в чат 12345"])
    agent_cli.run(args, fake_service(bodies), read=lambda prompt: pytest.fail("вопроса быть не должно"))
    assert bodies[1]["resume"] is False


def test_resume_mode_reads_pending_thread(capsys: pytest.CaptureFixture[str]) -> None:
    bodies: list[dict[str, Any]] = []
    thread = {"pending_approval": True, "checkpoint_id": "1f1c", "interrupts": [{"id": "i1", "value": INTERRUPT}]}
    args = agent_cli.parse(["--thread", "t1", "--resume", "--yes"])
    assert agent_cli.run(args, fake_service(bodies, thread)) == 0
    assert bodies == [{"thread_id": "t1", "resume": True}]
    assert "checkpoint 1f1c ждёт решения" in capsys.readouterr().out


def test_resume_mode_nothing_pending(capsys: pytest.CaptureFixture[str]) -> None:
    args = agent_cli.parse(["--thread", "t1", "--resume", "--yes"])
    assert agent_cli.run(args, fake_service([], {"pending_approval": False, "sent": True})) == 1
    assert "повторной отправки не будет" in capsys.readouterr().out


def test_wait_ready_survives_restart(capsys: pytest.CaptureFixture[str]) -> None:
    """Сразу после docker compose restart app: соединение рвётся, затем 503, затем 200 — без traceback."""
    replies: list[Any] = [httpx.RemoteProtocolError("Server disconnected"), httpx.ConnectError("refused"),
                          httpx.Response(503), httpx.Response(200)]

    def handler(request: httpx.Request) -> httpx.Response:
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    pauses: list[float] = []
    client = httpx.Client(base_url="http://svc", transport=httpx.MockTransport(handler))
    assert agent_cli.wait_ready(client, 90, sleep=pauses.append, clock=lambda: 0.0)
    assert not replies and len(pauses) == 3
    out = capsys.readouterr().out
    assert "жду до 90 с" in out and "сервис готов" in out


def test_wait_ready_gives_up_and_run_explains(capsys: pytest.CaptureFixture[str]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = httpx.Client(base_url="http://svc", transport=httpx.MockTransport(handler))
    ticks = iter([0.0, 5.0, 11.0])
    assert not agent_cli.wait_ready(client, 10, sleep=lambda s: None, clock=lambda: next(ticks))
    args = agent_cli.parse(["--thread", "t1", "--resume", "--wait", "0"])
    assert agent_cli.run(args, client) == 2
    assert "не ответил за 0 с" in capsys.readouterr().err


def test_main_turns_dropped_connection_into_hint(monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    def broken(args: Any, client: Any) -> int:
        raise httpx.RemoteProtocolError("Server disconnected without sending a response.")

    monkeypatch.setenv("ADMIN_TOKEN", "tok")
    monkeypatch.setattr(agent_cli, "run", broken)
    assert agent_cli.main(["--thread", "t5", "--resume"]) == 2
    err = capsys.readouterr().err
    assert "RemoteProtocolError" in err and "--thread t5 --resume" in err and "Traceback" not in err


def test_admin_token_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    env = tmp_path / ".env"
    env.write_text('ADMIN_TOKEN="abc"\n', encoding="utf-8")
    assert agent_cli.admin_token(env) == "abc"
    monkeypatch.setenv("ADMIN_TOKEN", "from-env")
    assert agent_cli.admin_token(env) == "from-env"
    monkeypatch.delenv("ADMIN_TOKEN")
    assert agent_cli.admin_token(tmp_path / "нет") is None


def test_postponed_decision_leaves_thread_pending(capsys: pytest.CaptureFixture[str]) -> None:
    """Ctrl+C или конец ввода на вопросе — resume не отправляется, thread ждёт (демо рестарта)."""
    bodies: list[dict[str, Any]] = []

    def eof(prompt: str) -> str:
        raise EOFError

    assert agent_cli.run(agent_cli.parse(["--thread", "t9", "--example"]), fake_service(bodies), read=eof) == 0
    assert len(bodies) == 1 and "--thread t9 --resume" in capsys.readouterr().out
