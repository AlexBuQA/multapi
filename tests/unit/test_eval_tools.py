"""
Скрипты блока 5.6 вокруг оценки, без сети: вычитка golden dataset (scripts/build_golden.py), трафик для
Phoenix (scripts/trace_queries.py, сервис — httpx.MockTransport) и HallucinationEvaluator по спанам
(scripts/phoenix_hallucinations.py, судья — MockTransport). build_golden и trace_queries работают без
группы eval; тесты phoenix_hallucinations пропускаются, если её нет.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW_COLUMNS = ["id", "user_input", "reference", "reference_contexts", "reference_sources", "synthesizer_name"]


def load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- scripts/build_golden.py
def write_raw(path: Path, count: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        for n in range(1, count + 1):
            writer.writerow({"id": f"raw_{n:03d}", "user_input": f"Вопрос номер {n}?", "reference": f"Ответ {n}.",
                             "reference_contexts": json.dumps([f"Контекст {n}."], ensure_ascii=False),
                             "reference_sources": json.dumps([f"api/doc{n}.md"]),
                             "synthesizer_name": "single_hop_specific_query_synthesizer"})


def write_review(path: Path, items: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump({"items": items}, allow_unicode=True), encoding="utf-8")


def test_build_golden_applies_review(tmp_path):
    script = load_script("build_golden")
    raw, review, out = tmp_path / "raw.csv", tmp_path / "review.yaml", tmp_path / "golden.json"
    write_raw(raw, 33)
    items: dict[str, Any] = {f"raw_{n:03d}": {"keep": True} for n in range(1, 34)}
    items["raw_002"] = {"keep": False, "reason": "дубль raw_001"}
    items["raw_005"] = {"keep": True, "reference": "Исправленный эталон: 30 минут.", "note": "в документе 30 минут"}
    items["raw_007"] = {"keep": True, "user_input": "  Переписанный вопрос?  "}
    write_review(review, items)

    assert script.main(["--raw", str(raw), "--review", str(review), "--out", str(out)]) == 0
    golden = json.loads(out.read_text(encoding="utf-8"))
    assert len(golden) == 32 and "raw_002" not in {g["id"] for g in golden}
    by_id = {g["id"]: g for g in golden}
    assert by_id["raw_005"]["reference"] == "Исправленный эталон: 30 минут." and by_id["raw_005"]["edited"] == ["reference"]
    assert by_id["raw_005"]["note"] == "в документе 30 минут"
    assert by_id["raw_007"]["user_input"] == "Переписанный вопрос?" and by_id["raw_007"]["edited"] == ["user_input"]
    assert by_id["raw_001"]["edited"] == [] and by_id["raw_001"]["reference_contexts"] == ["Контекст 1."]
    assert by_id["raw_001"]["reference_sources"] == ["api/doc1.md"]
    assert set(by_id["raw_001"]) == {"id", "synthesizer", "user_input", "reference", "reference_contexts",
                                     "reference_sources", "edited", "note"}


def test_build_golden_reports_unfinished_review(tmp_path, capsys):
    script = load_script("build_golden")
    raw, review, out = tmp_path / "raw.csv", tmp_path / "review.yaml", tmp_path / "golden.json"
    write_raw(raw, 4)
    write_review(review, {"raw_001": {"keep": True}, "raw_002": {"keep": False},
                          "raw_003": {"keep": True, "user_input": "Вопрос номер 1?"},
                          "raw_009": {"keep": True}})
    assert script.main(["--raw", str(raw), "--review", str(review), "--out", str(out)]) == 1
    err = capsys.readouterr().err
    for problem in ("raw_004: нет решения keep", "raw_009: такого id нет", "raw_002: выброшена без reason",
                    "raw_003: тот же вопрос, что raw_001", "оставлено 2 пар — меньше 30"):
        assert problem in err
    assert not out.exists()


def test_build_golden_check_does_not_write(tmp_path):
    script = load_script("build_golden")
    raw, review, out = tmp_path / "raw.csv", tmp_path / "review.yaml", tmp_path / "golden.json"
    write_raw(raw, 3)
    write_review(review, {f"raw_{n:03d}": {"keep": True} for n in range(1, 4)})
    assert script.main(["--raw", str(raw), "--review", str(review), "--out", str(out), "--min", "3", "--check"]) == 0
    assert not out.exists()
    write_review(review, {"raw_001": {"keep": True, "reference": ""}, "raw_002": {"keep": True},
                          "raw_003": {"keep": True}})
    assert script.main(["--raw", str(raw), "--review", str(review), "--out", str(out), "--min", "1"]) == 1
    assert script.main(["--raw", str(tmp_path / "missing.csv"), "--review", str(review)]) == 2


# ---------------------------------------------------------------- scripts/trace_queries.py
def rag_service(requests: list[str], *, limited_once: bool = False) -> httpx.MockTransport:
    """Сервис /rag/query: «борщ» и «чемпионат» — отказ score-guard, первый запрос — 429 (если limited_once)."""
    state = {"limited": limited_once}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rag/query"
        question = json.loads(request.content)["question"]
        if state["limited"]:
            state["limited"] = False
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "rate_limited"})
        requests.append(question)
        if "борщ" in question or "чемпионат" in question:
            return httpx.Response(200, json={"answer": "По базе не нашёл, могу эскалировать.", "top_score": 0.2,
                                             "confident": False, "sources": []})
        return httpx.Response(200, json={"answer": "Ответ.\nИсточники: [1]", "top_score": 0.71, "confident": True,
                                         "sources": [{"id": 1, "file_name": "a.md", "source": "api/a.md", "cited": True}]})

    return httpx.MockTransport(handler)


def test_trace_queries_golden_and_free_questions(tmp_path, monkeypatch, capsys):
    script = load_script("trace_queries")
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([{"user_input": f"Вопрос golden {n}?"} for n in range(1, 4)], ensure_ascii=False),
                      encoding="utf-8")
    asked: list[str] = []
    transport = rag_service(asked, limited_once=True)
    real_client = httpx.Client
    monkeypatch.setattr(script.httpx, "Client", lambda **kwargs: real_client(transport=transport, **kwargs))
    monkeypatch.setattr(script.time, "sleep", lambda seconds: None)
    out = tmp_path / "queries.jsonl"

    code = script.main(["--golden", "10", "--golden-path", str(golden), "--out", str(out), "--phoenix", "none"])
    assert code == 0
    kinds = [kind for kind, _ in script.golden_questions(golden, 10)]
    assert kinds == ["golden"] * 3 + ["base"] * 7                   # golden мал — добор вопросами по базе
    assert len(asked) == 22 and asked[:3] == ["Вопрос golden 1?", "Вопрос golden 2?", "Вопрос golden 3?"]
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["meta"] and lines[0]["queries"] == 22
    records = lines[1:]
    assert {r["kind"] for r in records} == {"golden", "base", "product", "offtopic"}
    assert all(r["status"] == 200 for r in records)
    offtopic = [r for r in records if r["kind"] == "offtopic"]
    assert all(r["confident"] is False for r in offtopic) and records[0]["cited"] == [1]
    assert "Ответов 200: 22 из 22" in capsys.readouterr().out


def test_trace_queries_without_golden_and_too_few(tmp_path, monkeypatch):
    script = load_script("trace_queries")
    transport = rag_service([])
    real_client = httpx.Client
    monkeypatch.setattr(script.httpx, "Client", lambda **kwargs: real_client(transport=transport, **kwargs))
    assert script.golden_questions(tmp_path / "none.json", 3) == [("base", q) for q in script.FALLBACK[:3]]
    code = script.main(["--golden", "0", "--golden-path", str(tmp_path / "none.json"),
                        "--out", str(tmp_path / "q.jsonl"), "--phoenix", "none"])
    assert code == 1                                                 # 12 свободных — меньше 20


def test_trace_queries_service_down(tmp_path, monkeypatch):
    script = load_script("trace_queries")

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    real_client = httpx.Client
    monkeypatch.setattr(script.httpx, "Client",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(refuse), **kwargs))
    assert script.main(["--out", str(tmp_path / "q.jsonl"), "--phoenix", "none"]) == 2


# ---------------------------------------------------------------- спаны Phoenix
QUESTION = "Каков срок возврата оплаты?"
FREE = "Есть ли приложение для Apple Watch?"
GUARD = "Как сварить борщ?"


def spans_frame() -> Any:
    """Спаны трёх трейсов /rag/query (как их отдаёт phoenix.client) и одного /chat."""
    pd = pytest.importorskip("pandas")
    start = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)
    rows: list[dict[str, Any]] = []

    def span(trace: str, span_id: str, parent: str | None, name: str, kind: str, offset: int, **attrs: Any) -> None:
        rows.append({"context.trace_id": trace, "context.span_id": span_id, "parent_id": parent, "name": name,
                     "span_kind": kind, "start_time": start + timedelta(seconds=offset),
                     "attributes.input.value": attrs.get("input"), "attributes.llm.input_messages": attrs.get("messages"),
                     "attributes.llm.output_messages": attrs.get("output")})

    def rag_trace(trace: str, offset: int, question: str, answer: str | None) -> None:
        span(trace, f"{trace}-root", None, "POST /rag/query", "UNKNOWN", offset)
        span(trace, f"{trace}-answer", f"{trace}-root", "RAGService.answer", "CHAIN", offset,
             input=json.dumps({"question": question}, ensure_ascii=False))
        span(trace, f"{trace}-retrieve", f"{trace}-answer", "RAGService.retrieve", "CHAIN", offset)
        span(trace, f"{trace}-retriever", f"{trace}-retrieve", "VectorIndexRetriever.retrieve", "RETRIEVER", offset)
        if answer is None:
            return
        messages = [{"message.role": "system", "message.content": "Отвечай только по фрагментам."},
                    {"message.role": "user", "message.content": f"[1] (refunds.md)\nВернуть оплату можно в течение "
                                                                f"14 дней.\n\nВопрос: {question}"}]
        output = [{"message.role": "assistant", "message.content": answer}]
        span(trace, f"{trace}-chat", f"{trace}-answer", "OpenAILike.chat", "LLM", offset + 1)
        span(trace, f"{trace}-chat-inner", f"{trace}-chat", "OpenAILike.chat", "LLM", offset + 1,
             messages=messages, output=output)
        # тот же вызов глазами инструментации OpenAI SDK — не под RAGService.answer, его не берём
        span(trace, f"{trace}-sdk", f"{trace}-root", "ChatCompletion", "LLM", offset + 1, messages=messages, output=output)

    rag_trace("t1", 0, QUESTION, "Вернуть оплату можно в течение 14 дней.\nИсточники: [1]")
    rag_trace("t2", 10, FREE, "Да, приложение для Apple Watch вышло в 2025 году.\nИсточники: [1]")
    rag_trace("t3", 20, GUARD, None)
    span("t4", "t4-root", None, "POST /chats/{chat_id}/messages", "UNKNOWN", 30)
    span("t4", "t4-llm", "t4-root", "ChatCompletion", "LLM", 30,
         messages=[{"message.role": "user", "message.content": "Привет"}],
         output=[{"message.role": "assistant", "message.content": "Здравствуйте"}])
    return pd.DataFrame(rows)


def judge_transport(seen: list[dict[str, Any]]) -> httpx.MockTransport:
    """Судья phoenix.evals (structured output): hallucinated, если в ответе есть число, которого нет во входе."""
    import re

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        text = body["messages"][-1]["content"]
        text = text if isinstance(text, str) else " ".join(part.get("text", "") for part in text)
        given = text.split("<input>", 1)[1].split("</input>", 1)[0]
        answer = text.split("<output>", 1)[1].split("</output>", 1)[0]
        extra = [n for n in re.findall(r"\d+", answer) if n not in given]
        verdict = {"explanation": f"нет во входе: {extra}" if extra else "всё есть во входе",
                   "label": "hallucinated" if extra else "grounded"}
        return httpx.Response(200, json={
            "id": "j", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(verdict, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

    return httpx.MockTransport(handler)


def phoenix_llm(seen: list[dict[str, Any]]) -> Any:
    pytest.importorskip("phoenix.evals", reason="группа eval: pip install -r requirements-eval.txt")
    from phoenix.evals import LLM

    transport = judge_transport(seen)
    return LLM(provider="openai", model="judge", api_key="test-key", base_url="http://judge.invalid/v1",
               sync_client_kwargs={"http_client": httpx.Client(transport=transport)},
               async_client_kwargs={"http_client": httpx.AsyncClient(transport=transport)})


def test_rag_answers_from_spans():
    script = load_script("phoenix_hallucinations")
    rows = script.rag_answers(spans_frame(), {QUESTION: "golden", FREE: "product"})
    assert [(r["question"], r["kind"], r["guard"]) for r in rows] == [
        (QUESTION, "golden", False), (FREE, "product", False), (GUARD, "unknown", True)]
    first = rows[0]
    assert first["root_span_id"] == "t1-root"
    assert first["input"].startswith("system: Отвечай только по фрагментам.\n\nuser: [1] (refunds.md)")
    assert first["output"] == "Вернуть оплату можно в течение 14 дней.\nИсточники: [1]"
    assert "input" not in rows[2]                                    # отказ score-guard — модель не вызывалась


def test_hallucination_evaluator_labels_and_summary(tmp_path):
    script = load_script("phoenix_hallucinations")
    rows = script.rag_answers(spans_frame(), {QUESTION: "golden", FREE: "product"})
    seen: list[dict[str, Any]] = []
    script.evaluate(rows, phoenix_llm(seen), {"temperature": 0.0, "max_tokens": 512})
    assert [r.get("label") for r in rows] == ["grounded", "hallucinated", None]
    assert rows[1]["score"] == 1.0 and "2025" in rows[1]["explanation"]
    assert len(seen) == 2 and seen[0]["response_format"]["type"] == "json_schema"
    assert seen[0]["temperature"] == 0.0 and seen[0]["max_tokens"] == 512
    assert "Отвечай только по фрагментам" in seen[0]["messages"][-1]["content"]
    lines = script.summary(rows)
    assert lines[0] == "свободные вопросы: трейсов 2, отказ score-guard 1, оценено 1, hallucinated 1 (100%)"
    assert lines[1] == "golden: трейсов 1, отказ score-guard 0, оценено 1, hallucinated 0 (0%)"
    out = tmp_path / "h.csv"
    script.write_csv(rows, out)
    with out.open(encoding="utf-8") as fh:
        table = list(csv.DictReader(fh))
    assert [r["label"] for r in table] == ["grounded", "hallucinated", "guard"] and list(table[0]) == list(script.COLUMNS)


def test_phoenix_hallucinations_main_annotates_root_spans(tmp_path, monkeypatch):
    pytest.importorskip("phoenix.client", reason="группа eval")
    script = load_script("phoenix_hallucinations")
    frame = spans_frame()
    logged: dict[str, Any] = {}
    asked: dict[str, Any] = {}

    class Spans:
        def get_spans_dataframe(self, **kwargs: Any) -> Any:
            asked.update(kwargs)
            return frame

        def log_span_annotations_dataframe(self, **kwargs: Any) -> None:
            logged.update(kwargs)

    class Client:
        def __init__(self, base_url: str) -> None:
            self.base_url, self.spans = base_url, Spans()

    import phoenix.client

    monkeypatch.setattr(phoenix.client, "Client", Client)
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(script, "build_llm", lambda cfg: (phoenix_llm(seen), {"temperature": 0.0}))
    queries = tmp_path / "q.jsonl"
    started = datetime(2026, 10, 12, 8, 59, tzinfo=UTC)
    queries.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"meta": True, "started": started.isoformat()}, {"kind": "golden", "question": QUESTION},
        {"kind": "product", "question": FREE}, {"kind": "offtopic", "question": GUARD}]), encoding="utf-8")
    out = tmp_path / "h.csv"

    assert script.main(["--project", "multapi-test", "--queries", str(queries), "--out", str(out)]) == 0
    assert asked["project_name"] == "multapi-test" and asked["start_time"] == started
    annotations = logged["dataframe"]
    assert list(annotations["span_id"]) == ["t1-root", "t2-root"] and list(annotations["label"]) == [
        "grounded", "hallucinated"]
    assert logged["annotation_name"] == "hallucination" and logged["annotator_kind"] == "LLM"
    assert out.is_file()


def test_phoenix_hallucinations_without_traces(tmp_path, monkeypatch):
    pytest.importorskip("phoenix.client", reason="группа eval")
    pd = pytest.importorskip("pandas")
    script = load_script("phoenix_hallucinations")
    import phoenix.client

    class Client:
        def __init__(self, base_url: str) -> None:
            self.spans = type("Spans", (), {"get_spans_dataframe": lambda self, **kw: pd.DataFrame()})()

    monkeypatch.setattr(phoenix.client, "Client", Client)
    assert script.main(["--queries", str(tmp_path / "none.jsonl"), "--no-annotate"]) == 1


def test_trace_queries_checks_tree_in_phoenix(monkeypatch):
    """После трафика скрипт смотрит в Phoenix: трейсы /rag/query пришли, retriever со score, LLM с токенами."""
    pytest.importorskip("phoenix.client", reason="группа eval")
    script = load_script("trace_queries")
    frame = spans_frame()
    frame["attributes.retrieval.documents"] = [
        [{"document.id": "n1", "document.score": 0.71, "document.content": "…"}] if kind == "RETRIEVER" else None
        for kind in frame["span_kind"]]
    frame["attributes.llm.token_count.total"] = [120.0 if name == "ChatCompletion" else None for name in frame["name"]]
    asked: dict[str, Any] = {}

    class Client:
        def __init__(self, base_url: str) -> None:
            asked["base_url"] = base_url
            self.spans = self

        def get_spans_dataframe(self, **kwargs: Any) -> Any:
            asked.update(kwargs)
            return frame

    import phoenix.client

    monkeypatch.setattr(phoenix.client, "Client", Client)
    since = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)
    ok, message = script.check_phoenix("http://127.0.0.1:6006", "multapi", since, 3, wait=0)
    assert ok and asked["project_name"] == "multapi" and asked["start_time"] == since
    assert "трейсов POST /rag/query 3 (ждали 3)" in message and "retriever со score 3" in message
    assert "LLM с токенами 3" in message
    ok, message = script.check_phoenix("http://127.0.0.1:6006", "multapi", since, 5, wait=0)
    assert not ok                                                    # не все запросы дошли до Phoenix


# ---------------------------------------------------------------- scripts/eval_report.py
def write_run(results: Path, stamp: str, label: str, means: dict[str, float | None], *, judge: str = "qwen3:4b-instruct",
              golden: str = "abc", citation_prompt: str = "c1", **config: Any) -> None:
    base_config = {"collection": "kb_docs", "chunking": "recursive", "chunk_size": 512, "chunk_overlap": 64,
                   "candidates_top_k": 10, "similarity_top_k": 5, "score_threshold": 0.5, "llm_model": "llama3.2",
                   "temperature": 0.1, "embeddings_model": "bge-m3",
                   "rerank": {"enabled": False, "provider": None, "model": None}}
    data = {"label": label, "timestamp": stamp, "config": {**base_config, **config},
            "judge": {"judge_model": judge, "judge_host": "127.0.0.1", "judge_embeddings": "bge-m3",
                      "citation_prompt": citation_prompt},
            "golden": {"path": "golden_dataset.json", "items": 30, "sha256": golden},
            "metrics": {m: {"mean": means.get(m), "n": 30, "nan": 0} for m in
                        ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "has_citation")},
            "latency_ms": {"mean": 9000, "p50": 8000, "p95": 15000}, "refusals": {"score_guard": 2, "model": 1}}
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{stamp}_{label}.json").write_text(json.dumps(data), encoding="utf-8")


def test_eval_report_ab_table(tmp_path, capsys):
    script = load_script("eval_report")
    results = tmp_path / "results"
    write_run(results, "2026-10-13_0900", "baseline", {"faithfulness": 0.70, "answer_relevancy": 0.80, "has_citation": 0.9})
    write_run(results, "2026-10-13_1100", "chunk_128", {"faithfulness": 0.75, "answer_relevancy": 0.78}, chunk_size=128)
    write_run(results, "2026-10-13_1300", "rerank", {"faithfulness": 0.72}, judge="other-judge",
              rerank={"enabled": True, "provider": "cross-encoder", "model": "BAAI/bge-reranker-v2-m3"})
    (results / "2026-10-13_1400_x.partial.jsonl").write_text("{}", encoding="utf-8")   # журнал — не прогон
    assert script.main(["--results", str(results)]) == 0
    table = capsys.readouterr().out
    lines = [line for line in table.splitlines() if line.startswith("| ")]
    assert lines[0].startswith("| Вариант | Изменение | faithfulness |")              # строка «|---|» — не «| »
    assert lines[1].startswith("| baseline | baseline | 0.700 | 0.800 | — | — | 0.900 | 9000 / 15000 | 2 / 1 |")
    assert lines[2].startswith("| chunk_128 | chunk_size 512→128 | 0.750 (+0.050) | 0.780 (-0.020) |")
    assert "| rerank ⚠️ другой судья | re-ranker off→on | 0.720 (+0.020) |" in lines[3]
    assert "`2026-10-13_1100_chunk_128.csv`" in lines[2] and "golden: вопросов 30" in table

    assert script.main(["--results", str(results), "--labels", "chunk_128", "baseline"]) == 0
    rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith("| ")]
    assert rows[1].startswith("| chunk_128 | baseline |") and "chunk_size 128→512" in rows[2]
    assert script.main(["--results", str(results), "--labels", "nope"]) == 2
    assert script.main(["--results", str(tmp_path / "empty")]) == 2



def test_eval_report_sources_fallback_missing_in_old_runs_means_off(tmp_path, capsys):
    """В JSON прогонов до блока 5.6 ключа sources_fallback нет — тогда строку источников не дописывали."""
    script = load_script("eval_report")
    results = tmp_path / "results"
    write_run(results, "2026-10-12_2147", "baseline", {"has_citation": 0.911})
    write_run(results, "2026-10-14_0100", "final", {"has_citation": 1.0}, sources_fallback=True)
    write_run(results, "2026-10-14_0900", "same", {"has_citation": 0.9}, sources_fallback=False)
    assert script.main(["--results", str(results), "--labels", "baseline", "final", "same"]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("| ")]
    assert lines[2].startswith("| final | sources_fallback False→True |")
    assert lines[3].startswith("| same | — |")


def test_eval_report_flags_other_citation_prompt(tmp_path, capsys):
    """has_citation с другим промптом — другая шкала: сдвиг есть, но строка помечена."""
    script = load_script("eval_report")
    results = tmp_path / "results"
    write_run(results, "2026-10-12_2040", "smoke", {"has_citation": 0.5}, citation_prompt="old")
    write_run(results, "2026-10-13_0400", "baseline", {"has_citation": 0.96})
    assert script.main(["--results", str(results), "--labels", "baseline", "smoke"]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("| smoke")]
    assert lines and "⚠️ другой промпт has_citation" in lines[0]

def test_eval_report_worst_and_diagnosis(tmp_path, capsys):
    script = load_script("eval_report")
    path = tmp_path / "run.csv"
    rows = [
        ("q1", "0.2", "0.1", ["Про тарифы."], ["billing/faq-billing.md"], ["billing/refunds_v2.docx"]),
        ("q2", "0.3", "1.0", ["Возврат — 14 дней."], ["billing/refunds_v2.docx"], ["billing/refunds_v2.docx"]),
        ("q3", "1.0", "1.0", ["Ссылка — 30 минут."], ["account/password-reset.html"], []),
        ("q4", "", "0.0", [], [], []),
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "user_input", "reference", "response", "faithfulness",
                                                "answer_relevancy", "context_precision", "context_recall",
                                                "has_citation", "retrieved_files", "reference_sources",
                                                "retrieved_contexts"])
        writer.writeheader()
        for rid, faith, recall, contexts, files, expected in rows:
            writer.writerow({"id": rid, "user_input": f"Вопрос {rid}?", "reference": "Эталон.", "response": "Ответ.",
                             "faithfulness": faith, "answer_relevancy": "0.9", "context_precision": "0.5",
                             "context_recall": recall, "has_citation": "1.0",
                             "retrieved_files": json.dumps(files), "reference_sources": json.dumps(expected),
                             "retrieved_contexts": json.dumps(contexts, ensure_ascii=False)})
    assert script.main(["--worst", str(path), "-n", "3"]) == 0
    out = capsys.readouterr().out
    assert "ответов 4, faithfulness есть у 3 (без него — 1" in out
    assert out.index("**1. q1**") < out.index("**2. q2**") < out.index("**3. q3**") and "q4**" not in out
    assert "диагноз (2×2): retriever" in out.split("**2. q2**")[0]
    assert "диагноз (2×2): generation" in out.split("**2. q2**")[1].split("**3. q3**")[0]
    assert "(эталон — `billing/refunds_v2.docx`)" in out and "«Про тарифы.…»" in out


def test_eval_report_judge_free_checks(tmp_path, capsys):
    """hit@5, MRR@5, маркеры [N] и куда они указывают — по CSV рядом с JSON прогона, без судьи."""
    script = load_script("eval_report")
    results = tmp_path / "results"
    write_run(results, "2026-10-13_0900", "baseline", {"has_citation": 0.75})
    write_run(results, "2026-10-13_1100", "rerank", {"has_citation": 1.0})       # без CSV — прочерки
    files = ["a.md", "ref.md", "b.md"]
    rows = [  # (ответ, вердикт судьи, файлы-источники эталона)
        ("Ответ [2].", "1.0", ["ref.md"]),          # ссылка на эталон, эталон на 2-м месте
        ("Ответ.\nИсточники: [1], [3]", "1.0", ["ref.md"]),   # ссылки мимо эталона
        ("Ответ без ссылок.", "1.0", ["ref.md"]),    # судья сказал yes, маркеров нет — расхождение
        ("Ответ [1].", "", ["zzz.md"]),              # эталона среди фрагментов нет; судья упал — NaN
    ]
    with (results / "2026-10-13_0900_baseline.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "response", "has_citation", "retrieved_files", "reference_sources"])
        writer.writeheader()
        for number, (response, cited, expected) in enumerate(rows, start=1):
            writer.writerow({"id": f"q{number}", "response": response, "has_citation": cited,
                             "retrieved_files": json.dumps(files), "reference_sources": json.dumps(expected)})
    assert script.main(["--results", str(results), "--labels", "baseline", "rerank"]) == 0
    out = capsys.readouterr().out
    assert "| baseline | 3/4 (0.750) | 0.375 | 3.0 | 3/4 | 2/3 | 1/3 (0.333) |" in out    # MRR: (1/2 + 1/2 + 1/2) / 4
    assert "| rerank | — | — | — | — | — | — |" in out

    assert script.main(["--worst", str(results / "2026-10-13_0900_baseline.csv"), "-n", "4"]) == 0


def test_eval_report_worst_shows_where_citations_point(tmp_path, capsys):
    script = load_script("eval_report")
    path = tmp_path / "run.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "user_input", "response", "faithfulness", "context_recall",
                                                "retrieved_files", "reference_sources", "retrieved_contexts"])
        writer.writeheader()
        writer.writerow({"id": "q1", "user_input": "Вопрос?", "response": "Ответ.\nИсточники: [2][7]",
                         "faithfulness": "0.0", "context_recall": "1.0",
                         "retrieved_files": json.dumps(["ref.md", "faq.md"]), "reference_sources": json.dumps(["ref.md"]),
                         "retrieved_contexts": json.dumps(["Фрагмент."], ensure_ascii=False)})
    assert script.main(["--worst", str(path)]) == 0
    assert "- ссылки ответа: [2] `faq.md`, [7] —; фрагмент эталона — [1]" in capsys.readouterr().out
    # фрагмента эталона в пятёрке нет, а context_recall высокий (щедрый судья) — всё равно retriever
    rows = script.read_rows(path)
    rows[0]["retrieved_files"] = json.dumps(["login.md", "faq.md"])
    assert script.diagnosis(rows[0]).startswith("retriever: фрагмента файла-источника эталона нет среди пяти "
                                                "(context_recall=1.00")


def test_eval_report_thresholds(tmp_path, capsys):
    script = load_script("eval_report")
    results = tmp_path / "results"
    write_run(results, "2026-10-13_1500", "final", {"faithfulness": 0.81, "answer_relevancy": 0.74, "has_citation": 0.97})
    write_run(results, "2026-10-13_1600", "weak", {"faithfulness": 0.65, "answer_relevancy": 0.74, "has_citation": 0.97})
    assert script.main(["--check", str(results / "2026-10-13_1500_final.json")]) == 0
    assert script.main(["--check", str(results / "2026-10-13_1600_weak.json")]) == 1
    assert "FAIL faithfulness = 0.650 (нужно > 0.7)" in capsys.readouterr().out


def test_committed_golden_matches_raw_and_review():
    """Golden в репозитории собран из актуальных сырого CSV и вычитки: правка golden_review.yaml без
    python scripts/build_golden.py (или новая генерация без вычитки) — тест падает."""
    raw = ROOT / "tests" / "eval" / "golden_dataset_raw.csv"
    review = ROOT / "tests" / "eval" / "golden_review.yaml"
    golden = ROOT / "tests" / "eval" / "golden_dataset.json"
    if not (raw.is_file() and review.is_file() and golden.is_file()):
        pytest.skip("golden dataset ещё не собран (scripts/generate_testset.py, build_golden.py)")
    script = load_script("build_golden")
    items, problems = script.build(script.read_raw(raw), script.read_review(review))
    assert problems == [], "вычитка неполная — python scripts/build_golden.py --check"
    assert len(items) >= 30
    assert json.loads(golden.read_text(encoding="utf-8")) == items, "golden устарел — python scripts/build_golden.py"
