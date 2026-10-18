"""
Трафик для Phoenix (блок 5.6, шаг 5): 20+ разных вопросов в POST /rag/query запущенного сервиса —
часть из golden dataset, часть свободные. Каждый запрос — трейс в Phoenix: POST /rag/query →
RAGService.answer → RAGService.retrieve → VectorIndexRetriever (фрагменты и их similarity score) →
эмбеддинг вопроса → OpenAILike.chat (промпт, ответ, токены).

    docker compose up -d --build                   # сервис + Phoenix (или uvicorn с PHOENIX_COLLECTOR_ENDPOINT)
    python scripts/trace_queries.py                # 10 вопросов из golden + 12 свободных
    python scripts/trace_queries.py --golden 15 --url http://127.0.0.1:8000

Свободные вопросы — разного рода, чтобы трейсы показали все ветки:
- base — ответ в базе есть, но вопрос сформулирован не как в golden (английский, разговорный, с опечаткой);
- product — о продукте, но ответа в базе нет: близость выше порога, отказывается модель;
- offtopic — не о продукте: score-guard отказывает без вызова модели, в трейсе нет LLM-span.

Без tests/eval/golden_dataset.json (golden ещё не вычитан) вместо вопросов golden идут вопросы по базе из
citation_experiment (блок 5.5) — их помечает kind=base.

Итог — таблица в консоли и var/trace_queries.jsonl (вопрос, вид, ответ, confident, top_score, время):
по нему scripts/phoenix_hallucinations.py находит эти трейсы и считает долю галлюцинаций на свободных
вопросах. Если установлен arize-phoenix (группа eval), скрипт сам заглядывает в Phoenix: сколько трейсов
пришло и есть ли в них retriever-spans со score и LLM-spans с токенами.

Код выхода: 0 — 20+ ответов и (если проверялся) Phoenix видит трейсы, 1 — ответов меньше или трейсов
нет, 2 — сервис недоступен.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOLDEN = ROOT / "tests" / "eval" / "golden_dataset.json"
OUT = ROOT / "var" / "trace_queries.jsonl"
MIN_QUERIES = 20
DEFAULT_PROJECT = "diploma-fastapi"

FREE: list[tuple[str, str]] = [
    ("base", "Как включить двухфакторную аутентификацию?"),
    ("base", "Какие ограничения на число запросов к API?"),
    ("base", "Можно ли выгрузить все свои данные в Excel?"),
    ("base", "Как передать права владельца аккаунта другому сотруднику?"),
    ("base", "Как сделать, чтобы уведомления не приходили ночью?"),
    ("base", "Работает ли мобильное приложение без интернета?"),
    ("base", "How do I reset my password?"),
    ("base", "скока стоит тариф бизнес в месяц"),
    ("product", "Можно ли оплатить тариф криптовалютой?"),
    ("product", "Есть ли приложение для Apple Watch?"),
    ("offtopic", "Как сварить борщ?"),
    ("offtopic", "Кто выиграл чемпионат мира по футболу 2018 года?"),
]
# Запасные вопросы по базе (scripts/citation_experiment.py, IN_BASE и ещё пять) — пока нет golden.
FALLBACK = [
    "Каков срок возврата оплаты?",
    "Сколько стоит тариф «Бизнес» в месяц?",
    "Сколько действует ссылка для сброса пароля?",
    "Сколько раз сервис повторяет доставку вебхука, если мой сервер не ответил 200?",
    "Как подключить уведомления в Telegram?",
    "Как пригласить сотрудника в команду?",
    "Как удалить аккаунт?",
    "Что делать, если не приходят письма от сервиса?",
    "Как настроить обмен с 1С?",
    "Как восстановить удалённые записи?",
]


def golden_questions(path: Path, count: int) -> list[tuple[str, str]]:
    """Первые count вопросов golden (вид golden); нет golden или в нём меньше — добор запасными вопросами
    по базе (вид base)."""
    if count <= 0:
        return []
    chosen: list[tuple[str, str]] = []
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data["items"] if isinstance(data, dict) else data
        questions = [str(item.get("user_input") or "").strip() for item in items]
        chosen = [("golden", q) for q in questions if q][:count]
    else:
        print(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} нет — вместо golden вопросы по "
              f"базе из citation_experiment", file=sys.stderr)
    taken = {q for _, q in chosen}
    spare = [("base", q) for q in FALLBACK if q not in taken]
    return chosen + spare[:max(0, count - len(chosen))]


def ask(http: httpx.Client, url: str, question: str, attempts: int = 3) -> dict[str, Any]:
    """Один вопрос: ответ сервиса или ошибка. На 429 (RATE_LIMIT_PER_MIN) ждёт Retry-After и повторяет."""
    started = time.perf_counter()
    for attempt in range(1, attempts + 1):
        response = http.post(f"{url}/rag/query", json={"question": question})
        if response.status_code == 429 and attempt < attempts:
            time.sleep(float(response.headers.get("Retry-After") or 10))
            continue
        break
    seconds = round(time.perf_counter() - started, 1)
    if response.status_code != 200:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text[:200]
        return {"status": response.status_code, "error": detail, "seconds": seconds}
    body = response.json()
    return {"status": 200, "answer": body["answer"], "confident": body["confident"], "top_score": body["top_score"],
            "sources": [s.get("source") or s.get("file_name") for s in body["sources"]],
            "cited": [s["id"] for s in body["sources"] if s.get("cited")], "seconds": seconds}


def project_name() -> str:
    """Проект Phoenix сервиса: PHOENIX_PROJECT_NAME из .env, без .env — значение по умолчанию."""
    try:
        from app.core.config import get_settings

        return get_settings().phoenix_project_name
    except Exception:  # noqa: BLE001 — .env неполный (скрипт запускают и без ключей сервиса)
        return DEFAULT_PROJECT


def check_phoenix(endpoint: str, project: str, since: datetime, expected: int, wait: float) -> tuple[bool, str]:
    """Сколько трейсов /rag/query с начала прогона видит Phoenix и раскрываются ли они до retriever и LLM."""
    try:
        from phoenix.client import Client
    except ImportError:
        return True, "проверка Phoenix пропущена: нет arize-phoenix (pip install -r requirements-eval.txt)"
    time.sleep(wait)                          # BatchSpanProcessor отправляет спаны пачкой, раз в ~5 с
    try:
        df = Client(base_url=endpoint).spans.get_spans_dataframe(project_name=project, start_time=since,
                                                                 limit=20000, timeout=30)
    except Exception as exc:  # noqa: BLE001 — Phoenix недоступен или проекта нет: сообщить и выйти с 1
        return False, f"Phoenix {endpoint}, проект {project}: {type(exc).__name__}: {exc}"
    if df.empty:
        return False, f"Phoenix {endpoint}: в проекте {project} нет спанов с начала прогона"
    roots = df[df["name"] == "POST /rag/query"]
    retrievers = df[df["span_kind"] == "RETRIEVER"]
    scored = retrievers["attributes.retrieval.documents"].apply(
        lambda docs: isinstance(docs, list) and any("document.score" in d for d in docs)) if len(retrievers) else []
    llm = df[df["span_kind"] == "LLM"]
    tokens = llm["attributes.llm.token_count.total"].notna().sum() if "attributes.llm.token_count.total" in llm else 0
    embeddings = (df["span_kind"] == "EMBEDDING").sum()
    ok = len(roots) >= expected and int(sum(scored)) > 0 and tokens > 0
    return ok, (f"Phoenix {endpoint}, проект {project}: трейсов POST /rag/query {len(roots)} (ждали {expected}), "
                f"спанов {len(df)}; retriever со score {int(sum(scored))}, LLM с токенами {int(tokens)}, "
                f"embedding {int(embeddings)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="20+ вопросов в /rag/query для трейсов Phoenix (блок 5.6)")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="адрес сервиса")
    parser.add_argument("--golden", type=int, default=10, help="сколько вопросов взять из golden dataset")
    parser.add_argument("--golden-path", type=Path, default=GOLDEN)
    parser.add_argument("--no-free", action="store_true", help="только вопросы golden")
    parser.add_argument("--out", type=Path, default=OUT, help="JSONL с ответами — для phoenix_hallucinations.py")
    parser.add_argument("--timeout", type=float, default=600.0, help="ожидание ответа, с (модель на CPU)")
    parser.add_argument("--phoenix", default="http://127.0.0.1:6006", help="адрес Phoenix; none — не проверять")
    parser.add_argument("--project", default=None, help="проект Phoenix; пусто => PHOENIX_PROJECT_NAME из .env")
    parser.add_argument("--wait", type=float, default=8.0, help="пауза перед проверкой Phoenix, с")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):     # вывод в файл на Windows (cp1251): «₽» и др. — не падать
        sys.stdout.reconfigure(errors="replace")

    queries = golden_questions(args.golden_path, args.golden) + ([] if args.no_free else FREE)
    url = args.url.rstrip("/")
    since = datetime.now(UTC)
    records: list[dict[str, Any]] = []
    print(f"{len(queries)} вопросов → {url}/rag/query\n")
    print(f"{'#':>2}  {'вид':<8} {'код':>3} {'conf':<5} {'top':>5} {'с':>6}  вопрос → ответ")
    try:
        with httpx.Client(timeout=httpx.Timeout(args.timeout, connect=5.0)) as http:
            for number, (kind, question) in enumerate(queries, start=1):
                result = ask(http, url, question)
                record = {"n": number, "kind": kind, "question": question, "at": datetime.now(UTC).isoformat(), **result}
                records.append(record)
                answer = result.get("answer") or json.dumps(result.get("error"), ensure_ascii=False)
                conf = {True: "да", False: "нет"}.get(result.get("confident"), "—")
                top = f"{result['top_score']:.3f}" if "top_score" in result else "—"
                print(f"{number:>2}  {kind:<8} {result['status']:>3} {conf:<5} {top:>5} {result['seconds']:>6}  "
                      f"{question[:50]} → {' '.join(answer.split())[:70]}")
    except httpx.HTTPError as exc:
        print(f"\nСервис {url} недоступен: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"meta": True, "started": since.isoformat(), "url": url, "queries": len(records)}
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [meta, *records]) + "\n", encoding="utf-8")
    answered = [r for r in records if r["status"] == 200]
    by_kind: dict[str, list[int]] = {}
    for r in answered:
        by_kind.setdefault(r["kind"], [0, 0])[0] += 1
        by_kind[r["kind"]][1] += int(bool(r["confident"]))
    print(f"\nОтветов 200: {len(answered)} из {len(records)}; уверенных (confident) по видам: "
          + ", ".join(f"{k} {c}/{n}" for k, (n, c) in by_kind.items()))
    print(f"Ответы → {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    ok = len(answered) >= MIN_QUERIES
    if not ok:
        print(f"Ответов меньше {MIN_QUERIES}: для задания нужно минимум {MIN_QUERIES} трейсов", file=sys.stderr)

    if args.phoenix.lower() != "none":
        project = args.project or project_name()
        phoenix_ok, message = check_phoenix(args.phoenix.rstrip("/"), project, since, len(answered), args.wait)
        print(message)
        ok = ok and phoenix_ok
        if phoenix_ok:
            print(f"Открыть: {args.phoenix.rstrip('/')} → проект {project} → трейс POST /rag/query")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
