"""
Прогон оценки качества по golden dataset (блок 3.7). Это не pytest-тест, а отдельный CLI:
он медленный и обращается к настоящим моделям. Запуск из корня проекта:

    python eval/run_evaluation.py                      # судья из .env (учебная связка — OpenRouter)
    python eval/run_evaluation.py --golden eval/golden_dataset.json --judge qwen3:4b-instruct
    python eval/run_evaluation.py --judge gpt-5.2 --judge-base-url https://api.openai.com/v1 \
        --out eval/runs/2026-10-08.json        # ключ судьи — в EVAL_JUDGE_API_KEY

Судью можно задать в .env: EVAL_JUDGE_MODEL, EVAL_JUDGE_BASE_URL, EVAL_JUDGE_API_KEY,
EVAL_JUDGE_REASONING и EVAL_JUDGE_MAX_TOKENS; флаги командной строки важнее .env.
Если base_url судьи внешний (OpenAI, OpenRouter), а в .env задан LLM__PROXY_URL, судья ходит через
прокси; локальный Ollama — напрямую. --model-base-url переключает на внешний API и
модель под тестом (например, gpt-4.1-mini на OpenAI), не трогая .env.

Что происходит:
1. Ответы. FastAPI-приложение запускается в этом же процессе (lifespan) и вызывается
   через httpx.AsyncClient + ASGITransport: POST /chat с temperature=0, как настоящий
   клиент. Кеш Redis на время прогона выключен — оценивается ответ модели, а не кеша.
   Модель под тестом — LLM__DEFAULT_MODEL из .env или --model.
2. Оценка. Судья — отдельная модель (--judge), промпт G-Eval reason-then-score
   (eval/judge.py), temperature=0, response_format={"type": "json_object"}. Судья
   видит эталон и те же статьи руководства, что сервис подставил модели (их находит
   тот же build_messages): верный факт из статьи сверх эталона — не выдумка.
   Сначала все ответы, потом все оценки: на локальном Ollama модели не перезагружаются
   на каждом вопросе.
3. Проверки без LLM: доля найденных expected_keywords (keyword_recall) и запрещённые
   слова must_not_contain.
4. Артефакт — eval/runs/<YYYY-MM-DD>.json (второй прогон за день — с временем в имени):
   run_id, timestamp, модели, версия golden, оценки по каждому кейсу и агрегаты.
   Читается jq: jq '.aggregates.correctness_avg' eval/runs/2026-10-07.json

Пороги «можно ли релизить» проверяет eval/check_thresholds.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
for path in (ROOT, EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# До импорта сервиса: настройки и логирование читаются при импорте app.main. Логи
# сервиса — только ошибки, чтобы не заслонять ход прогона.
os.environ.setdefault("LOG_LEVEL", "ERROR")

import httpx  # noqa: E402
import openai  # noqa: E402
from openai import AsyncOpenAI, DefaultAsyncHttpxClient  # noqa: E402

from judge import CRITERIA, JUDGE_PROMPT_VERSION, judge_answer  # noqa: E402

DEFAULT_JUDGE = "qwen3:4b-instruct"
DEFAULT_JUDGE_MAX_TOKENS = 1200
# Повторы вызова судьи — свои, а не SDK: SDK повторяет 429 через доли секунды, а у
# бесплатных моделей OpenRouter 429 значит «канал провайдера перегружен, повторите
# чуть позже» — быстрые повторы только расходуют дневной лимит ключа.
RATE_LIMIT_WAITS = (20.0, 40.0, 60.0)    # паузы при 429, если провайдер не прислал Retry-After
CONNECTION_WAITS = (5.0, 10.0)           # паузы при ошибке соединения и 5xx
FREE_MODEL_INTERVAL = 3.1                # бесплатные модели OpenRouter: не больше 20 запросов в минуту
STOP_AFTER_RATE_LIMITED = 2              # столько вопросов подряд без ответа после всех пауз — судья недоступен
PREVIEW = 70
ENV_FILE = ROOT / ".env"


def describe_exception(exc: BaseException) -> str:
    """Исключение с первопричиной: OpenAI SDK показывает «Connection error.», а за ним
    прячется, например, ProxyError: 407 Proxy Authentication Required."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__}: {current}".strip().rstrip(":")
        if not parts or parts[-1] != text:
            parts.append(text)
        current = current.__cause__ or current.__context__
    return " <- ".join(parts)


def env_value(name: str, env_file: Path | None = None) -> str | None:
    """Переменная окружения, а если её нет — значение из .env (как у настроек сервиса).
    Пустое значение — то же, что его нет."""
    value = os.environ.get(name)
    if not value:
        from dotenv import dotenv_values

        path = env_file or ENV_FILE
        value = dotenv_values(path).get(name) if path.exists() else None
    return value or None


def judge_max_tokens(cli_value: int | None, env_file: Path | None = None) -> int:
    """Предел длины ответа судьи: --judge-max-tokens, иначе EVAL_JUDGE_MAX_TOKENS, иначе 1200.
    Рассуждающему судье (nemotron на OpenRouter) нужно больше: лимит у OpenRouter общий
    для скрытых рассуждений и ответа."""
    if cli_value is not None:
        value, source = cli_value, "--judge-max-tokens"
    else:
        raw = env_value("EVAL_JUDGE_MAX_TOKENS", env_file)
        if raw is None:
            return DEFAULT_JUDGE_MAX_TOKENS
        source = f"EVAL_JUDGE_MAX_TOKENS={raw}"
        try:
            value = int(raw)
        except ValueError:
            raise SystemExit(f"{source}: нужно целое число, например 4000") from None
    if value <= 0:
        raise SystemExit(f"{source}: нужно число больше нуля, например 4000")
    return value


# --------------------------------------------------------------------------- #
# Проверки без LLM
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    """Для сравнения слов: регистр, ё/е, виды тире и неразрывный пробел не важны."""
    text = text.lower().replace("ё", "е").replace(" ", " ")
    for dash in ("‐", "‑", "‒", "–", "—", "−"):
        text = text.replace(dash, "-")
    return text


def keyword_report(answer: str, keywords: list[str]) -> tuple[float | None, list[str]]:
    """Доля ключевых слов эталона, найденных в ответе; варианты слова — через «|»."""
    if not keywords:
        return None, []
    text = normalize(answer)
    missing = [kw for kw in keywords if not any(normalize(v.strip()) in text for v in kw.split("|"))]
    return round(1 - len(missing) / len(keywords), 3), missing


def forbidden_hits(answer: str, forbidden: list[str]) -> list[str]:
    text = normalize(answer)
    return [word for word in forbidden if normalize(word) in text]


# --------------------------------------------------------------------------- #
# Агрегаты
# --------------------------------------------------------------------------- #
def _avg(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 2) if values else None


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [i for i in items if i.get("scores")]
    by_criterion = {c: [i["scores"][c] for i in scored] for c in CRITERIA}
    correctness = by_criterion["correctness"]
    recalls = [i["keyword_recall"] for i in items if i.get("keyword_recall") is not None]
    categories: dict[str, list[int]] = {}
    for item in scored:
        categories.setdefault(item["category"], []).append(item["scores"]["correctness"])
    worst = sorted(scored, key=lambda i: (i["scores"]["correctness"], i["id"]))[:5]
    return {
        "relevance_avg": _avg(by_criterion["relevance"]),
        "correctness_avg": _avg(correctness),
        "completeness_avg": _avg(by_criterion["completeness"]),
        "min_correctness": min(correctness) if correctness else None,
        "overall_avg": _avg([statistics.fmean(i["scores"][c] for c in CRITERIA) for i in scored]),
        "keyword_recall_avg": _avg(recalls),
        "must_not_contain_violations": sum(1 for i in items if i.get("must_not_contain_hits")),
        "items_total": len(items),
        "items_scored": len(scored),
        "errors": sum(1 for i in items if i.get("error")),
        "reasoning_first_rate": _avg([1.0 if i.get("reasoning_first") else 0.0 for i in scored]),
        "correctness_by_category": {cat: _avg(vals) for cat, vals in sorted(categories.items())},
        "worst_items": [{"id": i["id"], "correctness": i["scores"]["correctness"]} for i in worst],
    }


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #
def load_golden(path: Path, ids: list[str] | None, limit: int | None) -> tuple[int, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["items"]
    if ids:
        items = [i for i in items if i["id"] in set(ids)]
    if limit:
        items = items[:limit]
    return data["version"], items


def default_out(runs_dir: Path, now: datetime) -> Path:
    """eval/runs/<YYYY-MM-DD>.json; если за этот день прогон уже есть — с временем."""
    path = runs_dir / f"{now:%Y-%m-%d}.json"
    return path if not path.exists() else runs_dir / f"{now:%Y-%m-%d_%H%M%S}.json"


def _short(text: str | None) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= PREVIEW else text[: PREVIEW - 1] + "…"


async def collect_answers(items: list[dict[str, Any]], model: str | None, max_tokens: int,
                          run_id: str) -> list[dict[str, Any]]:
    """Шаг 1: ответы сервиса. Приложение работает в этом процессе со своим lifespan."""
    from app.main import app   # импорт здесь: до него выставлен LOG_LEVEL

    results = []
    async with app.router.lifespan_context(app):
        cache, app.state.cache = app.state.cache, None   # без кеша: оцениваем модель
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://eval", timeout=None) as http:
                for n, item in enumerate(items, 1):
                    payload: dict[str, Any] = {
                        "messages": [{"role": "user", "content": item["question"]}],
                        "temperature": 0,
                        "max_tokens": max_tokens,
                    }
                    if model:
                        payload["model"] = model
                    started = time.perf_counter()
                    response = await http.post("/chat", json=payload,
                                               headers={"X-Request-ID": f"eval-{run_id}-{item['id']}"})
                    latency_ms = round((time.perf_counter() - started) * 1000, 1)
                    body = response.json()
                    record = {"latency_ms": latency_ms, "answer": None, "usage": None, "error": None}
                    if response.status_code == 200:
                        record.update(answer=body["content"], usage=body["usage"], response_model=body["model"],
                                      finish_reason=body.get("finish_reason"))
                    else:
                        error = body.get("error", {})
                        record["error"] = f"chat_{response.status_code}: {error.get('code')} — {error.get('message')}"
                    results.append(record)
                    status = "ok" if record["answer"] is not None else record["error"]
                    print(f"  [{n}/{len(items)}] {item['id']}  {latency_ms / 1000:.1f} s  {status}  {_short(record['answer'])}")
        finally:
            app.state.cache = cache   # lifespan закроет подключение при выходе
    return results


def find_sources(items: list[dict[str, Any]], support: Any) -> list[list[dict[str, Any]]]:
    """Статьи руководства, которые сервис подставит модели для каждого вопроса, — тот же
    build_messages, что в /chat. Их видит судья: факт из статьи — не выдумка."""
    from app.schemas.chat import ChatRequest
    from app.services.knowledge import load_knowledge_base
    from app.services.prompts import build_messages

    try:
        kb = load_knowledge_base(support.knowledge_base_path)
    except (OSError, ValueError) as exc:
        print(f"  руководство не прочитано ({exc!r}): судья оценит только по эталону")
        kb = {"articles": []}
    by_id = {article["id"]: article for article in kb.get("articles", [])}
    found = []
    for item in items:
        req = ChatRequest(messages=[{"role": "user", "content": item["question"]}])
        found.append([by_id[i] for i in build_messages(req, support, kb).article_ids])
    return found


def is_daily_limit(exc: BaseException) -> bool:
    """429 из-за дневного лимита ключа (OpenRouter: «free-models-per-day»): ждать бесполезно."""
    text = str(exc).lower()
    return "per-day" in text or "per day" in text


def retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    try:
        return max(1.0, float(value)) if value else None
    except ValueError:
        return None


async def call_with_retries(call: Any, label: str, *, sleep: Any = asyncio.sleep) -> Any:
    """Вызов судьи с паузами: 429 — 20, 40, 60 с (или Retry-After), соединение и 5xx — 5, 10 с.
    Дневной лимит ключа не повторяется."""
    rate_limited = failed = 0
    while True:
        try:
            return await call()
        except openai.RateLimitError as exc:
            if is_daily_limit(exc) or rate_limited >= len(RATE_LIMIT_WAITS):
                raise
            wait = retry_after_seconds(exc) or RATE_LIMIT_WAITS[rate_limited]
            rate_limited += 1
            print(f"{label}  429: провайдер модели перегружен — пауза {wait:.0f} с "
                  f"(повтор {rate_limited} из {len(RATE_LIMIT_WAITS)})")
            await sleep(wait)
        except (openai.APIConnectionError, openai.InternalServerError) as exc:
            if failed >= len(CONNECTION_WAITS):
                raise
            wait = CONNECTION_WAITS[failed]
            failed += 1
            print(f"{label}  {type(exc).__name__} — пауза {wait:.0f} с (повтор {failed} из {len(CONNECTION_WAITS)})")
            await sleep(wait)


async def judge_all(items: list[dict[str, Any]], answers: list[dict[str, Any]],
                    sources: list[list[dict[str, Any]]], client: Any,
                    judge_model: str, *, max_tokens: int = 1200, reasoning: str | None = None,
                    require_parameters: bool = False, sleep: Any = asyncio.sleep) -> list[dict[str, Any]]:
    """Шаг 2: оценки судьи."""
    from app.services.prompts import format_article

    pace = FREE_MODEL_INTERVAL if judge_model.endswith(":free") else 0.0
    last_call: float | None = None
    stop_reason: str | None = None       # дневной лимит или перегруженный провайдер: дальше не тратим запросы
    rate_limited_in_row = 0
    results = []
    for n, (item, answer, articles) in enumerate(zip(items, answers, sources), 1):
        label = f"  [{n}/{len(items)}] {item['id']}"
        if answer["answer"] is None:
            results.append({"verdict": None, "reasoning_first": False, "error": None})
            print(f"{label}  пропущен: нет ответа")
            continue
        if stop_reason:
            results.append({"verdict": None, "reasoning_first": False, "error": stop_reason})
            print(f"{label}  не оценён: {stop_reason.split(': ', 1)[1]}")
            continue
        if pace and last_call is not None and (gap := time.monotonic() - last_call) < pace:
            await sleep(pace - gap)
        last_call = time.monotonic()
        try:
            result = await call_with_retries(
                lambda: judge_answer(client, judge_model, item, answer["answer"],
                                     sources=[format_article(a) for a in articles],
                                     max_tokens=max_tokens, reasoning=reasoning,
                                     require_parameters=require_parameters),
                label, sleep=sleep)
            error = result.error
            rate_limited_in_row = 0
        except openai.RateLimitError as exc:
            result, error = None, f"judge_call: {describe_exception(exc)}"[:400]
            rate_limited_in_row += 1
            if is_daily_limit(exc):
                stop_reason = error = ("judge_daily_limit: дневной лимит бесплатных запросов ключа исчерпан — "
                                       "повторите прогон завтра")
            elif rate_limited_in_row >= STOP_AFTER_RATE_LIMITED:
                stop_reason = ("judge_unavailable: провайдер модели перегружен (429 после всех пауз) — "
                               "повторите позже или смените EVAL_JUDGE_MODEL")
        except Exception as exc:  # noqa: BLE001 — один упавший кейс не должен ронять прогон
            result, error = None, f"judge_call: {describe_exception(exc)}"[:400]
        verdict = result.verdict if result else None
        results.append({"verdict": verdict, "reasoning_first": bool(result and result.reasoning_first),
                        "error": error})
        if verdict:
            s = verdict.scores
            print(f"{label}  relevance={s.relevance} correctness={s.correctness} "
                  f"completeness={s.completeness}  {_short(verdict.explanation)}")
        else:
            print(f"{label}  ошибка судьи: {error}")
    return results


def build_item(item: dict[str, Any], answer: dict[str, Any], judged: dict[str, Any],
               articles: list[dict[str, Any]]) -> dict[str, Any]:
    text = answer["answer"] or ""
    recall, missing = keyword_report(text, item.get("expected_keywords", [])) if answer["answer"] else (None, [])
    verdict = judged["verdict"]
    return {
        "id": item["id"],
        "question": item["question"],
        "category": item["category"],
        "difficulty": item["difficulty"],
        "answer": answer["answer"],
        "source_articles": [a["id"] for a in articles],
        "scores": verdict.scores.model_dump() if verdict else None,
        "reasoning": verdict.reasoning if verdict else None,
        "explanation": verdict.explanation if verdict else None,
        "reasoning_first": judged["reasoning_first"],
        "keyword_recall": recall,
        "missing_keywords": missing,
        "must_not_contain_hits": forbidden_hits(text, item.get("must_not_contain", [])),
        "latency_ms": answer["latency_ms"],
        "usage": answer["usage"],
        "response_model": answer.get("response_model"),   # "guardrail" — ответила проверка, а не модель
        "finish_reason": answer.get("finish_reason"),
        "error": answer["error"] or judged["error"],
    }


def make_judge_client(base_url: str | None, api_key: str, timeout: float,
                      proxy: str | None = None, *, use_system_certs: bool = False) -> AsyncOpenAI:
    from app.core.config import http_client_options, is_local_url, provider_headers

    options = http_client_options(proxy, use_system_certs and not is_local_url(base_url))
    http_client = DefaultAsyncHttpxClient(**options) if options else None
    # max_retries=0: повторы с паузами делает call_with_retries.
    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0,
                       http_client=http_client, default_headers=provider_headers(base_url, "multapi eval"))


def use_external_model(base_url: str, api_key: str | None) -> None:
    """--model-base-url: модель под тестом — у внешнего провайдера. Переменные окружения
    важнее .env, а кеш настроек сбрасывается — lifespan сервиса прочитает их заново."""
    from app.core.config import get_settings

    os.environ["LLM__BASE_URL"] = base_url
    if api_key:
        os.environ["LLM__OPENAI_API_KEY"] = api_key
    get_settings.cache_clear()


async def run(args: argparse.Namespace) -> Path:
    from app.core.config import get_settings, is_openrouter, proxy_display, proxy_for
    from app.services.prompts import PROMPT_VERSION

    if args.model_base_url:
        use_external_model(args.model_base_url, env_value(args.model_api_key_env))
    settings = get_settings()
    now = datetime.now(timezone.utc)
    run_id = f"{now:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    golden_version, items = load_golden(Path(args.golden), args.ids, args.limit)
    sources = find_sources(items, settings.support)
    model = args.model or settings.llm.default_model
    judge_model = args.judge or env_value("EVAL_JUDGE_MODEL") or DEFAULT_JUDGE
    judge_base_url = args.judge_base_url or env_value("EVAL_JUDGE_BASE_URL") or settings.llm.base_url
    judge_key = env_value(args.judge_api_key_env) or settings.llm.openai_api_key.get_secret_value()
    judge_proxy = proxy_for(judge_base_url, settings.llm.proxy_url)
    judge_tokens = judge_max_tokens(args.judge_max_tokens)
    judge_reasoning = args.judge_reasoning or env_value("EVAL_JUDGE_REASONING")
    if judge_reasoning == "off":         # --judge-reasoning off: не отправлять, даже если задано в .env
        judge_reasoning = None
    judge_on_openrouter = is_openrouter(judge_base_url)
    if judge_reasoning and not judge_on_openrouter:
        # OpenAI и Ollama параметр reasoning не знают: OpenAI ответил бы 400.
        print(f"  EVAL_JUDGE_REASONING={judge_reasoning} пропущен: он только для OpenRouter")
        judge_reasoning = None
    model_proxy = proxy_for(settings.llm.base_url, settings.llm.proxy_url)

    def where(base_url: str | None, proxy: str | None) -> str:
        return (base_url or "api.openai.com") + (f", через прокси {proxy_display(proxy)}" if proxy else "")

    print(f"Прогон {run_id}: {len(items)} кейсов")
    print(f"  модель под тестом: {model} ({where(settings.llm.base_url, model_proxy)})")
    if settings.llm.use_system_certs:
        print("  HTTPS: сертификаты проверяются по хранилищу ОС (LLM__USE_SYSTEM_CERTS=true)")
    print(f"  судья: {judge_model} ({where(judge_base_url, judge_proxy)}), max_tokens={judge_tokens}"
          + (f", reasoning.effort={judge_reasoning}" if judge_reasoning else ""))
    print("Шаг 1/2 — ответы сервиса (POST /chat, temperature=0):")
    started = time.perf_counter()
    answers = await collect_answers(items, args.model, args.max_tokens, run_id)

    print("Шаг 2/2 — оценки судьи (G-Eval, reason-then-score):")
    client = make_judge_client(judge_base_url, judge_key, args.judge_timeout, judge_proxy,
                               use_system_certs=settings.llm.use_system_certs)
    try:
        judged = await judge_all(items, answers, sources, client, judge_model,
                                 max_tokens=judge_tokens, reasoning=judge_reasoning,
                                 require_parameters=judge_on_openrouter)
    finally:
        await client.close()

    records = [build_item(item, answer, j, articles)
               for item, answer, j, articles in zip(items, answers, judged, sources)]
    run_data = {
        "run_id": run_id,
        "timestamp": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "model_under_test": model,
        "judge_model": judge_model,
        "golden_version": golden_version,
        "prompt_version": PROMPT_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "params": {"temperature": 0, "judge_temperature": 0, "max_tokens": args.max_tokens,
                   "judge_response_format": "json_object", "judge_max_tokens": judge_tokens,
                   "judge_reasoning_effort": judge_reasoning,
                   "judge_provider_require_parameters": judge_on_openrouter},
        "duration_s": round(time.perf_counter() - started, 1),
        "items": records,
        "aggregates": aggregate(records),
    }
    out = Path(args.out) if args.out else default_out(Path(args.runs_dir), now.astimezone())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    agg = run_data["aggregates"]
    print("\nАгрегаты:")
    for key in ("relevance_avg", "correctness_avg", "completeness_avg", "min_correctness", "overall_avg",
                "keyword_recall_avg", "must_not_contain_violations", "items_scored", "errors"):
        print(f"  {key:<28} {agg[key]}")
    print(f"  {'correctness_by_category':<28} {agg['correctness_by_category']}")
    if any((item.get("error") or "").startswith("judge_daily_limit") for item in records):
        print("\nДневной лимит бесплатных запросов ключа OpenRouter исчерпан (учебный ключ — общий для группы). "
              "Повторите прогон завтра.")
    elif any((item.get("error") or "").startswith(("judge_call", "judge_unavailable")) for item in records):
        print("\nСудья не ответил — причина в строках выше. 429 «rate-limited upstream» значит, что "
              "бесплатный канал провайдера модели перегружен: повторите позже или смените EVAL_JUDGE_MODEL. "
              "Сеть и прокси без вызова моделей проверяет python eval/check_connection.py")
    print(f"\nРезультат: {out} ({run_data['duration_s']} s)")
    print("Пороги: python eval/check_thresholds.py")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Прогон golden dataset: ответы сервиса + LLM-as-judge (G-Eval)")
    parser.add_argument("--golden", default=str(EVAL_DIR / "golden_dataset.json"))
    parser.add_argument("--judge", default=None,
                        help=f"модель-судья; по умолчанию EVAL_JUDGE_MODEL из .env, иначе {DEFAULT_JUDGE}")
    parser.add_argument("--judge-base-url", default=None,
                        help="адрес OpenAI-совместимого API судьи; по умолчанию EVAL_JUDGE_BASE_URL, иначе LLM__BASE_URL")
    parser.add_argument("--judge-api-key-env", default="EVAL_JUDGE_API_KEY",
                        help="переменная (окружение или .env) с ключом судьи; нет — ключ LLM__OPENAI_API_KEY")
    parser.add_argument("--judge-timeout", type=float, default=600.0, help="таймаут вызова судьи, с")
    parser.add_argument("--judge-max-tokens", type=int, default=None,
                        help=f"предел длины ответа судьи; по умолчанию EVAL_JUDGE_MAX_TOKENS, иначе {DEFAULT_JUDGE_MAX_TOKENS}")
    parser.add_argument("--judge-reasoning", default=None,
                        help="OpenRouter: reasoning.effort судьи (none — без скрытых рассуждений; "
                             "off — не отправлять); по умолчанию EVAL_JUDGE_REASONING")
    parser.add_argument("--model", default=None, help="модель под тестом; по умолчанию LLM__DEFAULT_MODEL")
    parser.add_argument("--model-base-url", default=None,
                        help="внешний API для модели под тестом, например https://api.openai.com/v1 (вместо LLM__BASE_URL)")
    parser.add_argument("--model-api-key-env", default="EVAL_JUDGE_API_KEY",
                        help="переменная с ключом для --model-base-url; по умолчанию тот же ключ, что у судьи")
    parser.add_argument("--max-tokens", type=int, default=600, help="предел длины ответа сервиса")
    parser.add_argument("--out", default=None, help="файл результата; по умолчанию eval/runs/<дата>.json")
    parser.add_argument("--runs-dir", default=str(EVAL_DIR / "runs"))
    parser.add_argument("--limit", type=int, default=None, help="только первые N кейсов — быстрая проверка")
    parser.add_argument("--ids", type=lambda s: [x.strip() for x in s.split(",") if x.strip()], default=None,
                        help="только эти кейсы, через запятую: faq_001,faq_019")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # консоль Windows: символы вне кодировки не роняют вывод
    asyncio.run(run(parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
