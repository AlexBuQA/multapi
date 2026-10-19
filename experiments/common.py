"""
Общее для обеих реализаций эксперимента 6.5 — чтобы сравнение было честным, отличаться у них может только
организация агентов:

- QUESTIONS — 5 одинаковых вопросов (3 по корпусу, 1 многошаговый, 1 вне базы) с эталоном и ключевыми фактами;
- search_knowledge_base — ОДИН объект инструмента для researcher и single-agent: RAG блока 5.5
  (app/services/rag.py, коллекция RAG__COLLECTION), до трёх фрагментов с файлом-источником и близостью;
- SEARCH_RULES и ANSWER_RULES — одинаковые правила поиска и формата ответа: researcher получает первые,
  writer — вторые, single-agent — обе части в одном промпте;
- limits() — одинаковые бюджеты create_agent: не больше MAX_SEARCHES поисков и MAX_MODEL_CALLS вызовов модели;
- Meter — callback: вызовы модели и токены (usage_metadata ответа) по каждому агенту;
- check_answer — детерминированные проверки ответа: ключевые факты, цитаты [n], честный отказ;
- save_records — запись в experiments/results.json по одной записи на (реализация, вопрос).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import LLMResult
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results.json"
IMPLEMENTATIONS = ("single_agent", "multi_agent")

TOP_N = 3                # фрагментов в ответе инструмента
MAX_SEARCHES = 4         # поисков за один запуск create_agent — у researcher и у single-agent одинаково
MAX_MODEL_CALLS = 6      # вызовов модели за один запуск create_agent — страховка от зацикливания
RESET_PROMPT = "Ответь одним словом: готово"


# ---------------------------------------------------------------- тестовый набор
@dataclass(frozen=True)
class Question:
    id: str
    kind: str                                   # corpus | multi_step | out_of_base
    text: str
    reference: str                              # эталон для судьи
    key_facts: tuple[tuple[str, ...], ...]      # каждый факт — варианты написания; все факты = ответ полный
    sources: tuple[str, ...]                    # где ответ в корпусе data/<категория>/


QUESTIONS: tuple[Question, ...] = (
    Question("q1", "corpus", "Сколько действует ссылка для сброса пароля и как быстро приходит письмо с ней?",
             "Письмо со ссылкой для сброса приходит в течение 10 минут, ссылка действует 30 минут.",
             (("30 минут", "30 мин"), ("10 минут", "10 мин")), ("account/password-reset.html",)),
    Question("q2", "corpus", "Какой лимит запросов к API на тарифе «Команда» и что ответит API, если его превысить?",
             "300 запросов в минуту; при превышении API отвечает кодом 429 и заголовком Retry-After — повторить "
             "запрос после указанного времени.",
             (("300",), ("429",)), ("api/faq-api.md", "api/api-limits.md")),
    Question("q3", "corpus", "За сколько дней до окончания периода приходит письмо об автопродлении и когда спишут "
             "деньги?",
             "Письмо приходит за 3 дня до окончания периода; оплата списывается с сохранённой карты в день "
             "окончания периода.",
             (("3 дня", "три дня", "3 дн"), ("в день окончания", "день окончания", "в последний день")),
             ("billing/auto-renewal.docx",)),
    Question("q4", "multi_step", "Не проходит оплата картой, а тариф заканчивается завтра. Что проверить, чем ещё "
             "можно оплатить и как быстро ответит поддержка, если оплатить так и не получится?",
             "Проверить баланс, срок действия карты и лимит на онлайн-платежи, подтвердить платёж 3-D Secure кодом "
             "из СМС; при повторной ошибке — другая карта или СБП. Не проходящая оплата картой — высокий приоритет: "
             "первый ответ поддержки за 1 час, решение за 1 рабочий день.",
             (("баланс", "срок действия"), ("сбп", "быстрых платежей", "другую карту", "другой карт"),
              ("1 час", "один час", "в течение часа", "60 минут")),
             ("billing/card-payment-fails.docx", "support/support-regulation_v2.3.pdf")),
    Question("q5", "out_of_base", "Как подключить интеграцию личного кабинета с amoCRM?",
             "В базе знаний нет информации об интеграции с amoCRM: в справке описаны только обмен с 1С и календарь "
             "сроков. Правильный ответ — честно сказать, что информации нет, и предложить обратиться в поддержку; "
             "инструкцию не выдумывать.",
             (), ()),
)


def pick_questions(ids: str | None) -> list[Question]:
    if not ids:
        return list(QUESTIONS)
    wanted = [part.strip() for part in ids.split(",") if part.strip()]
    known = {q.id: q for q in QUESTIONS}
    unknown = [q for q in wanted if q not in known]
    if unknown:
        raise SystemExit(f"Неизвестные вопросы: {', '.join(unknown)}; есть {', '.join(known)}")
    return [known[q] for q in wanted]


# ---------------------------------------------------------------- инструмент
class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(pattern=r"^[\s\S]{2,300}$", description="Запрос по-русски одной фразой, 2–300 символов, "
                       "например «сколько действует приглашение в команду»")


NOTHING_FOUND = "В базе знаний ничего не найдено"
SEARCH_DESCRIPTION = (
    "Что делает: ищет в базе знаний техподдержки «Личного кабинета» (аккаунт и вход, тарифы и оплата, возвраты, "
    f"API, уведомления, мобильное приложение, интеграции, команда, поддержка) до {TOP_N} самых близких фрагментов "
    "справки. Когда вызывать: перед любым ответом о правилах, сроках, ценах, лимитах и настройках продукта; по "
    "одному аспекту вопроса за вызов. Когда НЕ вызывать: повторно с тем же запросом. Что возвращает: фрагменты "
    "строками «[файл, близость 0.71] текст» или «В базе знаний ничего не найдено …» — тогда данных нет: не "
    "выдумывай ответ.")


def fragment(node: Any) -> str:
    source = node.metadata.get("source") or node.metadata.get("file_name") or "?"
    text = " ".join(node.node.get_content().split())
    return f"[{source}, близость {node.score or 0.0:.2f}] {text}"


def kb_search(query: str) -> str:
    """RAG блока 5.5: тот же RAGService, что у /rag/query и агентов 6.1–6.4 (score-guard, re-ranker)."""
    from app.tools.naive_tools import knowledge_base

    found = knowledge_base().retrieve(query)
    if not found.nodes:
        return (f"{NOTHING_FOUND} по запросу «{query}»: лучшая близость {found.top_score:.2f} ниже порога "
                f"{found.threshold}.")
    return "\n\n".join(fragment(node) for node in found.nodes[:TOP_N])


@tool("search_knowledge_base", args_schema=SearchArgs, description=SEARCH_DESCRIPTION)
def search_knowledge_base(query: str) -> str:
    try:
        return kb_search(query)
    except Exception as exc:  # noqa: BLE001 — сбой Qdrant модель видит строкой, граф не падает
        return f"Ошибка инструмента search_knowledge_base: {type(exc).__name__}: {exc}"


def limits() -> list[Any]:
    """Бюджеты create_agent — новые экземпляры на каждого агента, значения одни и те же у обеих реализаций.
    Лишний поиск блокируется ToolMessage-ошибкой, и модель продолжает (exit_behavior="continue"); лимит вызовов
    модели завершает агента (exit_behavior="end"). Лимит инструментов — без tool_name: инструмент у агентов один,
    а с tool_name узел назывался бы «ToolCallLimitMiddleware[search_knowledge_base]…», и квадратные скобки
    ломают Mermaid-схему draw_mermaid()."""
    from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware

    return [ToolCallLimitMiddleware(run_limit=MAX_SEARCHES),
            ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end")]


# ---------------------------------------------------------------- промпты (одинаковые части)
SEARCH_RULES = (
    "Факты бери только из базы знаний: инструмент search_knowledge_base, по одному аспекту вопроса за вызов. "
    f"Вопрос из нескольких частей — отдельный поиск на каждую часть, всего не больше {MAX_SEARCHES} поисков; тот же "
    "запрос повторно не отправляй. Используй только то, что есть во фрагментах, ничего не добавляй от себя.")
ANSWER_RULES = (
    "Ответ клиенту — по-русски, 2–5 предложений, только по фактам из базы знаний. После каждого утверждения ставь "
    "номер источника в квадратных скобках — [1], [2], — а в конце дай строку «Источники:» и под теми же номерами "
    "файлы, например «[1] team/invite-members.docx». Если в базе нет информации по вопросу или его части, так и "
    "напиши («В базе знаний нет информации о …»), предложи обратиться в поддержку и ничего не выдумывай.")
RESEARCHER_PROMPT = (
    "Ты — researcher в команде техподдержки «Личного кабинета»: собираешь факты для ответа на вопрос клиента. "
    f"{SEARCH_RULES} Итоговый ответ клиенту НЕ пиши — его напишет writer. Верни только маркированный список "
    "фактов, у каждого — файл-источник в квадратных скобках, например «- Приглашение действует 7 дней "
    "[team/invite-members.docx]». Если по какой-то части вопроса в базе ничего не нашлось — отдельный пункт "
    "«- По теме «…» в базе знаний ничего не найдено».")
WRITER_PROMPT = (
    "Ты — writer в команде техподдержки «Личного кабинета»: по списку фактов от researcher пишешь клиенту "
    f"итоговый ответ. Сам ничего не ищешь: в ответе — только факты из списка. {ANSWER_RULES}")
SINGLE_PROMPT = (
    "Ты — агент техподдержки «Личного кабинета»: сам находишь факты и сам пишешь клиенту ответ. Сначала найди "
    f"факты. {SEARCH_RULES} Затем ответь клиенту. {ANSWER_RULES}")


# ---------------------------------------------------------------- замеры
class Meter(BaseCallbackHandler):
    """Вызовы модели и токены за один вопрос. on_chat_model_start запоминает, чей вызов: lc_agent_name —
    у create_agent (researcher, writer, support_agent), иначе узел графа (supervisor). on_llm_end складывает
    usage_metadata ответа. Вызов без usage (провайдер его не вернул) считается в without_usage — токены тогда
    занижены, и это видно в results.json."""

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.without_usage = 0
        self.by_agent: dict[str, dict[str, int]] = {}
        self._owner: dict[UUID, str] = {}

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[BaseMessage]], *,
                            run_id: UUID, metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        meta = metadata or {}
        self._owner[run_id] = str(meta.get("lc_agent_name") or meta.get("langgraph_node") or "model")

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        usage = next((gen.message.usage_metadata for batch in response.generations for gen in batch
                      if getattr(getattr(gen, "message", None), "usage_metadata", None)), None)
        self._count(self._owner.pop(run_id, "model"), usage)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._count(self._owner.pop(run_id, "model"), None)

    def _count(self, owner: str, usage: dict[str, Any] | None) -> None:
        stats = self.by_agent.setdefault(owner, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        stats["calls"] += 1
        self.calls += 1
        if not usage:
            self.without_usage += 1
            return
        stats["input_tokens"] += int(usage.get("input_tokens") or 0)
        stats["output_tokens"] += int(usage.get("output_tokens") or 0)
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_record(self) -> dict[str, Any]:
        return {"total_tokens": self.total_tokens, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "llm_calls": self.calls,
                "llm_calls_without_usage": self.without_usage,
                "by_agent": {name: {**stats, "total_tokens": stats["input_tokens"] + stats["output_tokens"]}
                             for name, stats in self.by_agent.items()}}


def reset_cache(model: Any) -> None:
    """Как в scripts/bench_agents.py (блок 6.3): короткий посторонний запрос перед каждым вопросом. У Ollama
    с OLLAMA_NUM_PARALLEL=1 один слот KV-кеша — без сброса вторая реализация получила бы готовый префилл
    промпта первой. Вызов не входит в замер: Meter к нему не подключён."""
    model.invoke(RESET_PROMPT)


def warmup(model: Any) -> None:
    """Модель и эмбеддинги — в память до замеров: холодный старт Ollama — десятки секунд на первый вызов."""
    kb_search("лимит запросов к API")
    reset_cache(model)


def model_info(model: Any) -> dict[str, Any]:
    """Что записать в results.json о модели: имя, температура и хост провайдера — без ключей и логина прокси."""
    base_url = getattr(model, "openai_api_base", None)
    return {"model": getattr(model, "model_name", None) or type(model).__name__,
            "temperature": getattr(model, "temperature", None),
            "provider_host": urlsplit(base_url).hostname if base_url else None}


# ---------------------------------------------------------------- проверки ответа
CITATION = re.compile(r"\[(\d{1,2})\]")
REFUSAL = re.compile(r"нет (никакой |такой )?(информации|данных|сведений)|ничего не найден|не найден|"
                     r"не наш(ел|ла|ли|лось|лись)|отсутству|не описан|не содерж")


def normalize(text: str) -> str:
    return " ".join(str(text).lower().replace("ё", "е").replace(" ", " ").replace(" ", " ").split())


FILE_REF = re.compile(r"\[([^\]\s]*/[^\]\s]+)\]")                 # «[billing/auto-renewal.docx]» вместо номера
SOURCES_HEADER = re.compile(r"(?:^|\n)[ \t]*Источники[ \t]*:", re.IGNORECASE)
LISTED = re.compile(r"^[ \t]*\[(\d{1,2})\][ \t]*(.+?)[ \t]*$", re.MULTILINE)
SEARCHED = re.compile(r"^\[([^,\]\n]+), близость", re.MULTILINE)


def split_answer(answer: str) -> tuple[str, str]:
    """Текст ответа и список «Источники:» после него (если есть)."""
    parts = SOURCES_HEADER.split(answer, maxsplit=1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def check_answer(question: Question, answer: str, contexts: Iterable[str] = ()) -> dict[str, Any]:
    """Без модели, по тексту ответа до строки «Источники:»:
    - facts_hit — сколько ключевых фактов эталона есть в ответе; refused — есть ли честный отказ;
    - expected — ответ ведёт себя как нужно: все факты (вопрос по базе) или отказ (вне базы);
    - citations — номера [n] в тексте ответа (номера в самом списке источников не считаются);
    - file_refs — ссылки файлом в скобках вместо номера, «[billing/auto-renewal.docx]»;
    - sources — файлы из списка «Источники:», unknown_sources — те из них, которых не было в результатах
      search_knowledge_base (contexts): выдуманный или искажённый источник."""
    body, tail = split_answer(answer)
    text = normalize(body)
    hits = sum(any(normalize(alt) in text for alt in alts) for alts in question.key_facts)
    refused = bool(REFUSAL.search(text))
    sources = [match.group(2) for match in LISTED.finditer(tail)]
    searched = {name for context in contexts for name in SEARCHED.findall(context)}
    return {"facts_hit": hits, "facts_total": len(question.key_facts),
            "citations": sorted({int(n) for n in CITATION.findall(body)}), "file_refs": len(FILE_REF.findall(body)),
            "sources": sources, "unknown_sources": [s for s in sources if s not in searched], "refused": refused,
            "expected": refused if question.kind == "out_of_base" else hits == len(question.key_facts)}


# ---------------------------------------------------------------- сообщения и стрим
def last_text(messages: Iterable[BaseMessage]) -> str:
    """Текст последнего AIMessage без вызовов инструментов — итог create_agent."""
    final = [m for m in messages if isinstance(m, AIMessage) and not m.tool_calls and m.text.strip()]
    return final[-1].text.strip() if final else ""


def searches(messages: Iterable[BaseMessage]) -> tuple[list[str], list[str]]:
    """Запросы к search_knowledge_base и то, что инструмент вернул, — для отчёта и для судьи."""
    queries: list[str] = []
    contexts: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage):
            queries += [str(c["args"].get("query", "")) for c in message.tool_calls
                        if c["name"] == search_knowledge_base.name]
        elif isinstance(message, ToolMessage) and message.name == search_knowledge_base.name:
            contexts.append(str(message.content))
    return queries, contexts


def preview(text: Any, limit: int = 160) -> str:
    one = " ".join(str(text).split())
    return one if len(one) <= limit else one[: limit - 1] + "…"


def describe(message: BaseMessage) -> str | None:
    if isinstance(message, AIMessage) and message.tool_calls:
        return "; ".join(f"→ {c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in message.tool_calls)
    if isinstance(message, ToolMessage):
        return f"← {message.name}: {preview(message.content)}"
    if isinstance(message, AIMessage | HumanMessage):
        return f"{type(message).__name__}: {preview(message.content)}"
    return None


def print_update(chunk: Any) -> None:
    """Чанк app.stream(..., stream_mode="updates"): {узел: обновление}; с subgraphs=True — (namespace, {…}),
    namespace вида ("researcher:<id>",) — агент, внутри которого сработал узел."""
    namespace, update = chunk if isinstance(chunk, tuple) else ((), chunk)
    where = "/".join(part.split(":")[0] for part in namespace)
    for node, values in (update or {}).items():
        if not isinstance(values, dict) or "Middleware" in node:     # счётчики бюджетов create_agent — не шаги
            continue
        label = f"{where}/{node}" if where else node
        lines = [line for m in values.get("messages") or [] if (line := describe(m))]
        if not lines and not where:
            lines = ([f"→ {values['route'][-1]} без вызова модели: {preview(values.get('task', ''))}"]
                     if values.get("route") else [f"обновлено: {', '.join(values)}"])
        for line in lines:
            print(f"  [{label}] {line}")


# ---------------------------------------------------------------- результаты
def record(implementation: str, question: Question, *, info: dict[str, Any], answer: str, meter: Meter,
           latency_ms: float, handoff_count: int, queries: list[str], contexts: list[str], cache_reset: bool,
           error: str | None = None, **extra: Any) -> dict[str, Any]:
    return {"implementation": implementation, "question_id": question.id, "kind": question.kind,
            "question": question.text, **meter.as_record(), "latency_ms": round(latency_ms, 1),
            "handoff_count": handoff_count, "searches": len(queries), "queries": queries, **extra,
            "answer": answer, "checks": check_answer(question, answer, contexts), "contexts": contexts,
            "error": error, **info, "cache_reset": cache_reset,
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds")}


def load_records(path: Path = RESULTS) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def write_json(path: Path, data: Any) -> None:
    """LF и UTF-8 на любой ОС: иначе на Windows git видит изменённым каждый перезаписанный файл."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def save_records(records: list[dict[str, Any]], path: Path = RESULTS) -> list[dict[str, Any]]:
    """Свои записи заменяют прежние с тем же (реализация, вопрос), чужие остаются: скрипты пишут в один файл."""
    fresh = {(r["implementation"], r["question_id"]) for r in records}
    merged = [r for r in load_records(path) if (r["implementation"], r["question_id"]) not in fresh] + records
    order = {name: n for n, name in enumerate(IMPLEMENTATIONS)}
    merged.sort(key=lambda r: (order.get(r["implementation"], len(order)), r["question_id"]))
    write_json(path, merged)
    return merged


def print_result(rec: dict[str, Any]) -> None:
    checks = rec["checks"]
    print(f"\nОтвет:\n{rec['answer'] or '— (пусто)'}")
    print(f"tokens={rec['total_tokens']} llm_calls={rec['llm_calls']} latency_ms={rec['latency_ms']:.0f} "
          f"handoffs={rec['handoff_count']} searches={rec['searches']} facts={checks['facts_hit']}/"
          f"{checks['facts_total']} citations={checks['citations']} file_refs={checks['file_refs']} "
          f"unknown_sources={len(checks['unknown_sources'])} refused={checks['refused']}"
          + (f" ERROR={rec['error']}" if rec["error"] else ""))


# ---------------------------------------------------------------- CLI
def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", help="модель (по умолчанию AGENT__MODEL из .env)")
    p.add_argument("--questions", help=f"вопросы через запятую: {', '.join(q.id for q in QUESTIONS)}")
    p.add_argument("--no-cache-reset", action="store_true", help="не сбрасывать KV-кеш Ollama перед вопросом")
    p.add_argument("--out", type=Path, default=RESULTS, help="файл результатов (по умолчанию experiments/results.json)")
    return p


def console() -> None:
    """Консоль Windows (cp1251) не роняет печать «→» и «≈»; логи сервиса — в stderr и только WARNING."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    from app.observability.logging import setup_logging

    setup_logging("WARNING", stream=sys.stderr)
