"""
ReAct-агент блока 6.2 на LangGraph 1.x (блок 6.3): тот же агент — системный промпт, три инструмента,
лимит итераций — в orchestration-форме, двумя путями сборки.

- custom_graph — StateGraph руками: типизированный AgentState, узлы call_model / execute_tool /
  force_finish и условное ребро с router-функцией route_after_model;
- prebuilt_graph — тот же агент через langchain.agents.create_agent (рекомендуемый путь LangChain 1.0).

Модель — ChatOpenAI к тому же провайдеру, что у сервиса (LLM__BASE_URL, ключ, прокси, сертификаты ОС;
локально — Ollama), имя — AGENT__MODEL (канон задания — gpt-5.4-mini), температура — AGENT__TEMPERATURE.
В state — только сериализуемые данные: сообщения, счётчик итераций, результаты инструментов. Клиент
модели и инструменты живут в замыканиях узлов, поэтому граф готов к checkpointer (AsyncSqliteSaver) и
к interrupt без переделки state.

custom_graph и prebuilt_graph — атрибуты модуля, собираются при первом обращении (настройки из .env):

    from app.services.agent_graph import custom_graph
    await custom_graph.ainvoke(initial_state("Который час в Екатеринбурге?"),
                               config={"configurable": {"thread_id": "demo"}})

    python -m app.services.agent_graph "Сколько запросов в минуту можно делать к API на тарифе «Команда»?" --trace
"""
from __future__ import annotations

import argparse
import asyncio
import json
import operator
import sys
import time
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ValidationError

from app.observability.logging import get_logger
from app.services.agent_react import DEFAULT_MODEL, ITERATION_LIMIT, SYSTEM_PROMPT
from app.tools.graph_tools import TOOLS

log = get_logger()

MAX_ITERATIONS = 6                   # stop-кран: iteration_count >= 6 → force_finish
PREVIEW = 200                        # результат инструмента в tool_results — первые 200 символов
FORCE_FINISH = f"{ITERATION_LIMIT} ({MAX_ITERATIONS}): агент не дал финального ответа."
NOT_EXECUTED = "Не выполнено: превышен лимит итераций."


class AgentState(TypedDict):
    """Контракт состояния. messages — история диалога, add_messages дописывает новые сообщения (и
    заменяет сообщение с тем же id); iteration_count — число вызовов модели, без reducer: узел
    возвращает новое значение целиком; tool_results — результаты инструментов для отчёта и трейсинга,
    operator.add склеивает списки из всех шагов. Клиентов SDK, сессий и ключей здесь нет."""

    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int
    tool_results: Annotated[list[dict], operator.add]


def initial_state(task: str) -> AgentState:
    """Вход графа: задача пользователя. Системный промпт в state не кладётся — его добавляет call_model."""
    return {"messages": [HumanMessage(task)], "iteration_count": 0, "tool_results": []}


def route_after_model(state: AgentState) -> Literal["execute_tool", "force_finish"]:
    """Куда после модели: детерминированно, только чтение state, без сети и побочных эффектов."""
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "force_finish"
    last = state["messages"][-1]
    return "execute_tool" if isinstance(last, AIMessage) and last.tool_calls else "force_finish"


def build_custom_graph(model: BaseChatModel, tools: list[BaseTool] = TOOLS, *,
                       system_prompt: str = SYSTEM_PROMPT) -> Any:
    """StateGraph руками: START → call_model ⇄ execute_tool, call_model → force_finish → END.
    strict=True: без него langchain-openai выкидывает из схемы инструмента additionalProperties: false
    (extra="forbid" у pydantic-модели аргументов), и модель не видит, что лишние поля запрещены."""
    bound = model.bind_tools(tools, strict=True)
    by_name = {t.name: t for t in tools}

    async def call_model(state: AgentState) -> dict[str, Any]:
        response = await bound.ainvoke([SystemMessage(system_prompt), *state["messages"]])
        iteration = state["iteration_count"] + 1
        usage = response.usage_metadata or {}
        log.info("graph_model", iteration=iteration, tool_calls=[c["name"] for c in response.tool_calls],
                 input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))
        return {"messages": [response], "iteration_count": iteration}

    async def execute_tool(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        messages, results = [], []
        for call in last.tool_calls:
            started = time.perf_counter()
            content = await _invoke(by_name, call)
            messages.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
            results.append({"iteration": state["iteration_count"], "name": call["name"], "args": call["args"],
                            "result": content[:PREVIEW], "ms": round((time.perf_counter() - started) * 1000, 1)})
            log.info("graph_tool", iteration=state["iteration_count"], tool=call["name"], result=content[:80])
        return {"messages": messages, "tool_results": results}

    async def force_finish(state: AgentState) -> dict[str, Any]:
        """Граф не уходит в END молча. Ответ модели без tool_calls — уже финал, его прокидываем как есть.
        Лимит итераций при невыполненных вызовах — закрываем каждый вызов ToolMessage (история остаётся
        валидной для продолжения с checkpointer) и даём явный финальный ответ."""
        last = state["messages"][-1]
        if not (isinstance(last, AIMessage) and last.tool_calls):
            return {}
        log.warning("graph_force_finish", iteration=state["iteration_count"], pending=len(last.tool_calls))
        closing = [ToolMessage(content=NOT_EXECUTED, tool_call_id=call["id"], name=call["name"])
                   for call in last.tool_calls]
        return {"messages": [*closing, AIMessage(content=FORCE_FINISH, name="force_finish")]}

    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tool", execute_tool)
    builder.add_node("force_finish", force_finish)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", route_after_model,
                                  {"execute_tool": "execute_tool", "force_finish": "force_finish"})
    builder.add_edge("execute_tool", "call_model")
    builder.add_edge("force_finish", END)
    return builder.compile(name="custom_graph")


async def _invoke(by_name: dict[str, BaseTool], call: dict[str, Any]) -> str:
    """Инструмент по имени; неизвестное имя, неверные аргументы, сбой — строка-ошибка для модели."""
    name = call["name"]
    if name not in by_name:
        return f"Ошибка: инструмента {name!r} нет. Доступны только: {', '.join(by_name)}."
    try:
        return str(await by_name[name].ainvoke(call["args"]))
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'аргументы'}: {e['msg']}" for e in exc.errors())
        return f"Ошибка: аргументы {name} не по схеме — {problems}. Исправь и повтори вызов."
    except Exception as exc:  # noqa: BLE001 — сбой инструмента модель видит и может обойти
        return f"Ошибка инструмента {name}: {type(exc).__name__}: {exc}"


def build_prebuilt_graph(model: BaseChatModel, tools: list[BaseTool] = TOOLS, *,
                         system_prompt: str = SYSTEM_PROMPT) -> Any:
    """Тот же агент через create_agent (LangChain 1.0): цикл модель ⇄ инструменты, ToolNode и выход по
    ответу без tool_calls — внутри. То, что в кастомном графе написано руками, здесь — middleware:
    - ModelCallLimitMiddleware(run_limit=6, exit_behavior="end") — тот же stop-кран: перед седьмым вызовом
      модели агент уходит в конец с сообщением о лимите, а не в GraphRecursionError;
    - strict_tools — strict=True в bind_tools (create_agent связывает инструменты сам, параметр
      передаётся только через model_settings)."""
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware, wrap_model_call

    @wrap_model_call
    async def strict_tools(request: Any, handler: Any) -> Any:
        return await handler(request.override(model_settings={**request.model_settings, "strict": True}))

    return create_agent(model=model, tools=tools, system_prompt=system_prompt, name="prebuilt_graph",
                        middleware=[strict_tools, ModelCallLimitMiddleware(run_limit=MAX_ITERATIONS,
                                                                           exit_behavior="end")])


# Страховка от зацикливания самого графа: лимит вызовов модели держит middleware, а шаг prebuilt-агента —
# это 4 супершага (before_model, model, after_model, tools); 6 итераций — около 25, по умолчанию лимит 25.
PREBUILT_RECURSION_LIMIT = 50
PREBUILT_LIMIT_PREFIX = "Model call limits exceeded"     # сообщение ModelCallLimitMiddleware при exit_behavior="end"


def build_model(cfg: Any = None, *, model: str | None = None, temperature: float | None = None) -> BaseChatModel:
    """ChatOpenAI к провайдеру сервиса: те же base_url, ключ, прокси и сертификаты ОС, что у build_openai.
    Chat Completions (use_responses_api=False): Responses API у Ollama нет."""
    from langchain_openai import ChatOpenAI
    from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

    from app.core.config import (
        get_settings,
        http_client_options,
        is_local_url,
        provider_headers,
        proxy_for,
    )

    cfg = cfg or get_settings()
    proxy = proxy_for(cfg.llm.base_url, cfg.llm.proxy_url)
    options = http_client_options(proxy, cfg.llm.use_system_certs and not is_local_url(cfg.llm.base_url))
    return ChatOpenAI(
        model=model or cfg.agent.model or DEFAULT_MODEL,
        temperature=cfg.agent.temperature if temperature is None else temperature,
        api_key=cfg.llm.openai_api_key.get_secret_value(), base_url=cfg.llm.base_url,
        timeout=cfg.llm.request_timeout, max_retries=cfg.llm.max_retries,
        default_headers=provider_headers(cfg.llm.base_url, "multapi"),
        http_client=DefaultHttpxClient(**options) if options else None,
        http_async_client=DefaultAsyncHttpxClient(**options) if options else None,
        use_responses_api=False)


_BUILT: dict[str, Any] = {}


def graphs() -> dict[str, Any]:
    """Модель и оба графа из настроек .env — один раз на процесс."""
    if not _BUILT:
        model = build_model()
        _BUILT.update(model=model, custom_graph=build_custom_graph(model), prebuilt_graph=build_prebuilt_graph(model))
    return _BUILT


def __getattr__(name: str) -> Any:
    """custom_graph, prebuilt_graph и model — при первом обращении: импорт модуля не читает .env и не
    строит клиентов (тесты собирают графы со своей моделью через build_*)."""
    if name in ("model", "custom_graph", "prebuilt_graph"):
        return graphs()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------- разбор результата
def summarize(messages: list[AnyMessage]) -> dict[str, Any]:
    """Ответ, вызовы инструментов и токены по истории сообщений — одинаково для обоих графов (у
    prebuilt-агента своего tool_results нет)."""
    results = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    ai = [m for m in messages if isinstance(m, AIMessage)]
    calls = [{"name": c["name"], "args": c["args"], "result": str(results.get(c["id"], ""))}
             for m in ai for c in m.tool_calls]
    model_calls = [m for m in ai if not _limit_message(m)]
    usage = [m.usage_metadata or {} for m in model_calls]
    prompt, completion = sum(u.get("input_tokens", 0) for u in usage), sum(u.get("output_tokens", 0) for u in usage)
    final = ai[-1] if ai else None
    return {"answer": final.content if final else None, "force_finish": bool(final and _limit_message(final)),
            "steps": len(model_calls), "tool_calls": calls, "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion}


def _limit_message(message: AIMessage) -> bool:
    """Ответ «превышен лимит», а не модели: force_finish кастомного графа или ModelCallLimitMiddleware."""
    return message.name == "force_finish" or str(message.content).startswith(PREBUILT_LIMIT_PREFIX)


async def run_graph(graph: Any, task: str, *, thread_id: str, recursion_limit: int | None = None) -> dict[str, Any]:
    """Один прогон графа: config с thread_id — без checkpointer это no-op, но вызов уже совместим с
    персистентностью."""
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if recursion_limit:
        config["recursion_limit"] = recursion_limit
    state = await graph.ainvoke(initial_state(task), config=config)
    return {**summarize(state["messages"]), "iteration_count": state.get("iteration_count"),
            "tool_results": state.get("tool_results")}


def main(argv: list[str] | None = None) -> int:
    from langgraph.errors import GraphRecursionError
    from openai import OpenAIError

    from app.observability.logging import setup_logging

    parser = argparse.ArgumentParser(description="ReAct-агент на LangGraph (блок 6.3): кастомный граф или prebuilt")
    parser.add_argument("task", help="задача для агента")
    parser.add_argument("--prebuilt", action="store_true", help="prebuilt-агент create_agent вместо кастомного графа")
    parser.add_argument("--trace", action="store_true", help="напечатать вызовы инструментов")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # консоль Windows (cp1251) и «→» заглушки Telegram
    setup_logging("INFO", stream=sys.stderr)
    try:
        graph = graphs()["prebuilt_graph" if args.prebuilt else "custom_graph"]
        result = asyncio.run(run_graph(graph, args.task, thread_id="cli",
                                       recursion_limit=PREBUILT_RECURSION_LIMIT if args.prebuilt else None))
    except GraphRecursionError:
        print(f"Остановка: {ITERATION_LIMIT} (recursion_limit {PREBUILT_RECURSION_LIMIT})")
        return 1
    except (ValueError, OpenAIError) as exc:      # настройки (ValidationError — подкласс ValueError), провайдер
        print(f"Остановка: {type(exc).__name__}: {exc}")
        return 2
    print(result["answer"])
    print(f"\nШагов модели: {result['steps']}, инструментов: {len(result['tool_calls'])}, токенов: "
          f"{result['total_tokens']} (prompt {result['prompt_tokens']}, completion {result['completion_tokens']})")
    if args.trace:
        print("\n--- tool calls ---", *(json.dumps(c, ensure_ascii=False) for c in result["tool_calls"]), sep="\n")
    return 1 if result["force_finish"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
