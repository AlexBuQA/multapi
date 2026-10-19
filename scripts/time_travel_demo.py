"""
Time travel по checkpoint'ам персистентного агента (блок 6.4). Офлайн: AsyncSqliteSaver(":memory:"),
модель — детерминированный сценарий без сети, поиск по базе знаний — заглушка с фрагментом корпуса.
Отправка — TelegramOutbox над заглушкой блока 6.1 (печать в stdout, настоящий Telegram не вызывается).

    python scripts/time_travel_demo.py

Что печатает:
1. запуск до interrupt — payload __interrupt__ (превью сообщения клиенту), затем Command(resume=True);
2. историю чек-пойнтов thread — aget_state_history: (checkpoint_id, step, next);
3. состояние на прошлом чек-пойнте — aget_state с checkpoint_id: черновик готов, sent=False, next —
   узел подтверждения;
4. две ветки из одинакового входа на двух thread_id: отказ (Command(resume=False) → sent=False) и
   одобрение (sent=True);
5. нюанс resume: повторный resume того же interrupt-чек-пойнта с другим решением возвращает первое
   решение — оно лежит в checkpointer как pending write. Узел отправки при этом выполняется заново, и
   второе сообщение клиенту не уходит только благодаря ключу идемпотентности request_id.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from app.services.agent_persistent import (  # noqa: E402
    CONFIRM,
    SEND,
    build_agent,
    new_turn,
    thread_config,
)
from app.tools.graph_tools import TOOLS, SearchArgs  # noqa: E402

TASK = ("Найди в базе знаний, за сколько дней до окончания периода приходит письмо об автопродлении, и отправь "
        "клиенту в Telegram-чат 12345 короткое напоминание об этом.")
FRAGMENT = ("[billing/auto-renewal.docx, близость 0.73] Автопродление тарифа. Автопродление включено по умолчанию: "
            "за 3 дня до окончания периода приходит письмо, в день окончания оплата списывается с сохранённой карты.")
REMINDER = "Напоминаем: письмо об автопродлении тарифа приходит за 3 дня до окончания периода."


class ScriptedChat(BaseChatModel):
    """Модель-сценарий: решение — по последнему ToolMessage, без счётчиков и сети. Поиск → отправка →
    ответ по результату отправки. id вызовов постоянные: request_id зависит только от thread_id."""

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChat:
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        last = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)
        if last is None:
            reply = AIMessage(content="", tool_calls=[{"name": "search_knowledge_base", "id": "call_search_1",
                                                       "args": {"query": "за сколько дней приходит письмо об автопродлении"}}])
        elif last.name == "search_knowledge_base":
            reply = AIMessage(content="", tool_calls=[{"name": SEND, "id": "call_send_1",
                                                       "args": {"chat_id": "12345", "text": REMINDER}}])
        elif str(last.content).startswith(("Сообщение отправлено", "Уже отправлено")):
            reply = AIMessage(content="Напоминание отправлено клиенту в чат 12345.")
        else:
            reply = AIMessage(content="Сообщение клиенту не отправлено: оператор отклонил отправку.")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def kb_stub(query: str) -> str:
    return FRAGMENT


def line(title: str) -> None:
    print(f"\n=== {title} ===")


def answer(values: dict[str, Any]) -> str:
    last = next((m for m in reversed(values["messages"]) if isinstance(m, AIMessage)), None)
    return str(last.content) if last else "—"


async def demo() -> None:
    search = StructuredTool.from_function(func=kb_stub, name="search_knowledge_base",
                                          description=TOOLS[0].description, args_schema=SearchArgs)
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await saver.setup()
        graph = build_agent(saver, model=ScriptedChat(), tools=[search, *TOOLS[1:]])
        approve = thread_config("demo-approve")

        line("1. Запуск до interrupt (thread demo-approve)")
        result = await graph.ainvoke(new_turn([HumanMessage(TASK)]), approve)
        [pause] = result["__interrupt__"]
        print("__interrupt__ payload:")
        print(json.dumps(pause.value, ensure_ascii=False, indent=2))
        snapshot = await graph.aget_state(approve)
        print(f"snapshot.next = {snapshot.next}, sent = {snapshot.values['sent']}")
        print("\nCommand(resume=True):")
        result = await graph.ainvoke(Command(resume=True), approve)
        print(f"sent = {result['sent']}; ответ агента: {answer(result)}")

        line("2. История чек-пойнтов: aget_state_history(demo-approve), новые сверху")
        history = [s async for s in graph.aget_state_history(approve)]
        print(f"{'checkpoint_id':<38} {'step':>4}  next")
        for snap in history:
            print(f"{snap.config['configurable']['checkpoint_id']:<38} {snap.metadata.get('step', ''):>4}  "
                  f"{', '.join(snap.next) or '— (конец)'}")
        past = next(s for s in history if s.next == (CONFIRM,))
        past_id = past.config["configurable"]["checkpoint_id"]

        line("3. Состояние на прошлом чек-пойнте — до отправки")
        old = await graph.aget_state(thread_config("demo-approve", checkpoint_id=past_id))
        print(f"checkpoint_id = {past_id}")
        print(f"next = {old.next}")
        print(f"sent = {old.values['sent']}")
        print(f"draft = {json.dumps(old.values['draft'], ensure_ascii=False)}")
        print(f"interrupts = {[i.value['type'] for i in old.interrupts]}")

        line("4. Две ветки из одинакового входа (два thread_id)")
        reject = thread_config("demo-reject")
        await graph.ainvoke(new_turn([HumanMessage(TASK)]), reject)
        rejected = await graph.ainvoke(Command(resume=False), reject)
        current = await graph.aget_state(approve)
        print(f"{'thread':<14} {'resume':<6} {'sent':<5} ответ агента")
        print(f"{'demo-approve':<14} {'True':<6} {current.values['sent']!s:<5} {answer(current.values)}")
        print(f"{'demo-reject':<14} {'False':<6} {rejected['sent']!s:<5} {answer(rejected)}")

        line("5. Нюанс: повторный resume прошлого interrupt-чек-пойнта с другим решением")
        print(f"Command(resume=False) на checkpoint_id {past_id} (там уже было resume=True):")
        replay = await graph.ainvoke(Command(resume=False), thread_config("demo-approve", checkpoint_id=past_id))
        tool = next(m for m in reversed(replay["messages"]) if isinstance(m, ToolMessage))
        print(f"sent = {replay['sent']} — применилось первое решение (True), а не новое")
        print(f"узел отправки выполнился заново, ответ отправки: {tool.content}")
        print("Вывод: resume доставляет пропущенное значение в interrupt, а не переигрывает решение. Другое решение — "
              "другой thread или форк через update_state; от повторной отправки защищает ключ request_id.")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")     # консоль Windows (cp1251) и «→» заглушки Telegram
    from app.observability.logging import setup_logging

    setup_logging("WARNING", stream=sys.stderr)      # строки agent_model и т. п. не мешают выводу демо
    asyncio.run(demo())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
