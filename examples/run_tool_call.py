"""
Прогон Function Calling на трёх тест-запросах (блок 3.1):
  (а) запрос, который точно требует tool;
  (б) запрос, который точно не требует tool;
  (в) пограничный случай — фиксируем, что решит модель.

Запуск из корня проекта:
    python examples/run_tool_call.py                 # три тест-запроса
    python examples/run_tool_call.py "свой вопрос"   # произвольный запрос

Подробный лог каждого шага — logs/tool_calls.jsonl (JSON-строки).
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.config import tool_settings  # noqa: E402
from app.llm.client import AssistantReply, ToolCallingAssistant  # noqa: E402

CASES = [
    ("а", "Запрос, который точно требует tool",
     "Уже 40 минут не приходит письмо для сброса пароля. У вас что-то сломалось?"),
    ("б", "Запрос, который точно не требует tool",
     "Спасибо, всё заработало!"),
    ("в", "Пограничный случай",
     "Подойдёт ли пароль из 6 символов?"),
]


def _short(value: object, limit: int = 300) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…"


def print_reply(reply: AssistantReply) -> None:
    if reply.tool_calls:
        for call in reply.tool_calls:
            print(f"Вызов инструмента : {call.name}({_short(call.arguments, 200)})")
            print(f"Результат         : {_short(call.result)}")
    else:
        print("Вызов инструмента : нет — модель ответила сразу")
    print(f"Ответ             : {reply.answer}")
    print(
        f"Токены            : LLM-вызовов={reply.llm_calls} | "
        f"prompt={reply.prompt_tokens} | completion={reply.completion_tokens} | "
        f"total={reply.total_tokens}"
    )


def main(argv: list[str]) -> None:
    assistant = ToolCallingAssistant()
    print(f"Модель: {assistant.model} | промпт: system_{tool_settings.system_prompt_version}.j2")

    cases = [("—", "Свой запрос", " ".join(argv))] if argv else CASES
    for code, title, question in cases:
        print(f"\n=== Кейс ({code}): {title} ===")
        print(f"Запрос            : {question}")
        print_reply(assistant.ask(question))

    print(f"\nПодробный лог: {tool_settings.tool_log_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
