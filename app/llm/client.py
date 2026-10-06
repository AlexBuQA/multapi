"""
Полный цикл Function Calling (блок 3.1) на OpenAI SDK — тот же клиент работает
с локальным Ollama и с облачными провайдерами из src/config.py.

Цикл одного запроса:
  1. messages (system + user) + tools -> модель;
  2. если в ответе есть tool_calls:
       - в историю добавляется ассистентский ход с tool_calls (без него второй
         запрос отклоняется API),
       - каждый инструмент выполняется (аргументы проверяются по JSON Schema),
       - результат добавляется сообщением role="tool" с тем же tool_call_id,
     и запрос повторяется;
  3. если tool_calls нет — текст ответа и есть финальный ответ (случай, когда
     модель отвечает сразу, без инструментов).
Раундов с инструментами не больше TOOLS_MAX_ROUNDS; после этого модель просят
ответить без инструментов (tool_choice="none").

Небольшие модели (например, llama3.2 3B) иногда пишут вызов инструмента обычным
текстом — {"name": "search_knowledge_base", "parameters": {...}} — вместо поля
tool_calls. Такой текст распознаётся и выполняется как обычный tool_call, а
пользователю сырой JSON никогда не показывается.

System prompt не хранится в коде: он рендерится из app/prompts/system_<версия>.j2.
Каждый шаг пишется JSON-строкой в logs/tool_calls.jsonl (см. app/logging_utils.py).
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from app.config import ToolSettings, tool_settings
from app.logging_utils import get_event_logger, log_event
from app.prompts.loader import render_system_prompt
from app.tools.handlers import DISPATCH, execute_tool, parse_arguments
from app.tools.schemas import TOOLS
from src.robust_client import AllProvidersFailedError, RobustLLMClient

NO_ANSWER = "Не удалось сформировать ответ. Попробуйте переформулировать вопрос."


@dataclass
class ToolCallRecord:
    """Один выполненный вызов инструмента."""

    name: str
    arguments: Any
    result: dict[str, Any]
    call_id: str = ""


@dataclass
class AssistantReply:
    """Итог обработки запроса: ответ, вызванные инструменты и расход токенов."""

    answer: str
    run_id: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def used_tools(self) -> bool:
        return bool(self.tool_calls)


def _usage(response: Any) -> dict[str, int]:
    u = getattr(response, "usage", None)
    prompt = int(getattr(u, "prompt_tokens", 0) or 0)
    completion = int(getattr(u, "completion_tokens", 0) or 0)
    total = int(getattr(u, "total_tokens", 0) or 0) or prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_TOOL_NAME_IN_TEXT = re.compile(r'"name"\s*:\s*"([A-Za-z_]+)"')


def tool_calls_from_text(content: str | None, step: int) -> list[SimpleNamespace]:
    """
    Вызов инструмента, написанный моделью текстом, -> список tool_call.

    Понимает {"name": ..., "parameters"|"arguments": {...}} и список таких объектов
    (в том числе внутри ```json```). Если JSON битый, но имя известного инструмента
    видно, возвращает вызов с исходным текстом вместо аргументов: execute_tool
    ответит модели ошибкой invalid_json, и она сможет повторить вызов корректно.
    """
    text = _CODE_FENCE.sub("", (content or "").strip())
    if not text.startswith(("{", "[")):
        return []

    def call(i: int, name: str, arguments: Any) -> SimpleNamespace:
        raw = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
        return SimpleNamespace(id=f"call_text_{step}_{i}", type="function",
                               function=SimpleNamespace(name=name, arguments=raw))

    try:
        data = json.loads(text)
    except ValueError:
        match = _TOOL_NAME_IN_TEXT.search(text)
        return [call(0, match.group(1), text)] if match and match.group(1) in DISPATCH else []

    items = data if isinstance(data, list) else [data]
    calls = []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or item.get("name") not in DISPATCH:
            return []
        calls.append(call(i, item["name"], item.get("parameters", item.get("arguments", {}))))
    return calls


def _arguments_for_log(raw: Any) -> Any:
    try:
        return parse_arguments(raw)
    except ValueError:
        return raw  # битый JSON логируем как есть


class ToolCallingAssistant:
    """Ассистент техподдержки с инструментами search_knowledge_base и check_service_status."""

    def __init__(
        self,
        client: RobustLLMClient | None = None,
        *,
        config: ToolSettings | None = None,
    ) -> None:
        self.config = config or tool_settings
        self.client = client or RobustLLMClient()
        self.model = self.config.tools_model
        self.system_prompt = render_system_prompt(
            self.config.system_prompt_version,
            product_name=self.config.support_product_name,
            max_sentences=self.config.answer_max_sentences,
        )
        self.events = get_event_logger(self.config.tool_log_path)

    # ------------------------------------------------------------------ #
    def build_messages(
        self, user_input: str, history: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_input})
        return messages

    def ask(
        self, user_input: str, history: list[dict[str, Any]] | None = None
    ) -> AssistantReply:
        reply = AssistantReply(answer="", run_id=uuid.uuid4().hex[:12])
        self._log(reply, "user_input", text=user_input, model=self.model,
                  prompt_version=self.config.system_prompt_version)
        messages = self.build_messages(user_input, history)

        try:
            for step in range(1, self.config.tools_max_rounds + 2):
                tools_allowed = step <= self.config.tools_max_rounds
                message = self._request(messages, reply, step, tools_allowed)
                tool_calls = list(getattr(message, "tool_calls", None) or [])
                text_calls = [] if tool_calls else tool_calls_from_text(message.content, step)

                if not tools_allowed or not (tool_calls or text_calls):
                    # Вызов инструмента текстом на финальном шаге — это не ответ пользователю.
                    content = "" if text_calls else (message.content or "").strip()
                    reply.answer = content or NO_ANSWER
                    break

                if text_calls:
                    tool_calls = text_calls
                    self._log(reply, "tool_call_from_text", step=step, content=message.content)

                call_ids = [tc.id or f"call_{step}_{i}" for i, tc in enumerate(tool_calls)]
                messages.append(self._assistant_turn(
                    message, tool_calls, call_ids, content="" if text_calls else None,
                ))
                for tc, call_id in zip(tool_calls, call_ids):
                    record = self._run_tool(reply, tc, call_id, step)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(record.result, ensure_ascii=False),
                    })
        except AllProvidersFailedError as exc:
            reply.answer = RobustLLMClient.USER_FACING_FAILURE
            self._log(reply, "llm_error", level="error", error=str(exc))

        self._log(reply, "final_answer", text=reply.answer, tool_called=reply.used_tools,
                  tools=[r.name for r in reply.tool_calls])
        self._log(reply, "usage", llm_calls=reply.llm_calls,
                  prompt_tokens=reply.prompt_tokens,
                  completion_tokens=reply.completion_tokens,
                  total_tokens=reply.total_tokens)
        return reply

    # ------------------------------------------------------------------ #
    def _request(
        self, messages: list[dict[str, Any]], reply: AssistantReply,
        step: int, tools_allowed: bool,
    ) -> Any:
        tool_choice = "auto" if tools_allowed else "none"
        self._log(reply, "llm_request", step=step, model=self.model,
                  messages=len(messages), tool_choice=tool_choice)
        started = time.perf_counter()
        response = self.client.complete(
            messages,
            tools=TOOLS,
            tool_choice=tool_choice,
            temperature=self.config.tools_temperature,
            max_tokens=self.config.tools_max_tokens,
            label=f"tools/step{step}",
            model_override=self.model,
        )
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        usage = _usage(response)
        reply.llm_calls += 1
        reply.prompt_tokens += usage["prompt_tokens"]
        reply.completion_tokens += usage["completion_tokens"]
        reply.total_tokens += usage["total_tokens"]

        choice = response.choices[0]
        message = choice.message
        self._log(
            reply, "llm_response", step=step, finish_reason=choice.finish_reason,
            tool_calls=[
                {"name": tc.function.name, "arguments": _arguments_for_log(tc.function.arguments)}
                for tc in (getattr(message, "tool_calls", None) or [])
            ],
            content=message.content or "",
            usage=usage,
            duration_ms=duration_ms,
        )
        return message

    @staticmethod
    def _assistant_turn(
        message: Any, tool_calls: list[Any], call_ids: list[str], content: str | None = None,
    ) -> dict[str, Any]:
        """Ассистентский ход с tool_calls — обязателен перед сообщениями role="tool"."""
        return {
            "role": "assistant",
            # Пустая строка вместо None: Ollama отклоняет content=null.
            "content": message.content or "" if content is None else content,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                        if isinstance(tc.function.arguments, str)
                        else json.dumps(tc.function.arguments, ensure_ascii=False),
                    },
                }
                for tc, call_id in zip(tool_calls, call_ids)
            ],
        }

    def _run_tool(
        self, reply: AssistantReply, tool_call: Any, call_id: str, step: int
    ) -> ToolCallRecord:
        name = tool_call.function.name
        raw_arguments = tool_call.function.arguments
        arguments = _arguments_for_log(raw_arguments)
        self._log(reply, "tool_call", step=step, call_id=call_id, tool=name, arguments=arguments)

        started = time.perf_counter()
        result = execute_tool(name, raw_arguments)
        self._log(reply, "tool_result", step=step, call_id=call_id, tool=name,
                  ok="error" not in result,
                  duration_ms=round((time.perf_counter() - started) * 1000, 1),
                  result=result)

        record = ToolCallRecord(name=name, arguments=arguments, result=result, call_id=call_id)
        reply.tool_calls.append(record)
        return record

    def _log(self, reply: AssistantReply, event: str, level: str = "info", **fields: Any) -> None:
        if level == "error":
            self.events.error(event, extra={"fields": {"run_id": reply.run_id, **fields}})
        else:
            log_event(self.events, event, run_id=reply.run_id, **fields)
