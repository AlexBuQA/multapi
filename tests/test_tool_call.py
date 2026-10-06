"""
Тесты блока 3.1 (Function Calling). Сеть не используется: ответы модели
подменены сценарием, всё остальное — настоящий код (JSON Schema, обработчики,
цикл tool_call, retry-обёртка RobustLLMClient, JSON-лог).

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import copy
import inspect
import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jsonschema import Draft202012Validator  # noqa: E402

import app.llm.client as client_module  # noqa: E402
from app.config import ToolSettings  # noqa: E402
from app.llm.client import NO_ANSWER, ToolCallingAssistant  # noqa: E402
from app.logging_utils import close_event_logger  # noqa: E402
from app.prompts.loader import TOOL_PROMPTS_DIR, render_system_prompt  # noqa: E402
from app.tools import schemas  # noqa: E402
from app.tools import handlers  # noqa: E402
from app.tools.handlers import check_service_status, execute_tool, search_knowledge_base  # noqa: E402
from src.robust_client import RobustLLMClient  # noqa: E402
from src.utils import UsageTracker  # noqa: E402


# --------------------------------------------------------------------------- #
# Сценарная «модель»
# --------------------------------------------------------------------------- #
def tool_call(name: str, arguments: dict | str, call_id: str = "call_1") -> SimpleNamespace:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name, arguments=raw))


def response(content: str | None = None, tool_calls: list | None = None,
             prompt: int = 100, completion: int = 20) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message,
                                 finish_reason="tool_calls" if tool_calls else "stop")],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                              total_tokens=prompt + completion),
    )


class ScriptedSDK:
    """Отдаёт заранее заданные ответы и запоминает каждый запрос."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        item = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(item, Exception):
            raise item
        return item


class ScriptedClient(RobustLLMClient):
    def __init__(self, sdk: ScriptedSDK):
        super().__init__(usage=UsageTracker())
        self.sdk = sdk

    def _client_for(self, provider):
        return self.sdk


class AssistantTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.log_path = Path(self._tmp.name) / "tool_calls.jsonl"
        client_logger = logging.getLogger("client")
        client_logger.disabled = True
        self.addCleanup(setattr, client_logger, "disabled", False)

    def tearDown(self) -> None:
        for assistant in getattr(self, "_assistants", []):
            close_event_logger(assistant.events)
        self._tmp.cleanup()

    def make_assistant(self, sdk: ScriptedSDK, **overrides) -> ToolCallingAssistant:
        config = ToolSettings(tool_log_path=self.log_path, **overrides)
        assistant = ToolCallingAssistant(ScriptedClient(sdk), config=config)
        self._assistants = getattr(self, "_assistants", []) + [assistant]
        return assistant

    def log_events(self) -> list[dict]:
        with open(self.log_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# JSON Schema и промпты
# --------------------------------------------------------------------------- #
class TestSchemasAndPrompts(unittest.TestCase):
    def test_schemas_are_valid_json_schema(self):
        for name, params in schemas.TOOL_PARAMETERS.items():
            with self.subTest(tool=name):
                Draft202012Validator.check_schema(params)

    def test_tool_descriptions_come_from_files(self):
        for tool in schemas.TOOLS:
            name = tool["function"]["name"]
            file_text = (TOOL_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
            with self.subTest(tool=name):
                self.assertEqual(tool["function"]["description"], file_text)

    def test_parameter_descriptions_are_constants(self):
        props = schemas.SEARCH_KNOWLEDGE_BASE_PARAMETERS["properties"]
        self.assertIs(props["query"]["description"], schemas.QUERY_DESCRIPTION)
        self.assertIs(props["product"]["description"], schemas.PRODUCT_DESCRIPTION)

    def test_enums_match_data_files(self):
        kb = json.loads((ROOT / "data" / "knowledge_base.json").read_text(encoding="utf-8"))
        status = json.loads((ROOT / "data" / "service_status.json").read_text(encoding="utf-8"))
        self.assertLessEqual({a["product"] for a in kb["articles"]}, set(schemas.PRODUCTS))
        self.assertEqual(set(status["components"]) | {"all"}, set(schemas.COMPONENTS))

    def test_system_prompt_rendered_from_file(self):
        prompt = render_system_prompt("v1", product_name="Тест-Продукт", max_sentences=3)
        self.assertIn("«Тест-Продукт»", prompt)
        self.assertNotIn("{{", prompt)

    def test_client_has_no_inline_system_prompt(self):
        source = inspect.getsource(client_module)
        self.assertIn("render_system_prompt(", source)
        self.assertNotIn("Ты — ассистент", source)


# --------------------------------------------------------------------------- #
# Обработчики (реальное чтение файлов)
# --------------------------------------------------------------------------- #
class TestHandlers(unittest.TestCase):
    def test_search_finds_password_reset(self):
        result = search_knowledge_base("Как сбросить пароль, если я его забыла?")
        self.assertGreater(result["found"], 0)
        self.assertEqual(result["articles"][0]["id"], "KB-001")
        self.assertEqual(result["articles"][0]["section"], "2.1")

    def test_search_product_filter(self):
        result = search_knowledge_base("ошибка 429", product="api")
        self.assertTrue(all(a["id"] in {"KB-008", "KB-009"} for a in result["articles"]))
        self.assertEqual(result["articles"][0]["id"], "KB-009")

    def test_search_nothing_found(self):
        self.assertEqual(search_knowledge_base("погода в Москве")["found"], 0)

    def test_status_single_and_all(self):
        email = check_service_status("email")["components"][0]
        self.assertEqual(email["status"], "degraded")
        self.assertIn("incident", email)
        self.assertEqual(len(check_service_status("all")["components"]), 5)

    def test_dispatch_matches_tools(self):
        """Allowlist DISPATCH и описания TOOLS согласованы."""
        self.assertEqual(set(handlers.DISPATCH), set(schemas.TOOL_NAMES))

    def test_exception_inside_tool_returned_as_error(self):
        def broken(**kwargs):
            raise RuntimeError("база знаний недоступна")

        original = handlers.DISPATCH["search_knowledge_base"]
        handlers.DISPATCH["search_knowledge_base"] = broken
        logging.disable(logging.CRITICAL)
        try:
            result = execute_tool("search_knowledge_base", {"query": "пароль"})
        finally:
            handlers.DISPATCH["search_knowledge_base"] = original
            logging.disable(logging.NOTSET)
        self.assertEqual(result["error"], "tool_failed")
        self.assertIn("база знаний недоступна", result["message"])

    def test_schema_echo_with_value_is_unwrapped(self):
        """llama3.2 присылала {"query": {"type": ..., "description": ..., "value": "..."}}."""
        echoed = {"query": {"description": "Суть вопроса", "type": "string",
                            "value": "не приходит письмо для сброса пароля"}}
        result = execute_tool("search_knowledge_base", echoed)
        self.assertNotIn("error", result)
        self.assertEqual(result["query"], "не приходит письмо для сброса пароля")

    def test_schema_echo_without_value_gets_hint(self):
        result = execute_tool("search_knowledge_base",
                              {"query": {"type": "string", "description": "Суть вопроса"}})
        self.assertEqual(result["error"], "invalid_arguments")
        self.assertIn("описание параметра", result["message"])
        self.assertIn('{"query": "сброс пароля"}', result["message"])

    def test_execute_tool_errors_do_not_raise(self):
        self.assertEqual(execute_tool("check_service_status", {"component": "Почта"})["error"],
                         "invalid_arguments")
        self.assertEqual(execute_tool("search_knowledge_base", "{not json")["error"], "invalid_json")
        unknown = execute_tool("delete_everything", {})
        self.assertEqual(unknown["error"], "unknown_tool")
        self.assertIn("search_knowledge_base", unknown["message"])  # модель видит доступные


# --------------------------------------------------------------------------- #
# Полный цикл tool_call
# --------------------------------------------------------------------------- #
class TestToolCallCycle(AssistantTestCase):
    def test_tool_called_then_final_answer(self):
        """Кейс (а): tool_call -> результат функции -> финальный текст."""
        sdk = ScriptedSDK(
            response(tool_calls=[tool_call("check_service_status", {"component": "email"})]),
            response("Сейчас письма задерживаются до 30 минут.", prompt=300, completion=30),
        )
        reply = self.make_assistant(sdk).ask("Не приходит письмо для сброса пароля")

        self.assertEqual(len(sdk.requests), 2)
        first = sdk.requests[0]
        self.assertEqual([t["function"]["name"] for t in first["tools"]],
                         ["search_knowledge_base", "check_service_status"])
        self.assertEqual(first["tool_choice"], "auto")

        # Во втором запросе: ассистентский ход с tool_calls и ответ role="tool".
        second = sdk.requests[1]["messages"]
        self.assertEqual([m["role"] for m in second], ["system", "user", "assistant", "tool"])
        self.assertEqual(second[2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(second[2]["content"], "")
        self.assertEqual(second[3]["tool_call_id"], "call_1")
        self.assertEqual(json.loads(second[3]["content"])["components"][0]["status"], "degraded")

        self.assertTrue(reply.used_tools)
        self.assertEqual(reply.tool_calls[0].arguments, {"component": "email"})
        self.assertEqual(reply.answer, "Сейчас письма задерживаются до 30 минут.")
        self.assertEqual((reply.llm_calls, reply.total_tokens), (2, 120 + 330))

    def test_direct_text_answer(self):
        """Кейс (б): модель не вызывает tool — ответ без второго запроса, код не падает."""
        sdk = ScriptedSDK(response("Рада, что всё получилось!"))
        reply = self.make_assistant(sdk).ask("Спасибо, всё заработало!")
        self.assertEqual(len(sdk.requests), 1)
        self.assertFalse(reply.used_tools)
        self.assertEqual(reply.answer, "Рада, что всё получилось!")

    def test_parallel_tool_calls_in_one_response(self):
        """Два tool_calls в одном ответе: оба выполняются, у каждого свой tool_call_id."""
        sdk = ScriptedSDK(
            response(tool_calls=[
                tool_call("check_service_status", {"component": "email"}, "call_a"),
                tool_call("search_knowledge_base", {"query": "не приходит письмо"}, "call_b"),
            ]),
            response("Письма задерживаются; проверьте также «Спам» (раздел 3.1)."),
        )
        reply = self.make_assistant(sdk).ask("Не приходит письмо")
        second = sdk.requests[1]["messages"]
        self.assertEqual([m["role"] for m in second],
                         ["system", "user", "assistant", "tool", "tool"])
        self.assertEqual([c["id"] for c in second[2]["tool_calls"]], ["call_a", "call_b"])
        self.assertEqual([m["tool_call_id"] for m in second[3:]], ["call_a", "call_b"])
        self.assertEqual([r.name for r in reply.tool_calls],
                         ["check_service_status", "search_knowledge_base"])

    def test_tool_call_written_as_text_is_executed(self):
        """Модель написала вызов текстом вместо tool_calls — он выполняется как обычный."""
        sdk = ScriptedSDK(
            response('{"name": "check_service_status", "parameters": {"component": "email"}}'),
            response("Письма сейчас задерживаются до 30 минут."),
        )
        reply = self.make_assistant(sdk).ask("Не приходит письмо")
        assistant_turn = sdk.requests[1]["messages"][2]
        self.assertEqual(assistant_turn["content"], "")
        self.assertEqual(assistant_turn["tool_calls"][0]["function"]["name"], "check_service_status")
        self.assertEqual(reply.tool_calls[0].result["components"][0]["status"], "degraded")
        self.assertEqual(reply.answer, "Письма сейчас задерживаются до 30 минут.")
        self.assertIn("tool_call_from_text", [e["event"] for e in self.log_events()])

    def test_broken_text_tool_call_never_shown_to_user(self):
        """Битый JSON вызова (как у llama3.2 в кейсе (в)) — модель получает invalid_json."""
        broken = ('{"name": "search_knowledge_base", "parameters": '
                  '{"query": {"type":"string","description":"Суть вопроса"}"}}')
        sdk = ScriptedSDK(response(broken), response("Пароль должен быть не короче 8 символов (раздел 2.2)."))
        reply = self.make_assistant(sdk).ask("Подойдёт ли пароль из 6 символов?")
        tool_message = sdk.requests[1]["messages"][-1]
        self.assertEqual(json.loads(tool_message["content"])["error"], "invalid_json")
        self.assertNotIn('"name"', reply.answer)

    def test_text_tool_call_on_final_step_is_not_an_answer(self):
        text_call = response('{"name": "search_knowledge_base", "parameters": {"query": "пароль"}}')
        sdk = ScriptedSDK(text_call)
        reply = self.make_assistant(sdk, tools_max_rounds=1).ask("пароль")
        self.assertEqual(reply.answer, NO_ANSWER)

    def test_invalid_arguments_returned_to_model(self):
        sdk = ScriptedSDK(
            response(tool_calls=[tool_call("check_service_status", {"component": "Почта"})]),
            response("Уточните, пожалуйста, что именно не работает."),
        )
        reply = self.make_assistant(sdk).ask("Почта не работает")
        tool_message = sdk.requests[1]["messages"][-1]
        self.assertEqual(json.loads(tool_message["content"])["error"], "invalid_arguments")
        self.assertEqual(reply.answer, "Уточните, пожалуйста, что именно не работает.")

    def test_max_rounds_then_answer_without_tools(self):
        looping = response(tool_calls=[tool_call("search_knowledge_base", {"query": "пароль"})])
        sdk = ScriptedSDK(looping, looping, response("Итоговый ответ."))
        reply = self.make_assistant(sdk, tools_max_rounds=2).ask("пароль")
        self.assertEqual([r["tool_choice"] for r in sdk.requests], ["auto", "auto", "none"])
        self.assertEqual(len(reply.tool_calls), 2)
        self.assertEqual(reply.answer, "Итоговый ответ.")

    def test_model_unavailable_gives_user_message(self):
        sdk = ScriptedSDK(RuntimeError("model does not support tools"))
        reply = self.make_assistant(sdk).ask("Как сбросить пароль?")
        self.assertEqual(reply.answer, RobustLLMClient.USER_FACING_FAILURE)
        self.assertIn("llm_error", [e["event"] for e in self.log_events()])

    def test_log_has_every_step(self):
        """Лог: input -> tool name + args -> tool result -> final answer -> tokens."""
        sdk = ScriptedSDK(
            response(tool_calls=[tool_call("search_knowledge_base", {"query": "сброс пароля"})]),
            response("Нажмите «Забыли пароль?» (раздел 2.1).", prompt=250, completion=25),
        )
        reply = self.make_assistant(sdk).ask("Как сбросить пароль?")
        events = [e for e in self.log_events() if e["run_id"] == reply.run_id]
        self.assertEqual(
            [e["event"] for e in events],
            ["user_input", "llm_request", "llm_response", "tool_call", "tool_result",
             "llm_request", "llm_response", "final_answer", "usage"],
        )
        by_event = {e["event"]: e for e in events}
        self.assertEqual(by_event["user_input"]["text"], "Как сбросить пароль?")
        self.assertEqual(by_event["tool_call"]["tool"], "search_knowledge_base")
        self.assertEqual(by_event["tool_call"]["arguments"], {"query": "сброс пароля"})
        self.assertEqual(by_event["tool_result"]["result"]["articles"][0]["id"], "KB-001")
        self.assertEqual(by_event["final_answer"]["text"], reply.answer)
        self.assertEqual(by_event["usage"]["total_tokens"], 120 + 275)
        self.assertIn("duration_ms", by_event["llm_response"])


if __name__ == "__main__":
    unittest.main()
