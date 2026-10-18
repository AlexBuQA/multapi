# Агент на LangGraph: кастомный граф, prebuilt и цикл блока 6.2 (блок 6.3)

ReAct-агент блока 6.2 (`app/services/agent_react.py`, цикл `for` по `max_iterations`) переписан на
LangGraph 1.x в `app/services/agent_graph.py`. Агент тот же: системный промпт, три инструмента, лимит
итераций. Собран он двумя путями: `custom_graph` — `StateGraph` руками, `prebuilt_graph` —
`langchain.agents.create_agent`. Обе сборки и цикл 6.2 сравниваются на пяти задачах блока 6.2 по
latency и токенам.

## 1. Конфигурация

| Параметр | Значение |
|---|---|
| Версии | `langgraph` 1.2.14, `langchain` 1.4.4, `langchain-core` 1.6.9, `langchain-openai` 1.7.0 (`uv.lock`) |
| Модель | `AGENT__MODEL` у всех трёх реализаций; канон задания — `gpt-5.4-mini`, прогон — `qwen3:4b-instruct` в Ollama на CPU |
| Клиент модели | `ChatOpenAI` к тому же провайдеру, что у сервиса (`LLM__BASE_URL`, ключ, прокси, сертификаты ОС), Chat Completions (`use_responses_api=False`: Responses API у Ollama нет) |
| Температура | `AGENT__TEMPERATURE`, по умолчанию 0 — у графов и у цикла 6.2 в бенчмарке |
| Лимит итераций | графы — 6 вызовов модели (`MAX_ITERATIONS`); цикл 6.2 — `AGENT__MAX_ITERATIONS` (10), таймаут итерации — `AGENT__TIMEOUT_PER_ITERATION_SEC` |
| Системный промпт | `SYSTEM_PROMPT` блока 6.2 — один текст для всех трёх |
| Инструменты | `search_knowledge_base(query)` — RAG блока 5.5 по `kb_docs`; `get_current_time(timezone)`; `send_telegram_message(chat_id, text)` — заглушка (`app/tools/graph_tools.py`) |

Инструменты — те же функции блоков 6.1–6.2, оформленные `@tool` из `langchain_core.tools`. Docstring —
это description для модели. Текст тот же, что у ReAct-агента 6.2: четыре вопроса — что делает, когда
вызывать, когда НЕ вызывать, что возвращает. Тест сверяет описания, чтобы они не разошлись. Схема
аргументов — pydantic-модель с `extra="forbid"` и шаблонами строк. Неверные аргументы дают
`ValidationError` до вызова функции. Сбой внутри функции (Qdrant не отвечает, неизвестный часовой
пояс) возвращается строкой «Ошибка инструмента …»: `ToolNode` prebuilt-агента ловит только ошибки
аргументов, остальные уронили бы граф. Один и тот же список `TOOLS` идёт в оба графа.

Проверка версии: `python -c "import langgraph; print(langgraph.__version__)"` из задания в
`langgraph` 1.2 падает с `AttributeError` — атрибута `__version__` у пакета нет. Версию показывает
`python -c "import importlib.metadata as m; print(m.version('langgraph'))"` → `1.2.14`.

## 2. State contract

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int
    tool_results: Annotated[list[dict], operator.add]
```

| Поле | Reducer | Что хранит и почему |
|---|---|---|
| `messages` | `add_messages` | история: задача, ответы модели с `tool_calls`, `ToolMessage` с результатами. Узел возвращает только новые сообщения — reducer дописывает их, а сообщение с тем же `id` заменяет |
| `iteration_count` | нет (замена) | число вызовов модели. `call_model` возвращает `iteration_count + 1` целиком, по нему router включает stop-кран |
| `tool_results` | `operator.add` | результаты инструментов для отчёта и трейсинга: итерация, имя, аргументы, результат (200 символов), время. Каждый `execute_tool` добавляет свои записи к списку |

В state нет клиента модели, http-сессий и ключей. Модель с привязанными инструментами и словарь
инструментов живут в замыканиях узлов, собранных `build_custom_graph(model, tools)`. Поэтому state
целиком сериализуем: checkpointer сохранит его без изменений. Системный промпт в state тоже не
кладётся — `call_model` добавляет его перед каждым вызовом, как это делает `create_agent`.

## 3. Router и stop conditions

```python
def route_after_model(state: AgentState) -> Literal["execute_tool", "force_finish"]:
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "force_finish"
    last = state["messages"][-1]
    return "execute_tool" if isinstance(last, AIMessage) and last.tool_calls else "force_finish"
```

Router — чистая функция: только читает state, ничего в него не пишет, сети не трогает (тест сверяет
state до и после). Тип возврата — `Literal`.

**Условия остановки кастомного графа:**
- **Модель ответила без `tool_calls`** → `force_finish` прокидывает её ответ как есть → `END`.
- **`iteration_count >= 6`** — stop-кран, даже если модель снова просит инструмент → `force_finish`.
  Узел закрывает каждый невыполненный вызов сообщением `ToolMessage("Не выполнено: превышен лимит
  итераций.")`, чтобы история осталась валидной для продолжения, и добавляет явный финальный ответ
  «Превышен лимит итераций (6): агент не дал финального ответа.». Молча в `END` граф не уходит.
- **Ошибка инструмента** (неизвестное имя, неверные аргументы, исключение) — не остановка: она
  возвращается модели `ToolMessage`, и цикл идёт дальше до ответа или крана.

**У prebuilt-агента** выход по ответу без `tool_calls` встроен в `create_agent`. Stop-кран —
`ModelCallLimitMiddleware(run_limit=6, exit_behavior="end")`: перед седьмым вызовом модели агент уходит
в конец с сообщением «Model call limits exceeded: run limit (6/6)». `recursion_limit=50` в config —
страховка от зацикливания самого графа: одна итерация prebuilt-агента — 4 супершага (`before_model`,
`model`, `after_model`, `tools`), и лимит LangGraph по умолчанию (25) кончился бы раньше крана.

Кран проверен на задаче со сломанным инструментом:
- тесты `test_stop_tap_on_a_broken_tool` и `test_prebuilt_stops_after_six_model_calls` — подставная
  модель на каждом шаге снова просит упавший поиск, оба графа останавливаются ровно на 6-м вызове;
- на живой модели — `python scripts/bench_agents.py --stop-check`: поиск всегда отвечает «база знаний не
  ответила», результат — `docs/agent-graph-bench/stop_check.json`.

## 4. Схема кастомного графа

`python scripts/visualize_graph.py` → `docs/agent-graph-custom.mmd` (ниже) и `docs/agent-graph-prebuilt.mmd`.
Тест сверяет файлы и этот блок с тем, что строит код сейчас. Открываются в https://mermaid.live;
`--png` дополнительно сохраняет `docs/agent-graph.png` через mermaid.ink (нужна сеть).

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	call_model(call_model)
	execute_tool(execute_tool)
	force_finish(force_finish)
	__end__([<p>__end__</p>]):::last
	__start__ --> call_model;
	call_model -.-> execute_tool;
	call_model -.-> force_finish;
	execute_tool --> call_model;
	force_finish --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

Сплошные рёбра — безусловные, пунктир — ветки `route_after_model`. Схема prebuilt-агента — те же
`model` ⇄ `tools`, плюс узлы `ModelCallLimitMiddleware.before_model` / `.after_model` (крана).

## 5. Бенчмарк

`python scripts/bench_agents.py`: 5 задач блока 6.2 × 3 реализации × 3 повтора. Реализации:
- `b62_loop` — цикл блока 6.2: критик после каждого инструмента, проверка финального ответа;
- `custom` — `custom_graph`;
- `prebuilt` — `prebuilt_graph`.

Что измеряется:
- latency — `time.perf_counter()` вокруг вызова;
- токены — сумма `usage_metadata` ответов модели (у `b62_loop` — `usage`, вместе с критиком);
- `total_steps` — вызовы модели агентом (у `b62_loop` — итерации цикла);
- корректность — то же правило задачи, что в `scripts/compare_agents.py`.

Порядок прогонов — повтор → задача → реализация: реализации чередуются, и прогрев не достаётся
одной из них. Графы вызываются с `config={"configurable": {"thread_id": f"bench-{task_id}"}}`. Журнал —
`docs/agent-graph-bench/runs.jsonl`, сводка — `summary.json`. Таблицы ниже скрипт вписывает сам.

<!-- bench:start -->
Заполняется `python scripts/bench_agents.py`.
<!-- bench:end -->

## 6. Custom против prebuilt

| | `custom_graph` | `prebuilt_graph` |
|---|---|---|
| Пишется руками | `AgentState`, три узла, router, рёбра; разбор `tool_calls`, вызов инструмента, ошибки → `ToolMessage`; кран и финальный ответ при лимите; `strict=True` в `bind_tools` | одна строка `create_agent(model, tools, system_prompt)` и два middleware: кран (`ModelCallLimitMiddleware`) и `strict_tools` (`strict` в `model_settings`) |
| Делает сам фреймворк | — | цикл модель ⇄ `ToolNode`, параллельные вызовы, ошибки аргументов → сообщение модели, выход по ответу без `tool_calls`, системный промпт перед вызовом |
| State | свой: `iteration_count`, `tool_results` | свой у `create_agent` (`messages` и служебные поля middleware); `tool_results` нет — результаты берутся из `ToolMessage` |
| Лимит | 6 вызовов, шестой вызов инструмента не выполняется и закрывается `ToolMessage` | 6 вызовов, инструменты шестого вызова выполняются, седьмого вызова модели нет |
| Сообщение при лимите | своё, по-русски | «Model call limits exceeded: run limit (6/6)» |
| Схема | 3 узла | 4 узла: `model`, `tools` и два узла middleware |

Числа и вывод «какой путь когда» — после прогона бенчмарка.

## 7. Баг, найденный при отладке

**`extra="forbid"` не доходил до модели.** У схем аргументов pydantic-модели с `extra="forbid"`:
в JSON Schema у них `additionalProperties: false`, как у строгих схем блока 6.2. Граф прогнали через
настоящий `ChatOpenAI` на подставном OpenAI-совместимом сервере, который пишет каждый запрос в журнал.
В журнале у инструментов `additionalProperties` не было: `langchain-openai` при конвертации
`args_schema` в формат OpenAI выкидывает его, если не передан `strict=True`. Модель видела схему, где
лишние поля разрешены; отвергал их только pydantic уже после вызова.

Исправление:
- **`custom_graph`** — `model.bind_tools(tools, strict=True)`: в запросе появились
  `additionalProperties: false` и `strict: true`, как в 6.2.
- **`prebuilt_graph`** — так нельзя: `create_agent` связывает инструменты сам. `strict` передаётся
  только через `model_settings`, отсюда middleware `strict_tools` (`wrap_model_call`).

Тесты проверяют аргументы `bind_tools` у обеих сборок. Ollama поле `strict` не применяет, а
`ValidationError` ловит неверные аргументы в любом случае: исправление важно для облачной модели, у
которой strict-схема ограничивает саму генерацию.

## 8. Что блокирует переход к персистентности

- **`thread_id` уже передаётся.** Бенчмарк и CLI вызывают графы с
  `config={"configurable": {"thread_id": …}}`. Без checkpointer это no-op, но вызов менять не придётся.
- **Нет самого checkpointer.** `AsyncSqliteSaver` — отдельный пакет `langgraph-checkpoint-sqlite` (в
  зависимостях его нет), `AsyncPostgresSaver` — `langgraph-checkpoint-postgres`. Postgres у сервиса уже
  есть (блок 4.1), так что для сервиса естественнее второй. Подключение — `builder.compile(checkpointer=…)`
  и `create_agent(checkpointer=…)`.
- **Счётчики в одном thread.** `iteration_count` без reducer, и вход `initial_state` сбрасывает его в 0 на
  каждый запуск. Без этого второе сообщение в том же thread сразу ушло бы в `force_finish`.
  `tool_results` с `operator.add` будет копиться по всему thread — для трейсинга это нормально, но
  список растёт без ограничений. У prebuilt-агента — `run_limit`, а не `thread_limit`, по той же причине.
- **История должна быть валидной для повторной отправки.** После лимита `force_finish` закрывает
  невыполненные вызовы `ToolMessage`. Иначе в сохранённой истории остался бы `AIMessage` с
  `tool_calls` без ответов, и OpenAI отклонил бы её при продолжении диалога.
- **State сериализуем:** сообщения, числа и словари; клиентов и ключей в нём нет.
- **Human-in-the-loop не встроен.** Для `interrupt` перед `send_telegram_message` пишущий инструмент
  нужно вынести в отдельный узел, а не держать в общем `execute_tool`. Тогда можно поставить
  `interrupt_before` или вызвать `interrupt()` внутри узла.
- **Не перенесены критик и проверка финального ответа** блока 6.2. В графе это будут отдельные узлы
  после `execute_tool` и перед `END`, а не код внутри цикла.
