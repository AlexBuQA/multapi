# Архитектура мультиагента (блок 6.5)

Файл пишет `python experiments/multi_agent_langgraph.py` в конце прогона: `app.get_graph().draw_mermaid()`
supervisor-графа и `get_graph().draw_mermaid()` агента researcher. Руками не править — перезапишется.
Устройство, замеры и решение — [multi-agent-report.md](multi-agent-report.md).

## Supervisor-граф

Пунктир — переходы через `Command(goto=...)`. Supervisor выбирает агента handoff-инструментом
`transfer_to_researcher(task)` или `transfer_to_writer(task)`; researcher всегда возвращает управление
supervisor, writer — последний шаг (сплошная стрелка в `__end__`). Бюджет — 5 передач управления
на вопрос; последняя зарезервирована за writer.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	supervisor(supervisor)
	researcher(researcher)
	writer(writer)
	__end__([<p>__end__</p>]):::last
	__start__ --> supervisor;
	researcher -.-> supervisor;
	supervisor -.-> researcher;
	supervisor -.-> writer;
	writer --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

## Researcher изнутри

`create_agent(name="researcher", tools=[search_knowledge_base])` с бюджетами: `ToolCallLimitMiddleware` —
не больше 4 поисков, `ModelCallLimitMiddleware` — не больше 6 вызовов модели. Writer — тот же
`create_agent` без инструментов, один вызов модели. Single-agent baseline
(`experiments/single_agent_baseline.py`) устроен как researcher, с теми же бюджетами, но с промптом обеих
ролей.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	model(model)
	tools(tools)
	ToolCallLimitMiddleware\2eafter_model(ToolCallLimitMiddleware.after_model)
	ModelCallLimitMiddleware\2ebefore_model(ModelCallLimitMiddleware.before_model)
	ModelCallLimitMiddleware\2eafter_model(ModelCallLimitMiddleware.after_model)
	__end__([<p>__end__</p>]):::last
	ModelCallLimitMiddleware\2eafter_model --> ToolCallLimitMiddleware\2eafter_model;
	ModelCallLimitMiddleware\2ebefore_model -.-> __end__;
	ModelCallLimitMiddleware\2ebefore_model -.-> model;
	ToolCallLimitMiddleware\2eafter_model -.-> ModelCallLimitMiddleware\2ebefore_model;
	ToolCallLimitMiddleware\2eafter_model -.-> __end__;
	ToolCallLimitMiddleware\2eafter_model -.-> tools;
	__start__ --> ModelCallLimitMiddleware\2ebefore_model;
	model --> ModelCallLimitMiddleware\2eafter_model;
	tools -.-> ModelCallLimitMiddleware\2ebefore_model;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
