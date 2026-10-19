"""
Эксперимент блока 6.5: мультиагент (supervisor + researcher + writer) против одного агента с tools.

Прототипы живут здесь, а не в app/, пока не принято решение (docs/multi-agent-report.md, «## Решение»):
- common.py — общее для обеих реализаций: тестовые вопросы, инструмент search_knowledge_base, промпты,
  счётчик токенов и вызовов модели, проверки ответа, запись experiments/results.json;
- multi_agent_langgraph.py — supervisor-граф на LangGraph (ручной supervisor через Command);
- single_agent_baseline.py — один create_agent с тем же инструментом;
- compare.py — LLM-судья и сравнительная таблица по results.json.
"""
