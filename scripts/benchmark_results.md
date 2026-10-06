# Результаты бенчмарка sync vs async (блок 3.3)

Файл обновляет `scripts/benchmark.py`: у каждой цели свой раздел.

<!-- target:mock -->
## Цель: mock

мок OpenAI API, задержка 1 с на ответ; запросов: 20; дата: 2026-10-06 20:39.

| Режим | Время | Ускорение к sync | Ошибок |
|-------|-------|------------------|--------|
| sync, последовательно | 21.4 с | ×1.0 | 0 |
| async `batch_chat`, concurrency=1 | 20.3 с | ×1.1 | 0 |
| async `batch_chat`, concurrency=5 | 4.1 с | ×5.3 | 0 |
| async `batch_chat`, concurrency=10 | 2.1 с | ×10.2 | 0 |

Невалидная модель на 3-м запросе из 10 (concurrency=10):

| Метод | Время | Результат | Поведение |
|-------|-------|-----------|-----------|
| `batch_chat` (gather, return_exceptions) | 1.0 с | 9 из 10 ответов получены; ошибка на позиции [3] — `AllProvidersFailedError` | батч не упал |
| `batch_chat_strict` (TaskGroup) | 0.0 с | ни одного ответа: `ExceptionGroup` (ошибок: 1, `AllProvidersFailedError`), остальные задачи отменены | поймано через `except*` |
<!-- /target:mock -->

<!-- target:ollama -->
## Цель: ollama

локальный Ollama, модель llama3.2, http://localhost:11434/v1; запросов: 6; дата: 2026-10-06 20:45.

| Режим | Время | Ускорение к sync | Ошибок |
|-------|-------|------------------|--------|
| sync, последовательно | 30.0 с | ×1.0 | 0 |
| async `batch_chat`, concurrency=1 | 34.1 с | ×0.9 | 0 |
| async `batch_chat`, concurrency=5 | 35.8 с | ×0.8 | 0 |
| async `batch_chat`, concurrency=10 | 33.6 с | ×0.9 | 0 |

Невалидная модель на 3-м запросе из 6 (concurrency=10):

| Метод | Время | Результат | Поведение |
|-------|-------|-----------|-----------|
| `batch_chat` (gather, return_exceptions) | 27.2 с | 5 из 6 ответов получены; ошибка на позиции [3] — `AllProvidersFailedError` | батч не упал |
| `batch_chat_strict` (TaskGroup) | 0.3 с | ни одного ответа: `ExceptionGroup` (ошибок: 1, `AllProvidersFailedError`), остальные задачи отменены | поймано через `except*` |
<!-- /target:ollama -->
