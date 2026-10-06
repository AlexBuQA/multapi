# LiteLLM как готовый LLM Gateway — локальная проверка (блок 3.2)

Цель — поднять LiteLLM proxy с двумя провайдерами (primary и fallback) и увидеть, как он переключается при ошибке primary. Выводы и решение «берём LiteLLM / пишем сами» — в [`docs/architecture.md`](../architecture.md#6-litellm-как-готовый-llm-gateway).

| Файл | Что это |
|------|---------|
| `config.yaml` | Конфигурация proxy: `support-primary` (OpenRouter → `openai/gpt-4o-mini`) и `support-fallback` (локальный Ollama → `qwen3:4b-instruct`), цепочка fallback, cooldown как Circuit Breaker, master key |
| `send_requests.py` | Три запроса в proxy через OpenAI SDK; печатает, какая группа ответила и было ли переключение |
| `request.json` | Тело запроса для проверки через `curl.exe` |

## Установка (Windows, PowerShell)

LiteLLM ставится в **отдельное окружение**: `litellm[proxy]` тянет много собственных пакетов, и так они не смешиваются с зависимостями проекта. (На момент блока 3.2 была и прямая несовместимость: LiteLLM требует `openai` 2.x, а проект был закреплён на `openai<2.0`; в блоке 3.3 проект перешёл на `openai` 2.x.) Папка `.venv-litellm/` исключена из git.

```powershell
python -m venv .venv-litellm
.venv-litellm\Scripts\pip install "litellm[proxy]"
```

## Запуск proxy (терминал 1)

```powershell
$env:LITELLM_MASTER_KEY = "sk-" + [guid]::NewGuid().ToString("N")
$env:LITELLM_MASTER_KEY                      # скопируйте значение — оно понадобится во втором терминале
$env:OPENROUTER_API_KEY = "sk-or-invalid"     # неверный ключ: эмуляция отказа primary
$env:LITELLM_LOCAL_MODEL_COST_MAP = "True"   # не скачивать таблицу цен при старте
$env:PYTHONUTF8 = "1"                         # UTF-8 в консоли Windows
.venv-litellm\Scripts\litellm --config docs/litellm/config.yaml --port 4000
```

- Без `LITELLM_MASTER_KEY` текущие версии LiteLLM не запускаются: proxy отказывается работать без авторизации.
- Ollama должна быть запущена, модель `qwen3:4b-instruct` скачана.
- Если у вас есть настоящий ключ OpenRouter, подставьте его — тогда ответит primary и переключения не будет.

## Запросы (терминал 2)

В основном окружении проекта (`.venv`):

```powershell
$env:LITELLM_MASTER_KEY = "<значение из терминала 1>"
python docs/litellm/send_requests.py
```

Или через `curl.exe` (в PowerShell 5.1 команда `curl` — это другой инструмент, поэтому именно `curl.exe`):

```powershell
curl.exe -i http://localhost:4000/v1/chat/completions -H "Authorization: Bearer $env:LITELLM_MASTER_KEY" -H "Content-Type: application/json" -d "@docs/litellm/request.json"
```

## Что смотреть

| Заголовок ответа | Значение при отказе primary |
|------------------|-----------------------------|
| `x-litellm-model-group` | `support-fallback` — ответила резервная группа |
| `x-litellm-attempted-fallbacks` | `1` — было одно переключение |
| `x-litellm-attempted-retries` | `0` — ошибку авторизации LiteLLM не повторяет |
| `x-litellm-overhead-duration-ms` | накладные расходы proxy |

В консоли proxy (терминал 1) на уровне по умолчанию видны только строки `200 OK` — переключение подтверждают заголовки ответа. Чтобы увидеть саму ошибку OpenRouter, запустите proxy с флагом `--detailed_debug`.

После ошибки primary уходит в cooldown на `cooldown_time` (60 с), и следующие запросы к `support-primary` идут сразу на резервную модель, не обращаясь к OpenRouter, — так в LiteLLM работает Circuit Breaker. Это видно по накладным расходам proxy: на втором запросе они заметно меньше, чем на первом.

Результаты прогона — в [`docs/architecture.md`](../architecture.md#результаты-локального-прогона).
