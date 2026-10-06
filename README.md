# Мультимодальный ИИ-помощник техподдержки — ДЗ 2.6, блоки 3.1–3.3

**Автор:** Александра Бужор
**Репозиторий:** https://github.com/AlexBuQA/multapi

CLI-приложение с **мультимодальными возможностями**, построенное на наработках блоков 2.1–2.5. Сборка по умолчанию работает на **локальном Ollama** (OpenAI-совместимый API), модели `llama3.2` / `llama3.2-vision`.

Реализованы оба варианта задания:

- **Вариант А — Анализ изображений (Vision API):** путь к картинке → base64 → Vision-запрос → текстовый ответ. Демо на 3 изображениях разного типа (фото, скриншот, график). Работает локально на Ollama с vision-моделью.
- **Вариант Б — Голосовой пайплайн (Whisper + TTS):** аудио → транскрипция → классификация → ответ (LLM) → озвучка → аудиофайл.
- **Блок 3.2 — Архитектурный паспорт:** схема слоёв Gateway → Service → LLM → Data, ADR, точки отказа и проверка LiteLLM — [`docs/architecture.md`](docs/architecture.md).
- **Блок 3.1 — Function Calling:** ассистент техподдержки с инструментами `search_knowledge_base` и `check_service_status`, полный цикл tool_call на локальном Ollama — см. раздел [«Блок 3.1 — Function Calling»](#блок-31--function-calling).
- **Блок 3.3 — Асинхронная обработка запросов к ИИ:** `AsyncLLMClient` (семафор, таймауты, батч, стриминг), бенчмарк sync vs async и SSE-эндпоинт `/chat/stream` на FastAPI — см. раздел [«Блок 3.3»](#блок-33--асинхронная-обработка-запросов-к-ии).

## Важно про Ollama и модальности

Ollama по OpenAI-совместимому API поддерживает **чат и vision**, но **не поддерживает аудио** (нет эндпоинтов Whisper и TTS). Поэтому:

| Возможность | Где выполняется |
|-------------|-----------------|
| Чат, ответы помощника | локальный Ollama (`SUPPORT_PRIMARY_MODEL`, `llama3.2`) |
| Классификация обращений | локальный Ollama (`SUPPORT_CLASSIFIER_MODEL`, `llama3.2`) |
| Анализ изображений (вариант А) | локальный Ollama, **vision-модель** (`SUPPORT_VISION_MODEL`, `llama3.2-vision`) |
| Whisper + TTS (вариант Б) | отдельный OpenAI-совместимый аудио-эндпоинт (`AUDIO_API_KEY`) |

Текстовая `llama3.2` изображения не воспринимает — для варианта А установите vision-модель:

```bash
ollama pull llama3.2-vision
```

Для варианта Б нужен реальный ключ OpenAI в `AUDIO_API_KEY` (Ollama аудио не делает). Если ключ не задан — голосовой пайплайн аккуратно сообщит об этом, не падая.

> **Статус варианта Б (голос): проработан теоретически, не тестировался.**
> Whisper и TTS требуют платного доступа к OpenAI API (`AUDIO_API_KEY`) — Ollama их не поддерживает. Оплата доступа к OpenAI из России затруднена, поэтому по варианту Б проработано только теоретическое решение: код пайплайна (`src/voice.py`) написан целиком и проходит проверку управляющей логики на моках, но **на реальных аудио-сервисах не запускался**. В репозитории `AUDIO_API_KEY` пуст; при наличии ключа его достаточно вписать в `.env` — код менять не нужно. Полностью протестирован и работает локально **вариант А (анализ изображений)** на Ollama.

## Переиспользование наработок блока 2

| Из ДЗ | Что переиспользовано | Где |
|-------|----------------------|-----|
| 2.1 | OpenAI SDK, ключ/endpoint из `.env`, переключение провайдера через `LLM_PROVIDER`/`base_url` | `src/config.py`, `src/robust_client.py` |
| 2.2 | System prompt по РРФО + few-shot | `src/prompts.py` |
| 2.3 | `RobustLLMClient`: retry (exp. backoff + jitter), fallback-цепочка, логирование, usage | `src/robust_client.py` |
| 2.4 | `LLMCache`: SHA-256 ключ, TTL, hit rate | `src/cache.py` |
| 2.5 | Помощник техподдержки + классификация обращений (бонус) | `src/prompts.py`, `src/classifier.py` |

## Структура

```
multapi/
├── main.py                  # CLI: подкоманды vision / voice / ask
├── demo_vision.py           # демо варианта А (3 изображения + кеш)
├── demo_voice.py            # демо варианта Б (полный голосовой пайплайн)
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   ├── config.py            # .env, провайдеры (Ollama + fallback), аудио-эндпоинт
│   ├── robust_client.py     # надёжный клиент (retry + fallback + usage), vision, прокси-bypass для localhost
│   ├── cache.py             # LLMCache (TTL, hit rate)
│   ├── prompts.py           # РРФО system prompt + few-shot
│   ├── classifier.py        # классификация обращений (SUPPORT_CLASSIFIER_MODEL)
│   ├── vision.py            # вариант А: base64 + Vision (vision-модель)
│   ├── voice.py             # вариант Б: Whisper -> classify -> LLM -> TTS
│   └── utils.py             # логирование, base64, валидация файлов, UsageTracker
├── app/                     # блоки 3.1 и 3.3
│   ├── config.py            # настройки ассистента с tools и асинхронного клиента (pydantic-settings)
│   ├── logging_utils.py     # JSON-лог шагов -> logs/tool_calls.jsonl, logs/llm_calls.jsonl
│   ├── main.py              # блок 3.3: FastAPI — /health, /chat, /chat/stream (SSE)
│   ├── services/
│   │   └── llm_client.py    # блок 3.3: AsyncLLMClient (семафор, таймауты, батч, стриминг)
│   ├── prompts/
│   │   ├── system_v1.j2     # system prompt ассистента (Jinja2)
│   │   ├── tools/           # description инструментов (*.md)
│   │   └── loader.py        # render_system_prompt, load_tool_description
│   ├── tools/
│   │   ├── schemas.py       # JSON Schema tools + проверка jsonschema
│   │   └── handlers.py      # обработчики: поиск по руководству, статус сервиса
│   └── llm/
│       └── client.py        # полный цикл tool_call (ToolCallingAssistant)
├── data/
│   ├── knowledge_base.json  # руководство пользователя: 10 статей с разделами
│   └── service_status.json  # статус компонентов сервиса
├── examples/
│   └── run_tool_call.py     # прогон трёх тест-запросов
├── scripts/                 # блок 3.3
│   ├── benchmark.py         # бенчмарк sync vs async -> benchmark_results.md
│   ├── benchmark_results.md # результаты локального прогона (мок и Ollama)
│   ├── stream_demo.py       # демо stream_chat: TTFT и общее время
│   ├── mock_llm_server.py   # мок OpenAI API с задержкой (модель облачного провайдера)
│   ├── _target.py           # выбор цели: мок или локальный Ollama
│   └── stream_request.json  # тело запроса для curl.exe
├── docs/
│   ├── architecture.md      # блок 3.2: архитектурный паспорт (схема, ADR, точки отказа)
│   └── litellm/             # config.yaml LiteLLM proxy, скрипт запросов, инструкция
├── tools/
│   └── check_proxy.py       # проверка HTTP-прокси (egress-IP)
├── tests/
│   ├── test_review_fixes.py # тесты на моках (без сети): учёт аудио, классификатор, образцы
│   ├── test_tool_call.py    # блок 3.1: схемы, обработчики, цикл tool_call, лог
│   └── test_async_client.py # блок 3.3: семафор, батчи, таймаут, стриминг, SSE
├── samples/                 # входные файлы: photo.jpg, screenshot.png, chart.png, voice_question.wav
├── outputs/                 # сюда пишутся аудио-ответы TTS
└── logs/                    # sample_run.log (демо-лог), tool_calls_sample.jsonl (реальный прогон блока 3.1),
                             # app.log, tool_calls.jsonl, llm_calls.jsonl (локальные прогоны, в git не попадают)
```

`samples/voice_question.wav` — голосовой вопрос «Здравствуйте! Подскажите, как сбросить пароль от аккаунта?» (синтезированная речь, 16 кГц, моно, ~4,6 с). Его использует `demo_voice.py`.

## Установка

```bash
git clone https://github.com/AlexBuQA/multapi.git
cd multapi
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # значения для Ollama уже выставлены
```

Нужен Python 3.11 или новее: блок 3.3 использует `asyncio.TaskGroup`, `asyncio.timeout` и `except*`.

Запустите Ollama и подтяните модели:

```bash
ollama serve                 # если не запущен как сервис
ollama pull llama3.2          # чат и классификатор
ollama pull qwen3:4b-instruct # Function Calling (блок 3.1), см. «Результаты на трёх запросах»
ollama pull llama3.2-vision  # для варианта А (анализ изображений)
```

## Запуск

**Вариант А — Vision (локально на Ollama):**
```bash
python main.py vision samples/chart.png --question "Какой квартал лучший?"
python main.py vision samples/screenshot.png -q "Есть ли ошибка на экране?"
python demo_vision.py
```

**Вариант Б — Voice (нужен AUDIO_API_KEY):**
```bash
python main.py voice samples/voice_question.wav --out outputs/answer.mp3
python demo_voice.py
```

Свой голосовой вопрос можно сгенерировать через TTS (формат файла определяется расширением `--out`):
```bash
python main.py voice --make-sample "Как сбросить пароль?" --out samples/my_question.mp3
python main.py voice samples/my_question.mp3 --out outputs/answer.mp3
```

**Блок 3.1 — Function Calling (локально на Ollama):** в `.env` задайте `SUPPORT_TOOLS_MODEL=qwen3:4b-instruct` — на этой модели получены итоговые результаты (без неё используется `SUPPORT_PRIMARY_MODEL`).
```bash
python examples/run_tool_call.py                          # три тест-запроса
python examples/run_tool_call.py "Как включить 2FA?"       # свой запрос
python main.py ask "Не приходит письмо для сброса пароля"
```

**Блок 3.3 — асинхронный клиент:**
```bash
python scripts/benchmark.py                                    # мок с задержкой 1 с, 20 запросов
python scripts/benchmark.py --target ollama --n 6 --max-tokens 60
python scripts/stream_demo.py                                  # мок
python scripts/stream_demo.py --target ollama "Что такое event loop?"
uvicorn app.main:app --port 8000                               # сервис; запросы — во втором терминале
```

Проверка потока во втором терминале (PowerShell — именно `curl.exe`, тело запроса из файла):
```powershell
curl.exe -N -X POST http://localhost:8000/chat/stream -H "Content-Type: application/json" -d "@scripts/stream_request.json"
```
В Linux/macOS: `curl -N -X POST http://localhost:8000/chat/stream -H "Content-Type: application/json" -d '{"prompt": "Что такое event loop?"}'`.

**Тесты (без сети и без ключей):**
```bash
python -m unittest discover -s tests -v
```

## Блок 3.1 — Function Calling

Ассистент техподдержки продукта «Личный кабинет» получает два инструмента и проходит полный цикл: JSON Schema → запрос → разбор `tool_calls` → выполнение функции → возврат результата → финальный ответ. Работает через OpenAI SDK на локальном Ollama; retry и fallback — из `RobustLLMClient` (новый метод `complete()` возвращает ответ целиком, с `tool_calls` и `usage`).

### Инструменты

| Tool | Аргументы | Что делает |
|------|-----------|------------|
| `search_knowledge_base` | `query` (обязательный), `product`: `web` / `mobile` / `api` | Ищет статьи руководства пользователя в `data/knowledge_base.json` (10 статей с номерами разделов), возвращает до 3 статей |
| `check_service_status` | `component`: `auth` / `email` / `billing` / `api` / `mobile` / `all` | Читает статус компонентов из `data/service_status.json`; сейчас у `email` — инцидент «задержка писем до 30 минут» |

Оба обработчика читают файлы при каждом вызове — это реальная работа, а не константа: поправьте статус в JSON, и ассистент ответит иначе.

### Где что лежит

| Что | Где |
|-----|-----|
| JSON Schema инструментов | `app/tools/schemas.py`; схема проверяется `jsonschema` при импорте, аргументы от модели — перед вызовом обработчика |
| `description` инструментов | `app/prompts/tools/*.md` → именованные константы `SEARCH_KNOWLEDGE_BASE_DESCRIPTION`, `CHECK_SERVICE_STATUS_DESCRIPTION`; описания параметров — тоже константы |
| System prompt | `app/prompts/system_v1.j2` (Jinja2: `product_name`, `max_sentences`), загрузка — `app/prompts/loader.py`; в `app/llm/client.py` inline-строки нет |
| Обработчики | `app/tools/handlers.py`: allowlist `DISPATCH` и диспетчер `execute_tool()` |
| Цикл tool_call | `app/llm/client.py`, `ToolCallingAssistant.ask()` |
| Настройки | `app/config.py` (pydantic-settings): `SUPPORT_TOOLS_MODEL`, `SUPPORT_PRODUCT_NAME`, `SYSTEM_PROMPT_VERSION`, `TOOLS_MAX_ROUNDS` |

### Цикл tool_call

1. В модель уходят `messages` (system + user) и `tools`, `tool_choice="auto"`.
2. Если в ответе есть `tool_calls`: в историю добавляется ассистентский ход с `tool_calls`, каждый инструмент выполняется, результат добавляется сообщением `role: "tool"` с тем же `tool_call_id`, и запрос повторяется.
3. Если `tool_calls` нет — текст ответа и есть финальный ответ (случай, когда модель отвечает сразу, обрабатывается отдельно и не падает).
4. Раундов с инструментами — не больше `TOOLS_MAX_ROUNDS` (3), затем модель просят ответить без инструментов (`tool_choice="none"`).

Ошибки не роняют цикл: неизвестный инструмент (модель получает список доступных), битый JSON аргументов, нарушение схемы (например, `component: "Почта"`) или исключение внутри обработчика возвращаются модели как `{"error": ..., "message": ...}`, чтобы она поправилась или честно ответила. Если модель недоступна — пользователь получает «Сервис временно недоступен».

### Лог шагов

Каждый запрос пишет в `logs/tool_calls.jsonl` JSON-строки с общим `run_id`: `user_input` → `llm_request` / `llm_response` (токены и длительность шага) → `tool_call` (имя и аргументы) → `tool_result` → … → `final_answer` → `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`).

```json
{"ts": "2026-10-06T19:31:56.456+03:00", "level": "info", "event": "tool_call", "run_id": "d0e59967168f", "step": 1, "call_id": "call_ymhnvryd", "tool": "check_service_status", "arguments": {"component": "email"}}
```

Реальный лог всех прогонов из раздела ниже (12 запросов, три модели) сохранён в `logs/tool_calls_sample.jsonl`; у каждого запроса указан `run_id`.

### Результаты на трёх запросах

Тест-запросы (`examples/run_tool_call.py`):

| Кейс | Запрос |
|------|--------|
| (а) точно требует tool | «Уже 40 минут не приходит письмо для сброса пароля. У вас что-то сломалось?» |
| (б) точно не требует tool | «Спасибо, всё заработало!» |
| (в) пограничный | «Подойдёт ли пароль из 6 символов?» |

#### Прогон 1 — `llama3.2` (3B), первая версия кода

- **(а)** Модель вызвала `search_knowledge_base`, а не `check_service_status`, и вместо значения передала в `query` описание параметра из схемы: `{"type": "string", "description": "…", "value": "не приходит письмо для сброса пароля"}`. Проверка JSON Schema вернула модели `invalid_arguments`, код не упал. Повторно вызывать инструмент модель не стала и дала размытый ответ со ссылкой на раздел 3.1. 2 вызова LLM, 1541 токен.
- **(б)** На благодарность модель всё равно вызвала `search_knowledge_base("сброс пароля")` и ответила инструкцией по сбросу пароля — лишний вызов инструмента, критерий не выполнен. 2 вызова LLM, 1666 токенов.
- **(в)** Модель написала вызов инструмента обычным текстом (`{"name": "search_knowledge_base", "parameters": …}`, к тому же с битым JSON) вместо поля `tool_calls`. Код принял это за финальный ответ, и пользователь увидел сырой JSON. 1 вызов LLM, 954 токена.

Вывод: у 3B-модели слабое следование схеме и склонность вызывать инструмент на любой запрос.

#### Доработки по итогам прогона 1

1. **«Эхо схемы» вместо значения.** Если в аргументе пришло описание параметра со значением внутри, берётся само значение (`normalize_arguments`). Если значения нет, модель получает подсказку с примером правильных аргументов.
2. **Вызов инструмента текстом.** JSON вида `{"name": …, "parameters": …}` в тексте ответа распознаётся и выполняется как обычный `tool_call` (событие `tool_call_from_text` в логе). При битом JSON модель получает `invalid_json` и может повторить вызов. Сырой JSON пользователю больше не показывается.
3. **System prompt.** Добавлены явные правила выбора: «что-то не приходит / не работает → `check_service_status` с нужным компонентом», «благодарность, приветствие → ответ без инструментов», «аргументы — конкретные значения, при ошибке исправь и повтори». Описания параметров сокращены.

#### Прогон 2 — после доработок, сравнение трёх моделей

Код и промпт одинаковые, меняется только `SUPPORT_TOOLS_MODEL`. В ячейках — решение модели и `total_tokens` за запрос.

| Модель | (а) нужен tool | (б) tool не нужен | (в) пограничный | Итог |
|--------|----------------|-------------------|-----------------|------|
| `llama3.2` (3B) | ✅ `check_service_status({"component": "email"})`, 1702 | ❌ `search_knowledge_base({"query": "сброс пароля"})`, 1868 | вызвала поиск, ответ по разделу 2.2 верный, 1864 | (б) не выполнен |
| `qwen3-vl:2b-instruct` | ❌ «Проверю статус сервиса» без вызова инструмента, 1107 | ✅ ответ сразу, 1104 | без инструмента, ответ из общих знаний, 1108 | (а) не выполнен |
| `qwen3:4b-instruct` | ✅ `check_service_status({"component": "email"})`, 2339 | ✅ ответ сразу, 1033 | без инструмента, ответ с выдуманным разделом, 1076 | (а) и (б) выполнены |

Доработки сработали: `llama3.2` в (а) вызвала правильный инструмент с корректным аргументом, в (в) вызвала поиск штатно, сырого JSON в ответах больше нет. Но склонность 3B-модели вызывать инструмент на любое сообщение промптом не исправляется, а `qwen3-vl:2b-instruct`, наоборот, обещает проверить статус, не вызывая инструмент.

**Выбранная модель — `qwen3:4b-instruct`** (2,5 ГБ, без режима рассуждений): единственная из проверенных, которая выполняет оба обязательных кейса. Подробно по каждому кейсу:

- **(а) «Уже 40 минут не приходит письмо для сброса пароля…»** — модель вызвала `check_service_status` с аргументами `{"component": "email"}`: верный инструмент и верный компонент. Инструмент вернул статус `degraded` и инцидент о задержке писем до 30 минут. Финальный ответ пересказывает инцидент и добавляет совет проверить «Спам». Полный цикл из 4 шагов виден в логе (`run_id` `d0e59967168f`): 2 вызова LLM, 2339 токенов.
- **(б) «Спасибо, всё заработало!»** — инструмент не вызван, модель сразу ответила короткой фразой «Приятно слышать! Если понадобится — всегда на связи. 😊». Второго запроса не было, код отработал без ошибок (`bdf890cec70f`): 1 вызов LLM, 1033 токена.
- **(в) «Подойдёт ли пароль из 6 символов?»** — модель решила не вызывать инструмент и ответила из общих знаний. Минимум в 8 символов совпал с руководством (раздел 2.2), но она добавила спецсимволы, которых в требованиях нет, и сослалась на несуществующий «раздел 3.2», хотя промпт запрещает придумывать разделы (`f5a565000d5c`): 1 вызов LLM, 1076 токенов. Этот кейс показывает цену решения «без инструмента»: ответ правдоподобен, но не подтверждён данными. Для сравнения, `llama3.2` здесь вызвала поиск и ответила корректно по разделу 2.2.

Производительность на CPU без видеокарты: у `qwen3:4b-instruct` первый шаг кейса (а) занял 50 с (это первое обращение к модели, в него входит её загрузка в память), второй — 18 с; ответы без инструментов — 3–8 с.

### Соответствие критериям блока 3.1

| Критерий | Реализация |
|----------|------------|
| Отдельный модуль с JSON Schema, схема валидируется | `app/tools/schemas.py`, `Draft202012Validator.check_schema` + проверка аргументов |
| Функция делает реальную работу | чтение `data/knowledge_base.json` и `data/service_status.json` |
| System prompt вынесен в файл | `app/prompts/system_v1.j2`, `render_system_prompt()` |
| `description` — константа / из файла | `app/prompts/tools/*.md` → константы в `schemas.py` |
| Полный цикл на запросе (а), запрос (б) без tool | `ToolCallingAssistant.ask()`, тесты `tests/test_tool_call.py` |
| Ассистентский ход с `tool_calls` в истории | `_assistant_turn()`; проверяется в тестах |
| Лог: input → tool + args → result → answer → tokens | `logs/tool_calls.jsonl`; реальный прогон — `logs/tool_calls_sample.jsonl` |
| Три тест-кейса с наблюдениями | раздел «Результаты на трёх запросах» |

## Блок 3.3 — Асинхронная обработка запросов к ИИ

`AsyncLLMClient` — асинхронная версия `RobustLLMClient` из модуля 2 на `AsyncOpenAI`. От синхронного клиента сохранены fallback-цепочка провайдеров, кеш `LLMCache` и прокси только для удалённых провайдеров. Внутри `async def` нет ни синхронного `OpenAI`, ни `requests`, ни `time.sleep` — это проверяет тест `test_no_blocking_calls`.

### Где что лежит

| Что | Где |
|-----|-----|
| Асинхронный клиент | `app/services/llm_client.py`: `complete`, `batch_chat`, `batch_chat_strict`, `stream_chat` |
| Настройки | `app/config.py`, `AsyncClientSettings`: `LLM_CONCURRENCY`, `LLM_CALL_TIMEOUT`, `LLM_SDK_MAX_RETRIES` |
| FastAPI и SSE | `app/main.py`: `GET /health`, `POST /chat` (ответ целиком), `POST /chat/stream` (поток SSE) |
| Бенчмарк sync vs async | `scripts/benchmark.py` → результаты в `scripts/benchmark_results.md` (реальный прогон — в репозитории) |
| Демо стриминга | `scripts/stream_demo.py`: TTFT и общее время по `time.perf_counter()` |
| Мок OpenAI API | `scripts/mock_llm_server.py`: отвечает с заданной задержкой, умеет поток и `usage` |
| Тесты | `tests/test_async_client.py` (без сети: заглушка `AsyncOpenAI` на `asyncio.sleep`) |

### Как устроен клиент

- **Семафор — атрибут экземпляра.** `self._sem = asyncio.Semaphore(concurrency)` создаётся один раз в `__init__`. Его используют `complete` и `stream_chat`, поэтому лимит общий для всех вызовов клиента: и для батча, и для параллельных запросов к `/chat/stream`.
- **Лимит задаётся при создании клиента.** `batch_chat(prompts, concurrency)` принимает `concurrency`, как в задании, но не создаёт новый семафор: если значение не совпадает с лимитом клиента, будет `ValueError` с подсказкой создать `AsyncLLMClient(concurrency=N)`. Иначе семафор пришлось бы пересоздавать на каждый вызов, и лимит перестал бы быть общим.
- **Два уровня таймаута.** Таймаут SDK (`LLM_REQUEST_TIMEOUT`) действует на одну HTTP-попытку. `asyncio.timeout(LLM_CALL_TIMEOUT)` ограничивает всю операцию `complete()` вместе с повторами и fallback. Время ожидания в очереди семафора в этот бюджет не входит.
- **Повторы делает SDK.** `max_retries=LLM_SDK_MAX_RETRIES`: SDK сам повторяет 408, 409, 429, 5xx и ошибки соединения с экспоненциальной задержкой и учитывает `Retry-After`. Если провайдер так и не ответил, запрос уходит следующему провайдеру в цепочке.
- **`batch_chat`** — `asyncio.gather(..., return_exceptions=True)`. Ответы идут в порядке промптов; упавший запрос возвращается исключением на своей позиции и не роняет остальные.
- **`batch_chat_strict`** — `asyncio.TaskGroup`, «все ответы или ничего». При первой ошибке остальные задачи отменяются, вызывающий код ловит `ExceptionGroup` через `except*`. Сравнение двух методов — в `scripts/benchmark_results.md`.
- **`stream_chat`** — async-генератор фрагментов ответа, запрос с `stream_options={"include_usage": True}`. Переключиться на fallback можно только до первого токена. После закрытия потока в лог пишутся `usage`, время до первого токена и общее время. Если клиент перестал читать поток (закрыл SSE-соединение), генератор закрывается, запрос к модели прерывается, а в логе остаётся `status: "cancelled"`.

### Лог вызовов

Каждый вызов пишет JSON-строку в `logs/llm_calls.jsonl`:

- `llm.call` — `status` (`ok`, `cache_hit`, `timeout`, `cancelled`, `error:<тип>`), `duration_ms`, `queue_ms` (ожидание в семафоре), `model`, `provider`, `prompt_chars`, токены;
- `llm.stream` — то же для потока плюс `chunks`, `ttft_ms` и `last_token_ms`;
- `llm.fallback` — переход к следующему провайдеру;
- `llm.batch_strict_failed` — сколько задач строгого батча упало, отменено и успело завершиться.

Пример строки (прогон на моке):

```json
{"ts": "2026-10-06T20:32:10.452+03:00", "level": "info", "event": "llm.stream", "status": "ok", "provider": "ollama", "model": "mock-model", "prompt_chars": 49, "chunks": 31, "ttft_ms": 905.0, "last_token_ms": 2963.8, "duration_ms": 3034.4, "prompt_tokens": 12, "completion_tokens": 31, "total_tokens": 43}
```

### Бенчмарк: почему два прогона

Асинхронность ускоряет **I/O-bound** работу: пока один запрос ждёт ответа облачного API, клиент отправляет следующие. Локальный Ollama на CPU — **compute-bound**: модель на ноутбуке всё равно генерирует ответы по очереди, поэтому параллельные запросы почти не ускоряются. Бенчмарк запускается на двух целях:

- **мок** (`--target mock`, по умолчанию) — OpenAI-совместимый сервер с фиксированной задержкой ответа, как у облачного провайдера. Здесь видно ускорение от `concurrency`.
- **Ollama** (`--target ollama`) — реальная локальная модель из `.env`. Здесь видно, где асинхронность не помогает.

В обоих прогонах кеш выключен, и на каждый уровень `concurrency` создаётся новый клиент. Иначе повторный прогон тех же промптов отвечал бы из кеша, и ускорение было бы ложным.

### Результаты

Прогон 6 октября 2026 года на ноутбуке без видеокарты. Полные таблицы — в [`scripts/benchmark_results.md`](scripts/benchmark_results.md).

**Мок облачного API** (задержка 1 с на ответ, 20 запросов):

| Режим | Время | Ускорение к sync |
|-------|-------|------------------|
| sync, последовательно | 21,4 с | ×1,0 |
| async, concurrency=1 | 20,3 с | ×1,1 |
| async, concurrency=5 | 4,1 с | ×5,3 |
| async, concurrency=10 | 2,1 с | ×10,2 |

Критерий «не меньше 4–5× при concurrency=10» выполнен: ×10,2. Пока запросы только ждут ответа, ускорение растёт вместе с concurrency: при concurrency=10 двадцать запросов проходят двумя волнами примерно по секунде.

**Локальный Ollama** (`llama3.2`, 6 запросов, `max_tokens=60`):

| Режим | Время | Ускорение к sync |
|-------|-------|------------------|
| sync, последовательно | 30,0 с | ×1,0 |
| async, concurrency=1 | 34,1 с | ×0,9 |
| async, concurrency=5 | 35,8 с | ×0,8 |
| async, concurrency=10 | 33,6 с | ×0,9 |

Ускорения нет, как и ожидалось: модель считает ответы на процессоре ноутбука, и параллельные запросы делят те же ядра, а не ждут сеть. Объём вычислений не меняется, поэтому и общее время не уменьшается. Небольшое замедление — разброс замеров и накладные расходы на параллельную обработку. Асинхронный клиент здесь полезен в другом: пока модель генерирует ответ, event loop свободен, и сервис продолжает отвечать на другие запросы.

**Невалидная модель на 3-м запросе:**

| Метод | Мок (10 запросов) | Ollama (6 запросов) |
|-------|-------------------|---------------------|
| `batch_chat` (gather) | 9 из 10 ответов, на позиции 3 — `AllProvidersFailedError`; 1,0 с | 5 из 6 ответов, на позиции 3 — `AllProvidersFailedError`; 27,2 с |
| `batch_chat_strict` (TaskGroup) | ни одного ответа, `ExceptionGroup` пойман через `except*`; 0,0 с | ни одного ответа, `ExceptionGroup` пойман через `except*`; 0,3 с |

Строгий батч останавливается сразу после ответа 404 на невалидную модель и отменяет остальные запросы. Он подходит, когда частичный результат бесполезен — например, все ответы нужны для одного отчёта. `batch_chat` — когда каждый ответ ценен сам по себе.

**Стриминг** (`scripts/stream_demo.py`):

| Цель | TTFT | Общее время |
|------|------|-------------|
| мок, задержка 2 с | 1,36 с | 2,85 с |
| Ollama, `llama3.2`, «Что такое event loop?» | 3,11 с | 42,31 с |

TTFT меньше общего времени в обоих случаях. На Ollama пользователь видит начало ответа через 3 с вместо 42 с — для медленной локальной модели в этом и главный смысл стриминга.

**SSE** (`uvicorn` + `curl.exe -N`, `llama3.2`): ответ пришёл событиями `data:` по мере генерации и завершился событием `done`. Ollama отдаёт поток по токенам, поэтому слова приходят частями — клиент склеивает фрагменты подряд.

```
data:  цик

data: л

data:  провер

data: ки
...
event: done
data: [DONE]
```

Качество самого текста (например, «ЭVENT-ЛОК» в начале ответа) — это возможности 3B-модели `llama3.2`, к клиенту и стримингу оно отношения не имеет.

### Соответствие критериям блока 3.3

| Критерий | Реализация |
|----------|------------|
| Нет sync `OpenAI`, `requests`, `time.sleep` внутри `async def` | `AsyncOpenAI`, `asyncio.sleep` только в моке; тест `test_no_blocking_calls` |
| `self._sem` создаётся один раз в `__init__` | тест `test_semaphore_created_once_in_init`; число одновременных запросов проверяет `test_concurrency_limited_and_order_kept` |
| Таймаут и повторы | таймаут SDK + `asyncio.timeout`, `max_retries` SDK; тест `test_call_timeout` |
| `batch_chat` с `gather(return_exceptions=True)` | тест `test_failed_request_returned_in_place` |
| `concurrency=10` даёт ускорение ≥ 4–5× на 20 запросах | ×10,2 на моке облачного API — см. раздел «Результаты» и `scripts/benchmark_results.md` |
| `stream_chat` с `include_usage`, TTFT < общего времени | `scripts/stream_demo.py`: на Ollama TTFT 3,11 с при общем времени 42,31 с; тест `test_ttft_before_total_and_usage_logged` |
| Лог `llm.call`: `duration_ms`, `model`, `prompt_chars`, `status` | `logs/llm_calls.jsonl`; тест `test_answer_and_call_log` |
| Дополнительно: `TaskGroup` + `except*` | `batch_chat_strict`, тест `test_strict_batch_all_or_nothing`, сравнение в `scripts/benchmark_results.md` |
| Дополнительно: SSE-эндпоинт | `POST /chat/stream` в `app/main.py`, проверен через `curl.exe -N` на `llama3.2`; тест `test_chat_stream_sends_events` |

## Конфигурация (.env)

Ключевые переменные (полный список — в `.env.example`):

```
OPENAI_API_KEY=ollama                       # ключ-заглушка для Ollama
OPENAI_BASE_URL=http://localhost:11434/v1   # локальный эндпоинт Ollama
SUPPORT_PRIMARY_MODEL=llama3.2
SUPPORT_CLASSIFIER_MODEL=llama3.2
SUPPORT_VISION_MODEL=llama3.2-vision
LLM_PROVIDER=ollama
LLM_PROXY=                                   # к localhost не применяется; для Ollama не нужен
AUDIO_API_KEY=                               # реальный ключ OpenAI для Whisper/TTS (вариант Б)
```

Настройки асинхронного клиента (блок 3.3) необязательны — без них действуют значения по умолчанию:

```
LLM_CONCURRENCY=5          # одновременных запросов на один клиент (семафор)
LLM_CALL_TIMEOUT=180       # бюджет на весь вызов complete(), с: повторы и fallback включены
LLM_SDK_MAX_RETRIES=3      # повторы SDK на 408/409/429/5xx и ошибки соединения
```

## Прокси

HTTP-прокси (`LLM_PROXY`) поддерживается, но **к локальным endpoint (`localhost`/`127.0.0.1`) не применяется** — иначе локальный Ollama стал бы недоступен. Для удалённых провайдеров (OpenRouter, реальный OpenAI, аудио-эндпоинт) прокси применяется, если задан. Проверка прокси: `python tools/check_proxy.py`.

## Соответствие критериям ДЗ 2.6

| Критерий | Реализация |
|----------|------------|
| Пайплайн: от входного файла до результата | `vision.analyze_image` (А), `voice.run_pipeline` (Б) |
| Ошибки: файл не найден, формат, API | `utils.validate_file` + `InputFileError`; API — retry/fallback; аудио не настроено — `AudioNotConfiguredError` |
| Структура: функции разделены по ответственности | конфиг / клиент / кеш / промпты / классификатор / vision / voice / utils |
| Примеры входных файлов | `samples/` (3 изображения + аудио) |
| Лог работы программы | `logs/sample_run.log` |
| Учёт usage и стоимости | `UsageTracker`: `add_chat` (токены), `add_audio` (Whisper, по минутам), `add_tts` (TTS, по символам) |

## Доработки по итогам ревью

1. **Учёт аудио в `UsageTracker`.** `transcribe()` запрашивает у Whisper `verbose_json`, берёт из ответа длительность (для WAV есть запасной вариант — по заголовку файла) и учитывает её через `UsageTracker.add_audio()`. TTS больше не пишет напрямую в `audio_cost_usd`: для него добавлен метод `add_tts()`, так как TTS тарифицируется по символам, а не по минутам. `summary()` для голосового пайплайна теперь показывает Whisper, TTS, стоимость LLM и аудио отдельно и итог.
2. **Классификатор без учёта пунктуации.** Ответ модели и названия категорий нормализуются (регистр, «ё», пунктуация), сравнение идёт по целым словам. «тех. проблема», «тех проблема», «Тех.проблема», «техническая проблема» распознаются одинаково; «жалоба на технику» не превращается в «тех. проблему».
3. **Согласованы образцы аудио.** `demo_voice.py` использует `samples/voice_question.wav`, который лежит в репозитории (вместо прежнего тонального плейсхолдера — речевой образец с текстом демо-вопроса). TTS выбирает формат по расширению файла, поэтому при отсутствии образца демо сгенерирует настоящий WAV, а не MP3 с расширением `.wav`.

Дополнительно: если все провайдеры недоступны, ответ-заглушка «Сервис временно недоступен» больше не сохраняется в кеш (в `vision.analyze_image` и `voice.answer_text`). Раньше повторный запрос мог вернуть эту заглушку из кеша как «cache hit».

Все пункты покрыты тестами в `tests/test_review_fixes.py`.

## О демо-логе

`logs/sample_run.log` сгенерирован оффлайн (мок-режим): сетевой слой OpenAI SDK подменён заглушкой, остальная логика — реальный код проекта. Видно: retry с растущими задержками и jitter, fallback `ollama → openrouter`, cache hit, классификацию обращения (ответ модели «Тех проблема» без точки корректно распознан), голосовой пайплайн с учётом Whisper и TTS в usage и нулевую стоимость локального Ollama. С реальными сервисами тот же код пишет лог в `logs/app.log`.

## Безопасность

Ключи и endpoint читаются только из `.env`. Хардкода `sk-...` нет; `.env` исключён `.gitignore`.
