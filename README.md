# Мультимодальный ИИ-помощник техподдержки — ДЗ 2.6, блоки 3.1–3.4

**Автор:** Александра Бужор
**Репозиторий:** https://github.com/AlexBuQA/multapi

CLI-приложение с **мультимодальными возможностями**, построенное на наработках блоков 2.1–2.5; с блока 3.4 чат-ядро доступно и как HTTP-сервис на FastAPI. Сборка по умолчанию работает на **локальном Ollama** (OpenAI-совместимый API), модели `llama3.2` / `llama3.2-vision`.

Реализованы оба варианта задания:

- **Вариант А — Анализ изображений (Vision API):** путь к картинке → base64 → Vision-запрос → текстовый ответ. Демо на 3 изображениях разного типа (фото, скриншот, график). Работает локально на Ollama с vision-моделью.
- **Вариант Б — Голосовой пайплайн (Whisper + TTS):** аудио → транскрипция → классификация → ответ (LLM) → озвучка → аудиофайл.
- **Блок 3.2 — Архитектурный паспорт:** схема слоёв Gateway → Service → LLM → Data, ADR, точки отказа и проверка LiteLLM — [`docs/architecture.md`](docs/architecture.md).
- **Блок 3.1 — Function Calling:** ассистент техподдержки с инструментами `search_knowledge_base` и `check_service_status`, полный цикл tool_call на локальном Ollama — см. раздел [«Блок 3.1 — Function Calling»](#блок-31--function-calling).
- **Блок 3.3 — Асинхронная обработка запросов к ИИ:** `AsyncLLMClient` (семафор, таймауты, батч, стриминг), бенчмарк sync vs async и первый SSE-эндпоинт `/chat/stream` — см. раздел [«Блок 3.3»](#блок-33--асинхронная-обработка-запросов-к-ии).
- **Блок 3.4 — FastAPI-сервис для LLM:** `POST /chat` с кешем в Redis, `POST /chat/stream`, `GET /health`, `GET /models`; DI, middleware с `X-Request-ID`, CORS, единый формат ошибок, Swagger с примерами — см. раздел [«Блок 3.4»](#блок-34--fastapi-сервис-для-llm).

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
├── app/                     # блоки 3.1, 3.3 и 3.4
│   ├── main.py              # блок 3.4: FastAPI — lifespan, middleware, CORS, обработчики ошибок
│   ├── core/                # блок 3.4: config.py (Settings), exceptions.py (ошибки LLM)
│   ├── deps/providers.py    # блок 3.4: внедрение зависимостей
│   ├── routers/             # блок 3.4: chat.py, models.py, health.py
│   ├── schemas/             # блок 3.4: chat.py, models.py, errors.py
│   ├── services/
│   │   ├── llm.py           # блок 3.4: LLMService — кеш в Redis, поток, перевод ошибок
│   │   └── llm_client.py    # блок 3.3: AsyncLLMClient (семафор, таймауты, батч, стриминг)
│   ├── config.py            # настройки ассистента с tools и асинхронного клиента (pydantic-settings)
│   ├── logging_utils.py     # JSON-лог шагов -> logs/tool_calls.jsonl, logs/llm_calls.jsonl
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
│   ├── run_tool_call.py     # прогон трёх тест-запросов
│   └── requests/            # тела запросов к сервису для curl.exe (блок 3.4)
├── scripts/                 # блок 3.3
│   ├── benchmark.py         # бенчмарк sync vs async -> benchmark_results.md
│   ├── benchmark_results.md # результаты локального прогона (мок и Ollama)
│   ├── stream_demo.py       # демо stream_chat: TTFT и общее время
│   ├── mock_llm_server.py   # мок OpenAI API с задержкой (модель облачного провайдера)
│   └── _target.py           # выбор цели: мок или локальный Ollama
├── docs/
│   ├── architecture.md      # блок 3.2: архитектурный паспорт (схема, ADR, точки отказа)
│   └── litellm/             # config.yaml LiteLLM proxy, скрипт запросов, инструкция
├── tools/
│   └── check_proxy.py       # проверка HTTP-прокси (egress-IP)
├── tests/
│   ├── test_review_fixes.py # тесты на моках (без сети): учёт аудио, классификатор, образцы
│   ├── test_tool_call.py    # блок 3.1: схемы, обработчики, цикл tool_call, лог
│   ├── test_async_client.py # блок 3.3: семафор, батчи, таймаут, стриминг
│   └── test_service.py      # блок 3.4: настройки, ручки, кеш, ошибки, поток, CORS, Swagger
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

Для HTTP-сервиса (блок 3.4) впишите в `.env` переменные `LLM__*` из конца `.env.example` и, чтобы работал кеш, запустите Redis — см. раздел [«Блок 3.4»](#блок-34--fastapi-сервис-для-llm).

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
```

**Блок 3.4 — HTTP-сервис** (Swagger — http://localhost:8000/docs; запросы через `curl.exe` — в разделе [«Блок 3.4»](#блок-34--fastapi-сервис-для-llm)):
```bash
uvicorn app.main:app --reload --port 8000
```

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
| FastAPI и SSE | в блоке 3.3 — `app/main.py` на `sse-starlette` (`/chat`, `/chat/stream` с полем `prompt`); в блоке 3.4 заменены сервисом с роутерами, см. раздел «Блок 3.4» |
| Бенчмарк sync vs async | `scripts/benchmark.py` → результаты в `scripts/benchmark_results.md` (реальный прогон — в репозитории) |
| Демо стриминга | `scripts/stream_demo.py`: TTFT и общее время по `time.perf_counter()` |
| Мок OpenAI API | `scripts/mock_llm_server.py`: отвечает с заданной задержкой, умеет поток и `usage` |
| Тесты | `tests/test_async_client.py` (без сети: заглушка `AsyncOpenAI` на `asyncio.sleep`) |

### Как устроен клиент

- **Семафор — атрибут экземпляра.** `self._sem = asyncio.Semaphore(concurrency)` создаётся один раз в `__init__`. Его используют `complete` и `stream_chat`, поэтому лимит общий для всех вызовов клиента: и для батча, и для параллельных потоков.
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

**SSE** (`uvicorn` + `curl.exe -N`, `llama3.2`; формат блока 3.3 — текст прямо в `data:`, в блоке 3.4 он внутри JSON): ответ пришёл событиями `data:` по мере генерации и завершился событием `done`. Ollama отдаёт поток по токенам, поэтому слова приходят частями — клиент склеивает фрагменты подряд.

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
| Дополнительно: SSE-эндпоинт | `POST /chat/stream` на `sse-starlette`, проверен через `curl.exe -N` на `llama3.2`; в блоке 3.4 заменён на `StreamingResponse`, тесты — в `tests/test_service.py` |

## Блок 3.4 — FastAPI-сервис для LLM

Чат-ядро проекта стало HTTP-сервисом: `uvicorn app.main:app` отвечает на `POST /chat` и отдаёт ответ потоком через `POST /chat/stream`. Структура повторяет эталон из репозитория курса ([`m3_b4`](https://github.com/abat-voix/ai-tools-and-links/tree/main/m3_b4)) и адаптирована к проекту: провайдер по умолчанию — локальный Ollama, повторы делает SDK, а семафор из блока 3.3 ограничивает запросы всего сервиса.

### Структура

```
app/
├── main.py              # FastAPI: lifespan, middleware request_id, CORS, обработчики ошибок, роутеры
├── core/
│   ├── config.py        # Settings + вложенный LLMSettings (pydantic-settings v2), get_settings() с @lru_cache
│   └── exceptions.py    # LLMError, LLMRateLimitError, LLMTimeoutError, LLMAuthError, LLMUnavailableError
├── deps/
│   └── providers.py     # get_settings, get_openai, get_cache, get_llm_service + Annotated-алиасы
├── routers/
│   ├── chat.py          # POST /chat, POST /chat/stream
│   ├── models.py        # GET /models
│   └── health.py        # GET /health, GET /ready
├── services/
│   └── llm.py           # LLMService: complete() с кешем в Redis, stream(), перевод ошибок SDK
└── schemas/
    ├── chat.py          # ChatRequest, ChatResponse.from_openai(), ChatDelta, Usage
    ├── models.py        # ModelInfo и статический каталог моделей с ценами
    └── errors.py        # единый формат ошибок и описания ответов для Swagger
```

Код блоков 3.1 и 3.3 (`app/config.py`, `app/llm/`, `app/tools/`, `app/services/llm_client.py`) не менялся: ассистент с инструментами и бенчмарк работают как раньше.

### Эндпоинты

| Метод и путь | Что делает | Коды ответа |
|--------------|------------|-------------|
| `POST /chat` | Ответ целиком. Повторный одинаковый запрос берётся из Redis с `cached: true` | 200, 422, 429, 502, 504 |
| `POST /chat/stream` | Ответ потоком SSE: кадры `data: {"content": ...}`, затем `data: {"usage": {...}}` и `data: [DONE]` | 200, 422, 429, 502, 504 |
| `GET /health` | `{"status": "ok"}` без зависимостей: 200 даже без Redis и провайдера | 200 |
| `GET /ready` | Состояние Redis: `ready` или `degraded` — сервис работает без кеша | 200 |
| `GET /models` | Каталог: модели OpenAI со справочными ценами и локальные модели Ollama | 200 |

Swagger открывается на http://localhost:8000/docs. У `POST /chat` и `POST /chat/stream` есть два готовых примера запроса (список «Examples»), у каждого кода ответа — пример тела. Примеры не задают `model`, поэтому «Try it out» работает с любым провайдером из настроек. `/health`, `/ready` и `/models` не принимают тело и не обращаются к модели, поэтому у них в Swagger только код 200: 422, 429, 502 и 504 там возникнуть не могут.

### Как обрабатывается запрос

1. **Middleware** присваивает запросу `request_id` или берёт его из заголовка `X-Request-ID`, если там только буквы, цифры, `.`, `_` и `-`. Затем кладёт его в `request.state`, замеряет `duration_ms` через `time.perf_counter()`, пишет одну строку лога и возвращает `X-Request-ID` в ответе. Для `/chat/stream` `duration_ms` — время до первого фрагмента, потому что тело потока уходит позже.
   ```
   2026-10-06 21:11:04,839 INFO llm-service: request_id=rid-abc-123 method=GET path=/health status=200 duration_ms=0.5
   ```
2. **Внедрение зависимостей.** Ручка получает `LLMServiceDep`. `get_llm_service` собирает `LLMService(openai, cache, settings)` из объектов, которые `lifespan` создал один раз и положил в `app.state`. Глобальных клиентов на уровне модулей нет, поэтому тесты подменяют их через `app.dependency_overrides` и `app.state`.
3. **Кеш.** `complete()` строит ключ `chat:` + sha256 от запроса без `user_id`, `session_id` и `stream`; модель по умолчанию подставляется до расчёта ключа. Чтение — `await cache.get`, запись — `await cache.setex` с TTL `CACHE_TTL_SECONDS`. Попадание в кеш возвращается через `ChatResponse.model_validate_json(...)` с `cached: true`. Ответы кешируются при любой `temperature`, а `temperature` входит в ключ: ассистент техподдержки на одинаковый вопрос с теми же параметрами отвечает одинаково и не тратит токены. В эталоне курса кеш работает только при `temperature == 0`.
4. **Ошибки провайдера** `LLMService` переводит в доменные исключения, а обработчик в `app/main.py` — в JSON `{"error": {"code": ..., "message": ..., "request_id": ...}}`:

   | Ошибка OpenAI SDK | Исключение | HTTP | `error.code` |
   |-------------------|------------|------|--------------|
   | `RateLimitError` | `LLMRateLimitError` | 429 и `Retry-After`, если его прислал провайдер | `llm_rate_limit` |
   | `APITimeoutError` | `LLMTimeoutError` | 504 | `llm_timeout` |
   | `AuthenticationError`, `PermissionDeniedError` | `LLMAuthError` | 502 | `llm_auth` |
   | `APIConnectionError` (Ollama не запущен, неверный адрес) | `LLMUnavailableError` | 502 | `llm_unavailable` |
   | другие ошибки API, например 404 — модели нет у провайдера | `LLMError` | 502 | `llm_error` |

   Наружу уходит понятный текст, а исходная ошибка провайдера — только в лог сервиса вместе с `request_id`. Ошибка валидации возвращается с кодом 422: `{"error": {"code": "validation_error", "message": ..., "fields": [{"field": ..., "message": ...}]}}`. Любое другое исключение — 500 `internal_error` без трейсбека.
5. **Поток.** До ответа клиенту сервис дожидается первого фрагмента. Поэтому, если провайдер недоступен или ключ неверный, клиент получит обычный JSON с кодом 502, 429 или 504, а не поток с кодом 200 и ошибкой внутри. Обрыв посреди ответа приходит кадром `data: {"error": {...}}` перед `[DONE]`. Текст передаётся внутри JSON, как в эталоне курса: перевод строки в ответе модели (списки, код) не ломает разметку SSE, где пустая строка означает конец события. Если клиент закрыл соединение, сервис закрывает поток к провайдеру и освобождает слот семафора.
6. **Надёжность.** Повторы встроены в SDK (`LLM__MAX_RETRIES`): 408, 409, 429, 5xx и ошибки соединения повторяются с экспоненциальной задержкой и учётом `Retry-After`. Tenacity поверх не добавлен, иначе попытки перемножились бы: 3 × 3 = 9. Семафор из блока 3.3 создаётся один раз в `lifespan` и ограничивает число одновременных запросов к модели (`LLM__MAX_CONCURRENCY`) — это Bulkhead из архитектурного паспорта. Если Redis недоступен при старте, сервис работает без кеша до перезапуска. Если Redis упал во время работы, ошибка кеша пишется в лог, а запрос уходит в модель.
7. **CORS** разрешает только адреса из `CORS_ORIGINS` и открывает фронтенду заголовок `X-Request-ID`. Сочетание `CORS_ALLOW_CREDENTIALS=true` и `["*"]` настройки не пропускают: сервис не стартует и объясняет почему.

### Настройки

`app/core/config.py`: `Settings(BaseSettings)` с вложенным `LLMSettings`. Переменные вложенной секции задаются с префиксом `LLM__` (`env_nested_delimiter="__"`), ключ хранится как `SecretStr`. `get_settings()` обёрнут в `@lru_cache`. Пустое значение в `.env` (`КЛЮЧ=`) означает значение по умолчанию.

| Переменная | По умолчанию | Для локального Ollama |
|------------|--------------|-----------------------|
| `LLM__OPENAI_API_KEY` | — (обязательна) | `ollama` — любое непустое значение |
| `LLM__BASE_URL` | `https://api.openai.com/v1` | `http://localhost:11434/v1` |
| `LLM__DEFAULT_MODEL` | `gpt-4o-mini` | `llama3.2` |
| `LLM__REQUEST_TIMEOUT` | `30` с на одну попытку | `120` — на CPU ответ генерируется долго |
| `LLM__MAX_RETRIES` | `3` | `2` |
| `LLM__MAX_CONCURRENCY` | `10` | |
| `REDIS_URL` | `redis://localhost:6379/0` | |
| `CACHE_TTL_SECONDS` | `3600` | |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | |
| `CORS_ALLOW_CREDENTIALS` | `false` | |
| `APP_NAME` | `multapi — LLM-сервис техподдержки` | |

Переменные блоков 2–3 (`OPENAI_API_KEY`, `SUPPORT_PRIMARY_MODEL`, `LLM_REQUEST_TIMEOUT` и др.) сервис не читает: они по-прежнему нужны CLI, ассистенту с инструментами и бенчмарку.

Без `LLM__OPENAI_API_KEY` uvicorn не стартует — так pydantic-settings и должен себя вести:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for Settings
llm.openai_api_key
  Field required [type=missing, input_value={}, input_type=dict]
```

### Redis

Кеш нужен только для критерия «повторный запрос — `cached: true`»: без Redis сервис работает, просто без кеша. На Windows Redis удобно запустить в Docker Desktop — он понадобится и в следующем блоке.

```powershell
docker run -d --name multapi-redis -p 6379:6379 mirror.gcr.io/library/redis:7.4
docker exec multapi-redis redis-cli ping     # PONG
docker stop multapi-redis                    # выключить — проверка /health без Redis
docker start multapi-redis                   # включить снова
```

Две особенности, с которыми столкнулись при проверке:

- **Docker Hub недоступен** (`TLS handshake timeout` при обращении к `auth.docker.io`), поэтому образ берётся с зеркала Google `mirror.gcr.io` — это тот же официальный образ `redis:7.4`.
- **Docker Desktop забирает память у Ollama.** Его виртуальная машина WSL по умолчанию занимает несколько гигабайт, и Ollama не может загрузить модель: `llama runner process has terminated: ... failed to allocate compute pp buffers`. Помогает предел памяти в `%USERPROFILE%\.wslconfig` — Redis хватает и одного гигабайта:
  ```ini
  [wsl2]
  memory=1GB
  ```
  После правки: закрыть Docker Desktop, выполнить `wsl --shutdown` и запустить Docker Desktop снова.

Без Docker Redis можно поставить в WSL: `sudo apt install redis-server`, затем `sudo service redis-server start`; сервис на Windows видит его по `localhost:6379`.

### Проверка

Терминал 1 — сервис:
```powershell
uvicorn app.main:app --reload --port 8000
```

Терминал 2 — запросы. В PowerShell используется `curl.exe`, а тело запроса берётся из файла, чтобы не воевать с кавычками:
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
curl.exe -s -w "\ntime_total: %{time_total}s\n" -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "@examples/requests/chat.json"
curl.exe -s -w "\ntime_total: %{time_total}s\n" -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "@examples/requests/chat.json"
curl.exe -N -X POST http://localhost:8000/chat/stream -H "Content-Type: application/json" -d "@examples/requests/chat_stream.json"
curl.exe -N -X POST http://localhost:8000/chat/stream -H "Content-Type: application/json" -d "@examples/requests/chat_stream_long.json"
curl.exe -i http://localhost:8000/health
curl.exe http://localhost:8000/ready
```

`chat_stream.json` — запрос из задания («считай до пяти»), `chat_stream_long.json` — вопрос техподдержки с длинным ответом, на котором хорошо видно, как ответ приходит кусками.

В Linux и macOS — как в задании: `curl -X POST localhost:8000/chat -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"hi"}]}'`.

**Неверный ключ.** Ollama ключ не проверяет, а OpenRouter из нашей сети недоступен: сервис отвечает `502 llm_unavailable`, до проверки ключа дело не доходит. Поэтому ответ `502 llm_auth` проверяется на моке OpenAI API из блока 3.3: с параметром `--api-key` он, как настоящий API, отвечает 401 на чужой ключ. Переменные окружения важнее `.env`, поэтому сам `.env` менять не нужно.

Redis на время проверки выключается: иначе запрос, ответ на который уже лежит в кеше, вернётся оттуда с `cached: true`, и до провайдера с его проверкой ключа дело не дойдёт (см. «Наблюдения» ниже).

Терминал 3 — мок, принимающий только ключ `sk-correct`:
```powershell
python scripts/mock_llm_server.py --port 8001 --api-key sk-correct
```
Терминал 1 — Redis выключен, сервис с другим ключом:
```powershell
docker stop multapi-redis
$env:LLM__BASE_URL = "http://127.0.0.1:8001/v1"
$env:LLM__OPENAI_API_KEY = "sk-broken"
uvicorn app.main:app --port 8000
```
Запрос к `/chat` из терминала 2 вернёт `502` и `{"error": {"code": "llm_auth", ...}}`, а в логе сервиса будет исходная ошибка: `AuthenticationError("Error code: 401 - ... Incorrect API key provided ...")`. После проверки остановите мок и сервис, а в терминале 1 уберите переменные и включите Redis: `Remove-Item Env:LLM__BASE_URL, Env:LLM__OPENAI_API_KEY`, затем `docker start multapi-redis`.

### Результаты

Прогон 6 октября 2026 года на ноутбуке без видеокарты: Ollama `llama3.2`, Redis 7.4 в Docker Desktop.

| Проверка | Результат |
|----------|-----------|
| Старт `uvicorn app.main:app --reload` | `Redis redis://localhost:6379/0 доступен — кеш ответов включён`, `Модель по умолчанию: llama3.2` |
| `POST /chat` `{"messages":[{"role":"user","content":"hi"}]}` | 200, `"content":"How can I assist you today?"`, `usage` 26 + 8 = 34 токена, `cached: false`; 2,80 с (на сервере 2544 мс) |
| Тот же запрос повторно | 200, тот же ответ с `cached: true`; 0,22 с (на сервере 4,4 мс) |
| `POST /chat/stream` «считай до пяти» | кадры `data: {"content":"4"}`, `data: {"content":"."}`, затем `data: {"usage":{...}}` и `data: [DONE]`; до первого фрагмента 386 мс |
| `POST /chat/stream` с вопросом из `chat_stream_long.json` | 151 кадр `data: {"content":...}` по мере генерации, затем `data: {"usage":{"prompt_tokens":73,"completion_tokens":152,"total_tokens":225}}` и `data: [DONE]`; до первого фрагмента 10,5 с |
| `GET /health` при остановленном Redis | `HTTP/1.1 200 OK`, `{"status":"ok"}`, заголовок `x-request-id` |
| `GET /ready` при остановленном Redis | `{"status":"degraded","components":{"redis":false}}` |
| `POST /chat` при остановленном Redis | 200; в логе `cache.get failed, идём в модель без кеша` и `cache.setex failed, ответ не закеширован` |
| Swagger, «Try it out» с примером «Один вопрос» | 200: `POST /chat status=200 duration_ms=44315` в логе сервиса — длинный ответ `llama3.2` на CPU |
| Неверный ключ: мок с `--api-key sk-correct`, сервис с `sk-broken`, Redis выключен | `HTTP 502`, `{"error":{"code":"llm_auth","message":"Провайдер LLM отклонил ключ доступа сервиса.",...}}`; мок ответил `401 Unauthorized`, в логе сервиса — `llm_error=llm_auth cause=AuthenticationError("Error code: 401 - ... Incorrect API key provided: sk-***oken. ...")` |
| То же с включённым Redis | `HTTP 200`, `cached: true` за 13,8 мс: ответ на `hi` уже был в кеше, мок не получил ни одного запроса |
| Ollama не хватило памяти — ошибка 500 | `502 {"error":{"code":"llm_error","message":"Провайдер LLM вернул ошибку 500.",...}}` без трейсбека; текст Ollama — только в логе сервиса |
| Провайдер недоступен (OpenRouter из нашей сети) | `502 {"error":{"code":"llm_unavailable",...}}` |

Наблюдения:

- **Кеш.** Повтор отвечает за 4,4 мс на сервере вместо 2,5 с — модель не вызывается. `curl` показывает 0,22 с: для `localhost` он сначала 200 мс ждёт ответа по IPv6 (`::1`), а uvicorn слушает только `127.0.0.1`. С адресом `http://127.0.0.1:8000` эта задержка пропадает.
- **Поток.** 3B-модель `llama3.2` поняла «считай до пяти» по-своему и ответила «4.» — двумя фрагментами. Это качество маленькой модели, а не сервиса; на вопросе из `chat_stream_long.json` ответ приходит десятками фрагментов.
- **Redis выключен на ходу.** Запрос не падает, но каждое обращение к кешу ждёт таймаута подключения — до 1 с на `get` и столько же на `setex`. Если Redis недоступен уже при старте, сервис сразу работает без кеша и на него время не тратит.
- **Повторы SDK.** Ошибку 500 от Ollama SDK повторил ещё 2 раза (`LLM__MAX_RETRIES=2`), поэтому ответ 502 пришёл через 10–15 с. На 401 SDK не повторяет запрос: ответ `llm_auth` пришёл за 0,9 с.
- **Кеш и смена провайдера.** Ключ кеша строится по самому запросу — модель, сообщения, параметры, — как требует задание; адрес провайдера в него не входит. Поэтому после переключения на другой провайдер с той же моделью сервис ещё `CACHE_TTL_SECONDS` отдаёт прежние ответы из кеша. Для деградации это плюс: пока провайдер недоступен или ключ сломан, уже заданные вопросы получают ответ. Если провайдеры отвечают по-разному, в ключ стоит добавить `LLM__BASE_URL`.
- **Перевод строки внутри ответа.** Модель присылает фрагменты вроде `":\n\n"` — внутри JSON они безопасны. Без JSON пустая строка закрыла бы событие SSE посреди ответа.

### Соответствие критериям блока 3.4

| Критерий | Реализация |
|----------|------------|
| `uvicorn app.main:app --reload` стартует; без ключа — падает | `get_settings()` в `app/main.py`; тест `test_missing_api_key_stops_start` |
| `POST /chat` → 200, JSON с `content`, `usage`, `cached: false` | `routers/chat.py`, `LLMService.complete`; тест `test_chat_ok` |
| `POST /chat/stream` отдаёт ответ кусками, в конце `data: [DONE]` | `StreamingResponse`, кадры собираются вручную; тест `test_stream_frames_usage_and_done` |
| `GET /health` — 200 при выключенном Redis | ручка без зависимостей; тесты `test_health_without_dependencies`, `test_starts_without_redis` |
| Повторный запрос — `cached: true` и быстрее | Redis `get`/`setex`, ключ `chat:` + sha256; тест `test_repeat_request_served_from_cache` |
| Сломанный ключ → 502 `llm_auth`, а не 500 с трейсбеком | `provider_errors()` + обработчик `LLMError`; тест `test_provider_errors_mapped`; живая проверка — мок с `--api-key` |
| Swagger: примеры запроса, `summary`, `responses` 200/422/429/502/504; «Try it out» не даёт 422 | `json_schema_extra` и `openapi_examples`; тест `test_swagger_examples_summaries_and_responses` |
| DI без глобальных клиентов, `Annotated`-алиасы | `app/deps/providers.py` |
| Middleware: `request_id`, `duration_ms`, лог, `X-Request-ID` | `request_context` в `app/main.py`; тест `test_request_id_generated_propagated_and_logged` |
| CORS без `["*"]` вместе с `allow_credentials=True` | `Settings._check_cors`; тесты `test_cors_wildcard_with_credentials_rejected`, `test_cors_allows_only_configured_origin` |

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

HTTP-сервис (блок 3.4) читает свои переменные с префиксом `LLM__`; для локального Ollama:

```
LLM__OPENAI_API_KEY=ollama                  # обязательна: без неё сервис не стартует
LLM__BASE_URL=http://localhost:11434/v1
LLM__DEFAULT_MODEL=llama3.2
LLM__REQUEST_TIMEOUT=120
LLM__MAX_RETRIES=2
```

Остальные (`REDIS_URL`, `CACHE_TTL_SECONDS`, `CORS_ORIGINS` …) — в таблице раздела [«Блок 3.4»](#блок-34--fastapi-сервис-для-llm).

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
