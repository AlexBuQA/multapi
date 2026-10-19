# Развёртывание и демонстрация

Как поднять стенд локально к защите или записи видео, проверить его одной командой и что показать:
вопросы с источниками, агента с подтверждением отправки, отказ на вопрос вне базы, трейс в Phoenix.
Команды — для Windows PowerShell 5.1 из корня репозитория. Установку и сборку на видео не показываем —
только работающую систему.

## 1. Что нужно на машине

| Что | Зачем | Проверка |
|---|---|---|
| Docker Desktop (WSL 2) | стек `compose.yaml`: app, migrate, bot, redis, postgres, qdrant, phoenix | `docker compose version` |
| Ollama на хосте | модели — на CPU, контейнер ходит к ним по `host.docker.internal:11434` | `ollama list` |
| Модели Ollama | `llama3.2` — ответы по базе, `qwen3:4b-instruct` — агент, `bge-m3` — эмбеддинги, `gemma3:4b` — фото в чате (по желанию) | `ollama list` |
| Python 3.12+ и `pip install -r requirements.txt` | `scripts/preflight.py`, `scripts/agent_cli.py`, тесты | `python --version` |
| `.env` из `.env.example` | ключи и токены: `LLM__*`, `AGENT__MODEL`, `QDRANT_API_KEY`, `ADMIN_TOKEN`, `INTERNAL_TOKEN`, `BOT_TOKEN` | `python scripts/preflight.py` |

Память: три модели Ollama — около 6 ГБ, стек Docker — около 1,5 ГБ. На машине с 16 ГБ перед показом
закройте лишние программы (браузер с десятками вкладок, IDE с индексацией).

## 2. Накануне

```powershell
ollama pull llama3.2
ollama pull qwen3:4b-instruct
ollama pull bge-m3
python -m pytest -q                         # 1596 passed — регрессия без сети и ключей
docker compose stop                         # сборка при работающем стеке обрывалась (EOF) — нехватка памяти WSL
docker compose build app
docker compose up -d
python scripts/preflight.py --ask           # все строки OK, вопрос с источниками
```

Если что-то красное — раздел 6. Один раз пройдите все сценарии раздела 4: первый вызов каждой модели
загружает её в память (десятки секунд), на защите это время уже не нужно.

## 3. Перед показом (за 15–20 минут)

```powershell
docker compose up -d                        # если стек остановлен; Phoenix стартует до 5 минут
python scripts/preflight.py --ask           # прогрев: модель ответа и эмбеддинги — в памяти
```

Ожидаемый итог — `Стенд готов: критичных проблем — 0`. `WARN` по Phoenix до его старта или по
`BOT_TOKEN` — не критично для API, но бот без токена не ответит.

Для записи экрана:
- **Приглашение PowerShell без пути.** Путь содержит имя пользователя Windows. Перед записью выполните
  `function prompt { "PS> " }` — до закрытия окна приглашение будет коротким.
- **Шапка чата в Telegram.** Имя и аватар бота видны в шапке — если не хотите их показывать, обрежьте
  кадр. `X-Admin-Token` скрипты не печатают; `.env` на экране не открывайте.
- **Окна.** Telegram (Desktop или Web), PowerShell с крупным шрифтом (Ctrl+колесо), браузер с
  http://localhost:6006.

## 4. Сценарии демонстрации

Порядок — как в сценарии видео. Время — Ollama на CPU (Ryzen 5 7530U): ответ по базе — 20–40 с, агент
— 1–2 минуты. Пока модель думает, ответ в боте растёт правками сообщения.

### 4.1. Вопросы по базе с источниками (бот)

В Telegram боту:

1. «Сколько действует ссылка для сброса пароля?» → 30 минут, письмо — в течение 10 минут; внизу
   «Источники:» с `account/password-reset.html`.
2. «Каков срок возврата оплаты?» → 14 дней после списания, деньги — за 5–10 рабочих дней; источник —
   `billing/refunds_v2.docx` (новая редакция; старая, с 7 днями, в индекс не входит).
3. Follow-up «А при оплате по счёту?» → ответ учитывает предыдущий вопрос: на расчётный счёт плательщика.

Под ответом — 👍/👎: оценка сохраняется в Postgres вместе с источниками.

Без бота — те же вопросы через API (Swagger: http://localhost:8000/docs, `POST /rag/query`):

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
curl.exe -s -X POST http://127.0.0.1:8000/rag/query -H "Content-Type: application/json" -d "@examples/requests/rag_query_refund.json"
```

### 4.2. Вопрос вне базы

1. «Можно ли оплатить тариф криптовалютой?» — вопрос о продукте, ответа на который в базе нет. В нём
   слова «тариф» и «оплата», поэтому близость высокая (0.662 на Windows) и проходит порог 0.5. Отказывает
   сама модель по правилу промпта: «По базе не нашёл, могу эскалировать.», строка `rag_refused` в логе.
2. «Как приготовить борщ?» — посторонний вопрос: близость 0.327 < 0.5, отказ без вызова модели
   (score-guard), строка `rag_score_guard` в логе. Ответ приходит сразу — модель не тратит время.

```powershell
docker compose logs app --since 5m | Select-String "rag_score_guard|rag_refused"
```

Вопросы около порога (например, «интеграция с amoCRM» — 0.49) могут уйти в любой из двух слоёв: на
репетиции проверьте, что выбранные для видео вопросы дают ожидаемый отказ.

### 4.3. Агент: поиск, черновик, подтверждение перед отправкой

```powershell
python scripts/agent_cli.py --thread demo-video --example
```

Задача из `examples/requests/agent_stream_input.json`: «Найди в базе знаний, за сколько дней до окончания
периода приходит письмо об автопродлении, и отправь клиенту в Telegram-чат 12345 короткое напоминание».
На экране:
1. `[call_model] вызов search_knowledge_base(...)` — агент сам решил искать;
2. `[execute_tool] search_knowledge_base → [billing/auto-renewal.docx, …]` — результат инструмента;
3. `[call_model] вызов send_telegram_message(chat_id="12345", text=...)`, затем
   `[prepare_send_message] черновик готов`;
4. рамка «ПАУЗА: агент хочет написать клиенту» с текстом и вопрос «Отправить это сообщение клиенту?
   [y/n]».

**Пауза переживает перезапуск сервиса.** Сильная сторона проекта, показывать так:
1. на вопросе «Отправить?» нажмите **Ctrl+C, а не `y`** — решение отложено, thread ждёт. Если нажать
   `y`, сообщение уйдёт сразу и после перезапуска показывать будет нечего — начните заново с новым
   `--thread`;
2. `docker compose restart app`;
3. сразу `python scripts/agent_cli.py --thread demo-video --resume` — скрипт сам дождётся готовности
   сервиса («Сервис ещё не готов… жду до 90 с» → «сервис готов»), затем покажет тот же черновик из
   Postgres и тот же вопрос;
4. `y` → `send_telegram_message → Сообщение отправлено в 12345`, ответ агента;
5. `docker compose logs app --since 10m | Select-String TELEGRAM` — ровно одна строка
   `[TELEGRAM → 12345]` за этот показ. Без `--since` видна вся история контейнера: `restart` логи не
   очищает. Отправка — заглушка: настоящий Telegram API не вызывается;
6. ещё раз `--resume` → «уже завершён: сообщение отправлено (sent=True), повторной отправки не будет» —
   ключ идемпотентности и состояние в Postgres не дают отправить сообщение дважды.

Отказ: на вопросе `n` → «Сообщение клиенту не отправлено: оператор отклонил отправку.», `sent = нет`.
Для каждой репетиции и для записи — новое имя `--thread` (`demo-video`, `demo-video-2`, …): у
завершённого thread ждать уже нечего.

### 4.4. Трейс в Phoenix

http://localhost:6006 → проект `diploma-fastapi` → Traces → последний запрос бота или `POST /rag/query` →
дерево спанов:
- retriever — найденные фрагменты и их близость;
- embedding — эмбеддинг вопроса;
- LLM — промпт с нумерованными фрагментами и ответ.

Видно, из каких шагов состоял ответ и сколько занял каждый.

### 4.5. По желанию: документ и новый документ в базе

- **PDF в чате.** Отправьте боту `samples/support_rules.pdf` с подписью «Какие сроки ответа поддержки?» —
  сервис извлечёт текст документа и ответит по нему (блок 4.3). Проверьте на репетиции: ответ на CPU —
  до минуты.
- **Новый документ в базе знаний.** Загрузка администратором, через 30–60 с документ в индексе:

```powershell
$admin = (Select-String -Path .env -Pattern '^ADMIN_TOKEN=\s*"?([^"#\s]+)').Matches[0].Groups[1].Value   # токен не печатается
curl.exe -s -X POST http://127.0.0.1:8000/documents/upload -H "X-Admin-Token: $admin" -F "file=@samples/upload_tariffs_2027.pdf" -w " %{http_code}`n"
curl.exe -s -X POST http://127.0.0.1:8000/rag/query -H "Content-Type: application/json" -d "@examples/requests/rag_query_upload.json"
```

Ответ — «тариф «Квант» для студентов — 290 ₽ в месяц», источник — `uploads/upload_tariffs_2027.pdf`. Если
файл уже загружали, статус — `unchanged`, ответ тот же.

### 4.6. По желанию: сбой зависимости

- `docker compose stop redis` → `curl.exe -s http://127.0.0.1:8000/ready` — 503 `degraded`, а `/health`
  — 200 и бот отвечает: недоступный кеш — это промах, а не ошибка.
- `docker compose start redis` — `/ready` снова 200.

Остановку самой Ollama на защите не показываем: модель грузится заново десятки секунд. Поведение при
недоступном API модели описано на слайде «Поведение при сбоях» и в `docs/decisions.md`, раздел 8.

## 5. После показа

```powershell
docker compose stop                         # данные — в томах: история, индекс, checkpoint'ы
```

`docker compose down -v` удалит тома — индекс придётся строить заново (около минуты при старте app).

## 6. Если что-то не так

| Симптом | Причина и что делать |
|---|---|
| `preflight`: Ollama не отвечает | Ollama не запущена — запустите приложение Ollama или `ollama serve` |
| `preflight`: модели нет | `ollama pull <модель>` |
| `контейнер app: running, starting` | идёт старт или индексация корпуса — подождите минуту, `docker compose logs app --tail 30` |
| `Qdrant: коллекции kb_docs нет` или точек меньше 102 | индексация при старте ещё идёт или упала — `docker compose logs app \| Select-String ingest` |
| `/ready` — 503 | Redis не запущен: `docker compose up -d redis` |
| Бот молчит | `BOT_TOKEN` пуст или контейнер bot упал: `docker compose logs bot --tail 30`; прокси до Telegram — `BOT_PROXY_URL` |
| Первый ответ очень долгий | модель загружается в память — повторите вопрос; прогрев — `preflight.py --ask` |
| `docker compose build` обрывается с EOF | не хватает памяти WSL: `docker compose stop`, затем `build app`, затем `up -d` |
| `agent_cli`: 409 `approval_pending` | thread ждёт решения — `--resume` или новый `--thread` |
| `agent_cli`: «Связь с сервисом … оборвалась» | сервис перезапустился посреди ответа — дождитесь `(healthy)` в `docker compose ps app` и `--resume` |
| `agent_cli`: «не ответил за 90 с» | app не поднялся после рестарта — `docker compose logs app --tail 30` |
| `agent_cli`: 503 `agent_unavailable` | checkpointer не открылся — `docker compose logs app \| Select-String agent_` |
