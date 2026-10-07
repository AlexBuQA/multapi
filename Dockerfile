# syntax=docker/dockerfile:1.7
#
# Образ HTTP-сервиса ассистента техподдержки (блок 3.5).
# Две стадии: builder ставит зависимости через uv в /app/.venv, runtime получает
# только /app (venv + код) и запускается под непривилегированным пользователем.
#
#   docker build -t llm-service:v1 .
#   docker compose up -d --build        # сервис + Redis, см. compose.yaml

# ========== Стадия 1: builder ==========
FROM python:3.13-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

# Байткод компилируется при сборке (быстрее первый старт), файлы копируются, а не
# линкуются из кеша (кеш смонтирован только на время RUN), Python не скачивается —
# используется интерпретатор базового образа.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Зависимости — до кода: слой пересобирается, только если поменялись pyproject.toml
# или uv.lock. Правка кода приложения этот шаг не трогает.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

# Код приложения — последним слоем. В образ попадает пакет app/ и руководство
# пользователя: по нему /chat отвечает как ассистент техподдержки (блок 3.7).
COPY data/knowledge_base.json ./data/knowledge_base.json
COPY app/ ./app/

# ========== Стадия 2: runtime ==========
FROM python:3.13-slim-bookworm AS runtime

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

# Готовность: /ready отвечает 200, только когда доступен Redis (503 — degraded).
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready', timeout=3)"]

# exec-форма: uvicorn получает сигналы напрямую (корректная остановка по docker stop).
# --host 0.0.0.0 — иначе uvicorn слушает только loopback контейнера и снаружи недоступен.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
