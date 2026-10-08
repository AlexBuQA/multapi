"""
Проверка подсчёта токенов (блок 4.1): count_tokens() из app/chat/context.py против
usage.prompt_tokens, который возвращает провайдер на коротком диалоге. Критерий — ±10 %.

    python scripts/check_tokens.py                    # модель сервиса: LLM__BASE_URL, LLM__DEFAULT_MODEL
    python scripts/check_tokens.py --judge --model openai/gpt-4o-mini
                                                      # провайдер судьи блока 3.7 (OpenRouter):
                                                      # EVAL_JUDGE_BASE_URL, EVAL_JUDGE_API_KEY;
                                                      # модель платная — нужен ключ с кредитами

Ходит к провайдеру из .env (адрес, ключ, LLM__PROXY_URL, LLM__USE_SYSTEM_CERTS — как сервис
и eval блока 3.7) двумя запросами без потока, max_tokens=16 — несколько сотен токенов.

count_tokens считает словарём o200k_base (GPT-4o и новее) и +4 токена на сообщение, +2 на
запрос. Модели, для которых подсчёт точен, — GPT-4o и новее; на OpenRouter они платные, а
бесплатных вариантов gpt-oss (тот же словарь o200k) на 8 октября 2026 нет: «404 This model
is unavailable for free». У llama3.2 в Ollama и токенизатор свой, и шапка своя («Cutting
Knowledge Date…»): замер и вывод для SAFETY_MARGIN — в docs/chat.md.

Поэтому запросов два:
1. Диалог из DIALOG — сравнение с usage.prompt_tokens как есть: это критерий задания.
2. Одно короткое сообщение. Разница между провайдером и count_tokens на нём — постоянная
   часть шаблона модели: она одинакова в каждом запросе и не зависит от истории.
   Расхождение на диалоге без неё показывает точность подсчёта самой истории. Постоянную
   часть в бюджете покрывает SAFETY_MARGIN (256).

Код выхода: 0 — критерий выполнен на диалоге как есть, 1 — нет, 2 — сравнить не с чем
(ошибка провайдера, нет usage).

В каждый запрос добавляется случайная метка: Ollama кеширует промпты и на полностью
совпавший может не вернуть prompt_tokens.
"""
from __future__ import annotations

import argparse
import asyncio
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIALOG = [
    {"role": "system", "content": "Ты — ассистент техподдержки продукта «Личный кабинет». Отвечай по-русски, коротко."},
    {"role": "user", "content": "Привет, меня зовут Аня. Не приходит письмо для подтверждения email."},
    {"role": "assistant", "content": "Здравствуйте, Аня! Проверьте папку «Спам» и правильность адреса в профиле. "
                                     "Ссылка в письме действует 24 часа."},
    {"role": "user", "content": "Проверила, письма нет. Can you resend it? Что делать дальше?"},
]
TOLERANCE = 0.10
# Не 1: часть провайдеров рассуждающих моделей отвергает слишком малый max_tokens.
MAX_TOKENS = 16

HINTS = {
    402: "Модель платная, а у ключа нет кредитов. Модели со словарём o200k (GPT-4o и новее) на OpenRouter "
         "платные: пополните счёт или проверьте подсчёт на модели сервиса — python scripts/check_tokens.py.",
    404: "Модель не найдена, снята с бесплатного доступа или ни один её провайдер не подходит под "
         "настройки приватности OpenRouter. Проверьте --model.",
    429: "Лимит запросов бесплатной модели. Повторите через минуту.",
}


def env_value(name: str) -> str | None:
    """Переменная окружения или значение из .env; пустое — то же, что нет."""
    import os

    value = os.environ.get(name)
    if not value and (ROOT / ".env").exists():
        from dotenv import dotenv_values

        value = dotenv_values(ROOT / ".env").get(name)
    return value or None


def labeled(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Копия сообщений со случайной меткой в первом — против кеша промптов Ollama."""
    result = [dict(m) for m in messages]
    result[0]["content"] += f" Метка запроса: {secrets.token_hex(4)}."
    return result


@dataclass(frozen=True)
class Measurement:
    ours: int
    theirs: int
    model: str | None = None
    provider: str | None = None

    @property
    def delta(self) -> float:
        return (self.ours - self.theirs) / self.theirs


async def ask(client: Any, model: str, messages: list[dict[str, str]]) -> Measurement:
    from app.chat.context import count_tokens

    response = await client.chat.completions.create(model=model, messages=messages,
                                                    max_tokens=MAX_TOKENS, temperature=0)
    theirs = response.usage.prompt_tokens if response.usage else 0
    # OpenRouter называет провайдера, который ответил, в поле provider (вне схемы OpenAI).
    provider = (getattr(response, "model_extra", None) or {}).get("provider")
    return Measurement(ours=count_tokens(messages), theirs=theirs or 0, model=response.model, provider=provider)


def verdict(delta: float) -> str:
    return "[OK]" if abs(delta) <= TOLERANCE else "[!] вне допуска"


async def run(client: Any, model: str, base_url: str | None) -> int:
    """Два запроса и отчёт. client — AsyncOpenAI или его подмена в тестах."""
    full = await ask(client, model, labeled(DIALOG))
    if not full.theirs:
        print(f"Провайдер не вернул usage.prompt_tokens (модель {model}) — сравнить не с чем.")
        return 2
    short = await ask(client, model, labeled([{"role": "user", "content": "Привет!"}]))

    where = base_url or "api.openai.com"
    provider = f", провайдер {full.provider}" if full.provider else ""
    print(f"Модель:                      {full.model or model} ({where}{provider})")
    print(f"Диалог, сообщений:           {len(DIALOG)}")
    print(f"count_tokens (o200k_base):   {full.ours}")
    print(f"usage.prompt_tokens:         {full.theirs}")
    print(f"Расхождение:                 {full.delta * 100:+.1f} %  (допуск ±{TOLERANCE * 100:.0f} %)  "
          f"{verdict(full.delta)}")

    if short.theirs:
        constant = short.theirs - short.ours
        rest = full.theirs - constant
        print()
        print("Постоянная часть шаблона модели (запрос из одного сообщения «Привет!»):")
        print(f"  count_tokens / usage:      {short.ours} / {short.theirs}  -> {constant:+d} токенов в каждом запросе")
        if short.provider and short.provider != full.provider:
            print(f"  (ответил другой провайдер: {short.provider} — постоянная часть может отличаться)")
        if rest > 0:
            delta = (full.ours - rest) / rest
            print(f"  Диалог без неё:            {full.ours} против {rest}  -> {delta * 100:+.1f} %  {verdict(delta)}")

    ok = abs(full.delta) <= TOLERANCE
    if not ok:
        print()
        print("Расхождение на диалоге как есть — вне допуска: у модели другой токенизатор или своя шапка "
              "в каждом запросе (см. «Постоянная часть» выше и docs/chat.md).")
    return 0 if ok else 1


def describe_error(exc: Exception) -> str:
    import openai

    if isinstance(exc, openai.APIStatusError):
        body = exc.body if isinstance(exc.body, dict) else {}
        message = str(body.get("message") or exc.message)[:300]
        hint = HINTS.get(exc.status_code, "")
        return f"Провайдер ответил {exc.status_code}: {message}" + (f"\n{hint}" if hint else "")
    if isinstance(exc, openai.APIConnectionError):
        return (f"Нет соединения с провайдером: {exc!r}"[:300]
                + "\nПроверьте адрес, LLM__PROXY_URL и LLM__USE_SYSTEM_CERTS в .env.")
    return f"{type(exc).__name__}: {exc}"[:300]


async def measure(model: str | None, judge: bool = False) -> int:
    import openai
    from openai import AsyncOpenAI, DefaultAsyncHttpxClient

    from app.core.config import get_settings, http_client_options, is_local_url, proxy_for

    cfg = get_settings()
    base_url, api_key = cfg.llm.base_url, cfg.llm.openai_api_key.get_secret_value()
    if judge:
        base_url, api_key = env_value("EVAL_JUDGE_BASE_URL"), env_value("EVAL_JUDGE_API_KEY")
        if not base_url or not api_key or not model:
            print("--judge: нужны EVAL_JUDGE_BASE_URL и EVAL_JUDGE_API_KEY в .env и --model, "
                  "например openai/gpt-4o-mini")
            return 2
    proxy = proxy_for(base_url, cfg.llm.proxy_url)
    options = http_client_options(proxy, cfg.llm.use_system_certs and not is_local_url(base_url))
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=cfg.llm.request_timeout, max_retries=1,
                         http_client=DefaultAsyncHttpxClient(**options) if options else None)
    try:
        return await run(client, model or cfg.llm.default_model, base_url)
    except openai.OpenAIError as exc:
        print(describe_error(exc))
        return 2
    finally:
        await client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="count_tokens против usage.prompt_tokens провайдера")
    parser.add_argument("--model", default=None, help="модель; по умолчанию LLM__DEFAULT_MODEL")
    parser.add_argument("--judge", action="store_true", help="провайдер судьи: EVAL_JUDGE_BASE_URL и EVAL_JUDGE_API_KEY")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    return asyncio.run(measure(args.model, args.judge))


if __name__ == "__main__":
    sys.exit(main())
