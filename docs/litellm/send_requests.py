"""
Три запроса в локальный LiteLLM proxy (блок 3.2) и кто на них ответил.

Proxy должен быть запущен: litellm --config docs/litellm/config.yaml --port 4000
Запускать из основного окружения проекта (.venv) — нужен только пакет openai:
    python docs/litellm/send_requests.py

Какой провайдер ответил, видно по заголовкам LiteLLM:
  x-litellm-model-group         — группа, которая в итоге ответила;
  x-litellm-attempted-fallbacks — сколько раз proxy переключался на резервную;
  x-litellm-overhead-duration-ms — накладные расходы самого proxy.
"""
from __future__ import annotations

import os
import sys
import time

from openai import OpenAI

BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
MASTER_KEY = os.getenv("LITELLM_MASTER_KEY")

REQUESTS = [
    ("support-primary", "Как сбросить пароль в личном кабинете? Ответь одним предложением."),
    ("support-primary", "Что делать, если не приходит письмо? Ответь одним предложением."),
    ("support-fallback", "Ответь одним словом: привет."),
]


def main() -> int:
    if not MASTER_KEY:
        print("Задайте LITELLM_MASTER_KEY — тот же, с которым запущен proxy.", file=sys.stderr)
        return 2

    client = OpenAI(base_url=BASE_URL, api_key=MASTER_KEY, timeout=300, max_retries=0)
    for number, (group, question) in enumerate(REQUESTS, start=1):
        started = time.perf_counter()
        raw = client.chat.completions.with_raw_response.create(
            model=group,
            messages=[{"role": "user", "content": question}],
            max_tokens=80,
            temperature=0,
        )
        reply = raw.parse()
        headers = raw.headers
        print(f"\n=== Запрос {number}: группа {group}")
        print(f"Вопрос             : {question}")
        print(f"Ответила группа    : {headers.get('x-litellm-model-group')} "
              f"({headers.get('x-litellm-model-name')})")
        print(f"Переключений       : {headers.get('x-litellm-attempted-fallbacks')} | "
              f"повторов: {headers.get('x-litellm-attempted-retries')}")
        print(f"Время              : {time.perf_counter() - started:.1f} с, накладные proxy "
              f"{float(headers.get('x-litellm-overhead-duration-ms') or 0):.0f} мс")
        print(f"Ответ              : {reply.choices[0].message.content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
