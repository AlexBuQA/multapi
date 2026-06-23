"""
Проверка HTTP-прокси: показывает внешний IP напрямую и через прокси.

Адрес берётся из .env (LLM_PROXY) — без хардкода. Если LLM_PROXY пуст
(штатно для локального Ollama), проверяется только прямое соединение.

    python tools/check_proxy.py
"""
from __future__ import annotations

import asyncio
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

URL = "https://api.ipify.org?format=json"
HTTP_PROXY = (os.getenv("LLM_PROXY") or "").strip()


async def _show(label: str, proxy: str | None) -> None:
    try:
        async with httpx.AsyncClient(proxy=proxy, timeout=15) as client:
            r = await client.get(URL)
            print(f"{label}: {r.status_code} {r.text}")
    except Exception as exc:  # noqa: BLE001
        print(f"{label}: ОШИБКА — {type(exc).__name__}: {exc}")


async def main() -> None:
    await _show("direct     ", None)
    if HTTP_PROXY:
        await _show("http proxy ", HTTP_PROXY)
    else:
        print("http proxy : пропуск (LLM_PROXY не задан — штатно для локального Ollama)")


if __name__ == "__main__":
    asyncio.run(main())
