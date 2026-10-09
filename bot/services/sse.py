"""
Разбор потока Server-Sent Events (блоки 4.2–4.3) — так chat-сервис отдаёт ответ модели.
С блока 4.3 в data: каждого события — JSON ({"type": "token", "delta": ...}); его разбирает
BackendClient, а здесь — только рамки событий.

Событие — строки до пустой строки:
    event: <тип>      необязательно; по умолчанию message
    data: <текст>     одна или несколько; строки data: одного события склеиваются через \\n
    : комментарий     пропускается (keep-alive)
После "data:" один пробел отбрасывается, остальное — часть текста. Поля id: и retry:
сервис не шлёт, они пропускаются.

Строки режет sse_lines, а не httpx.Response.aiter_lines(): та делит текст как
str.splitlines() — ещё и по \\u2028, \\x85, \\x0b и другим символам, которые модель вполне
может вставить в ответ. Кусок после такого символа оказался бы строкой без «data:» и
пропал. По спецификации SSE конец строки — только \\r\\n, \\r или \\n.
"""
from __future__ import annotations

import re
from collections.abc import AsyncIterable, AsyncIterator
from typing import NamedTuple

_EOL = re.compile(r"\r\n|\r|\n")


class SSEEvent(NamedTuple):
    event: str
    data: str


async def sse_lines(chunks: AsyncIterable[str]) -> AsyncIterator[str]:
    """Строки потока по границам SSE: \\r\\n, \\r, \\n — даже если граница разорвана между кусками."""
    buffer = ""
    async for chunk in chunks:
        buffer += chunk
        while (match := _EOL.search(buffer)) is not None:
            if match.group() == "\r" and match.end() == len(buffer):
                break                              # может оказаться началом \r\n — ждём следующий кусок
            yield buffer[:match.start()]
            buffer = buffer[match.end():]
    if buffer:
        yield buffer.removesuffix("\r")


async def iter_sse(lines: AsyncIterable[str]) -> AsyncIterator[SSEEvent]:
    event, data = "message", list[str]()
    async for raw in lines:
        line = raw.rstrip("\r")
        if not line:
            if data:
                yield SSEEvent(event, "\n".join(data))
            event, data = "message", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "data":
            data.append(value)
        elif field == "event":
            event = value or "message"
    if data:                                   # поток кончился без пустой строки
        yield SSEEvent(event, "\n".join(data))
