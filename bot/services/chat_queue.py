"""
Очередь вопросов одного чата Telegram (блок 4.2).

aiogram обрабатывает апдейты параллельно. Если пользователь пришлёт второй вопрос, пока
модель отвечает на первый, оба запроса ушли бы в сервис сразу. Сервис ставит вопросы
одного чата в очередь (ChatLocks, блок 4.1), и второй ждал бы там без единого байта
ответа. Тогда ожидание входит в таймаут бота: ответ llama3.2 на CPU длиннее
BACKEND_TIMEOUT дал бы на втором вопросе «отвечает слишком долго». При этом сервис всё
равно ответил бы на него позже и сохранил ответ, которого пользователь не видел.

Поэтому бот сам отправляет вопросы и /clear одного чата по очереди: второй вопрос ждёт в
боте и уходит в сервис, когда первый ответ показан. Замки — в памяти процесса бота, по
одному на Telegram chat.id. Неиспользуемые удаляет WeakValueDictionary.
"""
from __future__ import annotations

import asyncio
import contextlib
import weakref
from collections.abc import AsyncIterator


class ChatQueue:
    def __init__(self) -> None:
        self._locks: weakref.WeakValueDictionary[int, asyncio.Lock] = weakref.WeakValueDictionary()

    @contextlib.asynccontextmanager
    async def turn(self, telegram_chat_id: int) -> AsyncIterator[None]:
        lock = self._locks.get(telegram_chat_id)
        if lock is None:
            lock = self._locks[telegram_chat_id] = asyncio.Lock()
        async with lock:
            yield
