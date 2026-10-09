"""
Роутеры бота (блок 4.2). Порядок подключения важен — aiogram отдаёт апдейт первому
подходящему handler:

1. commands — /cancel (первым: сбрасывает сценарий на любом шаге), /start, /help, /clear,
   /status;
2. fsm — /ask и шаги сценария; текст в состоянии сценария забирает этот роутер;
3. media (блок 4.3) — фото, голос, аудио, PDF и DOCX уходят в сервис файлом; в сценарии
   /ask на шаге вопроса файл — вопрос по теме;
4. text — остальной текст уходит в сервис как вопрос; неизвестные команды и прочие
   сообщения (стикеры, видео) — подсказки.

Ошибки, которые handler не обработал сам, ловит handlers/errors.py — его on_error
регистрируется на корневом роутере и видит исключения всех вложенных.
"""
from __future__ import annotations

from aiogram import Router

from bot.handlers import commands, errors, fsm, media, text


def setup_routers() -> Router:
    """Корневой роутер. Вызывается один раз: роутер aiogram нельзя подключить к двум родителям."""
    root = Router(name="root")
    root.include_routers(commands.router, fsm.router, media.router, text.router, errors.router)
    return root
