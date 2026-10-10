"""
Роутеры бота (блок 4.2). Порядок подключения важен — aiogram отдаёт апдейт первому
подходящему handler:

0. admin (блок 4.4) — /status, /stats, /users, /broadcast; фильтр IsAdmin на уровне роутера:
   не-администратора роутер пропускает дальше, и его команду забирает commands;
   feedback (блок 4.4) — нажатия 👍/👎 (callback_query, с сообщениями не пересекается);
1. commands — /cancel (первым: сбрасывает сценарий на любом шаге), /start, /help, /clear,
   отказ не-администраторам в admin-командах;
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

from bot.handlers import admin, commands, errors, feedback, fsm, media, text


def setup_routers() -> Router:
    """Корневой роутер. Вызывается один раз: роутер aiogram нельзя подключить к двум родителям."""
    root = Router(name="root")
    root.include_routers(admin.router, feedback.router, commands.router, fsm.router, media.router, text.router,
                         errors.router)
    return root
