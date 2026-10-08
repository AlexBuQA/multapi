"""Сценарий /ask (блок 4.2): тема кнопкой -> вопрос текстом -> ответ потоком."""
from aiogram.fsm.state import State, StatesGroup


class AskFlow(StatesGroup):
    waiting_for_topic = State()
    waiting_for_question = State()
    # Шаг подтверждения «Отправить / Заново» по заданию необязателен и не реализован:
    # для вопроса в поддержку лишний щелчок только мешает. Состояние оставлено, чтобы
    # группа совпадала с заданием.
    confirming = State()
