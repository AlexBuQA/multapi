"""
Тесты Telegram-бота (блок 4.2): фикстуры. Помощники — в bot_fakes.py.

- bot — Bot с MockedSession: без HTTP, запоминает вызовы Bot API.
- Переменные BOT_* и BACKEND_* окружения в тестах сброшены (clean_bot_env): src/config.py
  (блоки 2–3) при импорте вызывает load_dotenv(), и в полном прогоне .env разработчика —
  BOT_PROXY_URL, BOT_ADMIN_IDS — оказался бы в os.environ и перекрыл значения тестов.
- dispatcher — настоящий Dispatcher из bot/__main__.py, один на прогон: роутер aiogram
  нельзя подключить к двум диспетчерам. Чтобы тесты не мешали друг другу состоянием FSM,
  у каждого свой Telegram chat.id (new_chat_id). dp подставляет в него FakeBackend и
  настройки теста — так handlers получают их параметрами backend и settings.

    pytest tests/bot -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for path in (HERE.parents[1], HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aiogram import Bot, Dispatcher  # noqa: E402

from bot.config import BotSettings  # noqa: E402
from bot_fakes import ADMIN_ID, FakeBackend, MockedSession  # noqa: E402


BOT_ENV = ("BOT_TOKEN", "BACKEND_URL", "BOT_ADMIN_IDS", "BACKEND_TIMEOUT", "BOT_USE_SYSTEM_CERTS",
           "BOT_PROXY_URL", "BOT_PRODUCT_NAME")


@pytest.fixture(autouse=True)
def clean_bot_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in BOT_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings() -> BotSettings:
    return BotSettings(bot_token="42:TEST-TOKEN", backend_url="http://backend.test", bot_admin_ids=[ADMIN_ID],
                       _env_file=None)


@pytest.fixture
def session() -> MockedSession:
    return MockedSession()


@pytest.fixture
def bot(session: MockedSession) -> Bot:
    return Bot(token="42:TEST-TOKEN", session=session)


@pytest.fixture(scope="session")
def dispatcher() -> Dispatcher:
    from bot.__main__ import create_dispatcher

    with pytest.MonkeyPatch.context() as patch:            # фикстура на прогон: clean_bot_env ещё не действует
        for name in BOT_ENV:
            patch.delenv(name, raising=False)
        settings = BotSettings(bot_token="42:TEST-TOKEN", _env_file=None)
    return create_dispatcher(FakeBackend(), settings)  # type: ignore[arg-type]


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def dp(dispatcher: Dispatcher, backend: FakeBackend, settings: BotSettings) -> Dispatcher:
    dispatcher["backend"] = backend
    dispatcher["settings"] = settings
    return dispatcher
