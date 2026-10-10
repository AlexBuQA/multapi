"""
Общее для всех тестов pytest (блок 5.2): векторная база выключена.

В .env разработчика QDRANT_URL указывает на настоящий Qdrant, и тесты, которые запускают
lifespan сервиса (TestClient, lifespan_context), стали бы создавать в нём коллекцию.
QDRANT_URL=none выключает VectorStore поверх .env (пустое значение pydantic-settings
пропустил бы и взял .env). Тесты векторной базы задают настройки явно, а живой тест
(tests/integration/test_vector_store_live.py) читает .env сам.
"""
from __future__ import annotations

import os

os.environ["QDRANT_URL"] = "none"
