"""
Общее для всех тестов pytest (блок 5.2): векторная база выключена.

В .env разработчика QDRANT_URL указывает на настоящий Qdrant, и тесты, которые запускают
lifespan сервиса (TestClient, lifespan_context), стали бы создавать в нём коллекцию.
QDRANT_URL=none выключает VectorStore и RAG (блок 5.3) поверх .env (пустое значение
pydantic-settings пропустил бы и взял .env). Тесты векторной базы и RAG задают настройки явно, а
живые тесты (tests/integration/test_vector_store_live.py, test_rag_live.py) читают .env сами.

Блок 5.5: docstore инкрементальной индексации (RAG__DOCSTORE_DIR) — во временной папке, а не в
var/ingest проекта: иначе тест с подставной моделью эмбеддингов переписал бы состояние настоящей
индексации корпуса. Индексации при старте сервиса в тестах нет (RAG__INGEST_ON_START=false).

Блок 6.4: AGENT_CHECKPOINTER=memory — агент в тестах не пишет checkpoint'ы в файл проекта и в Postgres.
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile

os.environ["QDRANT_URL"] = "none"
_DOCSTORE = tempfile.mkdtemp(prefix="pytest-ingest-")
atexit.register(shutil.rmtree, _DOCSTORE, ignore_errors=True)
os.environ["RAG__DOCSTORE_DIR"] = _DOCSTORE
os.environ["RAG__INGEST_ON_START"] = "false"
# Блок 6.4: lifespan сервиса поднимает персистентного агента; в тестах — checkpointer в памяти, а не файл
# var/agent/checkpoints.sqlite проекта и не Postgres из .env разработчика.
os.environ["AGENT_CHECKPOINTER"] = "memory"
