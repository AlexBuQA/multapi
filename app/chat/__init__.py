"""
Чат с историей на сервере (блок 4.1).

- domain.py — Chat, ChatMessage (Pydantic v2) и доменные ошибки;
- repository.py — контракт ChatRepository (typing.Protocol);
- repositories/ — JsonChatRepository (файлы JSONL) и PostgresChatRepository (async
  SQLAlchemy 2.x, ORM-модели в pg_models.py);
- context.py — скользящее окно истории, подсчёт токенов (tiktoken) и бюджет;
- service.py — ChatService: история, контекст, поток ответа модели;
- routes.py — эндпоинты /chats с потоком SSE;
- deps.py — выбор хранилища по CHAT_REPOSITORY и сборка ChatService.

Архитектура, выбор стратегии контекста и примеры запросов — docs/chat.md.
"""
