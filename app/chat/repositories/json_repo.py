"""
JsonChatRepository (блок 4.1): история в файлах — для локальной разработки и одного
процесса сервиса.

Структура на диске (base_dir — CHAT_STORAGE_DIR, по умолчанию var/chats):
    chats/<chat_id>/chat.json        метаданные чата (Chat)
    chats/<chat_id>/messages.jsonl   одна ChatMessage на строку

- Запись — только дописыванием в конец (aiofiles.open(path, "a")), одна строка
  model_dump_json() на сообщение. Файл никогда не переписывается: ни при новом
  сообщении, ни при очистке.
- Очистка истории (/clear) — тоже дописывание: строка {"type": "soft_delete", "at": "<iso>"}.
  list_messages пропускает всё, что было до последнего такого маркера; сами сообщения
  остаются в файле.
- Чтение — readlines() всего файла, ChatMessage.model_validate_json(line), затем
  последние N.
- Процесс упал посреди записи — в конце файла остаётся строка без перевода строки.
  Перед дописыванием он добавляется, иначе следующая запись (в том числе маркер очистки)
  склеилась бы с обрывком и пропала. Сам обрывок при чтении пропускается с
  предупреждением chat_jsonl_line_skipped — остальная история читается.
- Переводы строк — всегда \\n (newline="\\n"), на Windows тоже: файл одинаковый на любой ОС.

Ограничение: блокировок нет. Два процесса сервиса, пишущие в один чат, могут перемешать
строки — для нескольких копий сервиса есть PostgresChatRepository.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import aiofiles
import aiofiles.os

from app.chat.domain import Chat, ChatMessage, ChatNotFoundError, ChatStorageError, utc_now
from app.observability.logging import get_logger

log = get_logger()

SOFT_DELETE = "soft_delete"
CHAT_FILE = "chat.json"
MESSAGES_FILE = "messages.jsonl"


def soft_delete_marker() -> str:
    return json.dumps({"type": SOFT_DELETE, "at": utc_now().isoformat()})


def is_soft_delete(line: str) -> bool:
    """Строка-маркер очистки. Быстрая проверка по подстроке: в сообщении поле "type" не
    встречается, а кавычки внутри текста экранированы (\\"type\\")."""
    if '"type"' not in line:
        return False
    try:
        data = json.loads(line)
    except ValueError:
        return False
    return isinstance(data, dict) and data.get("type") == SOFT_DELETE


async def append_line(path: Path, line: str) -> None:
    """Дописывает строку в конец файла. Если файл не кончается переводом строки (обрыв
    записи), сначала ставит его — обрывок остаётся отдельной битой строкой."""
    prefix = ""
    if await aiofiles.os.path.isfile(path) and await aiofiles.os.path.getsize(path) > 0:
        async with aiofiles.open(path, "rb") as fh:
            await fh.seek(-1, 2)
            if await fh.read(1) != b"\n":
                prefix = "\n"
    async with aiofiles.open(path, "a", encoding="utf-8", newline="\n") as fh:
        await fh.write(prefix + line + "\n")


class JsonChatRepository:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)

    def _chat_dir(self, chat_id: UUID) -> Path:
        return self.base_dir / "chats" / str(chat_id)

    async def _exists(self, chat_id: UUID) -> bool:
        return await aiofiles.os.path.isfile(self._chat_dir(chat_id) / CHAT_FILE)

    async def create_chat(self, owner_external_id: str, interface: str,
                          system_prompt: str | None = None) -> Chat:
        chat = Chat(owner_external_id=owner_external_id, interface=interface, system_prompt=system_prompt)
        folder = self._chat_dir(chat.id)
        try:
            await aiofiles.os.makedirs(folder, exist_ok=False)
            # Сначала временный файл, потом замена: чат без метаданных или с половиной
            # chat.json не появится, даже если процесс упадёт посреди записи.
            tmp = folder / (CHAT_FILE + ".tmp")
            async with aiofiles.open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                await fh.write(chat.model_dump_json(indent=2))
            await aiofiles.os.replace(tmp, folder / CHAT_FILE)
        except OSError as exc:
            raise ChatStorageError(f"Не удалось создать чат в {self.base_dir}: {exc.strerror or exc}") from exc
        return chat

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        path = self._chat_dir(chat_id) / CHAT_FILE
        try:
            if not await aiofiles.os.path.isfile(path):
                return None
            async with aiofiles.open(path, encoding="utf-8") as fh:
                raw = await fh.read()
        except OSError as exc:
            raise ChatStorageError() from exc
        try:
            return Chat.model_validate_json(raw)
        except ValueError as exc:
            raise ChatStorageError(f"Повреждён файл {CHAT_FILE} чата {chat_id}.") from exc

    async def append_message(self, chat_id: UUID, message: ChatMessage) -> ChatMessage:
        if message.chat_id != chat_id:
            raise ValueError(f"message.chat_id={message.chat_id} не совпадает с chat_id={chat_id}")
        try:
            if not await self._exists(chat_id):
                raise ChatNotFoundError(chat_id)
            await append_line(self._chat_dir(chat_id) / MESSAGES_FILE, message.model_dump_json())
        except OSError as exc:
            raise ChatStorageError() from exc
        return message

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        if limit <= 0:
            return []
        path = self._chat_dir(chat_id) / MESSAGES_FILE
        try:
            if not await aiofiles.os.path.isfile(path):
                return []
            async with aiofiles.open(path, encoding="utf-8") as fh:
                lines = await fh.readlines()
        except OSError as exc:
            raise ChatStorageError() from exc
        messages: list[ChatMessage] = []
        for number, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            if is_soft_delete(line):
                messages.clear()          # всё до последнего маркера скрыто
                continue
            try:
                messages.append(ChatMessage.model_validate_json(line))
            except ValueError as exc:
                log.warning("chat_jsonl_line_skipped", chat_id=str(chat_id), line=number,
                            error=type(exc).__name__)
        return messages[-limit:]

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        try:
            if not await self._exists(chat_id):
                return
            await append_line(self._chat_dir(chat_id) / MESSAGES_FILE, soft_delete_marker())
        except OSError as exc:
            raise ChatStorageError() from exc
