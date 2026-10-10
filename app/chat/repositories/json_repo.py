"""
JsonChatRepository (блок 4.1): история в файлах — для локальной разработки и одного
процесса сервиса.

Структура на диске (base_dir — CHAT_STORAGE_DIR, по умолчанию var/chats):
    chats/<chat_id>/chat.json        метаданные чата (Chat)
    chats/<chat_id>/messages.jsonl   одна ChatMessage на строку
    owners/<sha256>.json             {"chat_id": ...} — чат клиента для get_or_create_chat

- get_or_create_chat (блок 4.2): ключ — sha256 от interface и owner_external_id (в
  owner_external_id может быть что угодно, от email до символов, запрещённых в именах
  файлов). Файл owners/<ключ>.json указывает на чат клиента. Нет файла — один раз
  просматриваются все chat.json: чаты, созданные до блока 4.2 через create_chat, тоже
  находятся, берётся самый ранний. Одновременные вызовы с одним ключом в процессе
  сервиса идут по очереди (asyncio.Lock на ключ).

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

Блок 4.4 (OpsRepository) — три файла в корне base_dir:
    feedback.jsonl     оценки ответов; уникальность пары (owner_external_id, message_id)
                       проверяется под asyncio.Lock процесса перед дописыванием
    moderation.jsonl   инциденты модерации (без текста)
    broadcasts.json    очередь рассылки целиком; каждое изменение — новый файл и замена
Сводки (stats, recent_users) просматривают все чаты — для учебного стенда хватает; на
больших объёмах — Postgres с индексами по created_at.

Ограничение: блокировок нет. Два процесса сервиса, пишущие в один чат, могут перемешать
строки — для нескольких копий сервиса есть PostgresChatRepository.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import weakref
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import UUID

import aiofiles
import aiofiles.os

from app.chat.domain import (Broadcast, BroadcastNotClaimedError, Chat, ChatMessage, ChatNotFoundError, ChatStats,
                             ChatStorageError, FeedbackResult, FeedbackValue, ModerationIncident, TopQuestion, UserActivity, utc_now)
from app.chat.repository import normalize_question
from app.observability.logging import get_logger

log = get_logger()

SOFT_DELETE = "soft_delete"
CHAT_FILE = "chat.json"
MESSAGES_FILE = "messages.jsonl"
OWNERS_DIR = "owners"
FEEDBACK_FILE = "feedback.jsonl"
MODERATION_FILE = "moderation.jsonl"
BROADCASTS_FILE = "broadcasts.json"

# Замки get_or_create_chat — на процесс, а не на экземпляр: репозиторий создаётся заново
# на каждый запрос (app/chat/deps.py). Ключ — путь к файлу owners/<ключ>.json.
_owner_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
_file_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()


def file_lock(path: Path) -> asyncio.Lock:
    """Замок файла на процесс (блок 4.4): проверка «уже есть?» и запись — без гонки."""
    lock = _file_locks.get(str(path))
    if lock is None:
        lock = _file_locks[str(path)] = asyncio.Lock()
    return lock


def owner_key(owner_external_id: str, interface: str) -> str:
    return hashlib.sha256(f"{interface}\x00{owner_external_id}".encode()).hexdigest()


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

    def _owner_file(self, owner_external_id: str, interface: str) -> Path:
        return self.base_dir / OWNERS_DIR / f"{owner_key(owner_external_id, interface)}.json"

    async def get_or_create_chat(self, owner_external_id: str, interface: str,
                                 system_prompt: str | None = None) -> tuple[Chat, bool]:
        path = self._owner_file(owner_external_id, interface)
        lock = _owner_locks.get(str(path))
        if lock is None:
            lock = _owner_locks[str(path)] = asyncio.Lock()
        async with lock:
            chat = await self._chat_by_owner_file(path)
            if chat is not None:
                return chat, False
            chat = await self._scan_for_owner(owner_external_id, interface)
            created = chat is None
            if chat is None:
                chat = await self.create_chat(owner_external_id, interface, system_prompt)
            await self._write_owner_file(path, chat.id)
            return chat, created

    async def _chat_by_owner_file(self, path: Path) -> Chat | None:
        try:
            if not await aiofiles.os.path.isfile(path):
                return None
            async with aiofiles.open(path, encoding="utf-8") as fh:
                chat_id = UUID(json.loads(await fh.read())["chat_id"])
        except OSError as exc:
            raise ChatStorageError() from exc
        except (ValueError, KeyError, TypeError):
            log.warning("chat_owner_file_skipped", path=path.name)
            return None
        return await self.get_chat(chat_id)            # чат удалили с диска — None, создастся новый

    async def _scan_for_owner(self, owner_external_id: str, interface: str) -> Chat | None:
        """Самый ранний чат клиента среди всех chat.json — для чатов без файла owners/."""
        found: list[Chat] = []
        chats_dir = self.base_dir / "chats"
        try:
            if not await aiofiles.os.path.isdir(chats_dir):
                return None
            names = await aiofiles.os.listdir(chats_dir)
        except OSError as exc:
            raise ChatStorageError() from exc
        for name in names:
            try:
                chat = await self.get_chat(UUID(name))
            except (ValueError, ChatStorageError):
                continue                               # чужая папка или битый chat.json
            if chat is not None and chat.owner_external_id == owner_external_id and chat.interface == interface:
                found.append(chat)
        return min(found, key=lambda c: (c.created_at, str(c.id))) if found else None

    async def _write_owner_file(self, path: Path, chat_id: UUID) -> None:
        tmp = path.with_name(path.name + ".tmp")
        try:
            await aiofiles.os.makedirs(path.parent, exist_ok=True)
            async with aiofiles.open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                await fh.write(json.dumps({"chat_id": str(chat_id)}))
            await aiofiles.os.replace(tmp, path)
        except OSError as exc:
            raise ChatStorageError(f"Не удалось сохранить чат клиента в {self.base_dir}: {exc.strerror or exc}") from exc

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

    # ------------------------------------------------------------------ блок 4.4
    async def _read_lines(self, path: Path) -> list[str]:
        try:
            if not await aiofiles.os.path.isfile(path):
                return []
            async with aiofiles.open(path, encoding="utf-8") as fh:
                return [line.strip() for line in await fh.readlines() if line.strip()]
        except OSError as exc:
            raise ChatStorageError() from exc

    async def _append(self, path: Path, line: str) -> None:
        try:
            await aiofiles.os.makedirs(path.parent, exist_ok=True)
            await append_line(path, line)
        except OSError as exc:
            raise ChatStorageError() from exc

    async def _all_messages(self, chat_id: UUID) -> list[ChatMessage]:
        """Все сообщения чата в порядке записи — и скрытые /clear."""
        messages: list[ChatMessage] = []
        for line in await self._read_lines(self._chat_dir(chat_id) / MESSAGES_FILE):
            if is_soft_delete(line):
                continue
            try:
                messages.append(ChatMessage.model_validate_json(line))
            except ValueError:
                continue
        return messages

    async def _all_chats(self) -> list[Chat]:
        chats_dir = self.base_dir / "chats"
        try:
            names = await aiofiles.os.listdir(chats_dir) if await aiofiles.os.path.isdir(chats_dir) else []
        except OSError as exc:
            raise ChatStorageError() from exc
        chats: list[Chat] = []
        for name in sorted(names):
            try:
                chat = await self.get_chat(UUID(name))
            except (ValueError, ChatStorageError):
                continue
            if chat is not None:
                chats.append(chat)
        return chats

    async def get_message(self, chat_id: UUID, message_id: UUID) -> ChatMessage | None:
        return next((m for m in await self._all_messages(chat_id) if m.id == message_id), None)

    async def add_feedback(self, chat_id: UUID, message_id: UUID, owner_external_id: str,
                           value: FeedbackValue) -> FeedbackResult:
        path = self.base_dir / FEEDBACK_FILE
        async with file_lock(path):
            for line in await self._read_lines(path):
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("owner_external_id") == owner_external_id and row.get("message_id") == str(message_id):
                    return FeedbackResult(saved=False, value=row["value"])
            await self._append(path, json.dumps({
                "message_id": str(message_id), "chat_id": str(chat_id), "owner_external_id": owner_external_id,
                "value": value, "created_at": utc_now().isoformat()}, ensure_ascii=False))
        return FeedbackResult(saved=True, value=value)

    async def record_moderation(self, incident: ModerationIncident) -> None:
        await self._append(self.base_dir / MODERATION_FILE, incident.model_dump_json())

    async def _read_broadcasts(self) -> list[Broadcast]:
        path = self.base_dir / BROADCASTS_FILE
        try:
            if not await aiofiles.os.path.isfile(path):
                return []
            async with aiofiles.open(path, encoding="utf-8") as fh:
                raw = await fh.read()
            return [Broadcast.model_validate(item) for item in json.loads(raw or "[]")]
        except OSError as exc:
            raise ChatStorageError() from exc
        except ValueError as exc:
            raise ChatStorageError(f"Повреждён файл {BROADCASTS_FILE}.") from exc

    async def _write_broadcasts(self, items: list[Broadcast]) -> None:
        path = self.base_dir / BROADCASTS_FILE
        tmp = path.with_name(path.name + ".tmp")
        try:
            await aiofiles.os.makedirs(path.parent, exist_ok=True)
            async with aiofiles.open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                await fh.write(json.dumps([b.model_dump(mode="json") for b in items], ensure_ascii=False, indent=1))
            await aiofiles.os.replace(tmp, path)
        except OSError as exc:
            raise ChatStorageError() from exc

    async def enqueue_broadcast(self, message: str, interface: str) -> Broadcast:
        async with file_lock(self.base_dir / BROADCASTS_FILE):
            items = await self._read_broadcasts()
            item = Broadcast(id=max((b.id for b in items), default=0) + 1, message=message, interface=interface)
            await self._write_broadcasts([*items, item])
        return item

    async def claim_broadcast(self, stale_before: datetime, interface: str = "telegram") -> Broadcast | None:
        async with file_lock(self.base_dir / BROADCASTS_FILE):
            items = await self._read_broadcasts()
            ready = [b for b in items if b.interface == interface and (b.status == "pending" or (
                b.status == "sending" and b.claimed_at is not None and b.claimed_at < stale_before))]
            if not ready:
                return None
            chosen = min(ready, key=lambda b: (b.created_at, b.id))
            claimed = chosen.model_copy(update={"status": "sending", "claimed_at": utc_now()})
            await self._write_broadcasts([claimed if b.id == chosen.id else b for b in items])
        return claimed

    async def finish_broadcast(self, broadcast_id: int, sent: int, failed: int) -> Broadcast | None:
        async with file_lock(self.base_dir / BROADCASTS_FILE):
            items = await self._read_broadcasts()
            current = next((b for b in items if b.id == broadcast_id), None)
            if current is None:
                return None
            if current.status != "sending":
                raise BroadcastNotClaimedError(broadcast_id, current.status)
            done = current.model_copy(update={
                "status": "sent" if sent or not failed else "failed", "sent": sent, "failed": failed,
                "recipients": sent + failed, "finished_at": utc_now()})
            await self._write_broadcasts([done if b.id == broadcast_id else b for b in items])
        return done

    async def broadcast_recipients(self, interface: str) -> list[str]:
        first_seen: dict[str, datetime] = {}
        for chat in await self._all_chats():
            if chat.interface == interface:
                seen = first_seen.get(chat.owner_external_id)
                first_seen[chat.owner_external_id] = chat.created_at if seen is None else min(seen, chat.created_at)
        return sorted(first_seen, key=lambda owner: (first_seen[owner], owner))

    async def stats(self, since: datetime, top_n: int = 5) -> ChatStats:
        stats = ChatStats(since=since)
        active: set[tuple[str, str]] = set()
        latencies: list[float] = []
        questions: Counter[str] = Counter()
        for chat in await self._all_chats():
            previous: ChatMessage | None = None
            for message in await self._all_messages(chat.id):
                if message.created_at >= since:
                    stats.total_messages += 1
                    if message.role == "user":
                        stats.user_messages += 1
                        active.add((chat.interface, chat.owner_external_id))
                        question = normalize_question(message.content)
                        if question:
                            questions[question] += 1
                    elif message.role == "assistant" and previous is not None and previous.role == "user":
                        latencies.append((message.created_at - previous.created_at).total_seconds() * 1000)
                previous = message
        stats.active_users = len(active)
        stats.avg_latency_ms = round(sum(latencies) / len(latencies), 1) if latencies else None
        for line in await self._read_lines(self.base_dir / MODERATION_FILE):
            try:
                incident = ModerationIncident.model_validate_json(line)
            except ValueError:
                continue
            if incident.created_at >= since:
                stats.moderation_blocks += 1
                stats.moderation_input_blocks += incident.direction == "input"
        for line in await self._read_lines(self.base_dir / FEEDBACK_FILE):
            try:
                row = json.loads(line)
                created = datetime.fromisoformat(row["created_at"])
            except (ValueError, KeyError, TypeError):
                continue
            if created >= since:
                stats.feedback_up += row.get("value") == "up"
                stats.feedback_down += row.get("value") == "down"
        ranked = sorted(questions.items(), key=lambda item: (-item[1], item[0]))[:top_n]
        stats.top_questions = [TopQuestion(question=q, count=n) for q, n in ranked]
        return stats

    async def recent_users(self, limit: int = 50) -> list[UserActivity]:
        groups: dict[tuple[str, str], tuple[int, datetime]] = {}
        for chat in await self._all_chats():
            messages = await self._all_messages(chat.id)
            last = max([chat.created_at, *(m.created_at for m in messages)])
            key = (chat.owner_external_id, chat.interface)
            chats, seen = groups.get(key, (0, last))
            groups[key] = (chats + 1, max(seen, last))
        ordered = sorted(groups.items(), key=lambda item: (-item[1][1].timestamp(), item[0][0]))[:limit]
        return [UserActivity(owner_external_id=owner, interface=interface, chats=chats, last_seen_at=seen)
                for (owner, interface), (chats, seen) in ordered]
