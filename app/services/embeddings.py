"""
Эмбеддинги текстов (блок 5.1): embed_texts(), embed_query(), embed_documents().

Публичный интерфейс
- embed_texts(texts) -> list[list[float]] — векторы текстов в том же порядке, как есть, без
  префиксов. Длина каждого вектора — 1 (L2-нормализация): косинусная близость двух текстов —
  просто скалярное произведение, так их сравнивают и поиск, и векторные БД.
- embed_query(text) / embed_documents(texts) — для асимметричных моделей. E5 обучена с
  «query: » перед вопросом и «passage: » перед фрагментом базы знаний: без них похожесть
  вопроса и нужного фрагмента ниже (scripts/embeddings_benchmark.py это замеряет). Префиксы
  подбираются по имени модели (resolve_prefixes) или задаются в .env. У bge-m3 и
  text-embedding-3-* префиксов нет: embed_query(t) == embed_texts([t])[0].

Модель по умолчанию — bge-m3 в Ollama: EMBEDDINGS__MODEL, адрес EMBEDDINGS__BASE_URL (пусто —
тот же, что LLM__BASE_URL). Почему она — docs/embeddings.md.

Провайдеры (EMBEDDINGS__PROVIDER)
- openai — POST {base_url}/embeddings: Ollama (/v1), OpenAI, OpenRouter. Синхронный клиент
  OpenAI SDK с max_retries=0: повторы делает этот модуль (tenacity) — их видно в логе.
- sentence-transformers — модель Hugging Face в этом процессе, на CPU. torch — сотни МБ,
  поэтому пакеты отдельно: pip install -r requirements-embeddings.txt.

Батчи. Тексты уходят пачками по EMBEDDINGS__BATCH_SIZE (пусто — 32 для локальной модели на
CPU, 128 для облака). Для облака ещё и бюджет токенов на запрос: OpenAI принимает не больше
300 000 токенов во всех input одного запроса. Токены считает tiktoken словарём cl100k_base
(токенизатор text-embedding-3); без словаря — оценка байты UTF-8 / 4, она завышена, и пачки
выходят только меньше.

Сеть. Обрыв соединения, таймаут, 408/409, 429 и 5xx — до EMBEDDINGS__MAX_ATTEMPTS попыток с
экспоненциальной задержкой и случайным разбросом (1, 2, 4… с, не больше 20); на 429 с
заголовком Retry-After — столько, сколько сказал провайдер (до 60 с). То, что повтор не
исправит (401 ключ, 404 модель не скачана, 400), — сразу EmbeddingError с подсказкой.

Кеш. Два уровня: словарь в памяти процесса — повтор в том же процессе; файл SQLite
EMBEDDINGS__CACHE_PATH (var/embeddings_cache.sqlite) — повтор после перезапуска. Ключ —
sha256 от провайдера, места модели (локальный сервер — local:порт, облако — имя хоста),
имени модели, dimensions и текста вместе с префиксом. Сменили модель в .env — ключи другие,
векторы прежней модели не используются; код менять не нужно. В кеше вектор — float32 (4 байта
на число, как в векторных БД); ответ провайдера округляется до float32 сразу, поэтому первый
и повторный вызов возвращают одно и то же. Файл не открылся — кеш только в памяти, в лог —
embeddings_cache_unavailable.

Модуль синхронный, как интерфейс в задании. Из async-кода (FastAPI) — через
asyncio.to_thread(embed_texts, texts), чтобы не останавливать цикл событий.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import math
import re
import sqlite3
import sys
import threading
import time
from array import array
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlsplit

from tenacity import RetryCallState, Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from app.core.config import http_client_options, is_local_url, provider_headers, proxy_for
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from app.core.config import Settings

log = get_logger()

CACHE_VERSION = 1             # меняется, если меняется формат ключа или записи
LOCAL_BATCH = 32              # Ollama и sentence-transformers на CPU
CLOUD_BATCH = 128             # OpenAI, OpenRouter: меньше запросов — меньше накладных расходов
MAX_REQUEST_TOKENS = 250_000  # предел OpenAI — 300 000 на запрос; запас на неточность подсчёта
MEMORY_ITEMS = 20_000         # векторов в памяти: 1024 float32 — 4 КБ, всего до ~80 МБ
SQLITE_CHUNK = 500            # ключей в одном SELECT … IN (…)
RETRY_AFTER_MAX = 60.0        # с: дольше Retry-After не ждём
BACKOFF_MAX = 20.0            # с: потолок экспоненциальной задержки
DEFAULT_OPENAI_URL = "https://api.openai.com/v1"
CL100K_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"


class EmbeddingError(RuntimeError):
    """Векторы не получены. Текст — что случилось и что сделать."""


# ---------------------------------------------------------------- префиксы
@dataclass(frozen=True)
class Prefixes:
    query: str = ""
    document: str = ""
    family: str = "без префиксов"


E5_INSTRUCT_QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
# Порядок важен: e5-…-instruct раньше общего правила E5. Префиксы — из карточек моделей.
PREFIX_RULES: tuple[tuple[re.Pattern[str], Prefixes], ...] = (
    (re.compile(r"e5-(large-)?instruct|e5-mistral", re.I), Prefixes(E5_INSTRUCT_QUERY, "", "e5-instruct")),
    (re.compile(r"(^|[/_:-])(multilingual-)?e5([-_.:]|$)", re.I), Prefixes("query: ", "passage: ", "e5")),
    (re.compile(r"frida|rosberta", re.I), Prefixes("search_query: ", "search_document: ", "frida")),
    (re.compile(r"nomic-embed", re.I), Prefixes("search_query: ", "search_document: ", "nomic")),
    (re.compile(r"embeddinggemma", re.I),
     Prefixes("task: search result | query: ", "title: none | text: ", "embeddinggemma")),
)


def resolve_prefixes(model: str, query_prefix: str | None = None, document_prefix: str | None = None) -> Prefixes:
    """Префиксы для модели: по семейству из имени или из .env. None — по семейству,
    «none» — без префикса (пустая строка в .env означает «не задано»)."""
    auto = next((prefixes for pattern, prefixes in PREFIX_RULES if pattern.search(model)), Prefixes())
    if query_prefix is None and document_prefix is None:
        return auto

    def pick(value: str | None, default: str) -> str:
        if value is None:
            return default
        return "" if value.strip().lower() == "none" else value

    return Prefixes(pick(query_prefix, auto.query), pick(document_prefix, auto.document), "из .env")


# ---------------------------------------------------------------- векторы
def normalize(vector: Sequence[float]) -> list[float]:
    """L2-нормализация и округление до float32 (как в кеше). Нулевой вектор — как есть."""
    norm = math.sqrt(math.fsum(x * x for x in vector))
    scaled = [x / norm for x in vector] if norm > 0 else list(vector)
    return array("f", scaled).tolist()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Косинусная близость. Для нормализованных векторов — скалярное произведение."""
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(math.fsum(x * x for x in a)) * math.sqrt(math.fsum(y * y for y in b))
    return dot / norm if norm else 0.0


@lru_cache(maxsize=1)
def _cl100k() -> Any | None:
    """Словарь cl100k_base (text-embedding-3, ada-002). Скачивается один раз в кеш tiktoken;
    сеть проверяет HTTPS своим сертификатом — повтор с хранилищем ОС, как в app/chat/context.py."""
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as first:  # noqa: BLE001 — нет сети, сертификат, битый кеш
        try:
            from app.chat.context import download_with_system_certs

            download_with_system_certs(CL100K_URL)
            return tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # noqa: BLE001
            log.warning("tiktoken_unavailable", encoding="cl100k_base", error=repr(first)[:200],
                        retry_error=repr(exc)[:200], note="токены считаются по длине текста: байты UTF-8 / 4")
            return None


def count_tokens(text: str) -> int:
    """Токены текста для text-embedding-3 (cl100k_base); без словаря — байты UTF-8 / 4."""
    encoding = _cl100k()
    if encoding is not None:
        return len(encoding.encode(text, disallowed_special=()))
    return -(-len(text.encode("utf-8")) // 4)


# ---------------------------------------------------------------- провайдеры
class Backend(Protocol):
    provider: str      # openai | sentence-transformers
    namespace: str     # где модель: local:11434, api.openai.com, openrouter.ai, hf
    model: str
    dimensions: int | None
    local: bool        # на этом компьютере: батч меньше, бюджет токенов не нужен

    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def close(self) -> None: ...


class OpenAICompatibleBackend:
    """POST /embeddings: OpenAI, OpenRouter, Ollama (base_url …/v1)."""

    provider = "openai"

    def __init__(self, *, model: str, base_url: str | None, api_key: str, timeout: float = 60.0,
                 dimensions: int | None = None, proxy: str | None = None, use_system_certs: bool = False,
                 client: Any | None = None) -> None:
        self.model, self.base_url, self.dimensions = model, base_url, dimensions
        self.local = is_local_url(base_url)
        parts = urlsplit(base_url or DEFAULT_OPENAI_URL)
        # localhost, 127.0.0.1 и host.docker.internal — одна и та же Ollama: общий кеш.
        self.namespace = f"local:{parts.port or ''}" if self.local else (parts.hostname or "api.openai.com")
        if client is None:
            from openai import DefaultHttpxClient, OpenAI

            options = http_client_options(proxy, use_system_certs and not self.local)
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0,
                            http_client=DefaultHttpxClient(**options) if options else None,
                            default_headers=provider_headers(base_url, "multapi"))
        self._client = client

    def embed(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {"model": self.model, "input": texts, "encoding_format": "float"}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        response = self._client.embeddings.create(**kwargs)
        data = sorted(response.data, key=lambda item: item.index)   # порядок — по index, не по позиции
        return [list(item.embedding) for item in data]

    def close(self) -> None:
        self._client.close()


class SentenceTransformerBackend:
    """Модель Hugging Face в процессе (sentence-transformers 3+). Загружается при первом
    вызове: импорт torch и чтение весов — секунды, а CLI с кешем модель может не понадобиться."""

    provider = "sentence-transformers"
    namespace = "hf"
    local = True
    dimensions = None

    def __init__(self, model: str, *, device: str = "cpu", batch_size: int = LOCAL_BATCH,
                 use_system_certs: bool = False, loader: Callable[..., Any] | None = None) -> None:
        self.model, self.device, self.batch_size = model, device, batch_size
        self.use_system_certs = use_system_certs
        self._loader = loader
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                loader = self._loader or _sentence_transformer_class()
                if self.use_system_certs:       # скачивание весов в сети с подменой HTTPS
                    import truststore

                    truststore.inject_into_ssl()
                started = time.perf_counter()
                self._model = loader(self.model, device=self.device)
                log.info("embeddings_model_loaded", model=self.model, device=self.device,
                         ms=round((time.perf_counter() - started) * 1000))
            return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)
        return [[float(x) for x in vector] for vector in vectors]

    def close(self) -> None:
        self._model = None


def _sentence_transformer_class() -> Callable[..., Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError("EMBEDDINGS__PROVIDER=sentence-transformers: нет пакета sentence-transformers. "
                             "Установите: pip install -r requirements-embeddings.txt") from exc
    return SentenceTransformer


# ---------------------------------------------------------------- кеш
def cache_key(provider: str, namespace: str, model: str, dimensions: int | None, text: str) -> str:
    payload = json.dumps([CACHE_VERSION, provider, namespace, model, dimensions, text], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _to_blob(vector: Sequence[float]) -> bytes:
    data = array("f", vector)
    if sys.byteorder == "big":          # в файле — little-endian: один файл на любой машине
        data.byteswap()
    return data.tobytes()


def _from_blob(blob: bytes) -> array:
    data = array("f")
    data.frombytes(blob)
    if sys.byteorder == "big":
        data.byteswap()
    return data


class EmbeddingCache:
    """Память процесса (LRU на MEMORY_ITEMS векторов) + SQLite. path=None — только память."""

    def __init__(self, path: Path | None, *, memory_items: int = MEMORY_ITEMS) -> None:
        self.path = path
        self.memory_items = memory_items
        self._memory: OrderedDict[str, array] = OrderedDict()
        self._lock = threading.Lock()
        self._db = self._open(path) if path is not None else None

    @staticmethod
    def _open(path: Path) -> sqlite3.Connection | None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(path, timeout=30, check_same_thread=False)
            db.execute("PRAGMA journal_mode=WAL")      # CLI и сервис могут писать одновременно
            db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, model TEXT NOT NULL, "
                       "dim INTEGER NOT NULL, vector BLOB NOT NULL, created_at REAL NOT NULL)")
            db.commit()
            return db
        except (sqlite3.Error, OSError) as exc:
            log.warning("embeddings_cache_unavailable", path=str(path), error=repr(exc)[:200],
                        note="кеш только в памяти процесса")
            return None

    @property
    def persistent(self) -> bool:
        return self._db is not None

    def _remember(self, key: str, vector: array) -> None:
        self._memory[key] = vector
        self._memory.move_to_end(key)
        while len(self._memory) > self.memory_items:
            self._memory.popitem(last=False)

    def get_many(self, keys: Iterable[str]) -> dict[str, tuple[list[float], str]]:
        """{ключ: (вектор, "memory" | "disk")} для найденных ключей."""
        found: dict[str, tuple[list[float], str]] = {}
        with self._lock:
            missing = []
            for key in keys:
                vector = self._memory.get(key)
                if vector is None:
                    missing.append(key)
                else:
                    self._memory.move_to_end(key)
                    found[key] = (vector.tolist(), "memory")
            if self._db is None or not missing:
                return found
            for start in range(0, len(missing), SQLITE_CHUNK):
                chunk = missing[start:start + SQLITE_CHUNK]
                try:
                    rows = self._db.execute(f"SELECT key, dim, vector FROM embeddings WHERE key IN "
                                            f"({','.join('?' * len(chunk))})", chunk).fetchall()
                except sqlite3.Error as exc:
                    log.warning("embeddings_cache_read_failed", error=repr(exc)[:200])
                    return found
                for key, dim, blob in rows:
                    vector = _from_blob(blob)
                    if len(vector) != dim:          # запись повреждена — посчитаем заново
                        continue
                    self._remember(key, vector)
                    found[key] = (vector.tolist(), "disk")
        return found

    def put_many(self, items: Sequence[tuple[str, Sequence[float]]], model: str) -> None:
        with self._lock:
            for key, vector in items:
                self._remember(key, array("f", vector))
            if self._db is None:
                return
            now = time.time()
            try:
                with self._db:
                    self._db.executemany(
                        "INSERT OR REPLACE INTO embeddings (key, model, dim, vector, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        [(key, model, len(vector), _to_blob(vector), now) for key, vector in items])
            except sqlite3.Error as exc:            # диск полон, файл заблокирован — работаем дальше
                log.warning("embeddings_cache_write_failed", error=repr(exc)[:200])

    def clear(self) -> int:
        """Очистить кеш. Возвращает, сколько записей было в файле."""
        with self._lock:
            self._memory.clear()
            if self._db is None:
                return 0
            with self._db:
                removed = self._db.execute("DELETE FROM embeddings").rowcount
            return removed

    def summary(self) -> list[tuple[str, int, int]]:
        """[(модель, размерность, записей)] в файле — видно, что у разных моделей свои записи."""
        if self._db is None:
            return []
        with self._lock:
            return [tuple(row) for row in self._db.execute(
                "SELECT model, dim, COUNT(*) FROM embeddings GROUP BY model, dim ORDER BY model")]

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None


# ---------------------------------------------------------------- сервис
@dataclass
class EmbeddingStats:
    texts: int = 0             # текстов во всех вызовах (с повторами)
    memory_hits: int = 0       # уникальных текстов из памяти процесса
    disk_hits: int = 0         # … из файла кеша
    requests: int = 0          # обращений к модели (HTTP-запросов или encode), включая повторы
    embedded: int = 0          # текстов, которые посчитала модель
    retries: int = 0

    @property
    def cache_hits(self) -> int:
        return self.memory_hits + self.disk_hits


def is_transient(exc: BaseException) -> bool:
    """Ошибка, которую может исправить повтор: сеть, таймаут, лимит, сбой сервера."""
    import openai

    if isinstance(exc, (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)):
        return True            # APITimeoutError — подкласс APIConnectionError
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in (408, 409) or exc.status_code >= 500
    return isinstance(exc, (ConnectionError, TimeoutError))


def retry_after(exc: BaseException | None) -> float | None:
    """Пауза из заголовков ответа: retry-after-ms или retry-after в секундах."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            seconds = float(value) * scale
        except ValueError:          # HTTP-дата вместо секунд — обычная задержка
            continue
        if seconds >= 0:
            return seconds
    return None


def _checked(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, str):
        raise TypeError("Нужен список строк, а передана строка: embed_texts([text]), а не embed_texts(text)")
    texts = list(texts)
    for number, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"texts[{number}]: нужна строка, а не {type(text).__name__}")
        if not text.strip():
            raise ValueError(f"texts[{number}]: пустой текст — у него нет смысла, который можно закодировать")
    return texts


class EmbeddingService:
    def __init__(self, backend: Backend, *, prefixes: Prefixes | None = None, batch_size: int | None = None,
                 cache: EmbeddingCache | None = None, max_attempts: int = 5,
                 max_request_tokens: int | None = MAX_REQUEST_TOKENS,
                 token_counter: Callable[[str], int] = count_tokens,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.backend = backend
        self.prefixes = prefixes or resolve_prefixes(backend.model)
        self.batch_size = batch_size or (LOCAL_BATCH if backend.local else CLOUD_BATCH)
        # Локальной модели бюджет не нужен: Ollama сама обрезает текст длиннее контекста модели.
        self.max_request_tokens = None if backend.local else max_request_tokens
        self.cache = cache
        self.max_attempts = max_attempts
        self.stats = EmbeddingStats()
        self.dimension: int | None = None
        self._count = token_counter
        self._sleep = sleep
        self._backoff = wait_exponential_jitter(initial=1, max=BACKOFF_MAX, jitter=1)
        self._lock = threading.Lock()

    @property
    def label(self) -> str:
        """Модель для лога и записи в кеше: openai:local:11434/bge-m3. Без «@»: подпись вида
        openai@openrouter.ai маскирование PII в логе принимает за email (проверка на Windows)."""
        dims = f":{self.backend.dimensions}d" if self.backend.dimensions else ""
        return f"{self.backend.provider}:{self.backend.namespace}/{self.backend.model}{dims}"

    # -------------------------------------------------------- публичные методы
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Векторы текстов как есть, без префиксов."""
        return self._embed(_checked(texts), "")

    def embed_query(self, text: str) -> list[float]:
        """Вектор вопроса пользователя: с префиксом запроса модели («query: » у E5)."""
        return self._embed(_checked([text]), self.prefixes.query)[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Векторы фрагментов базы знаний: с префиксом документа («passage: » у E5)."""
        return self._embed(_checked(texts), self.prefixes.document)

    def close(self) -> None:
        self.backend.close()
        if self.cache is not None:
            self.cache.close()

    # -------------------------------------------------------- внутреннее
    def _key(self, text: str) -> str:
        return cache_key(self.backend.provider, self.backend.namespace, self.backend.model,
                         self.backend.dimensions, text)

    def _embed(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        full = [prefix + text for text in texts]
        unique = list(dict.fromkeys(full))                 # одинаковые тексты — один раз
        keys = {text: self._key(text) for text in unique}
        cached = self.cache.get_many(keys.values()) if self.cache is not None else {}
        vectors: dict[str, list[float]] = {}
        missing: list[str] = []
        memory_hits = disk_hits = 0
        for text in unique:
            hit = cached.get(keys[text])
            if hit is None:
                missing.append(text)
                continue
            vectors[text] = hit[0]
            memory_hits += hit[1] == "memory"
            disk_hits += hit[1] == "disk"
        with self._lock:
            self.stats.texts += len(texts)
            self.stats.memory_hits += memory_hits
            self.stats.disk_hits += disk_hits
        for batch in self._batches(missing):
            computed = self._request(batch)
            vectors.update(zip(batch, computed, strict=True))
            if self.cache is not None:
                self.cache.put_many([(keys[text], vector) for text, vector in zip(batch, computed, strict=True)],
                                    self.label)
        return [list(vectors[text]) for text in full]      # копии: кеш не испортить снаружи

    def _batches(self, texts: list[str]) -> Iterator[list[str]]:
        batch: list[str] = []
        tokens = 0
        for text in texts:
            size = self._count(text) if self.max_request_tokens else 0
            over_budget = self.max_request_tokens is not None and tokens + size > self.max_request_tokens
            if batch and (len(batch) >= self.batch_size or over_budget):
                yield batch
                batch, tokens = [], 0
            batch.append(text)
            tokens += size
        if batch:
            yield batch

    def _wait(self, state: RetryCallState) -> float:
        exc = state.outcome.exception() if state.outcome is not None else None
        delay = self._backoff(state)
        pause = retry_after(exc)
        return min(max(delay, pause), RETRY_AFTER_MAX) if pause is not None else delay

    def _before_sleep(self, state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome is not None else None
        with self._lock:
            self.stats.retries += 1
        log.warning("embeddings_retry", model=self.label, attempt=state.attempt_number,
                    wait_s=round(state.next_action.sleep if state.next_action else 0.0, 2),
                    error=type(exc).__name__, detail=str(exc)[:200])

    def _send(self, batch: list[str]) -> list[list[float]]:
        with self._lock:
            self.stats.requests += 1
        return self.backend.embed(batch)

    def _request(self, batch: list[str]) -> list[list[float]]:
        retrying = Retrying(stop=stop_after_attempt(self.max_attempts), wait=self._wait,
                            retry=retry_if_exception(is_transient), sleep=self._sleep,
                            before_sleep=self._before_sleep, reraise=True)
        started = time.perf_counter()
        try:
            raw = retrying(self._send, batch)
        except EmbeddingError:
            raise
        except Exception as exc:  # noqa: BLE001 — ошибки SDK и сети превращаются в понятный текст
            raise self._explain(exc) from exc
        if len(raw) != len(batch):
            raise EmbeddingError(f"{self.label}: на {len(batch)} текстов пришло {len(raw)} векторов")
        vectors = [normalize(vector) for vector in raw]
        sizes = {len(vector) for vector in vectors}
        if len(sizes) != 1 or (self.dimension is not None and sizes != {self.dimension}) or 0 in sizes:
            raise EmbeddingError(f"{self.label}: векторы разной длины {sorted(sizes)}, ожидалось {self.dimension}")
        self.dimension = sizes.pop()
        with self._lock:
            self.stats.embedded += len(batch)
        log.info("embeddings_request", model=self.label, texts=len(batch), dim=self.dimension,
                 ms=round((time.perf_counter() - started) * 1000))
        return vectors

    def _explain(self, exc: Exception) -> EmbeddingError:
        import openai

        where = getattr(self.backend, "base_url", None) or DEFAULT_OPENAI_URL
        model = self.backend.model
        if isinstance(exc, openai.APIConnectionError):
            hint = ("запущена ли Ollama (ollama serve) и верен ли EMBEDDINGS__BASE_URL" if self.backend.local
                    else "доступен ли провайдер; нужен ли прокси (LLM__PROXY_URL) и LLM__USE_SYSTEM_CERTS=true")
            return EmbeddingError(f"Нет связи с {where} после {self.max_attempts} попыток "
                                  f"({type(exc).__name__}). Проверьте, {hint}.")
        if isinstance(exc, openai.AuthenticationError):
            return EmbeddingError(f"{where} не принял ключ (401): EMBEDDINGS__API_KEY, а если пусто — "
                                  "LLM__OPENAI_API_KEY.")
        if isinstance(exc, openai.NotFoundError):
            hint = f"скачайте её: ollama pull {model}" if self.backend.local else "проверьте EMBEDDINGS__MODEL"
            return EmbeddingError(f"Модель {model} не найдена на {where} (404): {hint}.")
        if isinstance(exc, openai.APIStatusError) and exc.status_code == 402:
            return EmbeddingError(f"{where}: у ключа нет кредитов (402) — модели эмбеддингов OpenAI на OpenRouter "
                                  "платные. Пополните баланс ключа или возьмите локальную модель (bge-m3 в Ollama, "
                                  "multilingual-e5 через sentence-transformers).")
        if isinstance(exc, openai.RateLimitError):
            return EmbeddingError(f"{where}: лимит запросов (429) и после {self.max_attempts} попыток. "
                                  "Повторите позже или уменьшите EMBEDDINGS__BATCH_SIZE.")
        if isinstance(exc, openai.APIStatusError):
            return EmbeddingError(f"{where} отклонил запрос эмбеддингов модели {model} "
                                  f"({exc.status_code}): {str(exc)[:300]}")
        return EmbeddingError(f"Модель {model} не посчитала эмбеддинги: {type(exc).__name__}: {str(exc)[:300]}")


# ---------------------------------------------------------------- сборка из настроек
def build_backend(cfg: Settings, *, batch_size: int | None = None) -> Backend:
    emb = cfg.embeddings
    if emb.provider == "sentence-transformers":
        return SentenceTransformerBackend(emb.model, device=emb.device, batch_size=batch_size or LOCAL_BATCH,
                                          use_system_certs=cfg.llm.use_system_certs)
    base_url = emb.base_url or cfg.llm.base_url
    api_key = emb.api_key or cfg.llm.openai_api_key
    return OpenAICompatibleBackend(model=emb.model, base_url=base_url, api_key=api_key.get_secret_value(),
                                   timeout=emb.timeout, dimensions=emb.dimensions,
                                   proxy=proxy_for(base_url, cfg.llm.proxy_url),
                                   use_system_certs=cfg.llm.use_system_certs)


def build_service(cfg: Settings | None = None, *, use_cache: bool | None = None) -> EmbeddingService:
    """Сервис из настроек (.env). use_cache=False — без кеша (бенчмарк меряет модель, а не кеш)."""
    if cfg is None:
        from app.core.config import get_settings

        cfg = get_settings()
    emb = cfg.embeddings
    backend = build_backend(cfg, batch_size=emb.batch_size)
    enabled = emb.cache_enabled if use_cache is None else use_cache
    cache = EmbeddingCache(emb.cache_path) if enabled else None
    return EmbeddingService(backend, prefixes=resolve_prefixes(emb.model, emb.query_prefix, emb.document_prefix),
                            batch_size=emb.batch_size, cache=cache, max_attempts=emb.max_attempts)


_default: EmbeddingService | None = None
_default_lock = threading.Lock()
_atexit_registered = False


def default_service() -> EmbeddingService:
    """Один сервис на процесс (и одна память кеша): создаётся при первом вызове."""
    global _default, _atexit_registered
    with _default_lock:
        if _default is None:
            _default = build_service()
            if not _atexit_registered:          # файл кеша закрывается и при выходе из процесса
                atexit.register(reset_default_service)
                _atexit_registered = True
            log.info("embeddings_ready", model=_default.label, batch_size=_default.batch_size,
                     prefixes=_default.prefixes.family,
                     cache=str(_default.cache.path) if _default.cache is not None else None)
        return _default


def reset_default_service() -> None:
    """Закрыть сервис по умолчанию: следующий вызов прочитает настройки заново (тесты)."""
    global _default
    with _default_lock:
        if _default is not None:
            _default.close()
        _default = None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Векторы текстов (L2-нормализованные, в том же порядке) моделью из .env, с кешем."""
    return default_service().embed_texts(texts)


def embed_query(text: str) -> list[float]:
    """Вектор вопроса: с префиксом запроса, если модель его требует (E5 — «query: »)."""
    return default_service().embed_query(text)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Векторы документов: с префиксом документа, если модель его требует (E5 — «passage: »)."""
    return default_service().embed_documents(texts)
