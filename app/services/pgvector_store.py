"""
pgvector — та же база знаний в Postgres, для сравнения с Qdrant (блок 5.2, задача 7).

В сервис не подключён: поиск сервиса — Qdrant (app/services/vector_store.py). Замеры и выбор —
docs/vector_store.md, раздел «pgvector»; загрузка — scripts/load_to_pgvector.py, сравнение —
scripts/compare_pgvector.py.

Таблица в той же базе, где история чатов (DATABASE_URL), — Postgres из compose с расширением
pgvector 0.8 (образ pgvector/pgvector в compose.yaml):

    CREATE TABLE kb.documents (id uuid PRIMARY KEY, embedding vector(1024) NOT NULL, payload jsonb NOT NULL)

- Схема kb, а не public: Alembic сравнивает модели чата только со схемой public, и таблица
  вне моделей не попадёт в alembic revision --autogenerate как «лишняя» (drop_table).
- id и payload — те же, что у точек Qdrant (uuid5 фрагмента, payload из documents.py), векторы —
  те же (модуль эмбеддингов блока 5.1): одна база знаний в двух хранилищах.
- Индексы:
  * HNSW по embedding, vector_cosine_ops, m=16, ef_construction=100 — как у коллекции Qdrant;
  * HNSW по embedding::halfvec(N), halfvec_cosine_ops — половинная точность (2 байта на число
    вместо 4), индекс вдвое меньше; запрос сортирует по тому же выражению (search(half=True));
  * GIN по payload (jsonb_path_ops) — для условий payload @> '{"category": "billing"}'.
- <=> — косинусное расстояние (1 − косинус); score = 1 − расстояние — та же шкала, что у Qdrant
  с метрикой COSINE.
- Векторы ходят текстом '[0.1,0.2,…]' (codec asyncpg на типы vector и halfvec) — без пакета pgvector.

Ошибки — PgVectorError с текстом «что случилось и что сделать», как у VectorStore.
"""
from __future__ import annotations

import contextlib
import json
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import asyncpg

if TYPE_CHECKING:
    from app.core.config import Settings

SCHEMA = "kb"
TABLE = "documents"
HNSW_M = 16                   # как HNSW коллекции Qdrant (vector_store.HNSW)
HNSW_EF_CONSTRUCTION = 100    # у pgvector по умолчанию 64, у Qdrant ef_construct — 100
MIN_VERSION = (0, 8)          # 0.8: итеративное сканирование индекса (hnsw.iterative_scan)
CONNECT_TIMEOUT = 10
DEFAULT_BATCH = 500           # строк в одном INSERT … SELECT FROM unnest(…)
# Параметры сеанса, которые можно менять через settings(): план и поиск по HNSW.
SESSION_SETTINGS = {"enable_seqscan", "enable_bitmapscan", "hnsw.ef_search", "hnsw.iterative_scan",
                    "hnsw.max_scan_tuples"}
_NAME = re.compile(r"[a-z_][a-z0-9_]{0,62}")


class PgVectorError(RuntimeError):
    """Postgres отказал или таблица не подходит: текст — что случилось и что сделать."""


class PgVectorUnavailable(PgVectorError):
    """Нет связи с Postgres."""


@dataclass(frozen=True)
class Hit:
    """Результат поиска: как ScoredPoint Qdrant — id, близость и payload."""

    id: str
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class Where:
    """Условие WHERE: sql с местами {} под параметры и сами параметры по порядку.
    Номера $n расставляет render — после параметров самого запроса."""

    sql: str
    params: tuple[Any, ...] = field(default=())

    def render(self, start: int) -> str:
        if self.sql.count("{}") != len(self.params):
            raise ValueError(f"В условии {self.sql!r} мест {{}} не столько, сколько параметров ({len(self.params)})")
        numbers = iter(range(start, start + len(self.params)))
        return re.sub(r"\{\}", lambda _: f"${next(numbers)}", self.sql)


def and_(*parts: Where | None) -> Where | None:
    present = [part for part in parts if part is not None]
    if not present:
        return None
    return Where(" AND ".join(part.sql for part in present), tuple(p for part in present for p in part.params))


def sql_filter(*, category: str | None = None, source: str | None = None, created_after: str | date | None = None,
               include_archived: bool = False) -> Where | None:
    """Тот же фильтр, что build_filter для Qdrant, — на SQL. must — payload @> {...} (GIN-индекс),
    must_not — NOT payload @> {...}: как и must_not в Qdrant, проходят и строки без поля status
    (payload->>'status' <> 'archived' их бы потерял: NULL <> 'archived' — не true)."""
    must: list[Where] = []
    if category:
        must.append(Where("payload @> {}", ({"category": category},)))
    if source:
        must.append(Where("payload @> {}", ({"source": source},)))
    if created_after:
        must.append(Where("(payload->>'created_at')::timestamptz >= {}", (to_datetime(created_after),)))
    if not include_archived:
        must.append(Where("NOT payload @> {}", ({"status": "archived"},)))
    return and_(*must)


def to_datetime(value: str | date | datetime) -> datetime:
    """2026-09-10, 2026-09-10T00:00:00Z или date/datetime -> datetime с часовым поясом (UTC по умолчанию)."""
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    else:
        moment = datetime.fromisoformat(value.strip())       # Python 3.11+ понимает и «Z»
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def vector_literal(values: Sequence[float]) -> str:
    """[0.1,0.2,…] — текстовый вид vector и halfvec. 9 значащих цифр точно передают float32."""
    return "[" + ",".join(format(float(x), ".9g") for x in values) + "]"


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def parse_vector(text: str) -> list[float]:
    return [float(x) for x in text.strip("[]").split(",")] if text.strip("[]") else []


def asyncpg_dsn(database_url: str) -> str:
    """postgresql+asyncpg://… из настроек -> postgresql://… для asyncpg.connect."""
    from sqlalchemy.engine import make_url

    return make_url(database_url).set(drivername="postgresql").render_as_string(hide_password=False)


def describe_dsn(dsn: str) -> str:
    """Адрес для сообщений и вывода — без пароля: 127.0.0.1:5433/multapi."""
    from sqlalchemy.engine import make_url

    url = make_url(dsn)
    return f"{url.host or 'localhost'}:{url.port or 5432}/{url.database or ''}"


def plan_summary(plan: Any) -> str:
    """EXPLAIN (FORMAT JSON) -> «Limit → Index Scan (documents_embedding_hnsw)»: узлы сверху вниз
    по первой ветке; так видно, ищет ли Postgres по HNSW или перебором (Seq Scan + Sort)."""
    if isinstance(plan, str):
        plan = json.loads(plan)
    node = plan[0]["Plan"] if isinstance(plan, list) else plan["Plan"]
    steps = []
    while node:
        name = node["Node Type"]
        if node.get("Index Name"):
            name += f" ({node['Index Name']})"
        steps.append(name)
        children = node.get("Plans") or []
        node = children[0] if children else None
    return " → ".join(steps)


class PgVectorStore:
    def __init__(self, dsn: str, dim: int, *, schema: str = SCHEMA, table: str = TABLE,
                 timeout: float = CONNECT_TIMEOUT) -> None:
        for name in (schema, table):
            if not _NAME.fullmatch(name):
                raise ValueError(f"Имя схемы или таблицы {name!r}: только латиница в нижнем регистре, цифры и _")
        self.dsn = dsn
        self.dim = dim
        self.schema = schema
        self.table = table
        self.timeout = timeout
        self.address = describe_dsn(dsn)
        self.extension_version: str | None = None
        self.server_version: str | None = None
        self._extension_schema = "public"
        self._conn: asyncpg.Connection | None = None

    @classmethod
    def from_settings(cls, cfg: Settings, *, schema: str = SCHEMA) -> PgVectorStore:
        if cfg.embedding_dim is None:
            raise PgVectorError("EMBEDDING_DIM не задан: укажите в .env размерность модели эмбеддингов "
                                "(как для Qdrant)")
        return cls(asyncpg_dsn(cfg.database_url.get_secret_value()), cfg.embedding_dim, schema=schema)

    @property
    def qualified(self) -> str:
        return f"{self.schema}.{self.table}"

    # -------------------------------------------------------- подключение
    async def connection(self) -> asyncpg.Connection:
        """Одно соединение на объект (скрипт сравнения — один процесс, один запрос за раз)."""
        if self._conn is None or self._conn.is_closed():
            try:
                conn = await asyncpg.connect(self.dsn, timeout=self.timeout)
            except (OSError, TimeoutError, asyncpg.CannotConnectNowError) as exc:
                reason = str(exc) or f"{type(exc).__name__}: нет ответа за {self.timeout:g} с"
                raise PgVectorUnavailable(
                    f"Нет связи с Postgres ({self.address}): {reason}. Запущен ли он — "
                    "docker compose up -d postgres; верен ли DATABASE_URL") from exc
            except asyncpg.InvalidPasswordError as exc:
                raise PgVectorError(
                    f"Postgres ({self.address}) не принял пароль: проверьте DATABASE_URL в .env") from exc
            except asyncpg.InvalidCatalogNameError as exc:
                raise PgVectorError(f"В Postgres ({self.address}) нет такой базы: {exc}. Проверьте DATABASE_URL; "
                                    "база сервиса создаётся при первом docker compose up") from exc
            await conn.set_type_codec("jsonb", schema="pg_catalog", encoder=to_json, decoder=json.loads,
                                      format="text")
            self._conn = conn
            self.server_version = str(await conn.fetchval("SHOW server_version")).split()[0]   # 16.10
        return self._conn

    async def _register_vector_codecs(self, conn: asyncpg.Connection) -> None:
        for type_name in ("vector", "halfvec"):
            await conn.set_type_codec(type_name, schema=self._extension_schema, encoder=vector_literal,
                                      decoder=parse_vector, format="text")

    # -------------------------------------------------------- схема
    async def ensure_schema(self) -> None:
        """Расширение vector, схема, таблица и индексы. Повторный вызов ничего не меняет;
        таблица под другую размерность — PgVectorError (как коллекция Qdrant под другую модель)."""
        conn = await self.connection()
        await self._ensure_extension(conn)
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
        await conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self.qualified} ("
            f"id uuid PRIMARY KEY, embedding vector({self.dim}) NOT NULL, payload jsonb NOT NULL)")
        actual = await conn.fetchval(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = $1::regclass AND attname = 'embedding'", self.qualified)
        if actual != f"vector({self.dim})":
            size = re.search(r"\((\d+)\)", actual or "")
            what = f"из {size.group(1)} чисел ({actual})" if size else f"{actual}"
            raise PgVectorError(
                f"Таблица {self.qualified} создана для векторов {what}, а EMBEDDING_DIM={self.dim}: её заполняла "
                "другая модель эмбеддингов. Верните прежнюю модель или пересоздайте таблицу: "
                "python scripts/load_to_pgvector.py --recreate")
        for name, method in self.indexes().items():
            await conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {self.qualified} USING {method}")

    def indexes(self) -> dict[str, str]:
        """Имя индекса -> «метод (выражение) параметры» для CREATE INDEX … USING."""
        hnsw = f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
        return {
            f"{self.table}_embedding_hnsw": f"hnsw (embedding vector_cosine_ops) {hnsw}",
            f"{self.table}_embedding_half_hnsw": f"hnsw ((embedding::halfvec({self.dim})) halfvec_cosine_ops) {hnsw}",
            f"{self.table}_payload_gin": "gin (payload jsonb_path_ops)",
        }

    async def _ensure_extension(self, conn: asyncpg.Connection) -> None:
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except (asyncpg.FeatureNotSupportedError, asyncpg.UndefinedFileError) as exc:
            raise PgVectorError(
                f"В Postgres ({self.address}) нет расширения pgvector ({first_line(exc)}). В compose.yaml "
                "у postgres образ pgvector/pgvector — запустите его: docker compose up -d postgres") from exc
        except asyncpg.InsufficientPrivilegeError as exc:
            raise PgVectorError(
                f"Нет прав на CREATE EXTENSION vector ({first_line(exc)}): выполните его от суперпользователя Postgres "
                "(в compose это пользователь multapi)") from exc
        row = await conn.fetchrow(
            "SELECT e.extversion, n.nspname, a.default_version FROM pg_extension e "
            "JOIN pg_namespace n ON n.oid = e.extnamespace "
            "JOIN pg_available_extensions a ON a.name = e.extname WHERE e.extname = 'vector'")
        self.extension_version, self._extension_schema = row["extversion"], row["nspname"]
        if version_tuple(self.extension_version) < MIN_VERSION:
            hint = ("ALTER EXTENSION vector UPDATE" if version_tuple(row["default_version"]) >= MIN_VERSION
                    else "обновите образ pgvector/pgvector в compose.yaml: docker compose up -d postgres")
            raise PgVectorError(f"pgvector {self.extension_version}, а нужен 0.8 или новее (итеративное "
                                f"сканирование HNSW): {hint}")
        await self._register_vector_codecs(conn)

    async def _ready(self) -> asyncpg.Connection:
        """Соединение с codec-ами vector/halfvec: ensure_schema уже вызван или вызывается сейчас."""
        if self.extension_version is None:
            await self.ensure_schema()
        return await self.connection()

    async def drop(self) -> None:
        """Удалить таблицу (--recreate). Нет таблицы — не ошибка; схема и расширение остаются."""
        conn = await self.connection()
        await conn.execute(f"DROP TABLE IF EXISTS {self.qualified}")

    # -------------------------------------------------------- строки
    async def upsert(self, rows: Sequence[tuple[str, Sequence[float], dict[str, Any]]],
                     batch_size: int = DEFAULT_BATCH) -> tuple[int, int]:
        """(id, вектор, payload): тот же id — обновление (ON CONFLICT), а не копия; строка, где
        ничего не изменилось, не переписывается (повторная загрузка не плодит мёртвые версии строк).
        Пачками по batch_size в одной транзакции. Возвращает (новых, изменённых)."""
        if batch_size < 1:
            raise ValueError("batch_size должен быть больше 0")
        for point_id, vector, _ in rows:
            self._check_dim(vector, f"строка {point_id}")
        conn = await self._ready()
        sql = (f"INSERT INTO {self.qualified} AS d (id, embedding, payload) "
               f"SELECT u.id, u.embedding::vector({self.dim}), u.payload::jsonb "
               "FROM unnest($1::uuid[], $2::text[], $3::text[]) AS u(id, embedding, payload) "
               "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, payload = EXCLUDED.payload "
               "WHERE (d.embedding, d.payload) IS DISTINCT FROM (EXCLUDED.embedding, EXCLUDED.payload) "
               "RETURNING (xmax = 0) AS inserted")
        inserted = updated = 0
        async with conn.transaction():
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                result = await conn.fetch(sql, [point_id for point_id, _, _ in batch],
                                          [vector_literal(vector) for _, vector, _ in batch],
                                          [to_json(payload) for _, _, payload in batch])
                inserted += sum(1 for row in result if row["inserted"])
                updated += sum(1 for row in result if not row["inserted"])
        return inserted, updated

    async def ids(self) -> set[str]:
        conn = await self._ready()
        return {row["id"] for row in await conn.fetch(f"SELECT id::text AS id FROM {self.qualified}")}

    async def delete(self, ids: Sequence[str]) -> None:
        if ids:
            conn = await self._ready()
            await conn.execute(f"DELETE FROM {self.qualified} WHERE id = ANY($1::uuid[])", list(ids))

    async def count(self) -> int:
        conn = await self._ready()
        return int(await conn.fetchval(f"SELECT count(*) FROM {self.qualified}"))

    async def analyze(self) -> None:
        """Статистика для планировщика после загрузки (иначе он оценивает таблицу наугад)."""
        conn = await self._ready()
        await conn.execute(f"ANALYZE {self.qualified}")

    async def sizes(self) -> dict[str, int]:
        """Байты: table — строки с TOAST (векторы по 4 КБ хранятся в TOAST), индексы — по именам."""
        conn = await self._ready()
        result = {"table": int(await conn.fetchval("SELECT pg_table_size($1::regclass)", self.qualified))}
        for row in await conn.fetch(
                "SELECT c.relname, pg_relation_size(i.indexrelid) AS size FROM pg_index i "
                "JOIN pg_class c ON c.oid = i.indexrelid WHERE i.indrelid = $1::regclass ORDER BY c.relname",
                self.qualified):
            result[row["relname"]] = int(row["size"])
        return result

    # -------------------------------------------------------- поиск
    def search_sql(self, top_k: int = 5, where: Where | None = None, *, half: bool = False) -> str:
        """SELECT … ORDER BY embedding <=> $1 LIMIT k. Индекс HNSW подходит, только если ORDER BY —
        оператор расстояния над тем же выражением, что в индексе (для halfvec — embedding::halfvec(N))."""
        if not isinstance(top_k, int) or not 1 <= top_k <= 1000:
            raise ValueError("top_k — целое от 1 до 1000")
        column = f"embedding::halfvec({self.dim})" if half else "embedding"
        query = f"$1::halfvec({self.dim})" if half else f"$1::vector({self.dim})"
        sql = f"SELECT id::text AS id, payload, {column} <=> {query} AS distance FROM {self.qualified}"
        if where is not None:
            sql += f" WHERE {where.render(2)}"
        return sql + f" ORDER BY {column} <=> {query} LIMIT {top_k}"

    async def search(self, query_vector: Sequence[float], top_k: int = 5, where: Where | None = None, *,
                     half: bool = False) -> list[Hit]:
        """top_k ближайших строк с payload; where — условия WHERE (sql_filter или свой Where).
        half=True — поиск по halfvec-выражению (и его индексу)."""
        self._check_dim(query_vector, "вектор запроса")
        conn = await self._ready()
        rows = await conn.fetch(self.search_sql(top_k, where, half=half), list(query_vector),
                                *(where.params if where else ()))
        return [Hit(row["id"], 1.0 - float(row["distance"]), row["payload"]) for row in rows]

    async def explain(self, query_vector: Sequence[float], top_k: int = 5, where: Where | None = None, *,
                      half: bool = False) -> str:
        """План запроса search одной строкой (plan_summary)."""
        conn = await self._ready()
        plan = await conn.fetchval("EXPLAIN (FORMAT JSON) " + self.search_sql(top_k, where, half=half),
                                   list(query_vector), *(where.params if where else ()))
        return plan_summary(plan)

    async def execution_ms(self, query_vector: Sequence[float], top_k: int = 5, where: Where | None = None, *,
                           half: bool = False) -> float:
        """Время выполнения запроса search внутри Postgres, мс (EXPLAIN ANALYZE, Execution Time) —
        без сети и разбора ответа в Python."""
        conn = await self._ready()
        plan = await conn.fetchval("EXPLAIN (ANALYZE, FORMAT JSON) " + self.search_sql(top_k, where, half=half),
                                   list(query_vector), *(where.params if where else ()))
        plan = json.loads(plan) if isinstance(plan, str) else plan
        return float(plan[0]["Execution Time"])

    @contextlib.asynccontextmanager
    async def settings(self, **values: str | int) -> AsyncIterator[None]:
        """Параметры сеанса на время блока: settings(enable_seqscan="off") — заставить планировщик
        взять индекс; settings(**{"hnsw.iterative_scan": "strict_order"}) — итеративный поиск."""
        conn = await self._ready()
        applied = []
        try:
            for name, value in values.items():
                if name not in SESSION_SETTINGS or not re.fullmatch(r"[a-z0-9_]+", str(value)):
                    raise ValueError(f"Параметр {name}={value!r} не из списка {sorted(SESSION_SETTINGS)}")
                await conn.execute(f"SET {name} = {value}")
                applied.append(name)
            yield
        finally:
            for name in applied:
                await conn.execute(f"RESET {name}")

    async def close(self) -> None:
        if self._conn is not None and not self._conn.is_closed():
            await self._conn.close()
        self._conn = None

    def _check_dim(self, vector: Any, what: str) -> None:
        size = len(vector) if isinstance(vector, Sequence) else None
        if size != self.dim:
            raise PgVectorError(
                f"{what}: вектор из {size} чисел, а таблица {self.qualified} — для {self.dim} (EMBEDDING_DIM). "
                "Модель эмбеддингов в .env (EMBEDDINGS__MODEL) не та, под которую настроена таблица")


def first_line(exc: BaseException) -> str:
    """Текст ошибки Postgres без DETAIL и HINT — для сообщения в одну строку."""
    return (str(exc).splitlines() or [type(exc).__name__])[0]


def version_tuple(version: str | None) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version or "")[:3])
