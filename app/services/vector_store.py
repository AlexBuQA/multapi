"""
Векторное хранилище на Qdrant (блок 5.2): тонкая обёртка над AsyncQdrantClient.

Остальной код сервиса работает с VectorStore, а не с qdrant-client напрямую: на блоке 5.3
поиск переедет на LlamaIndex (QdrantVectorStore над этой же коллекцией), и интерфейс
upsert / search / ensure_collection при этом не меняется.

- Один клиент на процесс. В сервисе VectorStore создаёт lifespan (app/main.py) и кладёт в
  app.state; эндпоинты получают его через Depends(get_vector_store) (app/deps/providers.py).
  Скрипты создают свой — тоже один на запуск.
- ensure_collection() — коллекция и payload-индексы. Нет коллекции — создаёт: размерность
  EMBEDDING_DIM, метрика COSINE, HNSW m=16, ef_construct=100 (почему — HNSW ниже). Есть —
  сверяет размерность и метрику: коллекция под другую модель — VectorStoreError, сервис не
  стартует, а загрузка не льёт векторы не той длины. Недостающие индексы дописывает.
- upsert() — пачками по 256 точек; wait=True только у последней пачки: Qdrant применяет
  изменения по порядку, и когда ответила последняя, искать можно по всем. Длина каждого
  вектора проверяется до отправки — ошибка называет точку и обе длины.
- search() — query_points (search в Qdrant 1.19 устарел), возвращает list[ScoredPoint] с
  payload: в нём текст фрагмента для модели и метаданные для фильтров.

Ошибки Qdrant превращаются в понятный текст: VectorStoreUnavailable — нет связи (Qdrant не
запущен, неверный QDRANT_URL), VectorStoreError — отказ сервера (неверный ключ, размерность).
"""
from __future__ import annotations

import importlib.metadata
import warnings
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    Distance,
    Filter,
    HnswConfigDiff,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    Record,
    ScoredPoint,
    VectorParams,
)

from app.core.config import is_local_url
from app.observability.logging import get_logger

if TYPE_CHECKING:
    from app.core.config import Settings

log = get_logger()
T = TypeVar("T")

DEFAULT_BATCH = 256
SCROLL_PAGE = 1000
# Payload-индексы для полей, по которым фильтруется поиск. Без индекса Qdrant проверяет
# условие перебором точек — на сотне это незаметно, на миллионе — нет.
PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "source": PayloadSchemaType.KEYWORD,       # файл: release_notes.md, faq.md, …
    "created_at": PayloadSchemaType.DATETIME,  # дата редакции фрагмента, RFC 3339
    "category": PayloadSchemaType.KEYWORD,     # предметное поле: раздел справки (billing, api, …)
    "product": PayloadSchemaType.KEYWORD,      # web, mobile, api, support
    "status": PayloadSchemaType.KEYWORD,       # actual | archived
}
# HNSW — явно, со значениями по умолчанию Qdrant: m=16 (связей у вершины графа) и
# ef_construct=100 (ширина поиска при построении). Для баз до сотен тысяч векторов это
# стандартный баланс точности, памяти и времени индексации; увеличивать m и ef_construct стоит,
# если замер recall на своих данных покажет потери. Пока векторов меньше full_scan_threshold
# (10 000 КБ — около 2 500 векторов по 1024 числа), Qdrant ищет полным перебором, то есть
# точно, — у нашей базы в 110 фрагментов так и есть.
HNSW = HnswConfigDiff(m=16, ef_construct=100)


class VectorStoreError(RuntimeError):
    """Qdrant отказал или коллекция не подходит: текст — что случилось и что сделать."""


class VectorStoreUnavailable(VectorStoreError):
    """Нет связи с Qdrant."""


class VectorStore:
    def __init__(self, url: str | None, api_key: str | None, collection: str, dim: int, *,
                 distance: Distance = Distance.COSINE, timeout: int = 30,
                 client: AsyncQdrantClient | None = None) -> None:
        if client is None:
            if not url:
                raise VectorStoreError("QDRANT_URL не задан: укажите адрес Qdrant в .env (пример — в .env.example)")
            with warnings.catch_warnings():
                if is_local_url(url):
                    # Ключ по HTTP — норма для Qdrant на этом компьютере и в сети compose;
                    # для внешнего адреса предупреждение остаётся: там нужен https.
                    warnings.filterwarnings("ignore", message="Api key is used with an insecure connection")
                # check_compatibility=False: клиент сверял бы версии в фоновом потоке синхронным
                # запросом при создании; версию сверяет ensure_collection — асинхронно и в лог.
                client = AsyncQdrantClient(url=url, api_key=api_key, timeout=timeout, check_compatibility=False)
        self.client = client
        self.url = url or ":memory:"
        self.collection = collection
        self.dim = dim
        self.distance = distance
        self.server_version: str | None = None

    @classmethod
    def from_settings(cls, cfg: Settings, *, collection: str | None = None,
                      distance: Distance = Distance.COSINE) -> VectorStore:
        if not cfg.qdrant_url or cfg.embedding_dim is None:
            raise VectorStoreError("Векторная база не настроена: задайте QDRANT_URL и EMBEDDING_DIM в .env")
        key = cfg.qdrant_api_key.get_secret_value() if cfg.qdrant_api_key else None
        return cls(cfg.qdrant_url, key, collection or cfg.qdrant_collection, cfg.embedding_dim, distance=distance)

    # -------------------------------------------------------- коллекция
    async def ensure_collection(self) -> None:
        """Коллекция с нужной размерностью и метрикой + payload-индексы. Повторный вызов
        ничего не меняет; коллекция под другую модель — VectorStoreError."""
        await self._check_version()
        created = False
        if not await self._call(self.client.collection_exists(self.collection)):
            try:
                await self._call(self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=self.dim, distance=self.distance),
                    hnsw_config=HNSW,
                ))
                created = True
            except VectorStoreError as exc:
                if "already exists" not in str(exc):   # вторая копия сервиса успела создать — не ошибка
                    raise
        info = await self._call(self.client.get_collection(self.collection))
        vectors = info.config.params.vectors
        if not isinstance(vectors, VectorParams):
            raise VectorStoreError(f"Коллекция {self.collection} с именованными векторами — VectorStore работает "
                                   "с одним безымянным вектором на точку")
        if vectors.size != self.dim:
            raise VectorStoreError(
                f"Коллекция {self.collection} создана для векторов из {vectors.size} чисел, а EMBEDDING_DIM="
                f"{self.dim}: её заполняла другая модель эмбеддингов. Верните прежнюю модель или пересоздайте "
                f"коллекцию: python scripts/load_to_qdrant.py --recreate")
        if vectors.distance != self.distance:
            raise VectorStoreError(f"Коллекция {self.collection} с метрикой {vectors.distance.value}, а нужна "
                                   f"{self.distance.value}: пересоздайте её (scripts/load_to_qdrant.py --recreate)")
        existing = set(info.payload_schema or {})
        for field, schema in PAYLOAD_INDEXES.items():
            if field not in existing:
                await self._call(self.client.create_payload_index(self.collection, field, schema, wait=True))
        if created:
            log.info("vector_collection_created", collection=self.collection, dim=self.dim,
                     distance=self.distance.value, indexes=sorted(PAYLOAD_INDEXES))

    async def points_count(self) -> int:
        info = await self._call(self.client.get_collection(self.collection))
        return int(info.points_count or 0)

    async def collection_info(self) -> Any:
        return await self._call(self.client.get_collection(self.collection))

    async def drop(self) -> None:
        """Удалить коллекцию (эксперименты и --recreate). Нет коллекции — не ошибка."""
        await self._call(self.client.delete_collection(self.collection))

    # -------------------------------------------------------- точки
    async def upsert(self, points: Sequence[PointStruct], batch_size: int = DEFAULT_BATCH,
                     on_batch: Callable[[int], None] | None = None) -> None:
        """Записать точки пачками. Тот же id — перезапись, а не копия. on_batch(n) — после
        каждой пачки (прогресс-бар загрузки)."""
        if batch_size < 1:
            raise ValueError("batch_size должен быть больше 0")
        points = list(points)
        for point in points:
            self._check_dim(point.vector, f"точка {point.id}")
        for start in range(0, len(points), batch_size):
            batch = points[start:start + batch_size]
            last = start + batch_size >= len(points)
            await self._call(self.client.upsert(collection_name=self.collection, points=batch, wait=last))
            if on_batch is not None:
                on_batch(len(batch))

    async def search(self, query_vector: Sequence[float], top_k: int = 5,
                     query_filter: Filter | None = None) -> list[ScoredPoint]:
        """top_k ближайших точек с payload; query_filter — условия по payload (Filter Qdrant)."""
        self._check_dim(query_vector, "вектор запроса")
        result = await self._call(self.client.query_points(
            collection_name=self.collection, query=list(query_vector), query_filter=query_filter,
            limit=top_k, with_payload=True))
        return list(result.points)

    async def scroll(self, *, with_vectors: bool = False, with_payload: bool = True) -> list[Record]:
        """Все точки коллекции, страницами по SCROLL_PAGE."""
        records: list[Record] = []
        offset = None
        while True:
            page, offset = await self._call(self.client.scroll(
                collection_name=self.collection, limit=SCROLL_PAGE, offset=offset,
                with_payload=with_payload, with_vectors=with_vectors))
            records.extend(page)
            if offset is None:
                return records

    async def point_ids(self) -> set[str]:
        return {str(record.id) for record in await self.scroll(with_payload=False)}

    async def delete(self, ids: Sequence[str]) -> None:
        if ids:
            await self._call(self.client.delete(collection_name=self.collection,
                                                points_selector=PointIdsList(points=list(ids)), wait=True))

    async def close(self) -> None:
        await self.client.close()

    # -------------------------------------------------------- внутреннее
    async def _check_version(self) -> None:
        """Клиент и сервер должны совпадать по мажорной версии и отличаться по минорной не
        больше чем на 1 (правило qdrant-client): иначе часть запросов может не работать."""
        self.server_version = (await self._call(self.client.info())).version
        client_version = importlib.metadata.version("qdrant-client")
        server, client = (tuple(int(x) for x in v.split(".")[:2]) for v in (self.server_version, client_version))
        if server[0] != client[0] or abs(server[1] - client[1]) > 1:
            log.warning("qdrant_version_mismatch", server=self.server_version, client=client_version,
                        note="обновите образ qdrant/qdrant в compose.yaml или пакет qdrant-client")

    def _check_dim(self, vector: Any, what: str) -> None:
        size = len(vector) if isinstance(vector, Sequence) else None
        if size != self.dim:
            raise VectorStoreError(
                f"{what}: вектор из {size} чисел, а коллекция {self.collection} — для {self.dim} "
                "(EMBEDDING_DIM). Модель эмбеддингов в .env (EMBEDDINGS__MODEL) не та, под которую "
                "настроена коллекция")

    async def _call(self, call: Awaitable[T]) -> T:
        try:
            return await call
        except ResponseHandlingException as exc:
            raise VectorStoreUnavailable(
                f"Нет связи с Qdrant ({self.url}): {exc}. Запущен ли он — docker compose up -d qdrant; "
                "верен ли QDRANT_URL") from exc
        except UnexpectedResponse as exc:
            if exc.status_code in (401, 403):
                raise VectorStoreError(f"Qdrant ({self.url}) не принял ключ ({exc.status_code}): QDRANT_API_KEY в "
                                       ".env должен совпадать с ключом контейнера") from exc
            detail = exc.content.decode("utf-8", "replace")[:300] if exc.content else exc.reason_phrase
            raise VectorStoreError(f"Qdrant ответил {exc.status_code}: {detail}") from exc


# ---------------------------------------------------------------- фильтры и сервис
def build_filter(*, category: str | None = None, source: str | None = None, created_after: str | None = None,
                 include_archived: bool = False) -> Filter | None:
    """Filter Qdrant из параметров поиска: всё через must, архивные редакции — must_not.
    created_after — дата или момент RFC 3339 (2026-09-10 или 2026-09-10T00:00:00Z)."""
    from qdrant_client.models import DatetimeRange, FieldCondition, MatchValue

    must: list[Any] = []
    if category:
        must.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if source:
        must.append(FieldCondition(key="source", match=MatchValue(value=source)))
    if created_after:
        must.append(FieldCondition(key="created_at", range=DatetimeRange(gte=created_after)))
    must_not = [] if include_archived else [FieldCondition(key="status", match=MatchValue(value="archived"))]
    if not must and not must_not:
        return None
    return Filter(must=must or None, must_not=must_not or None)


async def open_vector_store(cfg: Settings) -> VectorStore | None:
    """Для lifespan сервиса: VectorStore и ensure_collection. QDRANT_URL не задан — None.
    Qdrant не отвечает — сервис всё равно стартует (чат от него не зависит), в лог —
    vector_store_unavailable. Коллекция под другую модель — исключение: старт с заведомо
    неверной конфигурацией хуже, чем понятная ошибка."""
    if not cfg.qdrant_url:
        log.info("vector_store_disabled", reason="QDRANT_URL не задан")
        return None
    store = VectorStore.from_settings(cfg)
    try:
        await store.ensure_collection()
        log.info("vector_store_ready", url=store.url, collection=store.collection, dim=store.dim,
                 points=await store.points_count(), server_version=store.server_version)
    except VectorStoreUnavailable as exc:
        log.warning("vector_store_unavailable", url=store.url, error=str(exc),
                    note="сервис работает без векторного поиска; коллекция проверится при следующем запуске")
    except Exception:
        await store.close()
        raise
    return store
