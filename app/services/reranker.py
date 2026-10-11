"""
Переранжирование кандидатов поиска (блок 5.4): вход — вопрос и список кандидатов из Qdrant, выход —
те же кандидаты, пересортированные по оценке re-ranker, первые top_n.

    reranker = build_reranker(get_settings())          # None, если RERANK__ENABLED=false
    top = reranker.rerank("Сколько стоит «Команда»?", candidates, top_n=3)

Векторный поиск сравнивает вопрос и фрагмент по отдельности посчитанным векторам. Re-ranker
(cross-encoder) читает вопрос и фрагмент вместе и оценивает их пару — точнее, но на каждую пару
нужен проход модели, поэтому переранжируются только первые RAG__CANDIDATES_TOP_K кандидатов.

- CrossEncoderReranker — BAAI/bge-reranker-v2-m3 (мультиязычный, на базе bge-m3) через
  sentence-transformers в процессе: бесплатно, без сети после первого скачивания весов
  (~2,3 ГБ), но на CPU — секунды на вопрос. Пакеты — requirements-embeddings.txt.
- CohereReranker — Cohere Rerank API (rerank-v3.5): быстро, но нужен ключ RERANK__API_KEY и
  доступ к api.cohere.com.

Оценки пар кешируются (память процесса + SQLite RERANK__CACHE_PATH): повторный прогон
эксперимента и одинаковые фрагменты в разных конфигурациях не считаются заново.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from app.observability.logging import get_logger

if TYPE_CHECKING:
    import httpx

    from app.core.config import Settings

log = get_logger()

DEFAULT_MODELS = {"cross-encoder": "BAAI/bge-reranker-v2-m3", "cohere": "rerank-v3.5"}


class RerankError(RuntimeError):
    """Re-ranker недоступен или ответил ошибкой: текст — что случилось и что сделать."""


@dataclass(frozen=True)
class Candidate:
    """Кандидат поиска. score — оценка re-ranker после rerank(), vector_score — близость из Qdrant."""

    text: str
    doc_id: str | None = None
    score: float = 0.0
    vector_score: float | None = None
    payload: Any = None


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: Sequence[Candidate], top_n: int | None = None) -> list[Candidate]: ...

    def close(self) -> None: ...


class ScoreCache:
    """Оценки пар «вопрос — фрагмент»: словарь в памяти и, если задан путь, таблица SQLite."""

    def __init__(self, path: Path | None = None) -> None:
        self._memory: dict[str, float] = {}
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(path), check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL NOT NULL)")
            self._db.commit()

    @staticmethod
    def key(model: str, query: str, text: str) -> str:
        return hashlib.sha256(json.dumps([model, query, text], ensure_ascii=False).encode()).hexdigest()

    def get_many(self, keys: Sequence[str]) -> dict[str, float]:
        with self._lock:
            found = {key: self._memory[key] for key in keys if key in self._memory}
            missing = [key for key in keys if key not in found]
            if self._db is not None and missing:
                for start in range(0, len(missing), 500):
                    part = missing[start:start + 500]
                    rows = self._db.execute(f"SELECT key, score FROM scores WHERE key IN ({','.join('?' * len(part))})",
                                            part).fetchall()
                    found.update(dict(rows))
                self._memory.update(found)
            return found

    def put_many(self, scores: dict[str, float]) -> None:
        with self._lock:
            self._memory.update(scores)
            if self._db is not None and scores:
                self._db.executemany("INSERT OR REPLACE INTO scores (key, score) VALUES (?, ?)", list(scores.items()))
                self._db.commit()

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None


class _CachedReranker:
    """Общее: кеш, сортировка по оценке, top_n. Подклассы считают оценки недостающих пар."""

    name = "reranker"
    model = ""

    def __init__(self, cache: ScoreCache | None = None) -> None:
        self.cache = cache or ScoreCache()
        self.pairs_scored = 0          # сколько пар посчитала модель (не из кеша) — для замеров

    def _score(self, query: str, texts: list[str]) -> list[float]:
        raise NotImplementedError

    def rerank(self, query: str, candidates: Sequence[Candidate], top_n: int | None = None) -> list[Candidate]:
        if not candidates:
            return []
        keys = [ScoreCache.key(self.model, query, c.text) for c in candidates]
        known = self.cache.get_many(keys)
        todo = list({key: c.text for key, c in zip(keys, candidates, strict=True) if key not in known}.items())
        if todo:
            scores = self._score(query, [text for _, text in todo])
            fresh = {key: float(score) for (key, _), score in zip(todo, scores, strict=True)}
            self.pairs_scored += len(fresh)
            self.cache.put_many(fresh)
            known.update(fresh)
        ranked = []
        for position, (key, candidate) in enumerate(zip(keys, candidates, strict=True)):
            vector = candidate.vector_score if candidate.vector_score is not None else candidate.score
            ranked.append((replace(candidate, score=known[key], vector_score=vector), position))
        ranked.sort(key=lambda pair: (-pair[0].score, pair[1]))    # при равенстве — порядок векторного поиска
        result = [candidate for candidate, _ in ranked]
        return result[:top_n] if top_n else result

    def close(self) -> None:
        self.cache.close()


class CrossEncoderReranker(_CachedReranker):
    """Cross-encoder Hugging Face через sentence-transformers. Модель загружается при первом
    вызове (секунды и ~2,3 ГБ памяти у bge-reranker-v2-m3) и дальше живёт в процессе."""

    name = "cross-encoder"

    def __init__(self, model: str = DEFAULT_MODELS["cross-encoder"], *, device: str = "cpu", max_length: int = 512,
                 batch_size: int = 8, use_system_certs: bool = False, cache: ScoreCache | None = None,
                 loader: Callable[..., Any] | None = None) -> None:
        super().__init__(cache)
        self.model, self.device, self.max_length, self.batch_size = model, device, max_length, batch_size
        self.use_system_certs = use_system_certs
        self._loader = loader
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                loader = self._loader or _cross_encoder_class()
                if self.use_system_certs:       # скачивание весов в сети с подменой HTTPS
                    import truststore

                    truststore.inject_into_ssl()
                started = time.perf_counter()
                self._model = loader(self.model, device=self.device, max_length=self.max_length)
                log.info("reranker_loaded", model=self.model, device=self.device,
                         ms=round((time.perf_counter() - started) * 1000))
            return self._model

    def _score(self, query: str, texts: list[str]) -> list[float]:
        scores = self._load().predict([(query, text) for text in texts], batch_size=self.batch_size,
                                      show_progress_bar=False, convert_to_numpy=True)
        return [float(score) for score in scores]

    def close(self) -> None:
        self._model = None
        super().close()


def _cross_encoder_class() -> Callable[..., Any]:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RerankError("RERANK__PROVIDER=cross-encoder: нет пакета sentence-transformers. Установите: "
                          "pip install -r requirements-embeddings.txt") from exc
    return CrossEncoder


class CohereReranker(_CachedReranker):
    """Cohere Rerank API v2: POST {base_url}/v2/rerank {model, query, documents, top_n}."""

    name = "cohere"

    def __init__(self, api_key: str, model: str = DEFAULT_MODELS["cohere"], *, base_url: str = "https://api.cohere.com",
                 timeout: float = 30.0, http_client: httpx.Client | None = None, cache: ScoreCache | None = None) -> None:
        import httpx

        super().__init__(cache)
        self.model, self.base_url = model, base_url.rstrip("/")
        self._client = http_client or httpx.Client(timeout=timeout)
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _score(self, query: str, texts: list[str]) -> list[float]:
        import httpx

        try:
            response = self._client.post(f"{self.base_url}/v2/rerank", headers=self._headers,
                                         json={"model": self.model, "query": query, "documents": texts,
                                               "top_n": len(texts)})
        except httpx.HTTPError as exc:
            raise RerankError(f"Cohere Rerank ({self.base_url}) не ответил: {type(exc).__name__}") from exc
        if response.status_code in (401, 403):
            raise RerankError(f"Cohere Rerank не принял ключ ({response.status_code}): RERANK__API_KEY в .env")
        if response.status_code >= 400:
            raise RerankError(f"Cohere Rerank ответил {response.status_code}: {response.text[:200]}")
        scores = [0.0] * len(texts)
        for item in response.json().get("results", []):
            scores[item["index"]] = float(item["relevance_score"])
        return scores

    def close(self) -> None:
        self._client.close()
        super().close()


def build_reranker(cfg: Settings) -> Reranker | None:
    """Re-ranker из RERANK__*; RERANK__ENABLED=false — None (поиск без переранжирования)."""
    settings = cfg.rerank
    if not settings.enabled:
        return None
    if settings.provider == "cross-encoder":
        _cross_encoder_class()      # нет пакета — понятная ошибка сразу, а не на первом вопросе
    cache = ScoreCache(settings.cache_path if settings.cache_enabled else None)
    model = settings.model or DEFAULT_MODELS[settings.provider]
    if settings.provider == "cohere":
        import httpx

        from app.core.config import http_client_options, proxy_for

        options = http_client_options(proxy_for(settings.base_url, cfg.llm.proxy_url), cfg.llm.use_system_certs)
        client = httpx.Client(timeout=settings.timeout, **options)
        return CohereReranker(settings.api_key.get_secret_value(), model, base_url=settings.base_url,
                              http_client=client, cache=cache)
    return CrossEncoderReranker(model, device=settings.device, max_length=settings.max_length,
                                batch_size=settings.batch_size, use_system_certs=cfg.llm.use_system_certs, cache=cache)
