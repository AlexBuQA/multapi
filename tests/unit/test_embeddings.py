"""
Эмбеддинги (блок 5.1): app/services/embeddings.py без сети и без моделей.

- FakeBackend вместо модели: записывает пачки, которые ему отправили, и возвращает
  ненормированные векторы — так видно, что нормализует сервис;
- OpenAI SDK — через httpx.MockTransport: тело запроса /embeddings и разбор ответа;
- sentence-transformers — через подставной загрузчик: torch тестам не нужен;
- кеш: память процесса, файл SQLite между «перезапусками» (новый EmbeddingService на тот
  же файл), смена модели — новый запрос;
- повторы tenacity: обрыв, 429 с Retry-After, 5xx — повтор; 401/404 — сразу понятная ошибка.

Данные задания: tests/eval/mini_benchmark.json (5–10 троек, фрагменты — дословно из базы)
и data/help_center.jsonl (50+ документов). Скрипты CLI, бенчмарка и стоимости — на
подставной модели.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

import httpx
import openai
import pytest

from app.core.config import EmbeddingSettings, LLMSettings, Settings
from app.services import embeddings as emb
from app.services.embeddings import (
    EmbeddingCache,
    EmbeddingError,
    EmbeddingService,
    OpenAICompatibleBackend,
    Prefixes,
    SentenceTransformerBackend,
    cache_key,
    cosine,
    normalize,
    resolve_prefixes,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "tests" / "eval" / "mini_benchmark.json"
CORPUS = ROOT / "data" / "help_center.jsonl"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module                  # dataclass в скрипте ищет свой модуль в sys.modules
    spec.loader.exec_module(module)
    return module


class FakeBackend:
    """Модель-заглушка: вектор [длина текста, 1, число букв «а», 2] — разный для разных текстов
    и с длиной не 1. errors — исключения по очереди перед успешными ответами."""

    provider = "openai"

    def __init__(self, model: str = "bge-m3", *, local: bool = True, namespace: str = "local:11434",
                 dimensions: int | None = None, errors: list[BaseException] | None = None, dim: int = 4) -> None:
        self.model, self.local, self.namespace, self.dimensions = model, local, namespace, dimensions
        self.errors = list(errors or [])
        self.dim = dim
        self.batches: list[list[str]] = []
        self.closed = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        if self.errors:
            raise self.errors.pop(0)
        return [([float(len(text)), 1.0, float(text.count("а")), 2.0] + [0.5] * self.dim)[:self.dim]
                for text in texts]

    @property
    def sent(self) -> list[str]:
        return [text for batch in self.batches for text in batch]

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "var" / "embeddings_cache.sqlite"


@pytest.fixture
def services():
    """Сервисы теста закрываются в конце: незакрытый файл SQLite — ResourceWarning."""
    created: list[EmbeddingService] = []

    def make(backend=None, **kwargs) -> EmbeddingService:
        kwargs.setdefault("sleep", lambda seconds: None)
        service = EmbeddingService(backend or FakeBackend(), **kwargs)
        created.append(service)
        return service

    yield make
    for service in created:
        service.close()


def request_for(url: str = "http://localhost:11434/v1/embeddings") -> httpx.Request:
    return httpx.Request("POST", url)


def status_error(cls, status: int, headers: dict[str, str] | None = None) -> openai.APIStatusError:
    response = httpx.Response(status, headers=headers or {}, request=request_for(),
                              json={"error": {"message": "boom"}})
    return cls("boom", response=response, body=None)


# ---------------------------------------------------------------- векторы
def test_vectors_are_normalized_and_in_order(services):
    backend = FakeBackend()
    vectors = services(backend).embed_texts(["а", "ббб", "ааааа"])
    assert len(vectors) == 3
    for vector in vectors:
        assert math.isclose(math.fsum(x * x for x in vector), 1.0, rel_tol=1e-6)
    raw = backend.embed(["ббб"])[0]
    assert cosine(vectors[1], raw) == pytest.approx(1.0)       # порядок ответа — порядок входа


def test_normalize_rounds_to_float32_and_keeps_zero_vector():
    assert normalize([3.0, 4.0]) == [pytest.approx(0.6), pytest.approx(0.8)]
    assert normalize([0.0, 0.0]) == [0.0, 0.0]
    assert normalize([1.0, 1.0, 1.0])[0] != 1 / math.sqrt(3)  # float32, как в кеше
    assert cosine([1.0, 0.0], [0.0, 2.0]) == 0.0 and cosine([0.0], [0.0]) == 0.0


def test_duplicates_are_embedded_once(services):
    backend = FakeBackend()
    vectors = services(backend).embed_texts(["тот же текст", "другой", "тот же текст"])
    assert backend.sent == ["тот же текст", "другой"]
    assert vectors[0] == vectors[2]


def test_empty_list_does_not_call_the_model(services):
    backend = FakeBackend()
    assert services(backend).embed_texts([]) == [] and backend.batches == []


@pytest.mark.parametrize("bad, error", [("строка, а не список", TypeError), (["ok", 42], TypeError),
                                        (["ok", "   "], ValueError), ([""], ValueError)])
def test_bad_input_is_rejected_before_the_model(services, bad, error):
    backend = FakeBackend()
    with pytest.raises(error):
        services(backend).embed_texts(bad)
    assert backend.batches == []


def test_returned_vectors_are_copies(services, cache_path):
    service = services(cache=EmbeddingCache(cache_path))
    first = service.embed_texts(["текст"])
    first[0][0] = 123.0
    assert service.embed_texts(["текст"])[0][0] != 123.0


def test_wrong_answers_from_model_are_errors(services):
    class Short(FakeBackend):
        def embed(self, texts):
            return super().embed(texts)[:-1]

    with pytest.raises(EmbeddingError, match="на 2 текстов пришло 1"):
        services(Short()).embed_texts(["а", "б"])

    class Ragged(FakeBackend):
        def embed(self, texts):
            vectors = super().embed(texts)
            vectors[-1] = vectors[-1][:2]
            return vectors

    with pytest.raises(EmbeddingError, match="разной длины"):
        services(Ragged()).embed_texts(["а", "б"])


def test_dimension_change_inside_one_service_is_an_error(services):
    backend = FakeBackend()
    service = services(backend)
    service.embed_texts(["а"])
    backend.dim = 6
    with pytest.raises(EmbeddingError, match="ожидалось 4"):
        service.embed_texts(["б"])


# ---------------------------------------------------------------- батчи
def test_local_model_batches_of_32(services):
    backend = FakeBackend(local=True)
    service = services(backend)
    service.embed_texts([f"документ {n}" for n in range(70)])
    assert service.batch_size == 32 and [len(b) for b in backend.batches] == [32, 32, 6]
    assert service.max_request_tokens is None


def test_cloud_model_batches_of_128_and_token_budget(services):
    backend = FakeBackend("text-embedding-3-small", local=False, namespace="api.openai.com")
    service = services(backend, token_counter=len)
    assert service.batch_size == 128 and service.max_request_tokens == emb.MAX_REQUEST_TOKENS
    service.embed_texts([f"документ {n:03d}" for n in range(300)])
    assert [len(b) for b in backend.batches] == [128, 128, 44]

    backend = FakeBackend("text-embedding-3-small", local=False, namespace="api.openai.com")
    service = services(backend, token_counter=len, max_request_tokens=25)   # 12 символов = 12 «токенов»
    service.embed_texts([f"документ {n:03d}" for n in range(5)])
    assert [len(b) for b in backend.batches] == [2, 2, 1]


def test_batch_size_from_settings(services):
    backend = FakeBackend()
    services(backend, batch_size=2).embed_texts(["а", "б", "в"])
    assert [len(b) for b in backend.batches] == [2, 1]


# ---------------------------------------------------------------- кеш
def test_repeat_in_same_process_comes_from_memory(services, cache_path):
    backend = FakeBackend()
    service = services(backend, cache=EmbeddingCache(cache_path))
    first = service.embed_texts(["тот же текст"])
    second = service.embed_texts(["тот же текст"])
    assert first == second and len(backend.batches) == 1
    assert (service.stats.memory_hits, service.stats.disk_hits, service.stats.requests) == (1, 0, 1)


def test_repeat_after_restart_comes_from_disk(services, cache_path):
    first = services(FakeBackend(), cache=EmbeddingCache(cache_path)).embed_texts(["тот же текст"])
    backend = FakeBackend()                                   # «перезапуск»: новый процесс, тот же файл
    service = services(backend, cache=EmbeddingCache(cache_path))
    assert service.embed_texts(["тот же текст"]) == first     # float32 в файле == float32 в памяти
    assert backend.batches == [] and service.stats.disk_hits == 1 and service.stats.requests == 0


def test_partial_hit_sends_only_new_texts(services, cache_path):
    services(FakeBackend(), cache=EmbeddingCache(cache_path)).embed_texts(["старый"])
    backend = FakeBackend()
    vectors = services(backend, cache=EmbeddingCache(cache_path)).embed_texts(["новый", "старый", "новый"])
    assert backend.sent == ["новый"] and vectors[0] == vectors[2] and vectors[0] != vectors[1]


@pytest.mark.parametrize("changed", [
    {"model": "bge-m3:567m"},
    {"model": "text-embedding-3-large", "local": False, "namespace": "api.openai.com"},
    {"dimensions": 512},
    {"namespace": "openrouter.ai", "local": False},
])
def test_model_change_does_not_reuse_old_vectors(services, cache_path, changed):
    """Сменили EMBEDDINGS__MODEL (или размерность, или провайдера) — запрос к новой модели."""
    services(FakeBackend(), cache=EmbeddingCache(cache_path)).embed_texts(["тот же текст"])
    backend = FakeBackend(**changed)
    service = services(backend, cache=EmbeddingCache(cache_path), token_counter=len)
    service.embed_texts(["тот же текст"])
    assert backend.sent == ["тот же текст"]
    assert len(service.cache.summary()) == 2                  # записи обеих моделей


def test_same_ollama_from_docker_and_host_shares_cache():
    host = OpenAICompatibleBackend(model="bge-m3", base_url="http://localhost:11434/v1", api_key="x", client=object())
    docker = OpenAICompatibleBackend(model="bge-m3", base_url="http://host.docker.internal:11434/v1", api_key="x",
                                     client=object())
    cloud = OpenAICompatibleBackend(model="bge-m3", base_url="https://openrouter.ai/api/v1", api_key="x",
                                    client=object())
    assert host.namespace == docker.namespace == "local:11434" and cloud.namespace == "openrouter.ai"
    assert host.local and not cloud.local


def test_prefix_is_part_of_the_key(services, cache_path):
    backend = FakeBackend("intfloat/multilingual-e5-small")
    service = services(backend, cache=EmbeddingCache(cache_path))
    service.embed_texts(["Как сменить пароль?"])
    service.embed_query("Как сменить пароль?")
    service.embed_documents(["Как сменить пароль?"])
    assert backend.sent == ["Как сменить пароль?", "query: Как сменить пароль?", "passage: Как сменить пароль?"]


def test_cache_key_depends_on_every_part():
    base = ("openai", "local:11434", "bge-m3", None, "текст")
    keys = {cache_key(*base)}
    for index, value in enumerate(["sentence-transformers", "api.openai.com", "bge-m3:567m", 512, "текст "]):
        changed = list(base)
        changed[index] = value
        keys.add(cache_key(*changed))
    assert len(keys) == 6 and all(len(key) == 64 for key in keys)


def test_cache_file_is_float32(cache_path):
    cache = EmbeddingCache(cache_path)
    cache.put_many([("k", [0.1, 0.2, 0.3])], "m")
    cache.close()
    with sqlite3.connect(cache_path) as db:
        dim, blob = db.execute("SELECT dim, vector FROM embeddings").fetchone()
    db.close()
    assert dim == 3 and len(blob) == 3 * 4


def test_broken_record_is_recomputed(services, cache_path):
    services(FakeBackend(), cache=EmbeddingCache(cache_path)).embed_texts(["текст"])
    with sqlite3.connect(cache_path) as db:
        db.execute("UPDATE embeddings SET vector = ?", (b"\x00" * 4,))
    db.close()
    backend = FakeBackend()
    services(backend, cache=EmbeddingCache(cache_path)).embed_texts(["текст"])
    assert backend.sent == ["текст"]


def test_unusable_cache_file_falls_back_to_memory(services, tmp_path):
    folder = tmp_path / "occupied"
    folder.mkdir()                                            # путь занят папкой — файл не открыть
    cache = EmbeddingCache(folder)
    assert not cache.persistent
    backend = FakeBackend()
    service = services(backend, cache=cache)
    service.embed_texts(["текст"])
    service.embed_texts(["текст"])
    assert len(backend.batches) == 1


def test_memory_is_bounded(cache_path):
    cache = EmbeddingCache(None, memory_items=2)
    cache.put_many([("a", [1.0]), ("b", [1.0]), ("c", [1.0])], "m")
    assert set(cache.get_many(["a", "b", "c"])) == {"b", "c"}


def test_clear_and_summary(services, cache_path):
    service = services(FakeBackend(), cache=EmbeddingCache(cache_path))
    service.embed_texts(["а", "б"])
    assert service.cache.summary() == [("openai:local:11434/bge-m3", 4, 2)]
    assert service.cache.clear() == 2 and service.cache.summary() == []
    service.embed_texts(["а"])
    assert service.stats.requests == 2


# ---------------------------------------------------------------- сеть и повторы
def test_connection_errors_are_retried_with_backoff(services):
    sleeps: list[float] = []
    backend = FakeBackend(errors=[openai.APIConnectionError(request=request_for()),
                                  openai.APITimeoutError(request=request_for())])
    service = services(backend, sleep=sleeps.append)
    assert len(service.embed_texts(["текст"])) == 1
    assert service.stats.requests == 3 and service.stats.retries == 2 and service.stats.embedded == 1
    assert 1 <= sleeps[0] <= 2 and 2 <= sleeps[1] <= 3          # 1 с, 2 с + разброс до 1 с


def test_server_errors_and_rate_limit_are_retried(services):
    sleeps: list[float] = []
    backend = FakeBackend(errors=[status_error(openai.InternalServerError, 503),
                                  status_error(openai.RateLimitError, 429, {"retry-after": "7"})])
    service = services(backend, sleep=sleeps.append)
    service.embed_texts(["текст"])
    assert len(backend.batches) == 3 and sleeps[1] >= 7       # Retry-After важнее своей задержки


def test_retry_after_is_capped():
    assert emb.retry_after(status_error(openai.RateLimitError, 429, {"retry-after-ms": "1500"})) == 1.5
    assert emb.retry_after(status_error(openai.RateLimitError, 429, {"retry-after": "Wed, 21 Oct 2026"})) is None
    assert emb.retry_after(ValueError()) is None


def test_long_retry_after_is_capped(services):
    sleeps: list[float] = []
    backend = FakeBackend(errors=[status_error(openai.RateLimitError, 429, {"retry-after": "3600"})])
    services(backend, sleep=sleeps.append).embed_texts(["текст"])
    assert sleeps == [emb.RETRY_AFTER_MAX]


def test_gives_up_after_max_attempts_with_hint(services):
    backend = FakeBackend(errors=[openai.APIConnectionError(request=request_for())] * 5)
    with pytest.raises(EmbeddingError, match=r"после 3 попыток.*ollama serve") as caught:
        services(backend, max_attempts=3).embed_texts(["текст"])
    assert len(backend.batches) == 3 and isinstance(caught.value.__cause__, openai.APIConnectionError)


@pytest.mark.parametrize("error, hint", [
    (status_error(openai.NotFoundError, 404), "ollama pull bge-m3"),
    (status_error(openai.AuthenticationError, 401), "EMBEDDINGS__API_KEY"),
    (status_error(openai.BadRequestError, 400), "отклонил запрос"),
])
def test_errors_that_retry_cannot_fix_are_not_retried(services, error, hint):
    backend = FakeBackend(errors=[error])
    with pytest.raises(EmbeddingError, match=hint):
        services(backend).embed_texts(["текст"])
    assert len(backend.batches) == 1


def test_no_credits_on_openrouter_is_explained(services):
    """Проверка на Windows: учебный ключ OpenRouter без кредитов — 402, модели эмбеддингов платные."""
    backend = FakeBackend("openai/text-embedding-3-small", local=False, namespace="openrouter.ai",
                          errors=[status_error(openai.APIStatusError, 402)])
    with pytest.raises(EmbeddingError, match="нет кредитов"):
        services(backend, token_counter=len).embed_texts(["текст"])
    assert len(backend.batches) == 1                                      # 402 не повторяется


def test_label_survives_pii_masking_in_logs():
    """Проверка на Windows: подпись openai@openrouter.ai в JSON-логе превращалась в [EMAIL]."""
    from app.observability.pii import redact_pii

    backend = FakeBackend("openai/text-embedding-3-small", local=False, namespace="openrouter.ai", dimensions=512)
    label = EmbeddingService(backend, cache=None, token_counter=len).label
    assert redact_pii(label) == label and "@" not in label


def test_cloud_connection_hint_mentions_proxy(services):
    backend = FakeBackend("text-embedding-3-small", local=False, namespace="openrouter.ai",
                          errors=[openai.APIConnectionError(request=request_for())])
    with pytest.raises(EmbeddingError, match="LLM__PROXY_URL"):
        services(backend, max_attempts=1, token_counter=len).embed_texts(["текст"])


def test_failed_batch_does_not_lose_finished_ones(services, cache_path):
    backend = FakeBackend()
    service = services(backend, cache=EmbeddingCache(cache_path), batch_size=2, max_attempts=1)
    original = backend.embed

    def second_fails(texts):
        if len(backend.batches) == 1:
            backend.batches.append(list(texts))
            raise status_error(openai.BadRequestError, 400)
        return original(texts)

    backend.embed = second_fails
    with pytest.raises(EmbeddingError):
        service.embed_texts(["а", "б", "в"])
    assert set(service.cache.get_many([service._key("а"), service._key("б")])) == {service._key("а"),
                                                                                     service._key("б")}


# ---------------------------------------------------------------- префиксы
@pytest.mark.parametrize("model, query, document", [
    ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    ("intfloat/multilingual-e5-base", "query: ", "passage: "),
    ("jeffh/intfloat-multilingual-e5-large:f16", "query: ", "passage: "),
    ("e5-small-v2", "query: ", "passage: "),
    ("intfloat/multilingual-e5-large-instruct", emb.E5_INSTRUCT_QUERY, ""),
    ("ai-forever/FRIDA", "search_query: ", "search_document: "),
    ("ai-forever/ru-en-RoSBERTa", "search_query: ", "search_document: "),
    ("nomic-embed-text", "search_query: ", "search_document: "),
    ("embeddinggemma", "task: search result | query: ", "title: none | text: "),
    ("bge-m3", "", ""),
    ("text-embedding-3-small", "", ""),
    ("openai/text-embedding-3-large", "", ""),
    ("deepvk/USER-bge-m3", "", ""),
])
def test_prefixes_by_model_family(model, query, document):
    prefixes = resolve_prefixes(model)
    assert (prefixes.query, prefixes.document) == (query, document)


def test_prefixes_from_env_override_family():
    assert resolve_prefixes("intfloat/multilingual-e5-small", "none", None) == Prefixes("", "passage: ", "из .env")
    assert resolve_prefixes("bge-m3", "Q: ", "D: ") == Prefixes("Q: ", "D: ", "из .env")


def test_query_and_documents_use_prefixes(services):
    backend = FakeBackend("intfloat/multilingual-e5-small")
    service = services(backend)
    query = service.embed_query("Забыл пароль")
    docs = service.embed_documents(["Нажмите «Забыли пароль?»", "Тарифы"])
    plain = service.embed_texts(["Забыл пароль"])
    assert backend.sent == ["query: Забыл пароль", "passage: Нажмите «Забыли пароль?»", "passage: Тарифы",
                            "Забыл пароль"]
    assert len(query) == 4 and len(docs) == 2 and plain[0] != query    # без префикса — другой вектор


def test_symmetric_model_query_equals_texts(services):
    backend = FakeBackend("bge-m3")
    service = services(backend)
    assert service.embed_query("вопрос") == service.embed_texts(["вопрос"])[0]
    assert backend.sent == ["вопрос", "вопрос"]                  # без кеша — дважды, но без префикса


# ---------------------------------------------------------------- OpenAI-совместимый API
def openai_backend(handler, base_url: str = "http://localhost:11434/v1", **kwargs) -> OpenAICompatibleBackend:
    client = openai.OpenAI(api_key="test", base_url=base_url, max_retries=0,
                           http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return OpenAICompatibleBackend(model=kwargs.pop("model", "bge-m3"), base_url=base_url, api_key="test",
                                   client=client, **kwargs)


def embeddings_response(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    data = [{"object": "embedding", "index": index, "embedding": [float(index + 1), 0.0, 0.0]}
            for index, _ in enumerate(body["input"])]
    return httpx.Response(200, json={"object": "list", "data": data[::-1], "model": body["model"],
                                     "usage": {"prompt_tokens": 1, "total_tokens": 1}})


def test_openai_request_and_unordered_answer(services):
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"url": str(request.url), "body": json.loads(request.content)})
        return embeddings_response(request)

    backend = openai_backend(handler)
    vectors = services(backend).embed_texts(["первый", "второй"])
    assert seen == [{"url": "http://localhost:11434/v1/embeddings",
                     "body": {"model": "bge-m3", "input": ["первый", "второй"], "encoding_format": "float"}}]
    assert vectors == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]          # нормализованы, порядок — по index
    raw = backend.embed(["первый", "второй"])
    assert raw == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]


def test_openai_dimensions_and_openrouter_headers(services):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return embeddings_response(request)

    backend = openai_backend(handler, base_url="https://openrouter.ai/api/v1", model="openai/text-embedding-3-small",
                             dimensions=512)
    backend.embed(["текст"])
    assert json.loads(seen[0].content)["dimensions"] == 512
    assert backend.namespace == "openrouter.ai" and not backend.local


def test_real_sdk_errors_go_through_retries(services):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": {"message": "loading model"}})
        if calls["n"] == 2:
            raise httpx.ConnectError("refused", request=request)
        return embeddings_response(request)

    service = services(openai_backend(handler))
    assert len(service.embed_texts(["текст"])) == 1
    assert calls["n"] == 3 and service.stats.retries == 2


def test_real_sdk_404_from_ollama(services):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": 'model "bge-m3" not found, try pulling it first'}})

    with pytest.raises(EmbeddingError, match="ollama pull bge-m3"):
        services(openai_backend(handler)).embed_texts(["текст"])


def test_built_client_uses_proxy_only_for_cloud(monkeypatch):
    made: list[dict] = []

    class Spy:
        def __init__(self, **kwargs):
            made.append(kwargs)

    monkeypatch.setattr(openai, "OpenAI", Spy)
    monkeypatch.setattr(openai, "DefaultHttpxClient", lambda **kwargs: ("httpx", kwargs))
    OpenAICompatibleBackend(model="bge-m3", base_url="http://localhost:11434/v1", api_key="k", proxy=None,
                            use_system_certs=True)
    OpenAICompatibleBackend(model="openai/text-embedding-3-small", base_url="https://openrouter.ai/api/v1",
                            api_key="k", proxy="http://proxy.invalid:8080", timeout=5)
    assert made[0]["max_retries"] == 0 and made[0]["http_client"] is None   # к Ollama — без прокси и truststore
    assert made[1]["http_client"] == ("httpx", {"proxy": "http://proxy.invalid:8080"})
    assert made[1]["default_headers"]["X-Title"] == "multapi" and made[1]["timeout"] == 5


# ---------------------------------------------------------------- sentence-transformers
class FakeSentenceTransformer:
    loaded: list[tuple[str, str]] = []

    def __init__(self, name: str, device: str) -> None:
        self.loaded.append((name, device))
        self.calls: list[dict] = []

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        return [[0.6, 0.8] for _ in texts]


def test_sentence_transformers_is_lazy_and_normalizes(services):
    FakeSentenceTransformer.loaded = []
    backend = SentenceTransformerBackend("intfloat/multilingual-e5-small", batch_size=16,
                                         loader=FakeSentenceTransformer)
    service = services(backend)
    assert FakeSentenceTransformer.loaded == []                  # модель не грузится заранее
    service.embed_query("вопрос")
    service.embed_documents(["фрагмент"])
    assert FakeSentenceTransformer.loaded == [("intfloat/multilingual-e5-small", "cpu")]
    calls = backend._model.calls
    assert calls[0]["texts"] == ["query: вопрос"] and calls[1]["texts"] == ["passage: фрагмент"]
    assert all(c["normalize_embeddings"] is True and c["batch_size"] == 16 for c in calls)
    assert service.batch_size == 32 and service.label == "sentence-transformers:hf/intfloat/multilingual-e5-small"


def test_sentence_transformers_missing_package(monkeypatch, services):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    service = services(SentenceTransformerBackend("intfloat/multilingual-e5-small"))
    with pytest.raises(EmbeddingError, match="requirements-embeddings.txt"):
        service.embed_texts(["текст"])


# ---------------------------------------------------------------- настройки
def settings(**embeddings) -> Settings:
    return Settings(_env_file=None, llm=LLMSettings(openai_api_key="llm-key", base_url="http://localhost:11434/v1"),
                    embeddings=EmbeddingSettings(**embeddings))


def test_defaults_bge_m3_in_ollama(tmp_path):
    cfg = settings(cache_path=tmp_path / "c.sqlite")
    service = emb.build_service(cfg)
    try:
        assert service.label == "openai:local:11434/bge-m3"
        assert service.batch_size == 32 and service.prefixes == Prefixes()
        assert service.backend.base_url == "http://localhost:11434/v1"      # LLM__BASE_URL
        assert service.cache is not None and service.cache.persistent
    finally:
        service.close()


def test_settings_from_environment(monkeypatch, tmp_path):
    for name in [name for name in os.environ if name.startswith("EMBEDDINGS__")]:
        monkeypatch.delenv(name)                                 # .env разработчика (src/config.py грузит его)
    monkeypatch.setenv("EMBEDDINGS__PROVIDER", "sentence-transformers")
    monkeypatch.setenv("EMBEDDINGS__MODEL", "intfloat/multilingual-e5-small")
    monkeypatch.setenv("EMBEDDINGS__BATCH_SIZE", "16")
    monkeypatch.setenv("EMBEDDINGS__QUERY_PREFIX", "вопрос: ")
    monkeypatch.setenv("EMBEDDINGS__CACHE_PATH", "var/other.sqlite")
    monkeypatch.setenv("EMBEDDINGS__CACHE_ENABLED", "false")
    cfg = Settings(_env_file=None)
    assert cfg.embeddings.provider == "sentence-transformers" and cfg.embeddings.batch_size == 16
    assert cfg.embeddings.query_prefix == "вопрос: "            # пробел в конце сохраняется
    assert cfg.embeddings.cache_path == ROOT / "var" / "other.sqlite"
    service = emb.build_service(cfg)
    assert service.cache is None and service.batch_size == 16
    assert service.prefixes == Prefixes("вопрос: ", "passage: ", "из .env")
    assert service.backend.batch_size == 16
    service.close()


def test_own_url_and_key_for_embeddings(tmp_path):
    cfg = settings(base_url="https://openrouter.ai/api/v1", api_key="emb-key", model="openai/text-embedding-3-small",
                   dimensions=512, cache_enabled=False)
    service = emb.build_service(cfg)
    try:
        assert service.backend.namespace == "openrouter.ai" and service.batch_size == 128
        assert service.backend._client.api_key == "emb-key" and service.backend.dimensions == 512
        assert service.label == "openai:openrouter.ai/openai/text-embedding-3-small:512d"
    finally:
        service.close()


@pytest.mark.parametrize("field, value", [("model", "  "), ("batch_size", 0), ("dimensions", 8),
                                          ("provider", "cohere"), ("max_attempts", 0)])
def test_bad_settings_are_rejected(field, value):
    with pytest.raises(ValueError):
        EmbeddingSettings(**{field: value})


def test_module_functions_use_one_default_service(monkeypatch, cache_path):
    backend = FakeBackend("intfloat/multilingual-e5-small")
    built: list[EmbeddingService] = []

    def build_service():
        built.append(EmbeddingService(backend, cache=EmbeddingCache(cache_path)))
        return built[-1]

    monkeypatch.setattr(emb, "build_service", build_service)
    emb.reset_default_service()
    try:
        vectors = emb.embed_texts(["тот же текст"])
        assert emb.embed_texts(["тот же текст"]) == vectors
        emb.embed_query("вопрос")
        emb.embed_documents(["фрагмент"])
        assert len(built) == 1 and backend.sent == ["тот же текст", "query: вопрос", "passage: фрагмент"]
    finally:
        emb.reset_default_service()
    assert backend.closed


# ---------------------------------------------------------------- данные задания
def corpus() -> list[dict]:
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_mini_benchmark_format():
    items = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    assert isinstance(items, list) and 5 <= len(items) <= 10
    for item in items:
        assert set(item) == {"query", "relevant", "irrelevant"}
        assert all(isinstance(value, str) and value.strip() for value in item.values())
        assert item["relevant"] != item["irrelevant"]
    assert len({item["query"] for item in items}) == len(items)


def test_benchmark_fragments_are_real_corpus_chunks():
    texts = {doc["text"] for doc in corpus()}
    for item in json.loads(BENCHMARK.read_text(encoding="utf-8")):
        assert item["relevant"] in texts and item["irrelevant"] in texts


def test_corpus_has_50_plus_documents_from_the_knowledge_base():
    docs = corpus()
    assert len(docs) >= 50 and len({doc["id"] for doc in docs}) == len(docs)
    assert all({"id", "section", "product", "category", "title", "text"} <= set(doc) for doc in docs)
    articles = json.loads((ROOT / "data" / "knowledge_base.json").read_text(encoding="utf-8"))["articles"]
    by_id = {doc["id"]: doc for doc in docs}
    for article in articles:
        assert by_id[article["id"]]["text"] == article["text"]


# ---------------------------------------------------------------- скрипты
@pytest.fixture
def fake_default(monkeypatch, cache_path):
    """build_service скрипта — подставная модель с файлом кеша cache_path. Каждый вызов —
    как новый процесс: новый бэкенд, тот же файл."""
    backends: list[FakeBackend] = []

    def build_service():
        backends.append(FakeBackend())
        return EmbeddingService(backends[-1], cache=EmbeddingCache(cache_path))

    monkeypatch.setattr(emb, "build_service", build_service)
    emb.reset_default_service()
    yield backends
    emb.reset_default_service()
    from log_capture import quiet_logs

    quiet_logs()


def test_cli_second_run_makes_no_requests(fake_default, capsys):
    cli = load_script("embeddings_cli")
    assert cli.main(["тот же текст"]) == 0
    first = capsys.readouterr().out
    emb.reset_default_service()                                # перезапуск
    assert cli.main(["тот же текст"]) == 0
    second = capsys.readouterr().out
    assert "Вызов 1, embed_texts: 1 текст, из кеша 0 (память 0, диск 0), запросов к модели 1" in first
    assert "Вызов 2 (тот же процесс), embed_texts: 1 текст, из кеша 1 (память 1, диск 0), запросов к модели 0" in first
    assert "Вызов 1, embed_texts: 1 текст, из кеша 1 (память 0, диск 1), запросов к модели 0" in second
    assert '"event": "embeddings_request"' in first and '"event": "embeddings_request"' not in second
    assert [len(b.batches) for b in fake_default] == [1, 0]


def test_cli_modes(fake_default, capsys):
    cli = load_script("embeddings_cli")
    assert cli.main(["--benchmark", "--repeat", "1"]) == 0
    assert "нужный фрагмент ближе ложного:" in capsys.readouterr().out
    assert cli.main(["--corpus", "--repeat", "1"]) == 0
    assert "embed_documents (56 документов)" in capsys.readouterr().out
    assert cli.main(["--cache-info"]) == 0
    assert "openai:local:11434/bge-m3: векторов" in capsys.readouterr().out
    assert cli.main(["--clear-cache"]) == 0
    assert "Удалено записей:" in capsys.readouterr().out


def test_cli_reports_model_errors(monkeypatch, cache_path, capsys):
    def build_service():
        return EmbeddingService(FakeBackend(errors=[status_error(openai.NotFoundError, 404)]), cache=None)

    monkeypatch.setattr(emb, "build_service", build_service)
    emb.reset_default_service()
    try:
        assert load_script("embeddings_cli").main(["текст"]) == 2
        assert "ollama pull bge-m3" in capsys.readouterr().err
    finally:
        emb.reset_default_service()
        from log_capture import quiet_logs

        quiet_logs()


class KeywordBackend(FakeBackend):
    """Близость по общим ключевым словам. Как у E5: без префикса к вектору добавляется «шум» —
    своё направление у каждого текста, и близость вопроса к нужному фрагменту падает."""

    WORDS = ("пароль", "аутентификатор", "спам", "счёт", "ключ", "429", "удал", "приоритет", "письм", "войти")
    NOISE = 1000

    def embed(self, texts):
        self.batches.append(list(texts))
        vectors = []
        for text in texts:
            lower = text.lower()
            vector = [float(lower.count(word)) for word in self.WORDS] + [0.0] * self.NOISE
            if not lower.startswith(("query: ", "passage: ")):
                bucket = int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:4], "little") % self.NOISE
                vector[len(self.WORDS) + bucket] = 2.0
            vectors.append(vector)
        return vectors


def test_benchmark_metrics_and_prefix_smoke():
    bench = load_script("embeddings_benchmark")
    items, docs = bench.load_items(), bench.load_corpus()
    service = EmbeddingService(KeywordBackend("intfloat/multilingual-e5-small"), cache=None)
    result = bench.evaluate(service, items, docs)
    assert result["docs"] == len(docs) and result["dim"] == len(KeywordBackend.WORDS) + KeywordBackend.NOISE
    assert 0 <= result["pair_accuracy"] <= 1 and 0 < result["mrr"] <= 1 and len(result["pairs"]) == len(items)
    assert result["prefixes"]["family"] == "e5"
    assert result["without_prefixes"]["mean_rel"] < result["mean_rel"]   # у этой заглушки без префиксов ниже
    assert result["query_without_prefix"]["mean_rel"] < result["mean_rel"]
    assert len(result["query_without_prefix"]["rel_by_pair"]) == len(items)
    table = bench.markdown([result], [{"spec": "st:x", "reason": "нет пакета"}])
    assert "вопрос без префикса, база с префиксом" in table and "без префиксов везде" in table
    assert "st:x: нет пакета" in table


def test_benchmark_rank_and_score():
    bench = load_script("embeddings_benchmark")
    docs = [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]
    assert bench.rank_of(1, [1.0, 0.0], docs) == 2 and bench.rank_of(0, [1.0, 0.0], docs) == 1
    items = [{"query": "q", "relevant": "a", "irrelevant": "c"}]
    scored = bench.score([[1.0, 0.0]], docs, items, {"a": 0, "b": 1, "c": 2})
    assert scored["pair_accuracy"] == 1 and scored["hit_at_1"] == 1 and scored["mrr"] == 1
    assert scored["pairs"][0]["margin"] == 1.0


def test_lexical_baseline_runs_without_models(tmp_path, capsys):
    bench = load_script("embeddings_benchmark")
    output = tmp_path / "bench.json"
    assert bench.main(["--models", "baseline:trigrams,unknown:x", "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    [result] = report["results"]
    assert result["model"] == "baseline:lexical/char-trigrams" and report["skipped"][0]["spec"] == "unknown:x"
    assert result["namespace"] == "lexical" and "without_prefixes" not in result      # у бейзлайна префиксов нет
    assert "hit@1" in capsys.readouterr().out
    from log_capture import quiet_logs

    quiet_logs()


def test_cost_arithmetic():
    cost = load_script("indexing_cost")
    data = cost.report([{"text": "x" * 100}, {"text": "y" * 300}], ["q" * 10], len, True,
                       [{"model": "openai:local:11434/bge-m3", "docs_per_s": 50.0, "query_ms_median": 40.0},
                        {"model": "baseline:lexical/char-trigrams", "docs_per_s": 1000.0}])
    small = data["prices"]["text-embedding-3-small"]
    assert data["tokens"] == 400 and data["tokens_avg"] == 200
    assert small["corpus"] == pytest.approx(400 * 0.02 / 1e6) and small["corpus_batch"] == small["corpus"] / 2
    assert small["100000"] == pytest.approx(200 * 100_000 * 0.02 / 1e6)        # 0.40 $
    assert data["storage"]["bge-m3"]["100000_bytes"] == 1024 * 4 * 100_000
    assert len(data["local"]) == 2 and data["local"][0]["100000_s"] == 2000
    models = ["openai:local:11434/bge-m3", "sentence-transformers:hf/e5", "openai:openrouter.ai/x",
              "baseline:lexical/char-trigrams",
              "openai@local:11434/bge-m3", "openai@openrouter.ai/x"]          # подписи JSON первого прогона
    assert [cost.is_local_result({"model": model}) for model in models] == [True, True, False, False, True, False]
    assert cost.is_local_result({"model": "что угодно", "namespace": "local:11434"})
    assert cost.money(0.000107) == "0.000107 $" and cost.money(1.25) == "1.25 $" and cost.money(1234.5) == "1 234.50 $"
    assert cost.duration(30) == "30 с" and cost.duration(600) == "10 мин" and cost.duration(36_000) == "10.0 ч"
    assert "| text-embedding-3-large | 0.13 |" in cost.render(data)
