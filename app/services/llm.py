"""
LLMService (блоки 3.4, 3.6) — слой работы с моделью для HTTP-сервиса.

Сервис создаётся на каждый запрос (app/deps/providers.py), а долгоживущие объекты
получает готовыми из app.state: клиент AsyncOpenAI, подключение к Redis и семафор.

- complete() — ответ целиком с кешем в Redis: ключ "chat:" + sha256 от параметров
  запроса без user_id/session_id/stream, TTL — CACHE_TTL_SECONDS;
- stream() — фрагменты ответа по мере генерации, без кеша, с итоговым usage;
- ошибки OpenAI SDK переводятся в доменные исключения из app/core/exceptions.py.

Из блока 3.3 перенесены: повторы средствами SDK (max_retries, без tenacity поверх —
иначе попытки перемножаются), stream_options.include_usage и семафор, созданный один
раз: теперь он живёт в lifespan и ограничивает запросы всего сервиса (Bulkhead).
Ошибки Redis запрос не роняют: кеш — ускорение, а не обязательная часть ответа.

Наблюдаемость (блок 3.6): каждый вызов оборачивается в span llm.chat с атрибутами
GenAI (gen_ai.request.model, gen_ai.usage.input_tokens, ...); span OpenAI SDK от
OpenInference становится его дочерним — в нём вход, ответ и токены. В лог пишется
строка llm_request_completed (модель, токены, latency_ms, finish_reason) —
request_id добавляется из contextvars. Сырого текста в логе нет: только
prompt_hash и prompt_preview после маскирования PII. С PII_PRESIDIO=true имена и
адреса в prompt_preview маскирует Presidio — фоновой задачей параллельно с вызовом
модели (app/observability/pii_presidio.py).

Ассистент техподдержки (блок 3.7): если в запросе нет system, сообщения для модели
собирает app/services/prompts.py — промпт ассистента со статьями руководства, найденными
по вопросу. Ключ кеша считается по итоговым сообщениям, поэтому правка промпта или
статьи сама «сбрасывает» старые ответы. В строке лога — версия промпта, найденные
статьи (kb_articles) и стоимость вызова по каталогу цен (cost_usd).

Проверки вокруг модели (app/services/guardrails.py, блок 3.7): просьба показать или
отменить инструкции получает готовый отказ без вызова модели (model="guardrail",
finish_reason="content_filter"); в /chat ответ, где дословно есть правила системного
промпта, заменяется тем же отказом. Оба случая — строка лога llm_guard_blocked с
причиной и атрибут guard.blocked на span.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import openai
import structlog
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from redis.exceptions import RedisError

from app.core.config import Settings
from app.core.exceptions import (
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.observability.logging import get_logger
from app.observability.pii import PREVIEW_CHARS, prompt_hash, prompt_preview, redact_pii
from app.observability.pii_presidio import NameRedactor
from app.schemas.chat import ChatDelta, ChatRequest, ChatResponse, Usage
from app.schemas.models import estimate_cost
from app.services.guardrails import leaks_instructions
from app.services.prompts import PreparedPrompt, build_messages

log = get_logger()
tracer = trace.get_tracer("multapi.llm")

# Поля, которые не влияют на ответ модели и не должны дробить кеш.
CACHE_KEY_EXCLUDE = {"user_id", "session_id", "stream"}
CACHE_ERRORS = (RedisError, OSError, asyncio.TimeoutError)
GUARD_MODEL = "guardrail"         # «модель» ответа, который дала проверка, а не LLM
GUARD_FINISH = "content_filter"   # как у OpenAI, когда ответ отфильтрован


def _retry_after(exc: openai.APIStatusError) -> int | None:
    value = exc.response.headers.get("retry-after") if exc.response is not None else None
    try:
        return max(1, round(float(value))) if value else None
    except ValueError:
        return None


@contextlib.contextmanager
def provider_errors(model: str) -> Iterator[None]:
    """Переводит исключения OpenAI SDK в доменные. Порядок важен: таймаут —
    подкласс ошибки соединения, а все HTTP-ошибки — подклассы APIStatusError."""
    try:
        yield
    except openai.RateLimitError as exc:
        raise LLMRateLimitError(retry_after=_retry_after(exc)) from exc
    except openai.APITimeoutError as exc:
        raise LLMTimeoutError() from exc
    except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
        raise LLMAuthError() from exc
    except openai.APIConnectionError as exc:
        raise LLMUnavailableError() from exc
    except openai.NotFoundError as exc:
        raise LLMError(f"Модель «{model}» не найдена у провайдера.") from exc
    except openai.APIStatusError as exc:
        raise LLMError(f"Провайдер LLM вернул ошибку {exc.status_code}.") from exc
    except openai.OpenAIError as exc:
        raise LLMError() from exc


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _trace_id(span: Span) -> str | None:
    ctx = span.get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


def _user_text(req: ChatRequest) -> str:
    """Вопрос пользователя — последнее сообщение с ролью user."""
    return next((m.content for m in reversed(req.messages) if m.role == "user"), req.messages[-1].content)


class LLMService:
    def __init__(
        self,
        openai_client: Any,
        cache: Any | None,
        settings: Settings,
        *,
        limiter: asyncio.Semaphore | None = None,
        redactor: NameRedactor | None = None,
    ) -> None:
        self.openai = openai_client
        self.cache = cache
        self.settings = settings
        self.limiter = limiter or asyncio.Semaphore(settings.llm.max_concurrency)
        self.redactor = redactor

    # ------------------------------------------------------------------ #
    def _resolve(self, req: ChatRequest) -> ChatRequest:
        """Подставляет модель по умолчанию — до расчёта ключа кеша, иначе после
        смены LLM__DEFAULT_MODEL из кеша вернулся бы ответ старой модели."""
        return req if req.model else req.model_copy(update={"model": self.settings.llm.default_model})

    @staticmethod
    def cache_key(req: ChatRequest, messages: list[dict[str, str]] | None = None) -> str:
        """messages — итоговые сообщения для модели (с промптом ассистента и статьями)."""
        payload = req.model_dump(mode="json", exclude=CACHE_KEY_EXCLUDE)
        if messages is not None:
            payload["messages"] = messages
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return "chat:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def _params(req: ChatRequest, prompt: PreparedPrompt) -> dict[str, Any]:
        return {
            "model": req.model,
            "messages": prompt.messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }

    async def _cache_get(self, key: str) -> str | None:
        if self.cache is None:
            return None
        try:
            return await self.cache.get(key)
        except CACHE_ERRORS as exc:
            log.warning("cache_unavailable", op="get", error=repr(exc))
            return None

    async def _cache_set(self, key: str, value: str) -> None:
        if self.cache is None:
            return
        try:
            # SET с EX, а не SETEX: в redis-py 8 setex() объявлен устаревшим (блок 3.7).
            await self.cache.set(key, value, ex=self.settings.cache_ttl_seconds)
        except CACHE_ERRORS as exc:
            log.warning("cache_unavailable", op="set", error=repr(exc))

    # ------------------------------------------------------------------ #
    # Наблюдаемость
    def _log_fields(self, req: ChatRequest, prompt: PreparedPrompt, *,
                    stream: bool) -> tuple[dict[str, Any], asyncio.Task | None]:
        """Поля строки лога о вызове модели. С Presidio prompt_preview считается фоновой
        задачей (её возвращаем вторым значением) и подставляется в _apply_preview."""
        raw = _user_text(req)
        fields = {"model": req.model, "stream": stream, "prompt_hash": prompt_hash(raw), "prompt_preview": None,
                  "prompt_version": prompt.version, "kb_articles": list(prompt.article_ids)}
        if self.redactor is None:
            fields["prompt_preview"] = prompt_preview(raw)
            return fields, None
        return fields, asyncio.create_task(self.redactor.preview(raw))

    @staticmethod
    async def _apply_preview(fields: dict[str, Any], task: asyncio.Task | None) -> None:
        """Ждёт фоновое маскирование. Ошибка Presidio — превью не пишем вовсе: текст
        только после regex мог бы оставить в логе имена."""
        if task is None:
            return
        try:
            fields["prompt_preview"] = await task
        except Exception as exc:  # noqa: BLE001 — любая ошибка NER не должна ронять запрос
            log.warning("presidio_failed", error=repr(exc)[:300])

    @staticmethod
    def _describe_root(root: Span, fields: dict[str, Any], answer: str | None) -> None:
        """Вход и выход — на корневой span запроса (HTTP от FastAPI): по нему Phoenix
        заполняет колонки input/output в списках трейсов и сессий. Только маскированный
        текст той же длины, что и в логе; полный текст — в дочернем span ChatCompletion."""
        if not root.is_recording():
            return
        if fields.get("prompt_preview"):
            root.set_attribute("input.value", fields["prompt_preview"])
        if answer:
            root.set_attribute("output.value", redact_pii(answer)[:PREVIEW_CHARS])

    @staticmethod
    def _preview_if_ready(fields: dict[str, Any], task: asyncio.Task | None) -> None:
        """Без ожидания (поток прерван): готово — подставляем, нет — превью не пишем."""
        if task is not None and task.done() and not task.cancelled() and task.exception() is None:
            fields["prompt_preview"] = task.result()

    @staticmethod
    def _span_attributes(req: ChatRequest, fields: dict[str, Any]) -> dict[str, Any]:
        attributes = {
            "openinference.span.kind": "CHAIN",          # как показывать span в Phoenix
            "gen_ai.operation.name": "chat",
            "gen_ai.system": "openai",
            "gen_ai.request.model": req.model,
            "gen_ai.request.temperature": req.temperature,
            "gen_ai.request.max_tokens": req.max_tokens,
            "prompt.hash": fields["prompt_hash"],
            "prompt.version": fields["prompt_version"],
            "kb.articles": fields["kb_articles"] or None,
            "request.id": structlog.contextvars.get_contextvars().get("request_id"),
            # Атрибуты OpenInference: по session.id Phoenix собирает трейсы диалога на
            # вкладке Sessions.
            "user.id": req.user_id,
            "session.id": req.session_id,
        }
        return {key: value for key, value in attributes.items() if value is not None}

    @staticmethod
    def _record_completion(span: Span, fields: dict[str, Any], *, model: str | None, usage: Usage,
                           finish_reason: str | None, started: float, **extra: Any) -> None:
        span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", usage.completion_tokens)
        if model:
            span.set_attribute("gen_ai.response.model", model)
        if finish_reason:
            span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])
        log.info(
            "llm_request_completed", **fields,
            response_model=model, input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens,
            cost_usd=estimate_cost(fields["model"], usage),
            latency_ms=_elapsed_ms(started), finish_reason=finish_reason, cached=False,
            trace_id=_trace_id(span), **extra,
        )

    @staticmethod
    def _record_failure(span: Span, fields: dict[str, Any], exc: LLMError, started: float) -> None:
        span.set_attribute("error.type", exc.code)
        span.set_status(Status(StatusCode.ERROR, exc.code))
        log.warning(
            "llm_request_failed", **fields, error=exc.code, cause=repr(exc.__cause__)[:300],
            latency_ms=_elapsed_ms(started), trace_id=_trace_id(span),
        )

    async def _blocked_response(self, span: Span, root: Span, fields: dict[str, Any],
                                preview_task: asyncio.Task | None, prompt: PreparedPrompt,
                                started: float) -> ChatResponse:
        """Запрос остановлен проверкой до модели: готовый отказ, токены не тратятся."""
        response = ChatResponse(content=prompt.refusal or "", model=GUARD_MODEL, usage=Usage(),
                                finish_reason=GUARD_FINISH)
        span.set_attribute("guard.blocked", prompt.blocked or "")
        await self._apply_preview(fields, preview_task)
        self._describe_root(root, fields, response.content)
        log.warning("llm_guard_blocked", **fields, reason=prompt.blocked,
                    latency_ms=_elapsed_ms(started), trace_id=_trace_id(span))
        return response

    @staticmethod
    def _guard_output(span: Span, fields: dict[str, Any], prompt: PreparedPrompt,
                      response: ChatResponse) -> ChatResponse:
        """Ответ модели с дословными правилами системного промпта заменяется отказом."""
        if not prompt.refusal or not prompt.messages:
            return response
        if not leaks_instructions(response.content, prompt.messages[0]["content"]):
            return response
        span.set_attribute("guard.blocked", "prompt_leak")
        log.warning("llm_guard_blocked", **fields, reason="prompt_leak", trace_id=_trace_id(span))
        return response.model_copy(update={"content": prompt.refusal, "finish_reason": GUARD_FINISH})

    # ------------------------------------------------------------------ #
    async def complete(self, req: ChatRequest) -> ChatResponse:
        req = self._resolve(req)
        prompt = build_messages(req, self.settings.support)
        fields, preview_task = self._log_fields(req, prompt, stream=False)
        started = time.perf_counter()
        root = trace.get_current_span()   # span HTTP-запроса, если трейсинг включён
        with tracer.start_as_current_span("llm.chat", attributes=self._span_attributes(req, fields)) as span:
            if prompt.blocked:
                return await self._blocked_response(span, root, fields, preview_task, prompt, started)
            key = self.cache_key(req, prompt.messages)
            cached = await self._cache_get(key)
            if cached is not None:
                try:
                    response = ChatResponse.model_validate_json(cached).model_copy(update={"cached": True})
                except ValueError:
                    log.warning("cache_entry_invalid", key=key)
                else:
                    span.set_attribute("cache.hit", True)
                    await self._apply_preview(fields, preview_task)
                    self._describe_root(root, fields, response.content)
                    log.info("llm_cache_hit", **fields, latency_ms=_elapsed_ms(started),
                             cached=True, trace_id=_trace_id(span))
                    return response
            span.set_attribute("cache.hit", False)

            try:
                async with self.limiter:
                    with provider_errors(req.model or ""):
                        raw = await self.openai.chat.completions.create(**self._params(req, prompt))
            except LLMError as exc:
                await self._apply_preview(fields, preview_task)
                self._record_failure(span, fields, exc, started)
                raise
            response = ChatResponse.from_openai(raw)
            await self._apply_preview(fields, preview_task)
            response = self._guard_output(span, fields, prompt, response)
            self._describe_root(root, fields, response.content)
            self._record_completion(span, fields, model=response.model, usage=response.usage,
                                    finish_reason=response.finish_reason, started=started)

        # Пустой ответ (например, модель сразу упёрлась в лимит) не кешируем.
        if response.content:
            await self._cache_set(key, response.model_dump_json())
        return response

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]:
        """Фрагменты ответа по мере генерации; последний кадр — usage. Без кеша."""
        req = self._resolve(req)
        prompt = build_messages(req, self.settings.support)
        fields, preview_task = self._log_fields(req, prompt, stream=True)
        started = time.perf_counter()
        root = trace.get_current_span()   # первый шаг генератора выполняется в обработчике запроса
        if prompt.blocked:
            with tracer.start_as_current_span("llm.chat", attributes=self._span_attributes(req, fields)) as span:
                response = await self._blocked_response(span, root, fields, preview_task, prompt, started)
            yield ChatDelta(content=response.content)
            yield ChatDelta(usage=response.usage)
            return
        # Span не делаем «текущим» на всё время генератора: между yield код идёт в другом
        # контексте. Текущим он становится только на вызов create(), чтобы span
        # OpenInference стал дочерним.
        span = tracer.start_span("llm.chat", attributes=self._span_attributes(req, fields))
        span.set_attribute("cache.hit", False)
        usage, finish_reason, ttft_ms = Usage(), None, None
        head = ""   # начало ответа — для output.value корневого span
        try:
            async with self.limiter:   # слот держится, пока идёт генерация
                with provider_errors(req.model or ""):
                    token = otel_context.attach(trace.set_span_in_context(span))
                    try:
                        stream = await self.openai.chat.completions.create(
                            **self._params(req, prompt), stream=True, stream_options={"include_usage": True}
                        )
                    finally:
                        otel_context.detach(token)
                    try:
                        async for chunk in stream:
                            if chunk.choices:
                                choice = chunk.choices[0]
                                text = choice.delta.content
                                if text:
                                    ttft_ms = ttft_ms or _elapsed_ms(started)
                                    if len(head) < 2 * PREVIEW_CHARS:
                                        head += text
                                    yield ChatDelta(content=text)
                                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                            if getattr(chunk, "usage", None):
                                usage = Usage.from_openai(chunk.usage)
                                yield ChatDelta(usage=usage)
                    finally:
                        # Клиент ушёл или поток оборвался — закрываем соединение с провайдером,
                        # чтобы модель не генерировала ответ впустую.
                        close = getattr(stream, "close", None)
                        if close is not None:
                            await close()
            await self._apply_preview(fields, preview_task)
            self._describe_root(root, fields, head)
            self._record_completion(span, fields, model=req.model, usage=usage,
                                    finish_reason=finish_reason, started=started, ttft_ms=ttft_ms)
        except LLMError as exc:
            await self._apply_preview(fields, preview_task)
            self._record_failure(span, fields, exc, started)
            raise
        except (GeneratorExit, asyncio.CancelledError):
            self._preview_if_ready(fields, preview_task)
            span.set_attribute("stream.cancelled", True)
            log.info("llm_stream_cancelled", **fields, latency_ms=_elapsed_ms(started), trace_id=_trace_id(span))
            raise
        finally:
            span.end()
