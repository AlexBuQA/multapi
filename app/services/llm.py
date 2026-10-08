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

Защитный слой (app/services/security/, блок 3.8), если SECURITY__ENABLED не false:
- до модели — screen_messages: каждое сообщение проверяет validate_input (длина,
  скрытые символы, закодированные вставки, шаблоны инъекции). Последний вопрос или
  system не прошли — готовый отказ без вызова модели, как в блоке 3.7
  (_blocked_response); прежние сообщения истории, не прошедшие проверку, выбрасываются.
  Проверка идёт первой, до сборки промпта и маскирования;
- в сообщения для модели добавляется системное сообщение с канарейкой (app.state.canary);
- после модели — filter_output: метка, начало или правила системного промпта, роль
  джейлбрейка -> ответ заменяется отказом (_filter_output); персональные данные в ответе
  маскируются. В потоке то же делает StreamGuard, придерживая последние 80 символов;
- в строке лога llm_request_completed — answer_preview: начало ответа после маскирования.
С SECURITY__ENABLED=false ничего из этого нет (прогон garak baseline), а ключ кеша
получает пометку guard=off — ответы «голого» сервиса не попадут к защищённому.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import openai
import structlog
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from redis.exceptions import RedisError

from app.core.config import Settings
from app.core.exceptions import (
    LLMAuthError,
    LLMContentFiltered,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.observability.logging import get_logger
from app.observability.pii import PREVIEW_CHARS, prompt_hash, prompt_preview, redact_pii
from app.observability.pii_presidio import NameRedactor
from app.schemas.chat import ChatDelta, ChatRequest, ChatResponse, Message, Usage
from app.schemas.models import estimate_cost
from app.services.prompts import PROMPT_VERSION, PreparedPrompt, build_messages
from app.services.security import refusal_for, screen_messages
from app.services.security.canary import with_canary
from app.services.security.input_validator import ValidationResult
from app.services.security.output_filter import OutputBlocked, StreamGuard, filter_output

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
    подкласс ошибки соединения, а все HTTP-ошибки — подклассы APIStatusError.

    Ошибки httpx — для потока: обрыв соединения или пауза дольше таймаута, пока ответ
    читается по частям, SDK не оборачивает в свои исключения (блок 4.1)."""
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
    except httpx.TimeoutException as exc:
        raise LLMTimeoutError() from exc
    except httpx.HTTPError as exc:
        raise LLMUnavailableError() from exc


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _trace_id(span: Span) -> str | None:
    ctx = span.get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


def _user_text(req: ChatRequest) -> str:
    """Вопрос пользователя — последнее сообщение с ролью user."""
    return next((m.content for m in reversed(req.messages) if m.role == "user"), req.messages[-1].content)


def _preview(text: str | None) -> str | None:
    return redact_pii(text)[:PREVIEW_CHARS] if text else None


class LLMService:
    def __init__(
        self,
        openai_client: Any,
        cache: Any | None,
        settings: Settings,
        *,
        limiter: asyncio.Semaphore | None = None,
        redactor: NameRedactor | None = None,
        canary: str | None = None,
    ) -> None:
        self.openai = openai_client
        self.cache = cache
        self.settings = settings
        self.limiter = limiter or asyncio.Semaphore(settings.llm.max_concurrency)
        self.redactor = redactor
        self.guarded = settings.security.enabled
        self.canary = canary if self.guarded else None

    # ------------------------------------------------------------------ #
    def _resolve(self, req: ChatRequest) -> ChatRequest:
        """Подставляет модель по умолчанию — до расчёта ключа кеша, иначе после
        смены LLM__DEFAULT_MODEL из кеша вернулся бы ответ старой модели."""
        return req if req.model else req.model_copy(update={"model": self.settings.llm.default_model})

    @staticmethod
    def cache_key(req: ChatRequest, messages: list[dict[str, str]] | None = None, *, guarded: bool = True) -> str:
        """messages — итоговые сообщения для модели (с промптом ассистента и статьями), но
        без канарейки: она меняется при каждом запуске. guarded=False (SECURITY__ENABLED=false)
        — отдельный ключ: ответ без проверки не должен попасть к защищённому сервису."""
        payload = req.model_dump(mode="json", exclude=CACHE_KEY_EXCLUDE)
        if messages is not None:
            payload["messages"] = messages
        if not guarded:
            payload["guard"] = "off"
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return "chat:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _params(self, req: ChatRequest, prompt: PreparedPrompt) -> dict[str, Any]:
        return {
            "model": req.model,
            "messages": with_canary(prompt.messages, self.canary),
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }

    # ------------------------------------------------------------------ #
    # Защитный слой (блок 3.8)
    def _screen(self, req: ChatRequest) -> tuple[ChatRequest, ValidationResult | None]:
        """Запрос после проверки истории и причина отказа (или None)."""
        if not self.guarded:
            return req, None
        screened = screen_messages([m.model_dump() for m in req.messages], self.settings.security.max_input_chars)
        if not screened.verdict.ok:
            return req, screened.verdict
        if screened.dropped:
            log.warning("llm_history_screened", dropped=len(screened.dropped), rules=list(screened.dropped))
            req = req.model_copy(update={"messages": [Message(**m) for m in screened.messages]})
        return req, None

    def _prepare(self, req: ChatRequest) -> tuple[ChatRequest, ValidationResult | None, PreparedPrompt]:
        """Проверка входа — до сборки промпта: отклонённый запрос не тратит время на поиск
        статей и маскирование. Для отказа нужен только признак режима ассистента."""
        req, verdict = self._screen(req)
        if verdict is None:
            return req, None, build_messages(req, self.settings.support)
        assistant = self.settings.support.enabled and not any(m.role == "system" for m in req.messages)
        return req, verdict, PreparedPrompt([], PROMPT_VERSION if assistant else None)

    def _refusal(self, rule: str, req: ChatRequest) -> str:
        return refusal_for(rule, self.settings.support.product_name,
                           length=max(len(m.content) for m in req.messages if m.role == "user") if rule == "length" else 0,
                           max_chars=self.settings.security.max_input_chars)

    @staticmethod
    def _system_prompt(prompt: PreparedPrompt) -> str | None:
        """Промпт ассистента, утечку которого проверяет фильтр; свой system клиента — не секрет."""
        return prompt.messages[0]["content"] if prompt.version is not None and prompt.messages else None

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
                           finish_reason: str | None, started: float, answer: str | None = None,
                           **extra: Any) -> None:
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
            answer_preview=_preview(answer), trace_id=_trace_id(span), **extra,
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
                                preview_task: asyncio.Task | None, req: ChatRequest,
                                verdict: ValidationResult, started: float) -> ChatResponse:
        """Запрос остановлен проверкой до модели: готовый отказ, токены не тратятся.
        Не HTTP 400, а ответ 200 с отказом (см. app/services/security/input_validator.py):
        клиент — чат поддержки, model="guardrail" и finish_reason="content_filter"
        отличают отказ от ответа модели."""
        response = ChatResponse(content=self._refusal(verdict.rule or "injection", req), model=GUARD_MODEL,
                                usage=Usage(), finish_reason=GUARD_FINISH)
        span.set_attribute("guard.blocked", verdict.rule or "")
        await self._apply_preview(fields, preview_task)
        self._describe_root(root, fields, response.content)
        log.warning("llm_guard_blocked", **fields, reason=verdict.rule, detail=verdict.reason,
                    answer_preview=_preview(response.content), latency_ms=_elapsed_ms(started),
                    trace_id=_trace_id(span))
        return response

    def _filter_output(self, span: Span, fields: dict[str, Any], prompt: PreparedPrompt,
                       req: ChatRequest, response: ChatResponse) -> ChatResponse:
        """Ответ модели с меткой, промптом или ролью джейлбрейка заменяется отказом;
        в остальных персональные данные маскируются."""
        if not self.guarded:
            return response
        try:
            content = filter_output(response.content, self._system_prompt(prompt), self.canary)
        except OutputBlocked as exc:
            span.set_attribute("guard.blocked", exc.rule)
            log.warning("llm_guard_blocked", **fields, reason=exc.rule, detail=exc.reason,
                        trace_id=_trace_id(span))
            return response.model_copy(update={"content": self._refusal(exc.rule, req),
                                               "finish_reason": GUARD_FINISH})
        if content != response.content:
            span.set_attribute("guard.pii_masked", True)
        return response.model_copy(update={"content": content})

    @staticmethod
    def _guard_chunk(span: Span, fields: dict[str, Any], guard: StreamGuard, text: str | None) -> str:
        """Фрагмент потока через StreamGuard; text=None — конец ответа. Утечка — поток
        обрывается ошибкой content_filter (кадр error в /chat/stream)."""
        try:
            return guard.feed(text) if text is not None else guard.finish()
        except OutputBlocked as exc:
            span.set_attribute("guard.blocked", exc.rule)
            log.warning("llm_guard_blocked", **fields, reason=exc.rule, detail=exc.reason,
                        trace_id=_trace_id(span))
            raise LLMContentFiltered() from exc

    # ------------------------------------------------------------------ #
    async def complete(self, req: ChatRequest) -> ChatResponse:
        req = self._resolve(req)
        req, verdict, prompt = self._prepare(req)          # защитный слой: до модели
        fields, preview_task = self._log_fields(req, prompt, stream=False)
        started = time.perf_counter()
        root = trace.get_current_span()   # span HTTP-запроса, если трейсинг включён
        with tracer.start_as_current_span("llm.chat", attributes=self._span_attributes(req, fields)) as span:
            if verdict is not None:
                return await self._blocked_response(span, root, fields, preview_task, req, verdict, started)
            key = self.cache_key(req, prompt.messages, guarded=self.guarded)
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
            response = self._filter_output(span, fields, prompt, req, response)   # защитный слой: после модели
            self._describe_root(root, fields, response.content)
            self._record_completion(span, fields, model=response.model, usage=response.usage,
                                    finish_reason=response.finish_reason, started=started,
                                    answer=response.content)

        # Пустой ответ (например, модель сразу упёрлась в лимит) не кешируем.
        if response.content:
            await self._cache_set(key, response.model_dump_json())
        return response

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatDelta]:
        """Фрагменты ответа по мере генерации; последний кадр — usage. Без кеша."""
        req = self._resolve(req)
        req, verdict, prompt = self._prepare(req)
        fields, preview_task = self._log_fields(req, prompt, stream=True)
        started = time.perf_counter()
        root = trace.get_current_span()   # первый шаг генератора выполняется в обработчике запроса
        if verdict is not None:
            with tracer.start_as_current_span("llm.chat", attributes=self._span_attributes(req, fields)) as span:
                response = await self._blocked_response(span, root, fields, preview_task, req, verdict, started)
            yield ChatDelta(content=response.content)
            yield ChatDelta(usage=response.usage)
            return
        guard = StreamGuard(self._system_prompt(prompt), self.canary) if self.guarded else None
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
                    got_usage = False
                    try:
                        async for chunk in stream:
                            if chunk.choices:
                                choice = chunk.choices[0]
                                text = choice.delta.content
                                if text and guard is not None:
                                    text = self._guard_chunk(span, fields, guard, text)
                                if text:
                                    ttft_ms = ttft_ms or _elapsed_ms(started)
                                    if len(head) < 2 * PREVIEW_CHARS:
                                        head += text
                                    yield ChatDelta(content=text)
                                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                            if getattr(chunk, "usage", None):
                                # Кадр usage — после всего текста: часть провайдеров шлёт usage
                                # в каждом фрагменте, и отдать придержанный хвост здесь было бы рано.
                                usage, got_usage = Usage.from_openai(chunk.usage), True
                        if guard is not None:                   # придержанный хвост ответа
                            tail = self._guard_chunk(span, fields, guard, None)
                            if tail:
                                ttft_ms = ttft_ms or _elapsed_ms(started)
                                head += tail if len(head) < 2 * PREVIEW_CHARS else ""
                                yield ChatDelta(content=tail)
                        if got_usage:
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
                                    finish_reason=finish_reason, started=started, ttft_ms=ttft_ms,
                                    answer=head)
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
