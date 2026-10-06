"""
Вариант Б — Голосовой пайплайн (Whisper + TTS).

СТАТУС: проработан теоретически, НЕ тестировался на реальных аудио-сервисах.
Whisper и TTS требуют платного доступа к OpenAI API (AUDIO_API_KEY); оплата
доступа к OpenAI из России затруднена, а Ollama аудио не поддерживает. Поэтому
код пайплайна написан целиком и проверен на моках (управляющая логика), но на
живых аудио-API не запускался. При наличии ключа его достаточно вписать в .env —
менять код не требуется.

Полный пайплайн:
  аудио -> Whisper (транскрипция) -> [классификация] -> LLM (ответ) -> TTS -> аудио.

ВАЖНО: Ollama НЕ предоставляет аудио-эндпоинты (Whisper/TTS). Поэтому аудио-шаги
идут на отдельный OpenAI-совместимый эндпоинт (AudioConfig, ключ AUDIO_API_KEY).
Если он не настроен — пайплайн поднимет понятную ошибку. Текстовый шаг (ответ)
по-прежнему идёт на локальный Ollama.

Учёт стоимости: Whisper — через UsageTracker.add_audio() (по длительности аудио),
TTS — через UsageTracker.add_tts() (по числу символов). Напрямую поля трекера
не изменяются, поэтому summary() включает все шаги пайплайна.
"""
from __future__ import annotations

import os
import wave
from dataclasses import dataclass
from typing import Any

from .cache import LLMCache
from .classifier import classify
from .config import AudioConfig, ProviderConfig, settings
from .prompts import build_assistant_messages
from .robust_client import RobustLLMClient
from .utils import InputFileError, get_logger, validate_file

logger = get_logger("voice")


class AudioNotConfiguredError(RuntimeError):
    """Аудио-эндпоинт (Whisper/TTS) не настроен — Ollama их не поддерживает."""


@dataclass
class VoiceResult:
    transcript: str
    category: str
    answer: str
    output_audio_path: str


def _audio_provider() -> ProviderConfig:
    """ProviderConfig-обёртка над AudioConfig для переиспользования retry/клиента."""
    a: AudioConfig = settings.audio
    if not a.configured:
        raise AudioNotConfiguredError(
            "Аудио-эндпоинт (Whisper/TTS) не настроен. Ollama их не поддерживает. "
            "Для варианта Б задайте в .env AUDIO_API_KEY (реальный ключ OpenAI) "
            "и при необходимости AUDIO_BASE_URL."
        )
    return ProviderConfig(
        name="audio",
        api_key_env=a.api_key_env,
        base_url=a.base_url,
        chat_model=a.whisper_model,
        vision_model=a.whisper_model,
    )


# Форматы, которые умеет отдавать TTS API (параметр response_format).
TTS_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "pcm"}


def _tts_format(output_path: str) -> str:
    """Формат TTS по расширению выходного файла (по умолчанию mp3)."""
    ext = os.path.splitext(output_path)[1].lower().lstrip(".")
    return ext if ext in TTS_FORMATS else "mp3"


def _audio_duration_seconds(resp: Any, audio_path: str) -> float:
    """
    Длительность распознанного аудио в секундах — база для стоимости Whisper.

    1) поле duration из ответа Whisper (response_format="verbose_json");
    2) иначе для WAV — по заголовку файла (стандартный модуль wave);
    3) иначе 0.0 с предупреждением в логе.
    """
    duration = getattr(resp, "duration", None)
    if isinstance(duration, (int, float)) and duration > 0:
        return float(duration)
    if audio_path.lower().endswith(".wav"):
        try:
            with wave.open(audio_path, "rb") as w:
                return w.getnframes() / float(w.getframerate())
        except (wave.Error, EOFError, OSError):
            pass
    logger.warning(
        "Не удалось определить длительность %s — стоимость Whisper учтена как 0.",
        audio_path,
    )
    return 0.0


def transcribe(audio_path: str, *, client: RobustLLMClient | None = None) -> str:
    """Шаг 1–2: приём аудио и транскрипция через Whisper API (+ учёт стоимости)."""
    abspath = validate_file(audio_path, kind="audio")
    client = client or RobustLLMClient()
    provider = _audio_provider()
    model = settings.audio.whisper_model
    # verbose_json возвращает duration (нужна для расчёта стоимости). Его
    # поддерживает whisper-1; для прочих моделей запрашиваем обычный json.
    response_format = "verbose_json" if model.startswith("whisper") else "json"
    logger.info("Транскрипция (Whisper): %s", abspath)

    def _do(c):
        with open(abspath, "rb") as f:
            return c.audio.transcriptions.create(
                model=model, file=f, language="ru", response_format=response_format
            )

    resp = client.call_with_retry(provider, _do)
    text = (resp.text or "").strip()

    seconds = _audio_duration_seconds(resp, abspath)
    client.usage.add_audio(
        seconds, settings.audio.price_whisper_per_min, label=f"whisper/{model}"
    )
    logger.info("Распознано (%.1f с): %s", seconds, text)
    return text


def answer_text(
    user_text: str,
    *,
    client: RobustLLMClient | None = None,
    cache: LLMCache | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Шаг 3: текст -> LLM -> ответ (промпт РРФО + few-shot, retry/fallback, кеш)."""
    client = client or RobustLLMClient()
    cache = cache or LLMCache(ttl=settings.cache_ttl)
    messages = build_assistant_messages(user_text, history)
    model = settings.primary().chat_model
    temperature = 0.3

    cached = cache.get(model, messages, temperature)
    if cached is not None:
        logger.info("Ответ из кеша (hit).")
        return cached

    answer = client.chat(messages, temperature=temperature, max_tokens=400, label="voice-llm")
    # Заглушку при недоступности всех провайдеров не кешируем.
    if answer != client.USER_FACING_FAILURE:
        cache.set(model, messages, temperature, answer)
    return answer


def text_to_speech(
    text: str, output_path: str, *,
    client: RobustLLMClient | None = None, voice: str | None = None,
) -> str:
    """
    Шаг 4: озвучка текста через TTS API -> аудиофайл (+ учёт стоимости).

    Формат аудио выбирается по расширению output_path (.mp3, .wav, .flac, ...),
    чтобы содержимое файла всегда соответствовало его расширению.
    """
    client = client or RobustLLMClient()
    provider = _audio_provider()
    voice = voice or settings.audio.tts_voice
    fmt = _tts_format(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    logger.info("Синтез речи (TTS): голос=%s формат=%s -> %s", voice, fmt, output_path)

    def _do(c):
        with c.audio.speech.with_streaming_response.create(
            model=settings.audio.tts_model, voice=voice, input=text,
            response_format=fmt,
        ) as response:
            response.stream_to_file(output_path)
        return output_path

    client.call_with_retry(provider, _do)
    client.usage.add_tts(
        len(text), settings.audio.price_tts_per_1m_chars,
        label=f"tts/{settings.audio.tts_model}",
    )
    return output_path


def run_pipeline(
    audio_path: str,
    output_path: str = "outputs/answer.mp3",
    *,
    client: RobustLLMClient | None = None,
    cache: LLMCache | None = None,
) -> VoiceResult:
    """Полный голосовой пайплайн: голосовой вопрос -> ответ голосом (+классификация)."""
    client = client or RobustLLMClient()
    cache = cache or LLMCache(ttl=settings.cache_ttl)

    transcript = transcribe(audio_path, client=client)
    category = classify(transcript, client=client)  # классификатор на Ollama
    answer = answer_text(transcript, client=client, cache=cache)
    out = text_to_speech(answer, output_path, client=client)
    return VoiceResult(
        transcript=transcript, category=category, answer=answer, output_audio_path=out
    )
