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
"""
from __future__ import annotations

import os
from dataclasses import dataclass

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


def transcribe(audio_path: str, *, client: RobustLLMClient | None = None) -> str:
    """Шаг 1–2: приём аудио и транскрипция через Whisper API."""
    abspath = validate_file(audio_path, kind="audio")
    client = client or RobustLLMClient()
    provider = _audio_provider()
    logger.info("Транскрипция (Whisper): %s", abspath)

    def _do(c):
        with open(abspath, "rb") as f:
            return c.audio.transcriptions.create(
                model=settings.audio.whisper_model, file=f, language="ru"
            )

    resp = client.call_with_retry(provider, _do)
    text = resp.text.strip()
    logger.info("Распознано: %s", text)
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
    cache.set(model, messages, temperature, answer)
    return answer


def text_to_speech(
    text: str, output_path: str, *,
    client: RobustLLMClient | None = None, voice: str | None = None,
) -> str:
    """Шаг 4: озвучка текста через TTS API -> аудиофайл."""
    client = client or RobustLLMClient()
    provider = _audio_provider()
    voice = voice or settings.audio.tts_voice
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    logger.info("Синтез речи (TTS): голос=%s -> %s", voice, output_path)

    def _do(c):
        with c.audio.speech.with_streaming_response.create(
            model=settings.audio.tts_model, voice=voice, input=text
        ) as response:
            response.stream_to_file(output_path)
        return output_path

    client.call_with_retry(provider, _do)
    client.usage.audio_cost_usd += (
        len(text) / 1_000_000 * settings.audio.price_tts_per_1m_chars
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
