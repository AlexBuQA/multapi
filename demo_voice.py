"""
Демо варианта Б — полный голосовой пайплайн: голосовой вопрос -> ответ голосом.

Шаги: Whisper (транскрипция) -> классификация (Ollama) -> LLM (ответ) -> TTS -> файл.

ВАЖНО: Ollama не поддерживает Whisper/TTS. Для этого демо нужен аудио-эндпоинт:
в .env задайте AUDIO_API_KEY (реальный ключ OpenAI). Если он не задан — демо
аккуратно сообщит об этом и завершится, не падая с трейсбеком.

Запуск:
    python demo_voice.py
"""
from __future__ import annotations

import os

from src.cache import LLMCache
from src.config import settings
from src.robust_client import RobustLLMClient
from src.voice import AudioNotConfiguredError, run_pipeline, text_to_speech

SAMPLE_QUESTION_TEXT = "Здравствуйте! Подскажите, как сбросить пароль от аккаунта?"
# Образец из репозитория (речь с текстом SAMPLE_QUESTION_TEXT). Если файл удалён,
# демо сгенерирует его заново через TTS — в том же формате WAV (по расширению).
SAMPLE_AUDIO = "samples/voice_question.wav"
OUTPUT_AUDIO = "outputs/answer.mp3"


def main() -> None:
    client = RobustLLMClient()
    cache = LLMCache(ttl=settings.cache_ttl)

    if not settings.audio.configured:
        print(
            "Вариант Б (голос): проработан теоретически, не тестировался.\n"
            "Whisper/TTS требуют платного доступа к OpenAI API (Ollama их не умеет),\n"
            "а оплата OpenAI из России затруднена — поэтому ключ AUDIO_API_KEY пуст.\n"
            "Чтобы запустить пайплайн, впишите AUDIO_API_KEY в .env (код менять не нужно).\n"
            "Вариант А (анализ изображений) полностью работает на локальном Ollama: "
            "python demo_vision.py"
        )
        return

    try:
        if not os.path.exists(SAMPLE_AUDIO):
            print(f"Генерирую образец вопроса через TTS: «{SAMPLE_QUESTION_TEXT}»")
            text_to_speech(SAMPLE_QUESTION_TEXT, SAMPLE_AUDIO, client=client)

        print(f"Вход: {SAMPLE_AUDIO}\nЗапускаю полный пайплайн...\n")
        result = run_pipeline(SAMPLE_AUDIO, OUTPUT_AUDIO, client=client, cache=cache)

        print(f"Транскрипция : {result.transcript}")
        print(f"Категория    : {result.category}")
        print(f"Ответ LLM    : {result.answer}")
        print(f"Аудио-ответ  : {result.output_audio_path}")
        print(f"\nUsage: {client.usage.summary()}")
    except AudioNotConfiguredError as exc:
        print(exc)


if __name__ == "__main__":
    main()
