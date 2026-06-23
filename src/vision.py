"""
Вариант А — Анализ изображений (Vision API).

Пайплайн: путь к файлу -> валидация -> base64 (data-URL) -> Vision-запрос -> текст.

Работает через локальный Ollama по OpenAI-совместимому API. ВАЖНО: для анализа
изображений нужна vision-модель (например, llama3.2-vision), задаётся переменной
SUPPORT_VISION_MODEL. Текстовая llama3.2 изображения не воспринимает.

Vision-запрос идёт через тот же RobustLLMClient (retry + fallback + usage) и через
LLMCache, что и обычный чат.
"""
from __future__ import annotations

from .cache import LLMCache
from .config import settings
from .prompts import VISION_SYSTEM_PROMPT
from .robust_client import RobustLLMClient
from .utils import encode_image_data_url, get_logger, validate_file

logger = get_logger("vision")

DEFAULT_QUESTION = "Опиши изображение и сделай главный вывод."


def build_vision_messages(data_url: str, question: str) -> list[dict]:
    """system + (текст + картинка) в одном user-блоке (мультимодальный content)."""
    return [
        {"role": "system", "content": VISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]


def analyze_image(
    image_path: str,
    question: str = DEFAULT_QUESTION,
    *,
    client: RobustLLMClient | None = None,
    cache: LLMCache | None = None,
    temperature: float = 0.2,
) -> str:
    """Анализирует изображение и возвращает текстовый ответ."""
    abspath = validate_file(image_path, kind="image")
    client = client or RobustLLMClient()
    cache = cache or LLMCache(ttl=settings.cache_ttl)

    vision_model = settings.primary().vision_model
    logger.info("Анализ изображения: %s | модель: %s | вопрос: %s",
                abspath, vision_model, question)

    data_url = encode_image_data_url(abspath)
    messages = build_vision_messages(data_url, question)

    cached = cache.get(vision_model, messages, temperature)
    if cached is not None:
        logger.info("Ответ из кеша (hit).")
        return cached

    answer = client.chat(
        messages, temperature=temperature, max_tokens=600,
        label="vision", model_override=vision_model,
    )
    cache.set(vision_model, messages, temperature, answer)
    return answer
