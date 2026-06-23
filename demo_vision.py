"""
Демо варианта А — анализ трёх изображений разного типа (фото, скриншот, график).

Покрывает критерий ДЗ 2.6 «Демо: 3 изображения разного типа» и показывает:
- единый Vision-пайплайн на base64 (через локальный Ollama);
- работу кеша (повторный запрос -> hit);
- трекинг usage и hit rate.

ВАЖНО: нужна vision-модель (SUPPORT_VISION_MODEL, например llama3.2-vision):
    ollama pull llama3.2-vision

Запуск:
    python demo_vision.py
"""
from __future__ import annotations

from src.cache import LLMCache
from src.config import settings
from src.robust_client import RobustLLMClient
from src.vision import analyze_image

CASES = [
    ("samples/photo.jpg", "Что изображено на фото? Опиши сцену."),
    ("samples/screenshot.png", "Что это за экран и есть ли на нём ошибка?"),
    ("samples/chart.png", "Что показывает график и какой квартал лучший?"),
]


def main() -> None:
    client = RobustLLMClient()
    cache = LLMCache(ttl=settings.cache_ttl)
    p = settings.primary()
    print(f"Провайдер: {p.name} | base_url: {p.base_url} | vision-модель: {p.vision_model}\n")

    for path, question in CASES:
        print(f"### {path}\nВопрос: {question}")
        answer = analyze_image(path, question, client=client, cache=cache)
        print(f"Ответ:\n{answer}\n{'-' * 60}")

    print("\n>>> Повтор первого запроса (ожидаем cache hit)")
    analyze_image(CASES[0][0], CASES[0][1], client=client, cache=cache)

    print(f"\nUsage: {client.usage.summary()}")
    print(f"Кеш: {cache.stats()}")


if __name__ == "__main__":
    main()
