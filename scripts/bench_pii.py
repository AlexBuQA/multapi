"""
Сколько стоит маскирование PII (блок 3.6, задача 5): regex против Presidio.

Нужны пакеты Presidio и русская модель spaCy (в requirements.txt их нет):
    pip install presidio-analyzer presidio-anonymizer
    python -m spacy download ru_core_news_md
    python scripts/bench_pii.py

Колонки таблицы:
- regex — redact_pii по всему тексту (то, что сервис делает всегда);
- Presidio, весь текст — если бы NER прогонялся по всему промпту;
- превью с Presidio — как в сервисе: regex по всему тексту, Presidio только по первым
  WINDOW_CHARS символам (в лог идут 120).
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.observability.pii import redact_pii  # noqa: E402
from app.observability.pii_presidio import NameRedactor  # noqa: E402

SAMPLE = ("Добрый день! Я Мария Сергеевна Кузнецова из Новосибирска, email maria.k@example.ru, "
          "тел. +7 (913) 555-12-34. Не могу войти в личный кабинет после смены пароля. ")
LENGTHS = (120, 1_000, 5_000, 20_000)


def measure(fn, text: str, runs: int) -> float:
    """Медиана времени вызова, мс (после прогрева)."""
    fn(text)
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        fn(text)
        times.append((time.perf_counter() - started) * 1000)
    return statistics.median(times)


def main() -> None:
    started = time.perf_counter()
    redactor = NameRedactor()
    load_s = time.perf_counter() - started
    print(f"Загрузка Presidio и ru_core_news_md: {load_s:.1f} с\n")
    print("| Длина текста | regex | Presidio, весь текст | превью с Presidio (как в сервисе) |")
    print("|---:|---:|---:|---:|")
    for length in LENGTHS:
        text = (SAMPLE * (length // len(SAMPLE) + 1))[:length]
        runs = 20 if length <= 1_000 else 5
        regex_ms = measure(redact_pii, text, runs * 5)
        full_ms = measure(redactor.redact, text, runs)
        preview_ms = measure(redactor.preview_sync, text, runs)
        print(f"| {length:,} | {regex_ms:.2f} мс | {full_ms:.1f} мс | {preview_ms:.1f} мс |".replace(",", " "))
    print("\nПример превью:", redactor.preview_sync(SAMPLE))
    redactor.close()


if __name__ == "__main__":
    main()
