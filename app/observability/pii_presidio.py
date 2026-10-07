"""
Опционально (блок 3.6, задача 5): имена и адреса через Microsoft Presidio поверх regex.

Regex из pii.py ловит то, что имеет формат: email, телефон, карту, ИНН, паспорт. Имя или
город формата не имеют — их находит NER-модель spaCy (ru_core_news_md), а Presidio
заменяет найденное на [PERSON] и [LOCATION].

Цена — время. Замеры scripts/bench_pii.py: загрузка модели — несколько секунд, разбор
текста растёт линейно с длиной (около 10 мс на 120 символов, 45 мс на 1000, 0,9 с на
20 000), regex на тех же текстах — доли миллисекунды. Поэтому:
- модель загружается один раз на старте (lifespan), а не на первом запросе;
- Presidio смотрит только на начало текста (WINDOW_CHARS): в лог всё равно попадают
  120 символов prompt_preview, так что длина промпта на стоимость не влияет;
- маскирование запускается фоновой задачей в отдельном потоке параллельно с вызовом
  модели (LLMService) и к моменту записи строки лога обычно уже готово — на время
  ответа оно не влияет;
- вызовы Presidio идут через один рабочий поток: потокобезопасность spaCy при
  параллельных вызовах не гарантируется.

Включается переменной PII_PRESIDIO=true. Зависимости ставятся отдельно и в Docker-образ
не входят (+~500 МБ):
    pip install presidio-analyzer presidio-anonymizer
    python -m spacy download ru_core_news_md
Если пакетов или модели нет, сервис пишет presidio_unavailable и работает на regex.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from app.observability.logging import get_logger
from app.observability.pii import PREVIEW_CHARS, redact_pii

SPACY_MODEL = "ru_core_news_md"
ENTITIES = ("PERSON", "LOCATION")
WINDOW_CHARS = 200   # с запасом к 120: имя на границе превью тоже должно найтись

log = get_logger()


class NameRedactor:
    """Presidio Analyzer + Anonymizer с русской моделью spaCy."""

    def __init__(self, model: str = SPACY_MODEL) -> None:
        import spacy
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig

        # Presidio сам скачивает отсутствующую модель — на старте сервиса это сюрприз,
        # поэтому проверяем заранее и просим поставить её явно.
        if not spacy.util.is_package(model):
            raise OSError(f"модель spaCy {model} не установлена: python -m spacy download {model}")
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "ru", "model_name": model}],
        })
        self.analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["ru"])
        self.anonymizer = AnonymizerEngine()
        self.operators = {name: OperatorConfig("replace", {"new_value": f"[{name}]"}) for name in ENTITIES}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="presidio")

    def redact(self, text: str) -> str:
        """Синхронно: имена и места в тексте -> [PERSON], [LOCATION]."""
        results = self.analyzer.analyze(text=text, language="ru", entities=list(ENTITIES))
        return self.anonymizer.anonymize(text=text, analyzer_results=results, operators=self.operators).text

    def preview_sync(self, raw: str) -> str:
        """prompt_preview с именами: regex по всему тексту, Presidio — по началу."""
        return self.redact(redact_pii(raw)[:WINDOW_CHARS])[:PREVIEW_CHARS]

    async def preview(self, raw: str) -> str:
        """То же в рабочем потоке — цикл событий не блокируется."""
        return await asyncio.get_running_loop().run_in_executor(self.executor, self.preview_sync, raw)

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


def load_redactor(model: str = SPACY_MODEL) -> NameRedactor | None:
    """Загружает Presidio; без пакетов или модели — None (маскирование только regex)."""
    started = time.perf_counter()
    try:
        redactor = NameRedactor(model)
        redactor.preview_sync("Прогрев модели: Иван Петров, Москва.")   # первый вызов заметно дольше
    except (ImportError, OSError) as exc:
        log.warning("presidio_unavailable", error=repr(exc)[:300],
                    note="маскирование только regex: pip install presidio-analyzer presidio-anonymizer "
                         f"&& python -m spacy download {model}")
        return None
    log.info("presidio_ready", model=model, load_ms=round((time.perf_counter() - started) * 1000))
    return redactor
