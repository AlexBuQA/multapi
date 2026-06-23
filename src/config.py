"""
Конфигурация проекта.

Все секреты и переключатели читаются из переменных окружения (.env).
Хардкода ключей в коде нет.

Текущая сборка ориентирована на локальный Ollama по OpenAI-совместимому API:
  OPENAI_API_KEY=ollama
  OPENAI_BASE_URL=http://localhost:11434/v1
  SUPPORT_PRIMARY_MODEL=llama3.2
  SUPPORT_CLASSIFIER_MODEL=llama3.2

Важно про модальности Ollama:
- Чат и классификация работают на текстовой модели (llama3.2).
- Vision требует vision-модель (например, llama3.2-vision): задаётся
  переменной SUPPORT_VISION_MODEL.
- Whisper (транскрипция) и TTS (озвучка) Ollama НЕ поддерживает — для варианта Б
  нужен отдельный OpenAI-совместимый аудио-эндпоинт (см. AudioConfig).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


@dataclass(frozen=True)
class ProviderConfig:
    """Параметры одного провайдера, совместимого с OpenAI SDK."""

    name: str
    api_key_env: str               # имя переменной окружения с ключом
    base_url: str | None = None    # None => дефолтный endpoint OpenAI
    chat_model: str = ""           # текстовая модель
    vision_model: str = ""         # vision-модель (может совпадать с chat_model)
    # Цена за 1M токенов в USD (для трекинга стоимости). Для локального Ollama = 0.
    price_in_per_1m: float = 0.0
    price_out_per_1m: float = 0.0

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)

    @property
    def is_local(self) -> bool:
        url = self.base_url or ""
        return "localhost" in url or "127.0.0.1" in url


# Текстовые/vision-модели. По умолчанию vision = SUPPORT_VISION_MODEL,
# а если она не задана — берём primary (с оговоркой, что текстовая llama3.2
# изображения не анализирует).
_PRIMARY = _env("SUPPORT_PRIMARY_MODEL", "llama3.2")
_VISION = _env("SUPPORT_VISION_MODEL") or _PRIMARY

PROVIDERS: dict[str, ProviderConfig] = {
    # Локальный Ollama (OpenAI-совместимый API).
    "ollama": ProviderConfig(
        name="ollama",
        api_key_env="OPENAI_API_KEY",
        base_url=_env("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        chat_model=_PRIMARY,
        vision_model=_VISION,
        price_in_per_1m=0.0,
        price_out_per_1m=0.0,
    ),
    # Резервный облачный провайдер (по умолчанию вне цепочки — нужен ключ).
    "openrouter": ProviderConfig(
        name="openrouter",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        chat_model=_env("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini"),
        vision_model=_env("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini"),
        price_in_per_1m=0.15,
        price_out_per_1m=0.60,
    ),
    # Реальный OpenAI как fallback для текста (по умолчанию вне цепочки).
    "openai": ProviderConfig(
        name="openai",
        api_key_env="OPENAI_FALLBACK_API_KEY",
        base_url=None,
        chat_model=_env("OPENAI_FALLBACK_MODEL", "gpt-4o-mini"),
        vision_model=_env("OPENAI_FALLBACK_MODEL", "gpt-4o-mini"),
        price_in_per_1m=0.15,
        price_out_per_1m=0.60,
    ),
}


@dataclass(frozen=True)
class AudioConfig:
    """
    Конфигурация аудио-эндпоинта (Whisper + TTS) для варианта Б.

    Ollama аудио не поддерживает, поэтому здесь отдельный ключ и base_url.
    Если api_key пуст — голосовой пайплайн сообщит, что эндпоинт не настроен.
    """

    api_key_env: str = "AUDIO_API_KEY"
    base_url: str | None = None
    whisper_model: str = "whisper-1"
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"
    price_whisper_per_min: float = 0.006
    price_tts_per_1m_chars: float = 15.0

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @property
    def is_local(self) -> bool:
        url = self.base_url or ""
        return "localhost" in url or "127.0.0.1" in url


@dataclass
class Settings:
    """Глобальные настройки приложения."""

    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "ollama"))
    fallback_order: list[str] = field(
        default_factory=lambda: [
            p.strip()
            for p in _env("LLM_FALLBACK_ORDER", "ollama").split(",")
            if p.strip()
        ]
    )
    request_timeout: float = field(
        default_factory=lambda: float(_env("LLM_REQUEST_TIMEOUT", "120"))
    )
    max_retries: int = field(default_factory=lambda: int(_env("LLM_MAX_RETRIES", "5")))
    cache_ttl: int = field(default_factory=lambda: int(_env("CACHE_TTL", "3600")))

    # Модель классификатора (FAQ / тех. проблема / жалоба) — из SUPPORT_CLASSIFIER_MODEL.
    classifier_model: str = field(
        default_factory=lambda: _env("SUPPORT_CLASSIFIER_MODEL", _PRIMARY)
    )

    # HTTP-прокси (опционально). К localhost не применяется (см. robust_client).
    proxy: str = field(default_factory=lambda: _env("LLM_PROXY"))

    # Аудио-эндпоинт (вариант Б).
    audio: AudioConfig = field(
        default_factory=lambda: AudioConfig(
            base_url=_env("AUDIO_BASE_URL") or None,
            whisper_model=_env("WHISPER_MODEL", "whisper-1"),
            tts_model=_env("TTS_MODEL", "tts-1"),
            tts_voice=_env("TTS_VOICE", "alloy"),
        )
    )

    def primary(self) -> ProviderConfig:
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"Неизвестный LLM_PROVIDER={self.provider!r}. "
                f"Допустимо: {', '.join(PROVIDERS)}"
            )
        return PROVIDERS[self.provider]

    def chain(self) -> list[ProviderConfig]:
        """Цепочка провайдеров для fallback: primary первым, остальные по порядку."""
        ordered: list[str] = [self.provider] + [
            p for p in self.fallback_order if p != self.provider
        ]
        result: list[ProviderConfig] = []
        for name in ordered:
            cfg = PROVIDERS.get(name)
            if cfg is not None and cfg not in result:
                result.append(cfg)
        return result


settings = Settings()
