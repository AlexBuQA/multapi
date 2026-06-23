"""
Вспомогательные утилиты, общие для обоих вариантов (Vision и Voice).

Содержит:
- настройку логгера (консоль + файл) с timestamp, как в ДЗ 2.3;
- кодирование файлов в base64 и определение MIME-типа;
- валидацию входных файлов (существование, размер, расширение);
- структуру UsageTracker для подсчёта токенов и стоимости.
"""
from __future__ import annotations

import base64
import logging
import mimetypes
import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Логирование
# --------------------------------------------------------------------------- #
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")


def get_logger(name: str = "multimodal") -> logging.Logger:
    """Логгер с выводом в консоль и в файл logs/app.log."""
    logger = logging.getLogger(name)
    if logger.handlers:  # уже сконфигурирован
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Работа с файлами
# --------------------------------------------------------------------------- #
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mpeg", ".mpga"}
MAX_IMAGE_MB = 20
MAX_AUDIO_MB = 25  # лимит Whisper API


class InputFileError(Exception):
    """Ошибка валидации входного файла (не найден / формат / размер)."""


def validate_file(path: str, *, kind: str) -> str:
    """
    Проверяет входной файл и возвращает абсолютный путь.

    kind: "image" или "audio". Бросает InputFileError с понятным сообщением.
    """
    if not path:
        raise InputFileError("Путь к файлу не задан.")
    abspath = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(abspath):
        raise InputFileError(f"Файл не найден: {path}")
    if not os.path.isfile(abspath):
        raise InputFileError(f"Это не файл: {path}")

    ext = os.path.splitext(abspath)[1].lower()
    allowed = IMAGE_EXTS if kind == "image" else AUDIO_EXTS
    if ext not in allowed:
        raise InputFileError(
            f"Неподдерживаемый формат {ext!r} для {kind}. "
            f"Допустимо: {', '.join(sorted(allowed))}"
        )

    size_mb = os.path.getsize(abspath) / (1024 * 1024)
    limit = MAX_IMAGE_MB if kind == "image" else MAX_AUDIO_MB
    if size_mb > limit:
        raise InputFileError(
            f"Файл слишком большой: {size_mb:.1f} МБ (лимит {limit} МБ)."
        )
    return abspath


def encode_image_data_url(path: str) -> str:
    """Кодирует изображение в data-URL вида data:image/png;base64,...."""
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# --------------------------------------------------------------------------- #
# Трекинг usage (ДЗ 2.3)
# --------------------------------------------------------------------------- #
@dataclass
class UsageTracker:
    """Накопительный учёт токенов и стоимости по всем вызовам."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    audio_seconds: float = 0.0  # для Whisper/TTS
    audio_cost_usd: float = 0.0
    _events: list[str] = field(default_factory=list)

    def add_chat(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        price_in_per_1m: float,
        price_out_per_1m: float,
        label: str = "",
    ) -> float:
        cost = (
            prompt_tokens * price_in_per_1m / 1_000_000
            + completion_tokens * price_out_per_1m / 1_000_000
        )
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost_usd += cost
        self.calls += 1
        self._events.append(
            f"{label or 'chat'}: in={prompt_tokens} out={completion_tokens} "
            f"cost=${cost:.6f}"
        )
        return cost

    def add_audio(self, seconds: float, price_per_min: float, label: str = "") -> float:
        cost = seconds / 60.0 * price_per_min
        self.audio_seconds += seconds
        self.audio_cost_usd += cost
        self._events.append(f"{label or 'audio'}: {seconds:.1f}s cost=${cost:.6f}")
        return cost

    def total_cost(self) -> float:
        return self.cost_usd + self.audio_cost_usd

    def summary(self) -> str:
        return (
            f"Вызовов: {self.calls} | "
            f"prompt_tokens={self.prompt_tokens} | "
            f"completion_tokens={self.completion_tokens} | "
            f"аудио={self.audio_seconds:.1f}s | "
            f"итого ≈ ${self.total_cost():.6f}"
        )
