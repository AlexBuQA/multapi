"""
Медиа в чате (блок 4.3): файл из POST /chats/{id}/messages -> content-part для модели.

    media_to_part(media: UploadFile) -> dict        готовый content-part OpenAI Chat Completions
    read_media(media) -> MediaRef                   он же с типом, MIME, размером и именем файла

Что получается из файла:
- картинка (JPEG, PNG, WEBP, GIF) — {"type": "image_url", "image_url": {"url": "data:<mime>;base64,..."}}:
  модель с поддержкой изображений видит исходные пиксели в том же chat.completions.create,
  отдельного Vision-вызова нет;
- голос (ogg/opus из Telegram, mp3, m4a, wav, webm, flac) — Whisper (/audio/transcriptions)
  -> {"type": "text", "text": "[пользователь сказал голосом]:\\n..."}. Whisper принимает эти
  форматы сам, конвертации (FFmpeg) нет. Почему не input_audio-часть: её принимают только
  модели gpt-*-audio и только wav/mp3, а Telegram присылает ogg/opus;
- PDF — pypdf, не больше MEDIA__MAX_PDF_PAGES страниц; DOCX — python-docx, абзацы и таблицы
  по порядку -> {"type": "text", "text": "[документ PDF]:\\n..."}. Текст — не длиннее
  MEDIA__MAX_DOCUMENT_CHARS (30 000) символов.

Тип определяется по MIME от клиента и проверяется по первым байтам файла: картинка с
подписью image/jpeg, внутри которой не JPEG, модели не уйдёт. Если клиент прислал
application/octet-stream (так бывает у документов в Telegram), тип берётся по первым байтам
и расширению имени файла.

Из текста документа и расшифровки убираются управляющие и невидимые символы (NUL Postgres
не хранит, мягкие переносы и символы нулевой ширины из PDF мешали бы проверке входа блока
3.8). Документ-скан без текстового слоя не угадывается: меньше 100 символов на 5 и больше
страницах — «похоже на скан», пустой текст — «нет текста»; распознавания (OCR) нет.

Ошибки — MediaError с HTTP-кодом и текстом для пользователя: 413 файл велик, 415 тип не
поддерживается, 422 файл не читается или в нём нет текста, 502/503/504 — расшифровка
голоса (сервис распознавания вернул ошибку, не настроен, не ответил вовремя).

Разбор PDF и DOCX идёт в потоке (asyncio.to_thread): 50 страниц pypdf разбирает секунды, и
цикл событий сервиса в это время отвечает другим.

Защита от файлов-«бомб» (файл от пользователя — недоверенный вход):
- размер проверяется до чтения файла в память (UploadFile.size), а файл читается не больше
  предела; тело запроса целиком ограничивает BodyLimitMiddleware (app/core/body_limit.py);
- DOCX — ZIP: до разбора проверяются распакованные размеры (document.xml не больше
  MAX_DOCX_XML_BYTES, весь архив — MAX_DOCX_UNZIPPED_BYTES), иначе 100 КБ архива
  распаковались бы в гигабайты;
- текст извлекается, пока его не набралось вдвое больше MEDIA__MAX_DOCUMENT_CHARS: дальше
  он всё равно был бы обрезан;
- разбор — не дольше MEDIA__PARSE_TIMEOUT и не больше PARSE_SLOTS одновременно: поток
  нельзя прервать, поэтому после таймаута он дорабатывает, но держит своё место, и новые
  документы ждут, а не забирают все потоки и процессор.
"""
from __future__ import annotations

import asyncio
import base64
import re
import unicodedata
import weakref
import zipfile
from collections.abc import Callable
from io import BytesIO
from pathlib import PurePath
from typing import Any

import openai
from fastapi import UploadFile

from app.chat.domain import MediaError, MediaKind, MediaRef
from app.core.config import MediaSettings
from app.observability.logging import get_logger

log = get_logger()

VOICE_PREFIX = "[пользователь сказал голосом]:\n"
PDF_PREFIX = "[документ PDF]:\n"
DOCX_PREFIX = "[документ DOCX]:\n"

PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
GENERIC_MIMES = {"", "application/octet-stream", "binary/octet-stream", "application/zip",
                 "application/x-zip-compressed"}

# Картинки, которые принимают и OpenAI, и Ollama: подпись в начале файла -> MIME.
IMAGE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)
# Форматы Whisper: расширение имени файла, по которому SDK и сервис распознавания
# определяют формат. Telegram присылает голосовые как audio/ogg (opus внутри).
AUDIO_EXTENSIONS = {
    "audio/ogg": ".ogg", "application/ogg": ".ogg", "audio/opus": ".ogg", "audio/x-opus+ogg": ".ogg",
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a",
    "audio/m4a": ".m4a", "audio/aac": ".m4a", "audio/wav": ".wav", "audio/x-wav": ".wav",
    "audio/wave": ".wav", "audio/webm": ".webm", "audio/flac": ".flac", "audio/x-flac": ".flac",
    "audio/mpga": ".mpga",
}
WHISPER_SUFFIXES = {".ogg", ".oga", ".mp3", ".m4a", ".mp4", ".mpeg", ".mpga", ".wav", ".webm", ".flac"}
SCAN_MIN_PAGES = 5
SCAN_MAX_CHARS = 100
MAX_DOCX_XML_BYTES = 10 * 1024 * 1024          # word/document.xml распакованный — это ~1 млн символов текста
MAX_DOCX_UNZIPPED_BYTES = 100 * 1024 * 1024    # весь архив распакованный (картинки внутри уже сжаты)
PARSE_SLOTS = 2                                # документов в разборе одновременно
_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")


def _mb(size: int) -> str:
    return f"{size / (1024 * 1024):.0f} МБ" if size >= 1024 * 1024 else f"{size // 1024} КБ"


def normalize_mime(mime: str | None) -> str:
    return (mime or "").split(";", 1)[0].strip().lower()


def sniff_image(data: bytes) -> str | None:
    for signature, mime in IMAGE_SIGNATURES:
        if data.startswith(signature):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def sniff_audio(data: bytes) -> str | None:
    if data.startswith(b"OggS"):
        return "audio/ogg"
    if data.startswith(b"ID3") or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return "audio/mpeg"                        # MP3 и AAC (ADTS): кадр начинается с 11 единичных бит
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data.startswith(b"fLaC"):
        return "audio/flac"
    if data.startswith(b"\x1aE\xdf\xa3"):
        return "audio/webm"
    if data[4:8] == b"ftyp":
        return "audio/mp4"
    return None


def is_pdf(data: bytes) -> bool:
    return b"%PDF-" in data[:1024]


def is_zip(data: bytes) -> bool:
    return data.startswith(b"PK\x03\x04")


def detect(mime: str | None, filename: str | None, data: bytes) -> tuple[MediaKind, str]:
    """Тип вложения и проверенный MIME. MIME от клиента сверяется с первыми байтами."""
    claimed = normalize_mime(mime)
    suffix = PurePath(filename or "").suffix.lower()
    if claimed.startswith("image/") or (claimed in GENERIC_MIMES and sniff_image(data)):
        real = sniff_image(data)
        if real is None:
            raise MediaError(415, "media_unsupported",
                             "Картинка не распознана: поддерживаются JPEG, PNG, WEBP и GIF.")
        return "image", real
    if claimed.startswith("audio/") or claimed == "application/ogg" or (claimed in GENERIC_MIMES and sniff_audio(data)):
        real = sniff_audio(data)
        if real is None:                           # не тратить платный вызов Whisper на не-аудио
            raise MediaError(415, "media_unsupported",
                             "Аудио не распознано: поддерживаются ogg, mp3, m4a, wav, webm и flac.")
        return "audio", claimed if claimed in AUDIO_EXTENSIONS else real
    if claimed == PDF_MIME or (claimed in GENERIC_MIMES and suffix == ".pdf"):
        if not is_pdf(data):
            raise MediaError(422, "media_unreadable", "Файл не похож на PDF: в начале нет заголовка %PDF.")
        return "document", PDF_MIME
    if claimed.endswith("wordprocessingml.document") or (claimed in GENERIC_MIMES and suffix == ".docx"):
        if not is_zip(data):
            raise MediaError(422, "media_unreadable", "Файл не похож на DOCX: это не архив Office Open XML.")
        return "document", DOCX_MIME
    if claimed == "application/msword" or suffix == ".doc":
        raise MediaError(415, "media_unsupported",
                         "Старый формат Word (.doc) не поддерживается: сохраните документ как DOCX или PDF.")
    raise MediaError(415, "media_unsupported",
                     "Такой файл не поддерживается. Можно прислать фото, голосовое сообщение, PDF или DOCX.")


def check_size(kind: MediaKind, size: int, limits: MediaSettings) -> None:
    if size == 0:
        raise MediaError(422, "media_empty", "Файл пустой.")
    limit = {"image": limits.max_image_bytes, "audio": limits.max_audio_bytes,
             "document": limits.max_document_bytes}[kind]
    if size > limit:
        what = {"image": "Картинка", "audio": "Аудио", "document": "Документ"}[kind]
        raise MediaError(413, "media_too_large", f"{what} больше {_mb(limit)} — пришлите файл поменьше.")


def clean_text(text: str) -> str:
    """Текст документа или расшифровки для модели: без управляющих и невидимых символов
    (кроме перевода строки и табуляции), без хвостовых пробелов и пачек пустых строк."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    kept = []
    for ch in text:
        if ch in "\n\t":
            kept.append(ch)
        elif unicodedata.category(ch) in {"Cc", "Cf", "Co", "Cs"}:
            continue
        else:
            kept.append(ch)
    text = _TRAILING_SPACES.sub("\n", "".join(kept))
    return _BLANK_LINES.sub("\n\n", text).strip()


def truncate(text: str, limit: int) -> str:
    """Не длиннее limit символов, по границе слова; в конце — пометка, сколько показано."""
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", int(limit * 0.9), limit)
    cut = cut if cut > 0 else limit
    return text[:cut].rstrip() + f"\n[…текст обрезан: показано {cut} из {len(text)} символов]"


# ------------------------------------------------------------------ документы
def extract_pdf_text(data: bytes, max_pages: int = 50, max_chars: int = 30_000) -> str:
    """Текст PDF по страницам (pypdf.PdfReader). Скан без текстового слоя -> MediaError.
    Страницы читаются, пока текста не наберётся вдвое больше max_chars."""
    from pypdf import PdfReader

    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise MediaError(422, "media_unreadable", "PDF защищён паролем — пришлите файл без защиты.")
        total = len(reader.pages)
        pages: list[str] = []
        collected = 0
        for number in range(min(total, max_pages)):
            try:
                text = reader.pages[number].extract_text() or ""
            except Exception:  # noqa: BLE001 — битая страница не должна ронять весь документ
                text = ""
            pages.append(text)
            collected += len(text)
            if collected > 2 * max_chars:          # дальше текст всё равно обрезали бы
                break
    except MediaError:
        raise
    except Exception as exc:  # noqa: BLE001 — файл от пользователя: любая ошибка разбора — 422, а не 500
        log.info("media_parse_failed", kind="pdf", error=repr(exc)[:300])
        raise MediaError(422, "media_unreadable", "PDF не читается: файл повреждён или в необычном формате.") from exc
    text = clean_text("\n\n".join(pages))
    if not text:
        raise MediaError(422, "media_empty",
                         "В PDF нет текста — похоже, это скан. Распознавания текста на картинках нет: "
                         "пришлите документ с текстовым слоем или сфотографируйте нужную страницу.")
    if len(pages) >= SCAN_MIN_PAGES and len(text) < SCAN_MAX_CHARS:
        raise MediaError(422, "media_empty",
                         f"В PDF на {len(pages)} страницах всего {len(text)} символов текста — похоже на скан. "
                         "Пришлите документ с текстовым слоем или фото нужной страницы.")
    if total > max_pages:
        text += f"\n\n[…показаны первые {max_pages} из {total} страниц]"
    return text


def _row_text(cells: list[str]) -> str:
    """Строка таблицы; объединённая ячейка python-docx повторяется — показываем её один раз."""
    out: list[str] = []
    for cell in cells:
        if not out or cell != out[-1]:
            out.append(cell)
    return " | ".join(out)


def check_docx_archive(data: bytes) -> None:
    """DOCX — ZIP-архив. До разбора — распакованные размеры: 100 КБ архива с повторяющимся
    текстом распаковываются в гигабайты (zip-бомба). Размеры в архиве объявлены, но zipfile
    больше объявленного и не распакует."""
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            infos = archive.infolist()
    except (zipfile.BadZipFile, ValueError) as exc:
        raise MediaError(422, "media_unreadable", "DOCX не читается: файл повреждён или это не документ Word.") from exc
    body = next((info for info in infos if info.filename == "word/document.xml"), None)
    if body is None:
        raise MediaError(422, "media_unreadable", "Это не документ Word: в архиве нет word/document.xml.")
    if body.file_size > MAX_DOCX_XML_BYTES or sum(info.file_size for info in infos) > MAX_DOCX_UNZIPPED_BYTES:
        log.warning("media_docx_too_large", xml_bytes=body.file_size, unzipped_bytes=sum(i.file_size for i in infos))
        raise MediaError(413, "media_too_large", "Документ слишком большой после распаковки — пришлите файл поменьше.")


def extract_docx_text(data: bytes, max_chars: int = 30_000) -> str:
    """Абзацы и таблицы DOCX в порядке документа (python-docx), пока текста не наберётся
    вдвое больше max_chars."""
    import docx
    from docx.table import Table

    check_docx_archive(data)
    try:
        document = docx.Document(BytesIO(data))
        blocks: list[str] = []
        collected = 0
        for block in document.iter_inner_content():
            if isinstance(block, Table):
                rows = [_row_text([cell.text.strip() for cell in row.cells]) for row in block.rows]
                blocks.append("\n".join(row for row in rows if row.strip(" |")))
            else:
                blocks.append(block.text)
            collected += len(blocks[-1])
            if collected > 2 * max_chars:          # дальше текст всё равно обрезали бы
                break
    except Exception as exc:  # noqa: BLE001 — файл от пользователя: любая ошибка разбора — 422, а не 500
        log.info("media_parse_failed", kind="docx", error=repr(exc)[:300])
        raise MediaError(422, "media_unreadable", "DOCX не читается: файл повреждён или это не документ Word.") from exc
    text = clean_text("\n".join(blocks))
    if not text:
        raise MediaError(422, "media_empty", "В документе нет текста.")
    return text


# ------------------------------------------------------------------ голос
def whisper_filename(filename: str | None, mime: str) -> str:
    """Имя файла для Whisper: формат он берёт из расширения, а у файла от клиента его может
    не быть (file.bin) или оно неверное."""
    suffix = PurePath(filename or "").suffix.lower()
    if suffix in WHISPER_SUFFIXES:
        return f"audio{suffix}"
    return "audio" + AUDIO_EXTENSIONS.get(mime, ".ogg")


async def whisper_transcribe(audio_bytes: bytes, filename: str, *, client: Any, model: str = "whisper-1",
                             language: str | None = None) -> str:
    """Расшифровка через /audio/transcriptions. Whisper-1 принимает ogg/m4a/mp3/wav/flac/webm напрямую."""
    f = BytesIO(audio_bytes)
    f.name = filename            # SDK берёт расширение для определения формата
    params: dict[str, Any] = {"model": model, "file": f}
    if language:
        params["language"] = language
    result = await client.audio.transcriptions.create(**params)
    return str(getattr(result, "text", result) or "")


async def transcribe(data: bytes, filename: str | None, mime: str, *, client: Any | None, model: str,
                     language: str | None) -> str:
    if client is None:
        log.warning("audio_not_configured", note="голос не принят: задайте AUDIO_API_KEY (и AUDIO_BASE_URL для "
                                                 "не-OpenAI сервиса распознавания)")
        raise MediaError(503, "audio_not_configured",
                         "Голосовые сообщения пока не принимаются: расшифровка речи не настроена. "
                         "Напишите вопрос текстом.")
    try:
        text = await whisper_transcribe(data, whisper_filename(filename, mime), client=client, model=model,
                                        language=language)
    except openai.APITimeoutError as exc:
        raise MediaError(504, "audio_timeout", "Расшифровка голоса заняла слишком много времени. "
                                               "Попробуйте сообщение покороче.") from exc
    except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
        log.warning("audio_auth_failed", error=repr(exc)[:300], note="проверьте AUDIO_API_KEY")
        raise MediaError(502, "audio_failed", "Расшифровка голоса сейчас недоступна. Напишите вопрос текстом.") from exc
    except openai.APIConnectionError as exc:
        log.warning("audio_failed", error=repr(exc)[:300], note="сервис распознавания недоступен (AUDIO_BASE_URL)")
        raise MediaError(502, "audio_failed", "Расшифровка голоса сейчас недоступна. Напишите вопрос текстом.") from exc
    except openai.APIStatusError as exc:
        log.warning("audio_failed", status=exc.status_code, error=repr(exc)[:300])
        message = ("Сервис распознавания не принял этот аудиоформат." if exc.status_code in (400, 415)
                   else f"Сервис распознавания голоса вернул ошибку {exc.status_code}.")
        raise MediaError(502, "audio_failed", message) from exc
    text = clean_text(text)
    if not text:
        raise MediaError(422, "media_empty", "В голосовом сообщении не удалось разобрать слов.")
    return text


# ------------------------------------------------------------------ сборка
async def build_part(kind: MediaKind, mime: str, data: bytes, filename: str | None, *,
                     settings: MediaSettings, audio_client: Any | None = None, whisper_model: str = "whisper-1",
                     language: str | None = None) -> dict[str, Any]:
    if kind == "image":
        b64 = base64.b64encode(data).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    if kind == "audio":
        transcript = await transcribe(data, filename, mime, client=audio_client, model=whisper_model,
                                      language=language)
        return {"type": "text", "text": VOICE_PREFIX + truncate(transcript, settings.max_document_chars)}
    if mime == PDF_MIME:
        text = await parse_in_thread(extract_pdf_text, data, settings.max_pdf_pages, settings.max_document_chars,
                                     timeout=settings.parse_timeout)
        return {"type": "text", "text": PDF_PREFIX + truncate(text, settings.max_document_chars)}
    text = await parse_in_thread(extract_docx_text, data, settings.max_document_chars, timeout=settings.parse_timeout)
    return {"type": "text", "text": DOCX_PREFIX + truncate(text, settings.max_document_chars)}


# Семафор — на свой цикл событий: asyncio.Semaphore привязывается к циклу, в котором его ждали.
_parse_slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = weakref.WeakKeyDictionary()


def _slots() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    slots = _parse_slots.get(loop)
    if slots is None:
        slots = _parse_slots[loop] = asyncio.Semaphore(PARSE_SLOTS)
    return slots


async def parse_in_thread(fn: Callable[..., str], *args: Any, timeout: float) -> str:
    """fn(*args) в потоке: не дольше timeout и не больше PARSE_SLOTS одновременно. Поток нельзя
    прервать — после таймаута он дорабатывает, но место освобождает только когда закончит."""
    slots = _slots()
    await slots.acquire()
    job = asyncio.ensure_future(asyncio.to_thread(fn, *args))

    def finished(future: asyncio.Future[str]) -> None:
        slots.release()
        if not future.cancelled():
            future.exception()                     # ошибку после таймаута никто не ждёт — не шуметь в лог

    job.add_done_callback(finished)
    try:
        return await asyncio.wait_for(asyncio.shield(job), timeout)
    except TimeoutError as exc:
        log.warning("media_parse_timeout", parser=getattr(fn, "__name__", "?"), timeout_s=timeout)
        raise MediaError(422, "media_unreadable", "Документ разбирается слишком долго — пришлите файл поменьше "
                                                  "или только нужные страницы.") from exc


async def read_media(media: UploadFile, *, settings: MediaSettings | None = None, audio_client: Any | None = None,
                     whisper_model: str = "whisper-1", language: str | None = None) -> MediaRef:
    """Файл запроса -> MediaRef с готовым content-part."""
    settings = settings or MediaSettings()
    # Размер — до чтения в память: тип ещё неизвестен, поэтому сначала общий предел (самый
    # большой из MEDIA__MAX_*_BYTES), после определения типа — свой для типа.
    limit = max(settings.max_image_bytes, settings.max_audio_bytes, settings.max_document_bytes)
    too_large = MediaError(413, "media_too_large", f"Файл больше {_mb(limit)} — пришлите файл поменьше.")
    if media.size is not None and media.size > limit:
        raise too_large
    data = await media.read(limit + 1)
    if len(data) > limit:
        raise too_large
    filename = clean_text(PurePath(media.filename).name)[:255] or None if media.filename else None
    kind, mime = detect(media.content_type, filename, data)
    check_size(kind, len(data), settings)
    part = await build_part(kind, mime, data, filename, settings=settings, audio_client=audio_client,
                            whisper_model=whisper_model, language=language)
    return MediaRef(kind=kind, mime=mime, size=len(data), filename=filename, part=part)


async def media_to_part(media: UploadFile, **options: Any) -> dict[str, Any]:
    """Готовый content-part для OpenAI Chat Completions API (options — как у read_media)."""
    return (await read_media(media, **options)).part


def placeholder(ref: MediaRef) -> str:
    """Текстовая копия сообщения без подписи — для истории и интерфейса."""
    if ref.kind == "image":
        return "[фото]"
    if ref.kind == "audio":
        return "[голосовое сообщение]"
    # Без имени файла: эта копия текста попадает в превью лога, а имя — нет (MediaRef).
    return f"[документ {'PDF' if ref.mime == PDF_MIME else 'DOCX'}]"
