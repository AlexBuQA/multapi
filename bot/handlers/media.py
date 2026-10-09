"""
Фото, голос и документы (блок 4.3). Бот только скачивает файл из Telegram и отправляет его
в сервис тем же BackendClient.send_message, что и текст, — с media и mime. Картинку,
расшифровку голоса и текст документа готовит сервис (app/chat/media.py), поэтому openai,
pypdf и python-docx боту не нужны.

- F.photo — Telegram хранит фото в нескольких размерах; берётся самый большой не больше
  PHOTO_LIMIT (2 МБ): модели этого хватает, а base64 в истории чата не раздувается.
  Фото из Telegram всегда JPEG;
- F.voice — голосовое как есть, .ogg (opus), mime audio/ogg: Whisper принимает ogg сам,
  конвертации нет ни в боте, ни в сервисе;
- F.audio — аудиофайл (mp3, m4a): mime от Telegram;
- F.document — PDF и DOCX до DOCUMENT_LIMIT (10 МБ). Картинка, отправленная файлом (без
  сжатия), — как фото, если не больше PHOTO_LIMIT. Остальные файлы — подсказка.

Подпись к файлу — вопрос; без подписи уходит вопрос по умолчанию (texts.*_PROMPT): сервису
нужен текст вопроса. В сценарии /ask на шаге вопроса файл — тоже вопрос: «Тема: … Вопрос:
<подпись>», как у текста; на шаге выбора раздела — подсказка выбрать раздел.

Скачивание — bot.get_file() и bot.download_file(destination=BytesIO()): Bot API отдаёт
файлы до 20 МБ. Файл живёт только в памяти бота, на диск не пишется. Ни файл, ни подпись в
лог бота не попадают — только тип и размер.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO

from aiogram import Bot, F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, PhotoSize
from aiogram.utils.chat_action import ChatActionSender

from bot import texts
from bot.config import BotSettings
from bot.handlers.common import backend_chat_id
from bot.handlers.fsm import build_prompt
from bot.services.backend_client import BACKEND_ERRORS, DOCX_MIME, BackendClient
from bot.services.chat_queue import ChatQueue
from bot.services.streaming import answer_with_stream
from bot.states import AskFlow

router = Router(name="media")
log = logging.getLogger(__name__)

MB = 1024 * 1024
PHOTO_LIMIT = 2 * MB
DOCUMENT_LIMIT = 10 * MB
AUDIO_LIMIT = 20 * MB              # больше Bot API не отдаёт (getFile)
DOCUMENT_SUFFIXES = (".pdf", ".docx")
DOCUMENT_MIMES = {".pdf": "application/pdf", ".docx": DOCX_MIME}


@dataclass(frozen=True)
class Upload:
    """Что скачать и как отправить в сервис."""

    kind: str                       # photo | voice | audio | document — для лога
    file_id: str
    size: int | None
    mime: str
    filename: str
    default_prompt: str


def pick_photo(sizes: list[PhotoSize], limit: int = PHOTO_LIMIT) -> PhotoSize | None:
    """Самый большой размер фото не больше limit; None — все больше."""
    fitting = [p for p in sizes if (p.file_size or 0) <= limit]
    return max(fitting, key=lambda p: (p.file_size or 0, p.width * p.height)) if fitting else None


async def download(bot: Bot, file_id: str) -> bytes:
    file = await bot.get_file(file_id)
    assert file.file_path is not None
    buffer = BytesIO()
    await bot.download_file(file.file_path, destination=buffer)
    return buffer.getvalue()


async def send_upload(message: Message, upload: Upload, *, bot: Bot, backend: BackendClient, chat_queue: ChatQueue,
                      settings: BotSettings, state: FSMContext) -> None:
    prompt = (message.caption or "").strip() or upload.default_prompt
    step = await state.get_state()
    if step == AskFlow.waiting_for_topic.state:
        await message.answer(texts.PICK_TOPIC)
        return
    if step == AskFlow.waiting_for_question.state:
        data = await state.get_data()
        prompt = build_prompt(data.get("topic", ""), prompt)
        await state.clear()
    async with chat_queue.turn(message.chat.id):        # второй вопрос — после ответа на первый
        try:
            chat_id = await backend_chat_id(backend, message.chat)
        except BACKEND_ERRORS as exc:
            log.warning("backend_error chat=%s during=chat error=%r", message.chat.id, exc)
            await message.answer(texts.user_message(exc))
            return
        try:
            async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
                content = await download(bot, upload.file_id)
        except Exception as exc:  # noqa: BLE001 — сеть до Telegram, файл больше 20 МБ: пользователю — текст
            log.warning("download_failed chat=%s kind=%s error=%r", message.chat.id, upload.kind, exc)
            await message.answer(texts.DOWNLOAD_FAILED)
            return
        renderer = await answer_with_stream(message, backend, chat_id, prompt, media=content, mime=upload.mime,
                                            filename=upload.filename, mode=settings.bot_streaming)
    log.info("answered telegram_chat=%s chat_id=%s media=%s bytes=%d mode=%s messages=%d updates=%d chars=%d",
             message.chat.id, chat_id, upload.kind, len(content), renderer.mode, len(renderer.all_messages),
             renderer.updates, renderer.chars)


@router.message(F.photo)
async def on_photo(message: Message, bot: Bot, backend: BackendClient, chat_queue: ChatQueue,
                   settings: BotSettings, state: FSMContext) -> None:
    photo = pick_photo(message.photo or [])
    if photo is None:
        await message.answer(texts.PHOTO_TOO_LARGE.format(limit=PHOTO_LIMIT // MB))
        return
    upload = Upload("photo", photo.file_id, photo.file_size, "image/jpeg", "photo.jpg", texts.PHOTO_PROMPT)
    await send_upload(message, upload, bot=bot, backend=backend, chat_queue=chat_queue, settings=settings, state=state)


@router.message(F.voice)
async def on_voice(message: Message, bot: Bot, backend: BackendClient, chat_queue: ChatQueue,
                   settings: BotSettings, state: FSMContext) -> None:
    voice = message.voice
    assert voice is not None
    if (voice.file_size or 0) > AUDIO_LIMIT:
        await message.answer(texts.FILE_TOO_LARGE.format(limit=AUDIO_LIMIT // MB))
        return
    upload = Upload("voice", voice.file_id, voice.file_size, voice.mime_type or "audio/ogg", "voice.ogg",
                    texts.VOICE_PROMPT)
    await send_upload(message, upload, bot=bot, backend=backend, chat_queue=chat_queue, settings=settings, state=state)


@router.message(F.audio)
async def on_audio(message: Message, bot: Bot, backend: BackendClient, chat_queue: ChatQueue,
                   settings: BotSettings, state: FSMContext) -> None:
    audio = message.audio
    assert audio is not None
    if (audio.file_size or 0) > AUDIO_LIMIT:
        await message.answer(texts.FILE_TOO_LARGE.format(limit=AUDIO_LIMIT // MB))
        return
    upload = Upload("audio", audio.file_id, audio.file_size, audio.mime_type or "audio/mpeg",
                    audio.file_name or "audio.mp3", texts.VOICE_PROMPT)
    await send_upload(message, upload, bot=bot, backend=backend, chat_queue=chat_queue, settings=settings, state=state)


def supported_document(message: Message) -> bool:
    """PDF или DOCX не больше DOCUMENT_LIMIT. Функцией, а не F.document.file_size <= …:
    размера у документа в Telegram может не быть (None), и сравнение упало бы."""
    doc = message.document
    return (doc is not None and (doc.file_name or "").lower().endswith(DOCUMENT_SUFFIXES)
            and (doc.file_size or 0) <= DOCUMENT_LIMIT)


@router.message(supported_document)
async def on_document(message: Message, bot: Bot, backend: BackendClient, chat_queue: ChatQueue,
                      settings: BotSettings, state: FSMContext) -> None:
    doc = message.document
    assert doc is not None and doc.file_name is not None
    suffix = doc.file_name.lower().rsplit(".", 1)[-1]
    # Telegram иногда присылает application/octet-stream — тогда MIME по расширению.
    mime = doc.mime_type if doc.mime_type in DOCUMENT_MIMES.values() else DOCUMENT_MIMES[f".{suffix}"]
    upload = Upload("document", doc.file_id, doc.file_size, mime, doc.file_name, texts.DOCUMENT_PROMPT)
    await send_upload(message, upload, bot=bot, backend=backend, chat_queue=chat_queue, settings=settings, state=state)


@router.message(F.document.mime_type.startswith("image/"))
async def on_image_document(message: Message, bot: Bot, backend: BackendClient, chat_queue: ChatQueue,
                            settings: BotSettings, state: FSMContext) -> None:
    """Картинка, отправленная файлом (без сжатия)."""
    doc = message.document
    assert doc is not None and doc.mime_type is not None
    if (doc.file_size or 0) > PHOTO_LIMIT:
        await message.answer(texts.PHOTO_TOO_LARGE.format(limit=PHOTO_LIMIT // MB))
        return
    upload = Upload("photo", doc.file_id, doc.file_size, doc.mime_type, doc.file_name or "image.jpg",
                    texts.PHOTO_PROMPT)
    await send_upload(message, upload, bot=bot, backend=backend, chat_queue=chat_queue, settings=settings, state=state)


@router.message(F.document)
async def on_other_document(message: Message) -> None:
    doc = message.document
    assert doc is not None
    if (doc.file_name or "").lower().endswith(DOCUMENT_SUFFIXES):     # нужный тип, но больше предела
        await message.answer(texts.FILE_TOO_LARGE.format(limit=DOCUMENT_LIMIT // MB))
    else:
        await message.answer(texts.DOCUMENT_ONLY_PDF_DOCX)
