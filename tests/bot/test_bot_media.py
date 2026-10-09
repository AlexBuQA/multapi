"""
Фото, голос и документы в боте (блок 4.3) через настоящий Dispatcher: MockedSession отдаёт
файлы по getFile, FakeBackend запоминает, что пришло в send_message.

Бот файл не разбирает — только скачивает и отправляет тем же send_message, что и текст.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from aiogram.types import Audio, Document, PhotoSize, Voice
from bot_fakes import FakeBackend, fsm, message_update, new_chat_id

from bot import texts
from bot.handlers.media import DOCUMENT_LIMIT, PHOTO_LIMIT, pick_photo
from bot.services.backend_client import DOCX_MIME
from bot.states import AskFlow

ROOT = Path(__file__).resolve().parents[2]
JPEG = b"\xff\xd8\xff\xe0" + bytes(3000)
OGG = b"OggS" + bytes(500)
PDF = b"%PDF-1.7\n" + bytes(2000)


def photo(file_id: str, size: int, side: int) -> PhotoSize:
    return PhotoSize(file_id=file_id, file_unique_id=f"u{file_id}", width=side, height=side, file_size=size)


def document(file_id: str, name: str | None, mime: str | None, size: int | None) -> Document:
    return Document(file_id=file_id, file_unique_id=f"u{file_id}", file_name=name, mime_type=mime, file_size=size)


async def test_photo_goes_to_backend_as_jpeg(dp, bot, session, backend):
    chat = new_chat_id()
    session.files["big"] = JPEG
    sizes = [photo("small", 1_500, 90), photo("big", len(JPEG), 1280), photo("huge", 3 * 1024 * 1024, 2560)]
    await dp.feed_update(bot, message_update(None, chat, photo=sizes, caption="Что за ошибка?"))
    assert backend.sent[-1][1] == "Что за ошибка?"                                  # подпись — вопрос
    assert backend.media == [{"media": JPEG, "mime": "image/jpeg", "filename": "photo.jpg"}]
    assert session.on_screen(chat) == [backend.answer]


async def test_photo_without_caption_gets_default_question(dp, bot, session, backend):
    chat = new_chat_id()
    session.files["p"] = JPEG
    await dp.feed_update(bot, message_update(None, chat, photo=[photo("p", len(JPEG), 800)]))
    assert backend.sent[-1][1] == texts.PHOTO_PROMPT


def test_pick_photo_prefers_largest_under_limit():
    sizes = [photo("a", 10, 90), photo("b", PHOTO_LIMIT, 1280), photo("c", PHOTO_LIMIT + 1, 2560)]
    assert pick_photo(sizes).file_id == "b"
    assert pick_photo([photo("c", PHOTO_LIMIT + 1, 2560)]) is None


async def test_photo_over_limit_is_refused_without_download(dp, bot, session, backend):
    chat = new_chat_id()
    await dp.feed_update(bot, message_update(None, chat, photo=[photo("x", PHOTO_LIMIT + 1, 2560)]))
    assert session.sent_texts(chat) == [texts.PHOTO_TOO_LARGE.format(limit=2)] and backend.sent == []


async def test_voice_goes_as_ogg_without_conversion(dp, bot, session, backend):
    chat = new_chat_id()
    session.files["v"] = OGG
    voice = Voice(file_id="v", file_unique_id="uv", duration=3, mime_type="audio/ogg", file_size=len(OGG))
    await dp.feed_update(bot, message_update(None, chat, voice=voice))
    assert backend.sent[-1][1] == texts.VOICE_PROMPT
    assert backend.media == [{"media": OGG, "mime": "audio/ogg", "filename": "voice.ogg"}]


async def test_audio_file_keeps_its_mime(dp, bot, session, backend):
    chat = new_chat_id()
    session.files["a"] = b"ID3" + bytes(100)
    audio = Audio(file_id="a", file_unique_id="ua", duration=5, mime_type="audio/mpeg", file_name="вопрос.mp3",
                  file_size=103)
    await dp.feed_update(bot, message_update(None, chat, audio=audio, caption="Послушайте"))
    assert backend.media[-1]["mime"] == "audio/mpeg" and backend.media[-1]["filename"] == "вопрос.mp3"


@pytest.mark.parametrize("name, mime, expected_mime", [
    ("Регламент.PDF", "application/pdf", "application/pdf"),
    ("notes.docx", DOCX_MIME, DOCX_MIME),
    ("notes.docx", "application/octet-stream", DOCX_MIME),        # Telegram не знает тип — по расширению
])
async def test_pdf_and_docx_are_sent(dp, bot, session, backend, name, mime, expected_mime):
    chat = new_chat_id()
    session.files["d"] = PDF
    await dp.feed_update(bot, message_update(None, chat, document=document("d", name, mime, len(PDF))))
    assert backend.sent[-1][1] == texts.DOCUMENT_PROMPT
    assert backend.media[-1] == {"media": PDF, "mime": expected_mime, "filename": name}


@pytest.mark.parametrize("doc, reply", [
    (document("t", "notes.txt", "text/plain", 100), texts.DOCUMENT_ONLY_PDF_DOCX),
    (document("o", "old.doc", "application/msword", 100), texts.DOCUMENT_ONLY_PDF_DOCX),
    (document("b", "big.pdf", "application/pdf", DOCUMENT_LIMIT + 1), texts.FILE_TOO_LARGE.format(limit=10)),
    (document("n", None, None, None), texts.DOCUMENT_ONLY_PDF_DOCX),                 # без имени и размера
])
async def test_other_documents_get_a_hint(dp, bot, session, backend, doc, reply):
    chat = new_chat_id()
    await dp.feed_update(bot, message_update(None, chat, document=doc))
    assert session.sent_texts(chat) == [reply] and backend.sent == []


async def test_image_sent_as_file_is_a_photo(dp, bot, session, backend):
    chat = new_chat_id()
    session.files["i"] = JPEG
    await dp.feed_update(bot, message_update(None, chat, document=document("i", "скрин.png", "image/png", len(JPEG))))
    assert backend.media[-1]["mime"] == "image/png" and backend.sent[-1][1] == texts.PHOTO_PROMPT


async def test_media_in_ask_question_step_keeps_topic(dp, bot, session, backend):
    chat = new_chat_id()
    state = fsm(dp, bot, chat)
    await state.set_state(AskFlow.waiting_for_question)
    await state.update_data(topic="Оплата и документы")
    session.files["d"] = PDF
    await dp.feed_update(bot, message_update(None, chat, document=document("d", "счёт.pdf", "application/pdf", 2009),
                                             caption="Почему два списания?"))
    assert backend.sent[-1][1] == "Тема: Оплата и документы. Вопрос: Почему два списания?"
    assert await state.get_state() is None


async def test_media_in_topic_step_asks_for_topic(dp, bot, session, backend):
    chat = new_chat_id()
    await fsm(dp, bot, chat).set_state(AskFlow.waiting_for_topic)
    await dp.feed_update(bot, message_update(None, chat, photo=[photo("p", 100, 90)]))
    assert session.sent_texts(chat) == [texts.PICK_TOPIC] and backend.sent == []


async def test_download_failure_is_a_message(dp, bot, session, backend):
    chat = new_chat_id()                                      # файла нет в session.files — getFile падает
    await dp.feed_update(bot, message_update(None, chat, photo=[photo("missing", 100, 90)]))
    assert session.sent_texts(chat) == [texts.DOWNLOAD_FAILED] and backend.sent == []


async def test_backend_media_error_text_reaches_user(dp, bot, session):
    import httpx

    request = httpx.Request("POST", "http://backend.test/chats/x/messages")
    message = "Голосовые сообщения пока не принимаются: расшифровка речи не настроена. Напишите вопрос текстом."
    response = httpx.Response(503, json={"error": {"code": "audio_not_configured", "message": message}},
                              request=request)
    dp["backend"] = FakeBackend(error=httpx.HTTPStatusError("503", request=request, response=response), error_after=0)
    chat = new_chat_id()
    session.files["v"] = OGG
    await dp.feed_update(bot, message_update(None, chat, voice=Voice(file_id="v", file_unique_id="uv", duration=1)))
    assert session.on_screen(chat) == [message]


def test_bot_does_not_import_media_libraries():
    """Критерий блока 4.3: openai, pypdf, python-docx — только в сервисе."""
    forbidden = {"openai", "pypdf", "docx", "PyPDF2", "fitz", "whisper"}
    for path in (ROOT / "bot").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                 for alias in node.names}
        names |= {(node.module or "").split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not names & forbidden, f"{path.relative_to(ROOT)}: {names & forbidden}"

