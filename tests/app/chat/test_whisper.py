"""
Голос -> Whisper -> text-part (блок 4.3, app/chat/media.py). Сервис распознавания подменён:
client.audio.transcriptions.create — unittest.mock.AsyncMock, сеть не нужна.

Голосовое из Telegram (ogg/opus) уходит в Whisper как есть: конвертации нет, FFmpeg и
subprocess в сервисе не используются.
"""
from __future__ import annotations

import ast
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from app.chat.domain import MediaError
from app.chat.media import VOICE_PREFIX, media_to_part, read_media, whisper_filename

ROOT = Path(__file__).resolve().parents[3]
OGG = b"OggS\x00\x02" + bytes(200)          # начало контейнера Ogg — как у голосового Telegram


def upload(data: bytes, filename: str, content_type: str) -> UploadFile:
    return UploadFile(file=BytesIO(data), filename=filename, headers=Headers({"content-type": content_type}))


def whisper(text: str = "Как сбросить пароль?") -> SimpleNamespace:
    create = AsyncMock(return_value=SimpleNamespace(text=text))
    return SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))


async def test_voice_becomes_text_part_with_prefix():
    client = whisper(" Как сбросить пароль? ")
    part = await media_to_part(upload(OGG, "voice.ogg", "audio/ogg"), audio_client=client)
    assert part == {"type": "text", "text": "[пользователь сказал голосом]:\nКак сбросить пароль?"}
    assert part["text"].startswith(VOICE_PREFIX)
    kwargs = client.audio.transcriptions.create.await_args.kwargs
    assert kwargs["model"] == "whisper-1"
    assert kwargs["file"].name == "audio.ogg" and kwargs["file"].getvalue() == OGG   # ogg как есть, без конвертации


async def test_whisper_model_and_language_from_settings():
    client = whisper()
    await read_media(upload(OGG, "voice.ogg", "audio/ogg"), audio_client=client, whisper_model="whisper-large-v3",
                     language="ru")
    kwargs = client.audio.transcriptions.create.await_args.kwargs
    assert (kwargs["model"], kwargs["language"]) == ("whisper-large-v3", "ru")


@pytest.mark.parametrize("filename, mime, expected", [
    ("voice.ogg", "audio/ogg", "audio.ogg"),
    ("file.bin", "audio/ogg", "audio.ogg"),           # имя от бота без расширения — по MIME
    ("file.bin", "audio/mpeg", "audio.mp3"),
    ("song.M4A", "audio/mp4", "audio.m4a"),
    (None, "audio/wav", "audio.wav"),
])
def test_whisper_gets_extension_it_understands(filename, mime, expected):
    assert whisper_filename(filename, mime) == expected


async def test_voice_without_audio_client_is_503():
    with pytest.raises(MediaError) as caught:
        await read_media(upload(OGG, "voice.ogg", "audio/ogg"), audio_client=None)
    assert (caught.value.status, caught.value.code) == (503, "audio_not_configured")
    assert "Напишите вопрос текстом" in caught.value.message


def openai_error(kind: str) -> Exception:
    request = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")
    if kind == "timeout":
        return openai.APITimeoutError(request=request)
    if kind == "connection":
        return openai.APIConnectionError(request=request)
    status = {"auth": 401, "format": 400, "server": 500}[kind]
    response = httpx.Response(status, request=request, json={"error": {"message": "x"}})
    cls = {401: openai.AuthenticationError, 400: openai.BadRequestError, 500: openai.InternalServerError}[status]
    return cls("x", response=response, body=None)


@pytest.mark.parametrize("kind, status", [("timeout", 504), ("connection", 502), ("auth", 502),
                                          ("format", 502), ("server", 502)])
async def test_whisper_errors_become_media_errors(kind, status):
    client = whisper()
    client.audio.transcriptions.create.side_effect = openai_error(kind)
    with pytest.raises(MediaError) as caught:
        await read_media(upload(OGG, "voice.ogg", "audio/ogg"), audio_client=client)
    assert caught.value.status == status and caught.value.message
    assert "x" not in caught.value.message                          # текст ошибки провайдера пользователю не уходит


async def test_silence_is_422():
    with pytest.raises(MediaError) as caught:
        await read_media(upload(OGG, "voice.ogg", "audio/ogg"), audio_client=whisper("  "))
    assert (caught.value.status, caught.value.code) == (422, "media_empty")


def test_backend_has_no_ffmpeg_or_subprocess():
    """Критерий блока 4.3: голос — в Whisper без конвертации; subprocess/FFmpeg в сервисе нет."""
    for path in (ROOT / "app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        imported |= {(node.module or "").split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not imported & {"subprocess", "ffmpeg", "pydub"}, path


async def test_audio_label_with_other_bytes_is_415_without_whisper():
    """Не тратить платный вызов Whisper на файл, который только назван аудио."""
    client = whisper()
    with pytest.raises(MediaError) as caught:
        await read_media(upload(b"%PDF-1.7 not audio", "voice.ogg", "audio/ogg"), audio_client=client)
    assert (caught.value.status, caught.value.code) == (415, "media_unsupported")
    client.audio.transcriptions.create.assert_not_awaited()


@pytest.mark.parametrize("head, mime", [(b"ID3\x04", "audio/mpeg"), (b"\xff\xfb\x90", "audio/mpeg"),
                                        (b"\xff\xf1\x50", "audio/mpeg"), (b"RIFF\x00\x00\x00\x00WAVE", "audio/wav"),
                                        (b"\x00\x00\x00\x20ftypM4A ", "audio/mp4"), (b"fLaC", "audio/flac")])
def test_audio_formats_are_recognised(head, mime):
    from app.chat.media import sniff_audio

    assert sniff_audio(head + bytes(20)) == mime


def test_audio_language_auto(tmp_path):
    from app.core.config import Settings

    base = {"llm": {"openai_api_key": "k"}, "_env_file": None}
    assert Settings(**base).audio_language == "ru"
    assert Settings(**base, audio_language="auto").audio_language is None       # Whisper определит сам
    assert Settings(**base, audio_language="en").audio_language == "en"
