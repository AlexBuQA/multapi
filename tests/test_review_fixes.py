"""
Тесты по замечаниям ревью (ДЗ 2.6). Сеть не используется: SDK-клиент OpenAI
подменён заглушкой, вся остальная логика (retry-обёртка, кеш, классификатор,
UsageTracker) — настоящая.

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import unittest
import wave
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.cache import LLMCache  # noqa: E402
from src.classifier import classify, match_category  # noqa: E402
from src.config import settings  # noqa: E402
from src.robust_client import RobustLLMClient  # noqa: E402
from src.utils import UsageTracker, validate_file  # noqa: E402
from src import voice  # noqa: E402

SAMPLE_WAV = os.path.join(ROOT, "samples", "voice_question.wav")


# --------------------------------------------------------------------------- #
# Заглушка SDK-клиента OpenAI
# --------------------------------------------------------------------------- #
class _StreamingResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def stream_to_file(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(self._payload)


class FakeSDK:
    """Минимальная имитация openai.OpenAI для chat / audio.transcriptions / audio.speech."""

    def __init__(self, transcript="Как сбросить пароль?", duration=12.0,
                 classifier_reply="тех проблема", answer="Нажмите «Забыли пароль?»."):
        self.transcription_kwargs: dict = {}
        self.speech_kwargs: dict = {}

        def transcribe(**kwargs):
            self.transcription_kwargs = kwargs
            resp = SimpleNamespace(text=transcript)
            if duration is not None:
                resp.duration = duration
            return resp

        def speech(**kwargs):
            self.speech_kwargs = kwargs
            return _StreamingResponse(b"FAKE-AUDIO")

        def chat_create(**kwargs):
            system = kwargs["messages"][0]["content"]
            text = classifier_reply if "классификатор" in system else answer
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
            )

        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(create=transcribe),
            speech=SimpleNamespace(
                with_streaming_response=SimpleNamespace(create=speech)
            ),
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=chat_create))


class FakeClient(RobustLLMClient):
    """Настоящий RobustLLMClient (retry, fallback, usage), но без сетевого SDK."""

    def __init__(self, sdk: FakeSDK):
        super().__init__(usage=UsageTracker())
        self.sdk = sdk

    def _client_for(self, provider):  # noqa: D401
        return self.sdk


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)
        patcher = mock.patch.dict(
            os.environ, {"AUDIO_API_KEY": "test-key", "OPENAI_API_KEY": "ollama"}
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(logging.disable, logging.NOTSET)


# --------------------------------------------------------------------------- #
# Замечание 1: Whisper и TTS учитываются через методы UsageTracker
# --------------------------------------------------------------------------- #
class TestVoiceUsage(_Base):
    def test_transcribe_counts_whisper_via_add_audio(self):
        client = FakeClient(FakeSDK(duration=12.0))
        with mock.patch.object(client.usage, "add_audio",
                               wraps=client.usage.add_audio) as spy:
            text = voice.transcribe(SAMPLE_WAV, client=client)

        self.assertEqual(text, "Как сбросить пароль?")
        spy.assert_called_once()
        self.assertEqual(client.sdk.transcription_kwargs["response_format"], "verbose_json")
        self.assertAlmostEqual(client.usage.audio_seconds, 12.0)
        expected = 12.0 / 60 * settings.audio.price_whisper_per_min
        self.assertAlmostEqual(client.usage.audio_cost_usd, expected)
        self.assertEqual(client.usage.audio_calls, 1)

    def test_transcribe_falls_back_to_wav_header_duration(self):
        client = FakeClient(FakeSDK(duration=None))  # ответ без поля duration
        voice.transcribe(SAMPLE_WAV, client=client)
        with wave.open(SAMPLE_WAV, "rb") as w:
            wav_seconds = w.getnframes() / w.getframerate()
        self.assertAlmostEqual(client.usage.audio_seconds, wav_seconds, places=3)

    def test_tts_counts_via_add_tts_not_direct_field(self):
        client = FakeClient(FakeSDK())
        text = "Ответ помощника"
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(client.usage, "add_tts",
                                  wraps=client.usage.add_tts) as spy:
            out = voice.text_to_speech(text, os.path.join(tmp, "a.wav"), client=client)
            self.assertTrue(os.path.exists(out))

        spy.assert_called_once()
        self.assertEqual(client.sdk.speech_kwargs["response_format"], "wav")
        self.assertEqual(client.usage.tts_chars, len(text))
        expected = len(text) / 1_000_000 * settings.audio.price_tts_per_1m_chars
        self.assertAlmostEqual(client.usage.audio_cost_usd, expected)

    def test_pipeline_summary_includes_all_steps(self):
        client = FakeClient(FakeSDK(duration=12.0))
        with tempfile.TemporaryDirectory() as tmp:
            result = voice.run_pipeline(
                SAMPLE_WAV, os.path.join(tmp, "answer.mp3"),
                client=client, cache=LLMCache(),
            )

        usage = client.usage
        self.assertEqual(result.category, "тех. проблема")
        self.assertEqual(usage.calls, 2)        # классификатор + ответ LLM
        self.assertEqual(usage.audio_calls, 2)  # Whisper + TTS
        self.assertEqual(usage.tts_chars, len(result.answer))
        self.assertAlmostEqual(usage.total_cost(), usage.cost_usd + usage.audio_cost_usd)
        summary = usage.summary()
        self.assertIn("Whisper=12.0s", summary)
        self.assertIn(f"TTS={len(result.answer)} симв.", summary)
        self.assertEqual(len(usage.events), 4)


# --------------------------------------------------------------------------- #
# Замечание 2: категории сравниваются без учёта пунктуации
# --------------------------------------------------------------------------- #
class TestClassifierMatching(_Base):
    CASES = {
        "тех. проблема": "тех. проблема",
        "тех проблема": "тех. проблема",
        "Тех.проблема": "тех. проблема",
        "Техническая проблема.": "тех. проблема",
        "техпроблема": "тех. проблема",
        "FAQ": "FAQ",
        "faq.": "FAQ",
        "«FAQ»": "FAQ",
        "Жалоба!": "жалоба",
        "Категория: жалоба": "жалоба",
        "Прочее": "прочее",
        "другое": "прочее",
    }

    def test_variants(self):
        for raw, expected in self.CASES.items():
            with self.subTest(raw=raw):
                self.assertEqual(match_category(raw), expected)

    def test_whole_words_only(self):
        # «тех» внутри другого слова не должно давать «тех. проблема».
        self.assertEqual(match_category("жалоба на технику"), "жалоба")
        self.assertIsNone(match_category("не знаю"))

    def test_classify_without_dot(self):
        client = FakeClient(FakeSDK(classifier_reply="Тех проблема"))
        self.assertEqual(classify("Не приходит письмо", client=client), "тех. проблема")

    def test_classify_unrecognized_is_other(self):
        client = FakeClient(FakeSDK(classifier_reply="затрудняюсь ответить"))
        self.assertEqual(classify("???", client=client), "прочее")


# --------------------------------------------------------------------------- #
# Замечание 3: demo_voice.py ссылается на файл, который лежит в samples/
# --------------------------------------------------------------------------- #
class TestDemoSample(_Base):
    def test_demo_points_to_existing_sample(self):
        import demo_voice

        path = os.path.join(ROOT, demo_voice.SAMPLE_AUDIO)
        self.assertTrue(os.path.exists(path), path)
        validate_file(path, kind="audio")  # не бросает InputFileError

    def test_tts_format_follows_extension(self):
        self.assertEqual(voice._tts_format("samples/voice_question.wav"), "wav")
        self.assertEqual(voice._tts_format("outputs/answer.mp3"), "mp3")
        self.assertEqual(voice._tts_format("outputs/answer.unknown"), "mp3")


# --------------------------------------------------------------------------- #
# Дополнительно: ответ-заглушка при сбое всех провайдеров не попадает в кеш
# --------------------------------------------------------------------------- #
class TestFailureNotCached(_Base):
    def _failing_client(self):
        client = FakeClient(FakeSDK())
        client.chat = lambda *a, **kw: client.USER_FACING_FAILURE
        return client

    def test_vision_failure_not_cached(self):
        from src.vision import analyze_image

        cache = LLMCache()
        image = os.path.join(ROOT, "samples", "chart.png")
        answer = analyze_image(image, client=self._failing_client(), cache=cache)
        self.assertEqual(answer, RobustLLMClient.USER_FACING_FAILURE)
        self.assertEqual(cache.stats()["size"], 0)

    def test_voice_failure_not_cached(self):
        cache = LLMCache()
        answer = voice.answer_text("Как сбросить пароль?",
                                   client=self._failing_client(), cache=cache)
        self.assertEqual(answer, RobustLLMClient.USER_FACING_FAILURE)
        self.assertEqual(cache.stats()["size"], 0)


if __name__ == "__main__":
    unittest.main()
