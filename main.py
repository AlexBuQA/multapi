"""
CLI мультимодального помощника (ДЗ 2.6, блок 3.1).

Автор: Александра Бужор
Репозиторий: https://github.com/AlexBuQA/multapi

Подкоманды:
  vision  — анализ изображения (вариант А)
  voice   — голосовой пайплайн Whisper -> LLM -> TTS (вариант Б)
  ask     — вопрос ассистенту техподдержки с инструментами (Function Calling, блок 3.1)

Примеры:
  python main.py vision samples/chart.png --question "Какой квартал лучший?"
  python main.py voice samples/voice_question.wav --out outputs/answer.mp3
  python main.py voice --make-sample "Как сбросить пароль?" --out samples/my_question.mp3
  python main.py ask "Не приходит письмо для сброса пароля"

Ключи берутся только из .env (хардкода нет). См. .env.example.
"""
from __future__ import annotations

import argparse
import sys

from src.cache import LLMCache
from src.config import settings
from src.robust_client import RobustLLMClient
from src.utils import InputFileError, get_logger
from src.vision import DEFAULT_QUESTION, analyze_image
from src.voice import AudioNotConfiguredError, run_pipeline, text_to_speech

logger = get_logger("main")


def _print_block(title: str, body: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'-' * 60}\n{body}\n{'=' * 60}")


def cmd_vision(args: argparse.Namespace) -> int:
    client = RobustLLMClient()
    cache = LLMCache(ttl=settings.cache_ttl)
    question = args.question or DEFAULT_QUESTION
    try:
        answer = analyze_image(args.image, question, client=client, cache=cache)
    except InputFileError as exc:
        print(f"Ошибка входного файла: {exc}", file=sys.stderr)
        return 2
    _print_block(f"Vision · {args.image}", answer)
    print(f"\n{client.usage.summary()}")
    print(f"Кеш: {cache.stats()}")
    return 0


def cmd_voice(args: argparse.Namespace) -> int:
    client = RobustLLMClient()

    # Режим генерации образца голосового вопроса через TTS (без микрофона).
    if args.make_sample:
        out = args.out or "samples/my_question.mp3"
        try:
            text_to_speech(args.make_sample, out, client=client, voice=args.voice)
        except AudioNotConfiguredError as exc:
            print(exc, file=sys.stderr)
            return 3
        print(f"Образец голосового вопроса сохранён: {out}")
        print(f"Текст: {args.make_sample}")
        return 0

    if not args.audio:
        print("Укажите путь к аудиофайлу или --make-sample <текст>.", file=sys.stderr)
        return 2

    cache = LLMCache(ttl=settings.cache_ttl)
    out = args.out or "outputs/answer.mp3"
    try:
        result = run_pipeline(args.audio, out, client=client, cache=cache)
    except InputFileError as exc:
        print(f"Ошибка входного файла: {exc}", file=sys.stderr)
        return 2
    except AudioNotConfiguredError as exc:
        print(exc, file=sys.stderr)
        return 3
    _print_block("Транскрипция (Whisper)", result.transcript)
    print(f"Категория обращения: {result.category}")
    _print_block("Ответ помощника (LLM)", result.answer)
    print(f"\nОзвученный ответ (TTS): {result.output_audio_path}")
    print(f"\n{client.usage.summary()}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    # Импорт здесь: зависимости блока 3 нужны только этой подкоманде.
    import json

    from app.config import tool_settings
    from app.llm.client import ToolCallingAssistant

    reply = ToolCallingAssistant().ask(args.question)
    for call in reply.tool_calls:
        print(f"Инструмент: {call.name}({json.dumps(call.arguments, ensure_ascii=False)})")
    if not reply.tool_calls:
        print("Инструмент: не вызывался")
    _print_block("Ответ ассистента", reply.answer)
    print(f"\nLLM-вызовов: {reply.llm_calls} | total_tokens={reply.total_tokens}")
    print(f"Лог шагов: {tool_settings.tool_log_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multimodal-assistant",
        description="Мультимодальный ИИ-помощник: Vision + Voice + Function Calling.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_vision = sub.add_parser("vision", help="Анализ изображения (вариант А)")
    p_vision.add_argument("image", help="Путь к изображению")
    p_vision.add_argument(
        "-q", "--question", default=None, help="Вопрос по изображению"
    )
    p_vision.set_defaults(func=cmd_vision)

    p_voice = sub.add_parser("voice", help="Голосовой пайплайн (вариант Б)")
    p_voice.add_argument("audio", nargs="?", default=None, help="Путь к аудиофайлу")
    p_voice.add_argument("--out", default=None, help="Путь для аудио-ответа")
    p_voice.add_argument("--voice", default=None, help="Голос TTS (alloy, echo, nova, ...)")
    p_voice.add_argument(
        "--make-sample",
        default=None,
        metavar="TEXT",
        help="Сгенерировать образец голосового вопроса из текста через TTS "
        "(формат по расширению --out: .mp3, .wav, ...)",
    )
    p_voice.set_defaults(func=cmd_voice)

    p_ask = sub.add_parser("ask", help="Вопрос ассистенту с инструментами (блок 3.1)")
    p_ask.add_argument("question", help="Текст вопроса")
    p_ask.set_defaults(func=cmd_ask)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
