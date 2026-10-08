"""
Проверка входа до модели (блок 3.8): validate_input(text) -> ValidationResult.

Порядок проверок — от дешёвой к дорогой, срабатывает первая:
1. length — сообщение длиннее MAX_INPUT_CHARS. Обычный вопрос в поддержку — десятки и
   сотни символов, а длинный текст обычно несёт «инструкцию» для модели (DAN-промпты —
   2,4–4,7 тыс. символов). Длинный текст ещё и дорог: LLM10, неограниченное потребление.
2. encoding — текст, спрятанный от человека: символы-теги Unicode (U+E0000–E007F, проба
   goodside.Tag), невидимые и управляющие символы, если их больше 10 %.
3. encoded_payload — закодированная вставка: слово «base64/hex/rot13…» рядом с
   «decode/encoded», или фрагмент base64 / base32 / hex, который раскодируется в
   читаемый текст. Случайный ключ или хеш при раскодировании дают мусор и не мешают.
4. injection — известные шаблоны инъекции и джейлбрейка, по-английски и по-русски
   (русские шаблоны перенесены из блока 3.7, app/services/guardrails.py).

Шаги 3 и 4 смотрят на текст после normalize(): NFKC и без невидимых символов, которыми
разбивают слова («Ig\u200bnore»). Ложных срабатываний на обычных вопросах в поддержку
избегают так: ряды из одних цифр (номера, счета, ИНН) не читаются как hex и base64, JWT
вырезается до поиска base64, неразрывный пробел и склейка эмодзи скрытыми не считаются,
«Developer Mode» и «jailbroken» без роли модели — не джейлбрейк (см. тесты).

Решение при срабатывании — готовый безопасный ответ, а не HTTP 400 (выбрано по сценарию
проекта, см. LLMService._blocked_response): клиент — чат поддержки, пользователь должен
увидеть понятную фразу, а не ошибку. Ответ помечен model="guardrail",
finish_reason="content_filter", поэтому API-клиент отличит его от ответа модели. Модель
не вызывается, токены не тратятся. garak получает текст отказа и оценивает его теми же
детекторами, что и ответ модели, — число оценённых попыток в прогонах baseline и after
одинаковое.

Шаблоны общие, а не под конкретный вопрос: отмена инструкций, просьба показать
промпт, смена роли, известные «режимы» джейлбрейка. Какую долю проб garak они ловят и
сколько обычных вопросов задевают — в README (блок 3.8) и в тестах.
"""
from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass
from typing import Final

MAX_INPUT_CHARS: Final[int] = 4000
NON_PRINTABLE_RATIO_LIMIT: Final[float] = 0.10

# --------------------------------------------------------------------------- шаблоны
# Между глаголом и «инструкциями» — до пяти слов: «ignore any previous and following
# instructions», «ignore all the instructions you got before». По-русски — уточнения из
# блока 3.7: «игнорируй все предыдущие инструкции», но не «можно ли забыть правила пароля?».
_RU_QUALIFIERS = r"(?:(?:все|всё|свои|твои|мои|эти|данные|предыдущ\w*|прошл\w*|прежн\w*|системн\w*|ранее|выше)\s+){0,3}"
INJECTION_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = tuple(
    (name, re.compile(pattern, flags)) for name, pattern, flags in (
        # --- отмена инструкций
        ("ignore_instructions",
         r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:[\w'-]+\s+){0,5}?"
         r"(?:instructions?|rules|directives|guidelines|prompts?|constraints|restrictions)\b", re.I),
        ("stop_everything", r"\bstop\s+everything\b", re.I),
        ("say_exactly", r"\bjust\s+(?:say|print|output|write|repeat|type)\s*[:\"“'«]", re.I),
        # «Отмени» — только с инструкциями и указаниями: «отмени все ограничения по карте» —
        # обычная просьба к поддержке.
        ("ignore_instructions_ru",
         r"\b(?:игнорируй|забудь|отбрось|не\s+обращай\s+внимания\s+на|не\s+следуй)\s+"
         + _RU_QUALIFIERS + r"(?:инструкц\w*|правил\w*|указани\w*|ограничени\w*)"
         r"|\bотмени\s+" + _RU_QUALIFIERS + r"(?:инструкц\w*|указани\w*)", re.I),
        # --- системный промпт
        ("system_prompt", r"\b(?:system|initial|hidden|original)\s+(?:prompt|instructions)\b", re.I),
        ("reveal_prompt",
         r"\b(?:reveal|show|print|repeat|output|display)\s+(?:me\s+)?(?:your|the)\s+"
         r"(?:\w+\s+)?(?:instructions|prompt|rules)\b", re.I),
        ("system_prompt_ru", r"\bсистемн\w*\s+(?:промпт\w*|промт\w*)", re.I),
        ("reveal_prompt_ru",
         r"\b(?:покажи|выведи|раскрой|напиши|повтори|перечисли|процитируй)\s+(?:мне\s+)?(?:полностью\s+)?"
         r"(?:сво\w*|тво\w*|исходн\w*|скрыт\w*)\s+(?:инструкц\w*|правил\w*|промпт\w*|промт\w*)", re.I),
        # --- смена роли и «режимы» джейлбрейка
        ("role_override",
         r"\byou\s+are\s+(?:now|going\s+to\s+(?:act|be|pretend))\b|\bpretend\s+(?:to\s+be|you\s+are)\b", re.I),
        # По-русски только «с этого момента ты»: «представь, что ты пользователь» и «ты теперь
        # работаешь по выходным?» — обычные вопросы (проверено в тестах).
        ("role_override_ru", r"\bс\s+этого\s+момента\s+ты\b", re.I),
        # «Developer Mode» и «jailbroken» сами по себе — не атака: «как включить Developer Mode
        # на телефоне?», «приложение на jailbroken iPhone». Атака — роль модели в таком режиме.
        ("jailbreak_mode",
         r"\b(?:DAN|STAN|DUDE|AIM|god|jailbreak)\s+mode\b|\bdeveloper\s+mode\s+(?:enabled|output|response)\b"
         r"|\b(?:with|in)\s+developer\s+mode\s+enabled\b|\bdo\s+anything\s+now\b|\bgodmode\b"
         r"|\bjailbroken\s+(?:AI|model|assistant|chatbot|mode|version)\b|\bjailbreak\s+prompt\b"
         r"|\bрежим\w*\s+(?:DAN|без\s+ограничений)", re.I),
        ("dan_persona", r"\b(?:DAN|STAN|DUDE)\b", 0),          # только заглавными: «Dan» — имя
    )
)

# Просьба раскодировать вставку — в форме команды, а не вопроса: «BASE64 encoded string:
# … decoded string:», «Decode the following BASE64», «Encoded: … Decoded:», «расшифруй этот
# base64». Вопрос «ключ API в hex — как его расшифровать?» под шаблоны не попадает.
_ENCODINGS = r"(?:base\s?64|base\s?32|base\s?16|hex(?:adecimal)?|rot\s?13|ascii85|uuencode\w*)"
ENCODING_MARKER: Final = re.compile(
    _ENCODINGS + r"\s+(?:encoded|decoded)\s+(?:string|text|message|data)\b"
    r"|\bdecode\s+the\s+following\b"
    r"|\bencoded\b\W{0,3}.{1,80}?\bdecoded\b"
    r"|\b(?:раскодируй|расшифруй|декодируй|переведи)\w*\s+(?:(?:этот|эту|это|следующ\w*|текст|строку)\s+){0,2}"
    r"(?:из\s+)?" + _ENCODINGS,
    re.I | re.S,
)
# JWT — три части base64url через точку; заголовок раскодируется в JSON. Это токен, а не
# вставка: пользователь API может прислать его с вопросом «почему не принимается».
_JWT: Final = re.compile(r"\beyJ[\w-]{5,}\.eyJ[\w-]{5,}\.[\w-]*")

_TOKEN: Final = re.compile(r"[A-Za-z0-9+/=_-]{4,}")
_HEX: Final = re.compile(r"(?:[0-9a-fA-F]{2}){4,}")
# Невидимые символы, которыми разбивают слова: «Ig\u200bnore», «Игнор\u00adируй». Перед
# поиском шаблонов они удаляются, а текст приводится к NFKC (полноширинные буквы и т. п.).
# Склейка эмодзи (ZWJ U+200D) и вариационный селектор U+FE0F невидимыми не считаются:
# «👨‍👩‍👧‍👦» — не атака.
_EMOJI_GLUE: Final = frozenset("\u200d\ufe0f")
_TAG_CHARS: Final = range(0xE0000, 0xE0080)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str | None = None
    rule: str | None = None          # length | encoding | encoded_payload | injection


def _is_plain(ch: str) -> bool:
    return " " <= ch <= "~" or ch in "\n\r\t" or "Ѐ" <= ch <= "ӿ"


def _readable(raw: bytes) -> str | None:
    """Раскодированные байты — читаемый текст (латиница, кириллица, знаки ASCII)?
    Обычное слово, прочитанное как base64, даёт байты, которые иногда складываются в
    валидный UTF-8 с иероглифами («feelings» -> '}祊x,'), — такое текстом не считаем."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    letters = sum(1 for ch in text if ch.isalpha() and _is_plain(ch))
    if letters < 3:
        return None
    plain = sum(1 for ch in text if _is_plain(ch))
    return text if plain / len(text) >= 0.95 else None


def _decodes_to_text(token: str) -> str | None:
    """Фрагмент base64 / base32 раскодируется в текст. Короткие фрагменты без «=»
    («TWFuZ2Fs» — уже 8 символов) неотличимы от обычных слов, поэтому для них — от 8."""
    stripped = token.rstrip("=")
    if (len(stripped) < 8 and not token.endswith("=")) or stripped.isdigit():   # одни цифры — номер, не base64
        return None
    padded = stripped + "=" * (-len(stripped) % 4)
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            text = _readable(decoder(padded, validate=True) if decoder is base64.b64decode else decoder(padded))
        except (binascii.Error, ValueError):
            continue
        if text:
            return text
    if re.fullmatch(r"[A-Z2-7]+", stripped):
        try:
            return _readable(base64.b32decode(stripped + "=" * (-len(stripped) % 8)))
        except (binascii.Error, ValueError):
            return None
    return None


def find_encoded_payload(text: str) -> str | None:
    """Описание найденной закодированной вставки или None."""
    marker = ENCODING_MARKER.search(text)
    if marker:
        return f"encoding marker: {marker.group(0)[:40]!r}"
    for match in _HEX.finditer(text):
        token = match.group(0)
        # Ряд из одних цифр — номер заявки, ИНН, карта или расчётный счёт, а не hex:
        # «41424344» прочитался бы как «ABCD», а 20-значный счёт — как текст почти в 1 %
        # случаев. Цена — hex слов только из букв a–i и p–y («hate» = 68617465) не ловится.
        if token.isdigit():
            continue
        decoded = _readable(bytes.fromhex(token))
        if decoded:
            return f"hex decodes to text: {decoded[:30]!r}"
    for match in _TOKEN.finditer(_JWT.sub(" ", text)):
        decoded = _decodes_to_text(match.group(0))
        if decoded:
            return f"base64 decodes to text: {decoded[:30]!r}"
    return None


def _is_hidden(ch: str) -> bool:
    """Управляющий (Cc) или невидимый форматирующий (Cf) символ. Пробелы любых видов
    (неразрывный из скопированного с сайта текста — тоже) и склейка эмодзи — не скрытые."""
    if ch in "\n\r\t" or ch in _EMOJI_GLUE:
        return False
    return unicodedata.category(ch) in {"Cc", "Cf"}


def hidden_characters(text: str) -> str | None:
    """Невидимые символы: теги Unicode — сразу, прочие управляющие — если их много."""
    if any(ord(ch) in _TAG_CHARS for ch in text):
        return "unicode tag characters"
    hidden = sum(1 for ch in text if _is_hidden(ch))
    if hidden / max(len(text), 1) > NON_PRINTABLE_RATIO_LIMIT:
        return f"non-printable ratio {hidden / len(text):.2f}"
    return None


def normalize(text: str) -> str:
    """Текст для поиска шаблонов: NFKC и без невидимых символов, разбивающих слова."""
    return "".join(ch for ch in unicodedata.normalize("NFKC", text) if not _is_hidden(ch) or ch in "\n\r\t")


def find_injection(text: str) -> str | None:
    """Имя сработавшего шаблона инъекции или None."""
    for name, pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            return name
    return None


def validate_input(text: str, max_chars: int = MAX_INPUT_CHARS) -> ValidationResult:
    if len(text) > max_chars:
        return ValidationResult(False, f"input too long: {len(text)} > {max_chars}", rule="length")
    hidden = hidden_characters(text)
    if hidden:
        return ValidationResult(False, hidden, rule="encoding")
    visible = normalize(text)
    payload = find_encoded_payload(visible)
    if payload:
        return ValidationResult(False, payload, rule="encoded_payload")
    pattern = find_injection(visible)
    if pattern:
        return ValidationResult(False, f"matched pattern {pattern}", rule="injection")
    return ValidationResult(True)
