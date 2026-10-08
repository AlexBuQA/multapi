"""
Проверка ответа модели (блок 3.8): filter_output(answer, system_prompt, canary) -> str.

Ответ с утечкой не отдаётся: функция поднимает OutputBlocked (подкласс ValueError, как
в стартер-коде), а LLMService заменяет ответ готовым отказом — тем же, что и на входе
(app/services/llm.py, _filter_output). Причины, по порядку:
1. canary — в ответе секретная метка из системного сообщения (app/services/security/
   canary.py), целиком или только её случайная часть, в том числе через пробелы;
2. prompt_leak — в ответе начало системного промпта (первые 80 символов после
   нормализации пробелов и регистра) или дословно два правила промпта и больше —
   проверка блока 3.7, app/services/guardrails.py;
3. jailbreak — модель приняла роль из джейлбрейка: «DAN Mode enabled», «[DAN]», «DAN:»,
   «Developer Mode enabled». Это вторая линия за проверкой входа: если перефразированный
   DAN-промпт прошёл валидатор, ответ в роли DAN всё равно не уйдёт.

Чистый ответ возвращается с маскированными персональными данными: email, телефон,
паспорт, карта, ИНН, СНИЛС, расчётный счёт, API-ключ — тот же redact_pii, что в логах
(app/observability/pii.py). Модель не получает персональных данных пользователя (они
маскируются до неё, блок 3.7), но может выдумать правдоподобные или повторить их из
истории диалога, которую прислал клиент.
"""
from __future__ import annotations

import re
from typing import Final

from app.observability.pii import PII_PATTERNS, redact_pii
from app.services.guardrails import leaks_instructions
from app.services.security.canary import CANARY_PREFIX

PROMPT_HEAD_CHARS: Final = 80
JAILBREAK_MARKERS: Final = re.compile(
    r"\[(?:DAN|STAN|DUDE)\]|\b(?:DAN|STAN|DUDE)\s*:|\b(?:DAN|Developer)\s+Mode\s+enabled\b", re.I
)


class OutputBlocked(ValueError):
    """Ответ модели нельзя отдавать. rule — canary | prompt_leak | jailbreak."""

    def __init__(self, rule: str, reason: str) -> None:
        super().__init__(f"{rule}: {reason}")
        self.rule = rule
        self.reason = reason


def _squash(text: str) -> str:
    """Текст без пробелов и в нижнем регистре: «C A N A R Y _ a 7 f 3 …» тоже метка."""
    return "".join(text.split()).lower()


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("ё", "е").split())


def leaked_canary(answer: str, canary: str | None) -> bool:
    if not canary:
        return False
    secret = canary.removeprefix(CANARY_PREFIX).lower()
    squashed = _squash(answer)
    return canary.lower() in squashed or (len(secret) >= 8 and secret in squashed)


def leaked_prompt_head(answer: str, system_prompt: str | None) -> bool:
    if not system_prompt:
        return False
    head = _normalize(system_prompt)[:PROMPT_HEAD_CHARS]
    return len(head) >= 20 and head in _normalize(answer)


def leaked_prompt(answer: str, system_prompt: str | None) -> bool:
    return leaked_prompt_head(answer, system_prompt) or (
        bool(system_prompt) and leaks_instructions(answer, system_prompt or ""))


def check_output(answer: str, system_prompt: str | None, canary: str | None) -> None:
    """Поднимает OutputBlocked, если ответ нельзя отдавать. Персональные данные не трогает —
    так проверку можно делать и на ещё не законченном ответе в потоке."""
    if leaked_canary(answer, canary):
        raise OutputBlocked("canary", "canary token in the answer")
    if leaked_prompt(answer, system_prompt):
        raise OutputBlocked("prompt_leak", "system prompt in the answer")
    marker = JAILBREAK_MARKERS.search(answer)
    if marker:
        raise OutputBlocked("jailbreak", f"jailbreak persona marker {marker.group(0)!r}")


def filter_output(answer: str, system_prompt: str | None, canary: str | None) -> str:
    """Режет утечку системного промпта и маскирует персональные данные в ответе."""
    check_output(answer, system_prompt, canary)
    return redact_pii(answer)


class StreamGuard:
    """Та же проверка для /chat/stream.

    Последние символы ответа придерживаются: клиент получает текст, только когда за ним
    пришло ещё HOLD_CHARS символов, и только до последнего пробела. За это время метка
    (15 символов, через пробелы — 29), начало промпта и роль джейлбрейка видны целиком и
    не уходят клиенту даже частично. HOLD_CHARS = 80 — когда есть промпт ассистента: его
    начало (80 символов) должно быть видно целиком. Без него (свой system клиента, чаты
    блока 4.1) достаточно SHORT_HOLD_CHARS = 40: метка через пробелы — 29 символов, роль
    джейлбрейка — до 22, номер с пробелами — до 23. Короткий ответ тогда тоже приходит
    кусками, а не одним блоком в конце. Резать по пробелу нужно для маскирования:
    email, ключ или номер без пробелов никогда не попадает на границу куска, а номера с
    пробелами (телефон, карта, паспорт, СНИЛС) граница обходит. Поэтому склеенный поток
    совпадает с redact_pii всего ответа. «Слово» без пробелов придерживается целиком,
    пока не закончится, — но не длиннее MAX_WORD_CHARS: такой ряд (base64-блоб, «xxxx…»)
    режется без пробела, иначе каждый фрагмент проверял бы весь растущий хвост.

    Работа на фрагмент — линейная: метка, начало промпта и роль ищутся в хвосте (ещё не
    отданное плюс CONTEXT_CHARS уже отданного), маскируется только отдаваемый кусок.
    Правила промпта целиком проверяются один раз, в конце ответа. Цена — первый фрагмент
    приходит позже, на 40–80 символов ответа."""

    HOLD_CHARS = 80
    SHORT_HOLD_CHARS = 40
    CONTEXT_CHARS = 200
    MAX_WORD_CHARS = 1000
    _SPACED_PII = tuple(PII_PATTERNS[name] for name in ("SNILS", "PHONE_RU", "CARD", "PASSPORT"))

    def __init__(self, system_prompt: str | None, canary: str | None) -> None:
        self.system_prompt = system_prompt
        self.canary = canary
        self.hold = self.HOLD_CHARS if system_prompt else self.SHORT_HOLD_CHARS
        self.raw = ""
        self.done = 0          # сколько символов ответа модели уже отдано (в маскированном виде)

    def _check_tail(self) -> None:
        window = self.raw[max(0, self.done - self.CONTEXT_CHARS):]
        if leaked_canary(window, self.canary):
            raise OutputBlocked("canary", "canary token in the answer")
        if leaked_prompt_head(window, self.system_prompt):
            raise OutputBlocked("prompt_leak", "system prompt in the answer")
        marker = JAILBREAK_MARKERS.search(window)
        if marker:
            raise OutputBlocked("jailbreak", f"jailbreak persona marker {marker.group(0)!r}")

    def _cut(self, tail: str) -> int:
        """Сколько символов хвоста можно отдать: до последнего пробела перед придержанными
        символами и не посреди номера с пробелами."""
        limit = len(tail) - self.hold
        if limit <= 0:
            return 0
        cut = max(tail.rfind(ch, 0, limit) for ch in " \n\t") + 1
        if cut == 0 and limit > self.MAX_WORD_CHARS:
            return limit
        for pattern in self._SPACED_PII:
            for match in pattern.finditer(tail, 0, limit + self.hold):
                if match.start() < cut < match.end():
                    cut = match.start()
        return cut

    def feed(self, text: str) -> str:
        """Новый фрагмент модели -> что можно отдать клиенту (может быть пусто)."""
        self.raw += text
        self._check_tail()
        tail = self.raw[self.done:]
        cut = self._cut(tail)
        if cut <= 0:
            return ""
        self.done += cut
        return redact_pii(tail[:cut])

    def finish(self) -> str:
        """Поток модели закончился — полная проверка (в том числе правил промпта) и остаток."""
        check_output(self.raw, self.system_prompt, self.canary)
        tail, self.done = self.raw[self.done:], len(self.raw)
        return redact_pii(tail)
