"""
Тесты маскирования PII (блок 3.6). Запуск любым из раннеров:
    python -m unittest discover -s tests -v
    pytest tests/test_pii.py -v

Тесты падают, если маскирование снять или ослабить шаблон: в prompt_preview не должно
остаться ни одного фрагмента исходных персональных данных.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.observability.pii import PREVIEW_CHARS, prompt_hash, prompt_preview, redact_pii  # noqa: E402


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


class TestTaskExamples(unittest.TestCase):
    def test_email_phone_card_from_assignment(self):
        raw = "Мой email ivan@mail.ru, тел +7 (999) 123-45-67, карта 4111 1111 1111 1111"
        preview = prompt_preview(raw)
        for placeholder in ("[EMAIL]", "[PHONE_RU]", "[CARD]"):
            self.assertIn(placeholder, preview)
        for fragment in ("ivan", "mail.ru", "999", "123-45-67", "4111", "1111"):
            self.assertNotIn(fragment, preview)
        self.assertEqual(_digits(preview), "")          # ни одной цифры исходных данных

    def test_criterion_example(self):
        preview = prompt_preview("email@example.com, +7 999 123 45 67")
        self.assertEqual(preview, "[EMAIL], [PHONE_RU]")


class TestPatterns(unittest.TestCase):
    def test_phone_formats(self):
        for phone in ("+7 (999) 123-45-67", "+7 999 123 45 67", "+79991234567",
                      "8 (999) 123-45-67", "89991234567", "8-999-123-45-67"):
            with self.subTest(phone=phone):
                self.assertEqual(redact_pii(f"звоните: {phone}."), "звоните: [PHONE_RU].")

    def test_phone_at_start_of_text(self):
        self.assertEqual(redact_pii("+7 999 123-45-67 — мой номер"), "[PHONE_RU] — мой номер")

    def test_cards(self):
        for card in ("4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111111", "8600 1234 5678 9012"):
            with self.subTest(card=card):
                self.assertEqual(redact_pii(f"карта {card}"), "карта [CARD]")

    def test_inn_and_passport(self):
        self.assertEqual(redact_pii("ИНН 7707083893"), "ИНН [INN]")
        self.assertEqual(redact_pii("ИНН 500100732259"), "ИНН [INN]")
        self.assertEqual(redact_pii("паспорт 45 09 123456"), "паспорт [PASSPORT]")
        self.assertEqual(redact_pii("паспорт 4509 123456"), "паспорт [PASSPORT]")

    def test_emails(self):
        for email in ("ivan@mail.ru", "i.petrov+support@example.co.uk", "user_1@sub.domain.org"):
            with self.subTest(email=email):
                self.assertEqual(redact_pii(f"<{email}>"), "<[EMAIL]>")

    def test_ordinary_text_is_untouched(self):
        text = "Заказ №12345 от 05.10.2026, сумма 1500 ₽. Не приходит письмо для сброса пароля."
        self.assertEqual(redact_pii(text), text)


class TestHashAndPreview(unittest.TestCase):
    def test_prompt_hash(self):
        value = prompt_hash("Как сбросить пароль?")
        self.assertRegex(value, r"^sha256:[0-9a-f]{16}$")
        self.assertEqual(value, prompt_hash("Как сбросить пароль?"))
        self.assertNotEqual(value, prompt_hash("Как сбросить пароль!"))

    def test_preview_is_short_and_masked(self):
        raw = "Мой email ivan@mail.ru. " + "Подробности. " * 30
        preview = prompt_preview(raw)
        self.assertLessEqual(len(preview), PREVIEW_CHARS)
        self.assertTrue(preview.startswith("Мой email [EMAIL]."))


if __name__ == "__main__":
    unittest.main()
