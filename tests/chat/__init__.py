"""
Тесты чата (блок 4.1) — для pytest: pytest tests/chat/.

Папка — пакет, чтобы её conftest.py не подменял tests/unit/conftest.py: у обоих одно имя
модуля conftest, а тесты tests/unit делают «from conftest import ...».
"""


def load_tests(loader, standard_tests, pattern):  # noqa: ARG001 — сигнатура протокола unittest
    """python -m unittest discover -s tests: тесты этой папки — async-функции с фикстурами
    pytest, unittest их не запускает."""
    import unittest

    return unittest.TestSuite()
