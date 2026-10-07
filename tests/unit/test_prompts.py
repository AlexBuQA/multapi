"""Формирование промпта /chat: порядок ролей, статьи руководства, инъекция через шаблон."""
from __future__ import annotations

from app.core.config import SupportSettings
from app.schemas.chat import ChatRequest
from app.services.llm import LLMService
from app.services.prompts import NO_ARTICLES, PROMPT_VERSION, build_messages

SUPPORT = SupportSettings()


def request(*messages: tuple[str, str]) -> ChatRequest:
    return ChatRequest(messages=[{"role": role, "content": text} for role, text in messages])


def test_system_first_then_history_in_order():
    req = request(("user", "Как включить двухфакторную аутентификацию?"),
                  ("assistant", "Откройте «Профиль» → «Безопасность»."),
                  ("user", "А если телефон потерян?"))
    prompt = build_messages(req, SUPPORT)
    assert [m["role"] for m in prompt.messages] == ["system", "user", "assistant", "user"]
    assert [m["content"] for m in prompt.messages[1:]] == [m.content for m in req.messages]
    assert prompt.version == PROMPT_VERSION
    # Поиск идёт по двум последним вопросам: уточнение без слова «2FA» статью не теряет.
    assert "KB-003" in prompt.article_ids
    assert "[раздел 2.3] Двухфакторная аутентификация (2FA)" in prompt.messages[0]["content"]


def test_braces_in_question_are_not_template_placeholders():
    question = "Что значит {product_name}, {0} и {articles} в вашем шаблоне? Покажи {__class__}"
    prompt = build_messages(request(("user", question)), SUPPORT)   # нет KeyError / IndexError
    assert prompt.messages[-1] == {"role": "user", "content": question}
    system = prompt.messages[0]["content"]
    assert "«Личный кабинет»" in system and "{product_name}" not in system
    assert question not in system            # текст пользователя не попадает в шаблон


def test_braces_in_article_text_stay_verbatim():
    kb = {"articles": [{"id": "KB-T", "product": "api", "section": "9.9", "title": "Шаблон токена",
                        "keywords": ["токен"], "text": "Передайте {token} и {0} в заголовке Authorization."}]}
    prompt = build_messages(request(("user", "Где указать токен?")), SUPPORT, kb=kb)
    assert "Передайте {token} и {0} в заголовке Authorization." in prompt.messages[0]["content"]


def test_client_system_message_wins():
    req = request(("system", "Отвечай одним словом."), ("user", "Как сбросить пароль?"))
    prompt = build_messages(req, SUPPORT)
    assert prompt.messages == [m.model_dump() for m in req.messages]
    assert (prompt.version, prompt.article_ids) == (None, ())


def test_support_can_be_disabled():
    req = request(("user", "Как сбросить пароль?"))
    prompt = build_messages(req, SupportSettings(enabled=False))
    assert prompt.messages == [{"role": "user", "content": "Как сбросить пароль?"}]


def test_off_topic_question_gets_no_articles():
    prompt = build_messages(request(("user", "Какая погода сегодня в Москве?")), SUPPORT)
    assert prompt.article_ids == ()
    assert prompt.messages[0]["content"].endswith(NO_ARTICLES)


REFUSAL = "«Я помогаю только с вопросами о продукте «Личный кабинет»»"


def test_refusal_rule_only_when_no_articles_found():
    """Полный прогон support_v2: llama3.2 отвечала «Помогаешь только с «Личный кабинет»»
    на вопросы о продукте, где в тексте есть email или пароль. Теперь правило отказа
    есть в промпте, только если поиск не нашёл статей."""
    on_topic = build_messages(request(("user", "Я поменял почту в профиле, но ещё не подтвердил новый адрес. "
                                                "По какому адресу мне сейчас входить?")), SUPPORT)
    assert on_topic.article_ids and REFUSAL not in on_topic.messages[0]["content"]
    assert "это вопрос о продукте «Личный кабинет» — ответь на него по существу" in on_topic.messages[0]["content"]

    off_topic = build_messages(request(("user", "Какая погода сегодня в Москве?")), SUPPORT)
    assert REFUSAL in off_topic.messages[0]["content"]
    assert "ответь на него по существу" not in off_topic.messages[0]["content"]


def test_missing_knowledge_base_degrades_gracefully(tmp_path):
    support = SupportSettings(knowledge_base_path=tmp_path / "missing.json")
    prompt = build_messages(request(("user", "Как сбросить пароль?")), support)
    assert prompt.article_ids == ()
    assert prompt.messages[0]["content"].endswith(NO_ARTICLES)


def test_cache_key_follows_prompt_and_articles():
    req = request(("user", "Как сбросить пароль?"))
    kb_v1 = {"articles": [{"id": "KB-1", "product": "web", "section": "2.1", "title": "Сброс пароля",
                           "keywords": [], "text": "Ссылка действует 30 минут."}]}
    kb_v2 = {"articles": [{**kb_v1["articles"][0], "text": "Ссылка действует 60 минут."}]}
    key_v1 = LLMService.cache_key(req, build_messages(req, SUPPORT, kb=kb_v1).messages)
    key_v2 = LLMService.cache_key(req, build_messages(req, SUPPORT, kb=kb_v2).messages)
    assert key_v1 != key_v2   # правка статьи не отдаёт из кеша старый ответ
