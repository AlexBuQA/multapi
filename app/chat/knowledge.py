"""
База знаний в чатах (блок 5.5): RAG для POST /chats/{id}/messages.

История диалога уже на сервере (блок 4.1: ChatService, окно по бюджету токенов), второй store не
нужен. На каждый текстовый вопрос:

1. condense (RAG__CONDENSE, по умолчанию включён) — только если в чате уже были вопросы: один вызов
   модели переписывает follow-up («А при оплате по счёту?») в самостоятельный вопрос с учётом
   последних CONDENSE_TURNS сообщений этого чата (ключ — chat_id). Он нужен только поиску: вектор
   видит одну строку, и на коротком follow-up она бессмысленна. Модель для ответа всё равно получает
   историю целиком. Модель не ответила — ищем по вопросу как есть;
2. поиск — RAGService.retrieve (тот же, что у /rag/query): 10 кандидатов, score-guard, re-ranker, 5;
3. лучший фрагмент ниже RAG__SCORE_THRESHOLD — ответ «По базе не нашёл, могу эскалировать.» без
   вызова модели (score-guard, в лог — rag_score_guard);
4. иначе в запрос к модели добавляются правило «только по контексту, с цитатами [N]» (в конце
   системного промпта чата) и контекст — отдельным системным сообщением сразу после него, текстом
   вложения (media="document"): защитный слой (блок 3.8) проверяет его шаблонами инъекции, но не
   пределом длины вопроса, маскирование персональных данных его не трогает (телефон поддержки — не
   чужие данные), а бюджет токенов не вытесняет — режется старая история;
5. после ответа — событие sources (номера, файлы, страницы, близость, cited) и done.

Цитаты [N] в ответах из истории в следующем ходу вычищаются (strip_citations): у фрагментов нового
контекста другие номера, и старые [1] модель приняла бы за ссылки на них.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.chat.domain import ChatMessage, RequestError
from app.services.rag_common import CHAT_CONTEXT, mark_cited, strip_citations

CONDENSE_TURNS = 6            # последних сообщений чата для переписывания follow-up
CONDENSE_CHARS = 400          # столько символов каждого сообщения видит condense
CONDENSE_MAX_TOKENS = 96
QUERY_MAX_CHARS = 300
CONDENSE_SYSTEM = (
    "Ты переписываешь вопросы для поиска по базе знаний. Перепиши последний вопрос пользователя так, чтобы он "
    "был понятен без диалога: подставь из диалога, о чём идёт речь. Ответь одним вопросом на русском языке, без "
    "пояснений и кавычек. Если вопрос и так понятен без диалога — повтори его без изменений."
)
CONDENSE_USER = "Диалог:\n{dialog}\n\nПоследний вопрос: {question}\nСамостоятельный вопрос:"
# Образцы перед вопросом: llama3.2 на Windows без них переписала «А при оплате по счёту?» после
# вопроса о возврате в «Какова сумма, которую я могу заплатить по счёту?» — тема возврата пропала, и
# поиск ушёл не туда (docs/rag.md, «Condense (Windows)»). Три случая: подставить новую деталь, добавить
# условие, оставить самостоятельный вопрос как есть. Тема — библиотека, чтобы модель не перенесла
# факты образцов в вопрос о продукте.
CONDENSE_EXAMPLES = (
    ("Пользователь: На сколько дней выдают книгу?\nАссистент: Книгу выдают на 21 день.", "А учебник?",
     "На сколько дней выдают учебник?"),
    ("Пользователь: Как продлить книгу?\nАссистент: Продлить книгу можно в разделе «Мои книги».", "А через приложение?",
     "Как продлить книгу через приложение?"),
    ("Пользователь: Как продлить книгу?\nАссистент: Продлить книгу можно в разделе «Мои книги».",
     "Где находится читальный зал?", "Где находится читальный зал?"),
)
ROLE_NAMES = {"user": "Пользователь", "assistant": "Ассистент"}


@dataclass
class KnowledgeTurn:
    """Поиск по базе для одного хода чата."""

    retrieval: Any            # app.services.rag.Retrieval
    query: str                # вопрос, по которому искали (после condense — переписанный)
    condensed: bool

    @property
    def confident(self) -> bool:
        return bool(self.retrieval.confident)

    def sources(self, answer: str) -> list[dict[str, Any]]:
        return mark_cited(self.retrieval.sources(), answer)

    def texts(self) -> list[str]:
        """Тексты фрагментов контекста в порядке номеров [1], [2]… — для дописывания ссылки."""
        return [node.node.get_content() for node in self.retrieval.nodes]

    def stats(self) -> dict[str, Any]:
        return {"rag_top_score": round(self.retrieval.top_score, 3), "rag_sources": len(self.retrieval.nodes),
                "rag_condensed": self.condensed}


def _short(text: str) -> str:
    text = " ".join(strip_citations(text).split())
    return text if len(text) <= CONDENSE_CHARS else text[:CONDENSE_CHARS - 1] + "…"


def condense_messages(prior: list[ChatMessage], question: str, *,
                      examples: bool = True) -> list[dict[str, Any]] | None:
    """Запрос condense по последним сообщениям чата (до текущего вопроса): системное правило,
    образцы CONDENSE_EXAMPLES парами «вопрос — переписанный вопрос» и сам вопрос последним
    сообщением. Вопросов ещё не было — None: переписывать нечего."""
    turns = [m for m in prior if m.role in ROLE_NAMES][-CONDENSE_TURNS:]
    if not any(m.role == "user" for m in turns):
        return None
    dialog = "\n".join(f"{ROLE_NAMES[m.role]}: {_short(m.content)}" for m in turns)
    messages: list[dict[str, Any]] = [{"role": "system", "content": CONDENSE_SYSTEM}]
    for demo_dialog, demo_question, demo_answer in CONDENSE_EXAMPLES if examples else ():
        messages += [{"role": "user", "content": CONDENSE_USER.format(dialog=demo_dialog, question=demo_question)},
                     {"role": "assistant", "content": demo_answer}]
    messages.append({"role": "user", "content": CONDENSE_USER.format(dialog=dialog, question=question)})
    return messages


def clean_query(text: str) -> str | None:
    """Ответ condense → вопрос: первая непустая строка без кавычек и подписи «Самостоятельный вопрос:»."""
    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    for prefix in ("Самостоятельный вопрос:", "Вопрос:"):
        if line.lower().startswith(prefix.lower()):
            line = line[len(prefix):].strip()
    line = line.strip("«»\"'“”„ ").strip()
    return line[:QUERY_MAX_CHARS] or None


def with_knowledge(messages: list[dict[str, Any]], retrieval: Any) -> list[dict[str, Any]]:
    """Контекст — системным сообщением после системного промпта; цитаты в прежних ответах — убрать."""
    head = next((i for i, m in enumerate(messages) if m["role"] != "system"), len(messages))
    context = {"role": "system", "content": [{"type": "text", "media": "document",
                                              "text": CHAT_CONTEXT.format(context_str=retrieval.context())}]}
    body = [{**m, "content": strip_citations(m["content"])} if m["role"] == "assistant" and isinstance(m["content"], str)
            else m for m in messages[head:]]
    return [*messages[:head], context, *body]


def rag_error(exc: BaseException) -> RequestError | None:
    """Ошибка поиска по базе → ответ клиенту до начала потока (503). Не ошибка RAG — None."""
    from app.services import embeddings
    from app.services.rag_common import RAGError, RAGNotIndexed, RAGUnavailable
    from app.services.reranker import RerankError

    if isinstance(exc, RAGNotIndexed):
        return RequestError(503, "rag_not_indexed", "База знаний ещё индексируется — повторите вопрос через минуту.")
    if isinstance(exc, RAGUnavailable):
        return RequestError(503, "vector_store_unavailable", "База знаний недоступна: нет связи с Qdrant.")
    if isinstance(exc, embeddings.EmbeddingError):
        return RequestError(503, "embeddings_unavailable", "База знаний недоступна: модель эмбеддингов не отвечает.")
    if isinstance(exc, RerankError):
        return RequestError(503, "rerank_unavailable", "База знаний недоступна: re-ranker не отвечает.")
    if isinstance(exc, RAGError):
        return RequestError(503, "rag_error", "База знаний недоступна.")
    return None
