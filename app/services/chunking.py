"""
Три стратегии нарезки документов на фрагменты (блок 5.4) — сравниваются в
scripts/chunking_experiment.py, итог — docs/chunking_experiment.md.

    fixed      TokenTextSplitter(chunk_size, chunk_overlap) — baseline: окна по токенам, границы
               предложений не учитываются (режет по пробелу ближе к пределу).
    recursive  SentenceSplitter(chunk_size, chunk_overlap, paragraph_separator="\\n\\n",
               chunking_tokenizer_fn=split_sentences_ru) — аналог RecursiveCharacterTextSplitter
               из LangChain: абзацы → предложения → части предложений → слова, пока кусок не
               влезет в chunk_size, затем куски склеиваются в фрагменты с перекрытием.
    semantic   SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95,
               embed_model=модель индекса) — граница там, где смысл соседних предложений
               расходится сильнее, чем у 95 % соседних пар. Размер фрагмента не задаётся.

Токены считает tiktoken (cl100k_base), как у SentenceSplitter в блоке 5.3. Предложения — русская
модель NLTK Punkt (punkt_tab/russian), она лежит в данных NLTK внутри llama-index-core: без неё
сплиттеры LlamaIndex режут по необученной модели и путают сокращения («т. е.», «руб.») с концом
предложения.
"""
from __future__ import annotations

import statistics
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.node_parser import NodeParser
    from llama_index.core.schema import BaseNode

Strategy = Literal["fixed", "recursive", "semantic"]
STRATEGIES: tuple[Strategy, ...] = ("fixed", "recursive", "semantic")
SEMANTIC_BUFFER_SIZE = 1
SEMANTIC_BREAKPOINT = 95

_punkt: Any = None
_punkt_lock = threading.Lock()


def _russian_punkt() -> Any:
    """Модель Punkt для русского из данных NLTK, которые везёт llama-index-core (ничего не скачивает)."""
    global _punkt
    with _punkt_lock:
        if _punkt is None:
            from llama_index.core.utils import globals_helper
            from nltk.tokenize.punkt import PunktTokenizer

            _ = globals_helper.punkt_tokenizer   # добавляет папку данных NLTK из llama-index-core в nltk.data.path
            _punkt = PunktTokenizer("russian")
        return _punkt


def split_sentences_ru(text: str) -> list[str]:
    """Предложения русского текста. Пробелы и переносы между предложениями остаются в конце
    предыдущего — склеенные части дают исходный текст, как у сплиттеров LlamaIndex."""
    spans = list(_russian_punkt().span_tokenize(text))
    if not spans:
        return [text] if text else []
    starts = [0] + [start for start, _ in spans[1:]]
    ends = [start for start, _ in spans[1:]] + [len(text)]
    return [text[start:end] for start, end in zip(starts, ends, strict=True)]


# ---------------------------------------------------------------- стратегии
def fixed_size(chunk_size: int = 512, chunk_overlap: int = 64) -> TokenTextSplitter:
    """Baseline: окна по chunk_size токенов с перекрытием, граница — пробел, а не конец предложения."""
    return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def recursive(chunk_size: int = 512, chunk_overlap: int = 64) -> SentenceSplitter:
    """Абзацы (пустая строка) → русские предложения → части предложений → слова."""
    return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, paragraph_separator="\n\n",
                            chunking_tokenizer_fn=split_sentences_ru)


def semantic(embed_model: BaseEmbedding, buffer_size: int = SEMANTIC_BUFFER_SIZE,
             breakpoint_percentile_threshold: int = SEMANTIC_BREAKPOINT) -> SemanticSplitterNodeParser:
    """Границы по смыслу: эмбеддинг каждого предложения с buffer_size соседями с каждой стороны;
    граница — где косинусное расстояние до следующей группы выше перцентиля. embed_model — тот же,
    что у индекса: иначе «смысл» при нарезке и при поиске меряется разными моделями."""
    return SemanticSplitterNodeParser.from_defaults(
        embed_model=embed_model, buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold, sentence_splitter=split_sentences_ru)


def build_splitter(strategy: str, *, chunk_size: int = 512, chunk_overlap: int = 64,
                   embed_model: BaseEmbedding | None = None, buffer_size: int = SEMANTIC_BUFFER_SIZE,
                   breakpoint_percentile_threshold: int = SEMANTIC_BREAKPOINT) -> NodeParser:
    if strategy == "fixed":
        return fixed_size(chunk_size, chunk_overlap)
    if strategy == "recursive":
        return recursive(chunk_size, chunk_overlap)
    if strategy == "semantic":
        if embed_model is None:
            raise ValueError("semantic: нужна модель эмбеддингов индекса (embed_model)")
        return semantic(embed_model, buffer_size, breakpoint_percentile_threshold)
    raise ValueError(f"Неизвестная стратегия нарезки {strategy!r}: {', '.join(STRATEGIES)}")


def label(strategy: str, *, chunk_size: int = 512, chunk_overlap: int = 64, buffer_size: int = SEMANTIC_BUFFER_SIZE,
          breakpoint_percentile_threshold: int = SEMANTIC_BREAKPOINT) -> str:
    """Метка нарезки для payload коллекции: «recursive:512/64», «semantic:1/95»."""
    if strategy == "semantic":
        return f"semantic:{buffer_size}/{breakpoint_percentile_threshold}"
    return f"{strategy}:{chunk_size}/{chunk_overlap}"


# ---------------------------------------------------------------- статистика
def chunk_stats(nodes: Sequence[BaseNode], documents: int | None = None) -> dict[str, float | int]:
    """Сколько фрагментов, сколько на документ (документ — файл корпуса), средняя длина в токенах
    cl100k_base и в символах, самый короткий и самый длинный."""
    from app.services.embeddings import count_tokens

    tokens = [count_tokens(node.get_content()) for node in nodes]
    chars = [len(node.get_content()) for node in nodes]
    files = documents or len({node.metadata.get("file_name") for node in nodes}) or 1
    return {
        "chunks": len(nodes),
        "documents": files,
        "chunks_per_doc": round(len(nodes) / files, 1),
        "avg_tokens": round(statistics.mean(tokens)) if tokens else 0,
        "avg_chars": round(statistics.mean(chars)) if chars else 0,
        "min_tokens": min(tokens, default=0),
        "max_tokens": max(tokens, default=0),
    }
