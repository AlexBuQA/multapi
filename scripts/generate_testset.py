"""
Golden dataset для оценки RAG (блок 5.6): RAGAS TestsetGenerator на корпусе data/<категория>/ блока 5.5.

    python scripts/generate_testset.py                          # граф знаний + 50 пар → tests/eval/golden_dataset_raw.csv
    python scripts/generate_testset.py --size 60 --seed 7       # граф уже построен (var/testset_kg.json) — только вопросы
    python scripts/generate_testset.py --rebuild-kg             # построить граф заново

Как устроено:
1. Документы — те же, что индексирует сервис (app/services/ingestion.py: read_file по list_files, с
   .ragignore): раздел Markdown, страница PDF, статья HTML, файл DOCX — 102 документа по 78–200 токенов.
   Загрузки (data/uploads/) не входят: это не проверенная база знаний, а то, что прислали пользователи.
2. Граф знаний (KnowledgeGraph): узел на документ и преобразования RAGAS для документов 101–500 токенов —
   краткое содержание и его эмбеддинг, темы, именованные сущности, связи по близости содержаний и по
   общим сущностям. Фильтр узлов CustomNodeFilter (вызов судьи на каждый узел: годится ли он для
   вопроса) убран: корпус — проверенные статьи справки, а на CPU это ещё ~100 вызовов. Граф —
   самая дорогая часть (~300 вызовов модели), поэтому он сохраняется в var/testset_kg.json, и
   повторная генерация вопросов его не строит.
3. Вопросы — три синтезатора RAGAS: single-hop specific (вопрос по одному документу), multi-hop
   specific (по двум документам с общими сущностями) и multi-hop abstract (обобщающий по двум
   близким документам). Доли — 0.6 / 0.2 / 0.2, синтезатор без подходящих узлов графа пропускается.
   Персоны заданы, а не генерируются: четыре типичных пользователя «Личного кабинета». К промпту
   генерации добавлено требование писать вопрос и ответ по-русски: образцы RAGAS — английские, и
   маленькая модель иначе отвечает по-английски.
4. Сырой результат — tests/eval/golden_dataset_raw.csv: id, user_input, reference,
   reference_contexts (JSON-список), reference_sources (файлы корпуса, из которых контексты),
   synthesizer_name. Дальше — обязательная ручная вычитка (tests/eval/golden_review.yaml) и
   scripts/build_golden.py → tests/eval/golden_dataset.json.

Модель — судья RAGAS (RAGAS__JUDGE_*, по умолчанию qwen3:4b-instruct в Ollama), эмбеддинги — модель
поиска (bge-m3, кеш эмбеддингов сервиса). На CPU — 2–4 часа: граф ~300 вызовов, вопросы ~2 на пару.
Код выхода: 0 — готово, 2 — ошибка.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
# TestsetGenerator в RAGAS 0.4 принимает только «старые» LLM и эмбеддинги (BaseRagasLLM): llm_factory —
# для метрик, а генератору нужны LlamaIndexLLMWrapper и LlamaIndexEmbeddingsWrapper. Их пометка
# deprecated — про будущие версии, здесь другого пути нет.
warnings.filterwarnings("ignore", message=r"LlamaIndex(LLM|Embeddings)Wrapper is deprecated", category=DeprecationWarning)

DEFAULT_OUT = ROOT / "tests" / "eval" / "golden_dataset_raw.csv"
DEFAULT_KG = ROOT / "var" / "testset_kg.json"
EXCLUDED_PREFIXES = ("uploads/",)
PERSONAS = [
    ("Владелец небольшого бизнеса",
     "Сам платит за тариф «Личного кабинета» и настраивает аккаунт; спрашивает коротко и по делу."),
    ("Бухгалтер компании",
     "Отвечает за оплату по счёту, возвраты, закрывающие документы и обмен с 1С."),
    ("Разработчик интеграции",
     "Подключает API и вебхуки; спрашивает о ключах, лимитах, ошибках и версиях API."),
    ("Администратор команды",
     "Приглашает и удаляет сотрудников, назначает роли, следит за безопасностью входа и уведомлениями."),
]
RUSSIAN = ("\n5. **Language**: the documents are in Russian. Write both the query and the answer in Russian "
           "(вопрос и ответ — на русском языке), as a real user of the service would ask.")
LLM_CONTEXT = ("Вопросы пользователей сервиса «Личный кабинет» в поддержку: как что-то сделать, сколько стоит, "
               "какие сроки и ограничения. Вопрос и ответ — на русском языке, ответ — только по документам.")
WEIGHTS = {"single_hop_specific_query_synthesizer": 0.6, "multi_hop_specific_query_synthesizer": 0.2,
           "multi_hop_abstract_query_synthesizer": 0.2}


def corpus_documents(data_dir: Path) -> list[Any]:
    """Документы корпуса, как их видит индексация сервиса: текст и метаданные (source, page, …)."""
    from app.services.ingestion import list_files, read_file

    docs = []
    for path in list_files(data_dir):
        relative = path.relative_to(data_dir).as_posix()
        if relative.startswith(EXCLUDED_PREFIXES):
            continue
        docs.extend(read_file(path, data_dir))
    return docs


def transforms_for(docs: list[Any], llm: Any, embeddings: Any) -> list[Any]:
    """Преобразования графа RAGAS по длине документов, без CustomNodeFilter (см. описание модуля)."""
    from langchain_core.documents import Document as LCDocument
    from ragas.testset.transforms import default_transforms
    from ragas.testset.transforms.filters import CustomNodeFilter

    transforms = default_transforms(documents=[LCDocument(page_content=d.text) for d in docs], llm=llm,
                                    embedding_model=embeddings)
    return [t for t in transforms if not isinstance(t, CustomNodeFilter)]


def build_graph(docs: list[Any], llm: Any, embeddings: Any, run_config: Any) -> Any:
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.transforms import apply_transforms

    nodes = [Node(type=NodeType.DOCUMENT, properties={"page_content": d.text, "document_metadata": {
        "source": d.metadata.get("source"), "file_name": d.metadata.get("file_name"),
        "page": d.metadata.get("page"), "title": d.metadata.get("title")}}) for d in docs]
    graph = KnowledgeGraph(nodes=nodes)
    apply_transforms(graph, transforms_for(docs, llm, embeddings), run_config)
    return graph


def russian_synthesizer(synthesizer: Any) -> Any:
    """Требование писать по-русски — в конец инструкции промпта генерации вопроса и ответа."""
    prompts = synthesizer.get_prompts()
    prompt = prompts["query_answer_generation_prompt"]
    if RUSSIAN not in prompt.instruction:
        prompt.instruction += RUSSIAN
    synthesizer.set_prompts(query_answer_generation_prompt=prompt)
    return synthesizer


def query_distribution(llm: Any, graph: Any) -> list[tuple[Any, float]]:
    """Три синтезатора с долями WEIGHTS; без подходящих узлов графа синтезатор пропускается, доли
    остальных нормируются."""
    from ragas.testset.synthesizers import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )

    available = []
    for cls in (SingleHopSpecificQuerySynthesizer, MultiHopSpecificQuerySynthesizer, MultiHopAbstractQuerySynthesizer):
        synthesizer = russian_synthesizer(cls(llm=llm, llm_context=LLM_CONTEXT))
        try:
            clusters = synthesizer.get_node_clusters(graph)
        except Exception as exc:  # noqa: BLE001 — нет нужных свойств узлов: синтезатор не подходит
            print(f"  {synthesizer.name}: пропущен ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue
        if clusters:
            available.append(synthesizer)
        else:
            print(f"  {synthesizer.name}: пропущен — нет подходящих узлов графа", file=sys.stderr)
    if not available:
        raise ValueError("Ни один синтезатор RAGAS не нашёл в графе подходящих узлов")
    total = sum(WEIGHTS[s.name] for s in available)
    return [(s, WEIGHTS[s.name] / total) for s in available]


def _clean_context(text: str) -> str:
    """Контекст multi-hop синтезаторов начинается с метки «<1-hop>» — для поиска файла она не нужна."""
    lines = [line for line in str(text).splitlines() if not (line.startswith("<") and line.endswith("-hop>"))]
    return "\n".join(lines).strip()


def sources_of(contexts: list[str], docs: list[Any]) -> list[str]:
    """Файлы корпуса, из которых взяты эталонные контексты: по совпадению текста документа."""
    found: list[str] = []
    for context in contexts:
        text = _clean_context(context)
        for doc in docs:
            if text and (text in doc.text or doc.text in text) and doc.metadata.get("source") not in found:
                found.append(doc.metadata.get("source"))
    return found


def to_rows(frame: Any, docs: list[Any]) -> Any:
    """DataFrame RAGAS → строки сырого golden: id, вопрос, эталон, контексты и их файлы, синтезатор."""
    import pandas as pd

    rows = []
    for number, record in enumerate(frame.to_dict("records"), start=1):
        contexts = record.get("reference_contexts") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        contexts = [str(c) for c in contexts]
        rows.append({"id": f"raw_{number:03d}", "user_input": str(record.get("user_input") or "").strip(),
                     "reference": str(record.get("reference") or "").strip(),
                     "reference_contexts": json.dumps(contexts, ensure_ascii=False),
                     "reference_sources": json.dumps(sources_of(contexts, docs), ensure_ascii=False),
                     "synthesizer_name": record.get("synthesizer_name") or ""})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Golden dataset через RAGAS TestsetGenerator (блок 5.6)")
    parser.add_argument("--size", type=int, default=50, help="сколько пар сгенерировать (после вычитки нужно ≥ 30)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="сырой CSV")
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG, help="граф знаний: есть — берётся, нет — строится и сохраняется")
    parser.add_argument("--rebuild-kg", action="store_true", help="построить граф заново, даже если файл есть")
    parser.add_argument("--seed", type=int, default=42, help="seed выбора узлов и персон")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    from pydantic import ValidationError

    from app.core.config import get_settings
    from app.observability.logging import setup_logging
    from app.services.embeddings import EmbeddingError, default_service
    from app.services.ingestion import ProjectEmbedding
    from app.services.rag import no_implicit_event_loop

    setup_logging("WARNING", stream=sys.stderr)
    no_implicit_event_loop()
    try:
        from ragas.embeddings import LlamaIndexEmbeddingsWrapper
        from ragas.llms import LlamaIndexLLMWrapper
        from ragas.run_config import RunConfig
        from ragas.testset import TestsetGenerator
        from ragas.testset.graph import KnowledgeGraph
        from ragas.testset.persona import Persona

        from app.eval.metrics import build_judge_llama_index, judge_info

        cfg = get_settings()
        started = time.perf_counter()
        docs = corpus_documents(cfg.rag.data_dir)
        print(f"Корпус {cfg.rag.data_dir.name}/: {len(docs)} документов; судья {judge_info(cfg)['judge_model']}",
              file=sys.stderr)
        llm = LlamaIndexLLMWrapper(build_judge_llama_index(cfg))
        embeddings = LlamaIndexEmbeddingsWrapper(ProjectEmbedding(default_service()))
        run_config = RunConfig(timeout=int(cfg.ragas.judge_timeout), max_workers=cfg.ragas.concurrency,
                               max_retries=3, seed=args.seed)
        if args.kg.is_file() and not args.rebuild_kg:
            graph = KnowledgeGraph.load(str(args.kg))
            print(f"Граф знаний — из {args.kg} ({len(graph.nodes)} узлов, {len(graph.relationships)} связей)",
                  file=sys.stderr)
        else:
            print("Строю граф знаний: краткие содержания, темы, сущности, связи…", file=sys.stderr)
            graph = build_graph(docs, llm, embeddings, run_config)
            args.kg.parent.mkdir(parents=True, exist_ok=True)
            graph.save(str(args.kg))
            print(f"Граф — {args.kg}: {len(graph.nodes)} узлов, {len(graph.relationships)} связей", file=sys.stderr)
        generator = TestsetGenerator(llm=llm, embedding_model=embeddings, knowledge_graph=graph,
                                     persona_list=[Persona(name=n, role_description=d) for n, d in PERSONAS],
                                     llm_context=LLM_CONTEXT)
        testset = generator.generate(testset_size=args.size, query_distribution=query_distribution(llm, graph),
                                     num_personas=len(PERSONAS), run_config=run_config)
        rows = to_rows(testset.to_pandas(), docs)
    except (EmbeddingError, ValidationError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out, index=False, encoding="utf-8")
    by_kind = rows["synthesizer_name"].value_counts().to_dict() if len(rows) else {}
    print(f"\n{len(rows)} пар → {args.out} ({', '.join(f'{k}: {v}' for k, v in by_kind.items())}); "
          f"{(time.perf_counter() - started) / 60:.1f} мин. Дальше — ручная вычитка: tests/eval/golden_review.yaml "
          "и python scripts/build_golden.py", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
