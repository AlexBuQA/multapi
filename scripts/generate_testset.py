"""
Golden dataset для оценки RAG (блок 5.6): RAGAS TestsetGenerator на корпусе data/<категория>/ блока 5.5.

    python scripts/generate_testset.py                          # граф знаний + 40 пар → tests/eval/golden_dataset_raw.csv
    python scripts/generate_testset.py --size 50 --seed 7       # граф уже построен (var/testset_kg.json) — только вопросы
    python scripts/generate_testset.py --rebuild-kg             # построить граф заново
    python scripts/generate_testset.py --abstract               # плюс обобщающие multi-hop вопросы (+~200 вызовов)
    python scripts/generate_testset.py --append --size 16 --single-hop-only   # добавить пары по ещё не взятым документам

Как устроено:
1. Документы — те же, что индексирует сервис (app/services/ingestion.py: read_file по list_files, с
   .ragignore): раздел Markdown, страница PDF, статья HTML, файл DOCX — 102 документа по 78–200 токенов.
   Загрузки (data/uploads/) не входят: это не проверенная база знаний, а то, что прислали пользователи.
2. Модель — судья RAGAS (RAGAS__JUDGE_*, по умолчанию qwen3:4b-instruct в Ollama) через llm_factory, как у
   метрик: JSON-режим и проверка ответа по схеме с повтором. Первая версия шла через LlamaIndexLLMWrapper —
   без JSON-режима qwen3:4b-instruct на Windows отвечал краткими содержаниями обычным текстом, и все
   вызовы SummaryExtractor падали с «Invalid json output». Ответы модели — в кеше судьи (var/ragas_cache):
   после сбоя повторный запуск не платит за готовое.
3. Граф знаний (KnowledgeGraph): узел на документ, именованные сущности каждого документа (NERExtractor —
   один вызов модели на документ, до 6 сущностей) и связи по общим сущностям (OverlapScoreBuilder, без модели). Этого
   хватает двум синтезаторам: single-hop specific (вопрос по одному документу) и multi-hop specific (по
   двум документам с общими сущностями). С --abstract — ещё краткие содержания, их эмбеддинги, темы и
   связи по близости содержаний для multi-hop abstract (обобщающий вопрос по двум близким документам):
   +2 вызова модели на документ, а такие вопросы на справке («как связаны X и Y?») чаще всего уходят при
   вычитке как слишком общие. Фильтр узлов CustomNodeFilter (вызов на документ: годится ли он для
   вопроса) не используется: корпус — проверенные статьи справки. К инструкциям извлечения добавлено
   «текст на русском — ответ на русском»: образцы RAGAS английские, и маленькая модель иначе переводит.
   Граф сохраняется в var/testset_kg.json; граф без сущностей (сбой модели) не используется повторно.
4. Вопросы — доли 0.7 / 0.3 (с --abstract — 0.6 / 0.2 / 0.2), синтезатор без подходящих узлов графа
   пропускается. Single-hop RAGAS берёт первые n узлов графа по порядку, поэтому узлы перед генерацией идут
   по кругу по категориям корпуса (внутри категории — в случайном порядке с --seed), связи — вперемешку:
   первая генерация на Windows шла по порядку файлов, и 28 вопросов по одному документу пришлись только на
   account, api и billing. --append дописывает пары в существующий CSV (id продолжаются) по документам,
   которых в нём ещё нет. Персоны заданы, а не генерируются: четыре типичных пользователя «Личного кабинета». К
   промпту генерации добавлено требование писать вопрос и ответ по-русски.
5. Сырой результат — tests/eval/golden_dataset_raw.csv: id, user_input, reference,
   reference_contexts (JSON-список), reference_sources (файлы корпуса, из которых контексты),
   synthesizer_name. Дальше — обязательная ручная вычитка (tests/eval/golden_review.yaml) и
   scripts/build_golden.py → tests/eval/golden_dataset.json.

На CPU — около 2 часов: граф ~100 вызовов модели, вопросы — ~2 на пару (с --abstract граф ~300 вызовов).
Код выхода: 0 — готово, 2 — ошибка.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
# Эмбеддинги графа (только --abstract) — модель поиска через LlamaIndexEmbeddingsWrapper: EmbeddingExtractor
# RAGAS 0.4 работает со «старым» интерфейсом эмбеддингов. Пометка deprecated — про будущие версии.
warnings.filterwarnings("ignore", message=r"LlamaIndexEmbeddingsWrapper is deprecated", category=DeprecationWarning)

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
RUSSIAN_EXTRACT = ("\nThe text is in Russian: write the output in Russian, with names, terms and section titles "
                   "exactly as in the text (ответ — на русском, как в тексте).")
LLM_CONTEXT = ("Вопросы пользователей сервиса «Личный кабинет» в поддержку: как что-то сделать, сколько стоит, "
               "какие сроки и ограничения. Вопрос и ответ — на русском языке, ответ — только по документам.")
WEIGHTS = {"single_hop_specific_query_synthesizer": 0.7, "multi_hop_specific_query_synthesizer": 0.3}
WEIGHTS_ABSTRACT = {"single_hop_specific_query_synthesizer": 0.6, "multi_hop_specific_query_synthesizer": 0.2,
                    "multi_hop_abstract_query_synthesizer": 0.2}
MIN_ENTITY_SHARE = 0.5      # граф, где сущности есть меньше чем у половины документов, — сбой модели
MAX_ENTITIES = 6


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


def russian_extractor(extractor: Any) -> Any:
    """Своя копия промпта извлечения с «текст на русском — ответ на русском» (промпт по умолчанию — общий
    объект класса, его не трогаем)."""
    prompt = copy.deepcopy(extractor.prompt)
    if RUSSIAN_EXTRACT not in prompt.instruction:
        prompt.instruction += RUSSIAN_EXTRACT
    extractor.prompt = prompt
    return extractor


FAILURES: list[str] = []


def tolerant(extractor: Any, empty: Any) -> Any:
    """Сбой модели на одном документе (не JSON и после повторов, таймаут) не обрывает граф: RAGAS дожидается
    всех задач преобразования и поднимает первое исключение — и сотня готовых извлечений пропала бы. Документ
    со сбоем получает пустое значение (в вопросы не попадёт), причина — в FAILURES и в итоге графа."""
    extract = extractor.extract

    async def safe_extract(node: Any) -> tuple[str, Any]:
        try:
            return await extract(node)
        except Exception as exc:  # noqa: BLE001 — любая ошибка модели на одном документе
            source = (node.properties.get("document_metadata") or {}).get("source") or str(node.id)
            FAILURES.append(f"{type(extractor).__name__} · {source}: {type(exc).__name__}: "
                            f"{' '.join(str(exc).split())[:160]}")
            return extractor.property_name, empty

    extractor.extract = safe_extract
    return extractor


def transforms_for(llm: Any, embeddings: Any, *, abstract: bool = False) -> list[Any]:
    """Преобразования графа (см. описание модуля): сущности и связи по ним; с abstract — как ветка
    default_transforms RAGAS для документов 101–500 токенов, без CustomNodeFilter."""
    from ragas.testset.graph import NodeType
    from ragas.testset.transforms import (
        CosineSimilarityBuilder,
        EmbeddingExtractor,
        OverlapScoreBuilder,
        Parallel,
        SummaryExtractor,
    )
    from ragas.testset.transforms.extractors.llm_based import (
        NERExtractor,
        ThemesExtractor,
    )
    from ragas.utils import num_tokens_from_string

    # 6 сущностей вместо 10 по умолчанию: ответ модели короче (на CPU это время), а для выбора темы вопроса
    # и связей между документами справки хватает.
    ner = tolerant(russian_extractor(NERExtractor(llm=llm, max_num_entities=MAX_ENTITIES)), [])
    overlap = OverlapScoreBuilder(threshold=0.01)
    if not abstract:
        return [ner, overlap]

    def long_doc(node: Any) -> bool:
        return node.type == NodeType.DOCUMENT and num_tokens_from_string(node.properties["page_content"]) > 100

    def summarized(node: Any) -> bool:            # краткое содержание есть — сбой модели его не оставил
        return long_doc(node) and bool(node.get_property("summary"))

    summary = tolerant(russian_extractor(SummaryExtractor(llm=llm, filter_nodes=long_doc)), None)
    summary_embedding = EmbeddingExtractor(embedding_model=embeddings, property_name="summary_embedding",
                                           embed_property_name="summary", filter_nodes=summarized)
    themes = tolerant(russian_extractor(ThemesExtractor(llm=llm, filter_nodes=lambda node: node.type == NodeType.DOCUMENT)), [])
    similarity = CosineSimilarityBuilder(property_name="summary_embedding", new_property_name="summary_similarity",
                                         threshold=0.5, filter_nodes=lambda node: bool(node.get_property("summary_embedding")))
    return [summary, Parallel(summary_embedding, themes, ner), Parallel(similarity, overlap)]


def build_graph(docs: list[Any], llm: Any, embeddings: Any, run_config: Any, *, abstract: bool = False) -> Any:
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.transforms import apply_transforms

    nodes = [Node(type=NodeType.DOCUMENT, properties={"page_content": d.text, "document_metadata": {
        "source": d.metadata.get("source"), "file_name": d.metadata.get("file_name"),
        "page": d.metadata.get("page"), "title": d.metadata.get("title")}}) for d in docs]
    graph = KnowledgeGraph(nodes=nodes)
    apply_transforms(graph, transforms_for(llm, embeddings, abstract=abstract), run_config)
    return graph


def graph_stats(graph: Any) -> dict[str, int]:
    """Сколько узлов получили сущности (и краткие содержания) и сколько связей каждого вида; сущности — непустой
    список (пустой — сбой модели на этом документе)."""
    stats = {"nodes": len(graph.nodes),
             "entities": sum(1 for n in graph.nodes if n.get_property("entities")),
             "summaries": sum(1 for n in graph.nodes if n.get_property("summary"))}
    for rel in graph.relationships:
        stats[rel.type] = stats.get(rel.type, 0) + 1
    return stats


def graph_usable(stats: dict[str, int], *, abstract: bool) -> bool:
    """Граф годится, если сущности есть хотя бы у половины документов (а с abstract — и краткие содержания)."""
    nodes = stats["nodes"] or 1
    ok = stats["entities"] / nodes >= MIN_ENTITY_SHARE
    return ok and (not abstract or stats["summaries"] / nodes >= MIN_ENTITY_SHARE)


def describe_graph(stats: dict[str, int]) -> str:
    links = ", ".join(f"{k} {v}" for k, v in stats.items() if k not in ("nodes", "entities", "summaries")) or "нет"
    return (f"{stats['nodes']} узлов, сущности у {stats['entities']}, краткие содержания у {stats['summaries']}; "
            f"связи: {links}")


def russian_synthesizer(synthesizer: Any) -> Any:
    """Требование писать по-русски — в конец инструкции промпта генерации вопроса и ответа."""
    prompts = synthesizer.get_prompts()
    prompt = prompts["query_answer_generation_prompt"]
    if RUSSIAN not in prompt.instruction:
        prompt.instruction += RUSSIAN
    synthesizer.set_prompts(query_answer_generation_prompt=prompt)
    return synthesizer


def category_of(node: Any) -> str:
    source = (node.properties.get("document_metadata") or {}).get("source") or ""
    return source.split("/", 1)[0]


def source_of(node: Any) -> str:
    return (node.properties.get("document_metadata") or {}).get("source") or ""


def balanced(graph: Any, seed: int, exclude_sources: set[str] | frozenset[str] = frozenset()) -> Any:
    """Тот же граф, но узлы — по кругу по категориям (account, api, billing, …), внутри категории — в
    случайном порядке; связи — вперемешку; без документов exclude_sources и их связей."""
    from ragas.testset.graph import KnowledgeGraph

    rng = random.Random(seed)
    groups: dict[str, list[Any]] = defaultdict(list)
    for node in graph.nodes:
        if source_of(node) not in exclude_sources:
            groups[category_of(node)].append(node)
    for nodes in groups.values():
        rng.shuffle(nodes)
    order: list[Any] = []
    while any(groups.values()):
        for name in sorted(groups):
            if groups[name]:
                order.append(groups[name].pop(0))
    kept = {node.id for node in order}
    relationships = [r for r in graph.relationships if r.source.id in kept and r.target.id in kept]
    rng.shuffle(relationships)
    return KnowledgeGraph(nodes=order, relationships=relationships)


def query_distribution(llm: Any, graph: Any, *, abstract: bool = False,
                       single_only: bool = False) -> list[tuple[Any, float]]:
    """Синтезаторы с долями WEIGHTS (с abstract — WEIGHTS_ABSTRACT); без подходящих узлов графа синтезатор
    пропускается, доли остальных нормируются."""
    from ragas.testset.synthesizers import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )

    weights = WEIGHTS_ABSTRACT if abstract else WEIGHTS
    classes = [SingleHopSpecificQuerySynthesizer] if single_only else [SingleHopSpecificQuerySynthesizer,
                                                                        MultiHopSpecificQuerySynthesizer]
    if abstract and not single_only:
        classes.append(MultiHopAbstractQuerySynthesizer)
    available = []
    for cls in classes:
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
    total = sum(weights[s.name] for s in available)
    return [(s, weights[s.name] / total) for s in available]


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


def to_rows(frame: Any, docs: list[Any], start: int = 1) -> Any:
    """DataFrame RAGAS → строки сырого golden: id, вопрос, эталон, контексты и их файлы, синтезатор."""
    import pandas as pd

    rows = []
    for number, record in enumerate(frame.to_dict("records"), start=start):
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


def read_existing(path: Path) -> tuple[list[dict[str, str]], set[str], int]:
    """Строки сырого CSV, файлы корпуса, по которым уже есть пары, и следующий номер id."""
    if not path.is_file():
        return [], set(), 1
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    used = {source for row in rows for source in json.loads(row.get("reference_sources") or "[]")}
    numbers = [int(row["id"].rsplit("_", 1)[-1]) for row in rows if row.get("id", "").rsplit("_", 1)[-1].isdigit()]
    return rows, used, max(numbers, default=0) + 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Golden dataset через RAGAS TestsetGenerator (блок 5.6)")
    parser.add_argument("--size", type=int, default=40, help="сколько пар сгенерировать (после вычитки нужно ≥ 30)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="сырой CSV")
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG, help="граф знаний: есть — берётся, нет — строится и сохраняется")
    parser.add_argument("--rebuild-kg", action="store_true", help="построить граф заново, даже если файл есть")
    parser.add_argument("--seed", type=int, default=42, help="seed выбора узлов и персон")
    parser.add_argument("--abstract", action="store_true",
                        help="ещё multi-hop abstract: краткие содержания и темы в графе (+2 вызова модели на документ)")
    parser.add_argument("--append", action="store_true",
                        help="дописать пары в --out (id продолжаются) по документам, которых в нём ещё нет")
    parser.add_argument("--single-hop-only", action="store_true", help="только вопросы по одному документу")
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
        from ragas.run_config import RunConfig
        from ragas.testset import TestsetGenerator
        from ragas.testset.graph import KnowledgeGraph
        from ragas.testset.persona import Persona

        from app.eval.metrics import build_judge, judge_cache, judge_info

        cfg = get_settings()
        started = time.perf_counter()
        docs = corpus_documents(cfg.rag.data_dir)
        print(f"Корпус {cfg.rag.data_dir.name}/: {len(docs)} документов; модель {judge_info(cfg)['judge_model']}",
              file=sys.stderr)
        llm = build_judge(cfg, cache=judge_cache(cfg))
        embeddings = LlamaIndexEmbeddingsWrapper(ProjectEmbedding(default_service()))
        run_config = RunConfig(timeout=int(cfg.ragas.judge_timeout), max_workers=cfg.ragas.concurrency,
                               max_retries=3, seed=args.seed)
        graph = None
        if args.kg.is_file() and not args.rebuild_kg:
            graph = KnowledgeGraph.load(str(args.kg))
            stats = graph_stats(graph)
            if graph_usable(stats, abstract=args.abstract):
                print(f"Граф знаний — из {args.kg}: {describe_graph(stats)}", file=sys.stderr)
            else:
                print(f"Граф {args.kg} не годится ({describe_graph(stats)}) — строю заново", file=sys.stderr)
                graph = None
        if graph is None:
            print("Строю граф знаний: " + ("краткие содержания, темы, " if args.abstract else "")
                  + "сущности, связи…", file=sys.stderr)
            FAILURES.clear()
            graph = build_graph(docs, llm, embeddings, run_config, abstract=args.abstract)
            stats = graph_stats(graph)
            print(f"Граф: {describe_graph(stats)}", file=sys.stderr)
            if FAILURES:
                print(f"Сбоев модели при извлечении: {len(FAILURES)} (документ без сущностей в вопросы не попадёт):\n  "
                      + "\n  ".join(FAILURES[:10]) + ("\n  …" if len(FAILURES) > 10 else ""), file=sys.stderr)
            if not graph_usable(stats, abstract=args.abstract):
                raise ValueError("модель не вернула сущности для большинства документов — граф не сохранён, "
                                 "см. ошибки выше (python scripts/verify_eval.py --live)")
            args.kg.parent.mkdir(parents=True, exist_ok=True)
            graph.save(str(args.kg))
            print(f"Граф сохранён — {args.kg}", file=sys.stderr)
        existing, used, start = read_existing(args.out) if args.append else ([], set(), 1)
        ordered = balanced(graph, args.seed, used)
        if args.append:
            print(f"Дописываю в {args.out.name}: в нём {len(existing)} пар по {len(used)} файлам — их документы "
                  f"пропускаются, осталось {len(ordered.nodes)} документов", file=sys.stderr)
        generator = TestsetGenerator(llm=llm, embedding_model=embeddings, knowledge_graph=ordered,
                                     persona_list=[Persona(name=n, role_description=d) for n, d in PERSONAS],
                                     llm_context=LLM_CONTEXT)
        distribution = query_distribution(llm, ordered, abstract=args.abstract, single_only=args.single_hop_only)
        testset = generator.generate(testset_size=args.size, query_distribution=distribution,
                                     num_personas=len(PERSONAS), run_config=run_config)
        rows = to_rows(testset.to_pandas(), docs, start=start)
    except (EmbeddingError, ValidationError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    import pandas as pd

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([pd.DataFrame(existing, columns=list(rows.columns)), rows]) if existing else rows
    combined.to_csv(args.out, index=False, encoding="utf-8")
    by_kind = rows["synthesizer_name"].value_counts().to_dict() if len(rows) else {}
    by_category = Counter(source.split("/", 1)[0] for value in rows.get("reference_sources", [])
                          for source in json.loads(value))
    print(f"\n{len(rows)} пар → {args.out}" + (f" (всего {len(combined)})" if existing else "")
          + f": {', '.join(f'{k}: {v}' for k, v in by_kind.items())}; категории: "
          + ", ".join(f"{k} {v}" for k, v in sorted(by_category.items()))
          + f"; {(time.perf_counter() - started) / 60:.1f} мин. Дальше — ручная вычитка: tests/eval/golden_review.yaml "
          "и python scripts/build_golden.py", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
