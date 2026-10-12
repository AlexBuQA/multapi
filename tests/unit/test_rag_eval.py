"""
Оценка RAG на RAGAS 0.4 (блок 5.6) без сети и без моделей: метрики и has_citation на подставном судье
(tests/unit/ragas_fakes.py), сборка судьи из настроек, scripts/run_eval.py целиком (два прохода,
журнал, --resume, CSV и JSON) и scripts/generate_testset.py на подставной модели.

Нужна группа eval (pip install -r requirements-eval.txt); без неё файл пропускается. Числа настоящего
судьи — прогоны на Windows, docs/rag_evaluation.md.
"""
from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

pytest.importorskip("ragas", reason="группа eval: pip install -r requirements-eval.txt")

from rag_fakes import TrigramEmbeddings  # noqa: E402
from ragas_fakes import FakeEmbeddings, FakeJudge, make_generator_judge  # noqa: E402

from app.core.config import RagasSettings, Settings  # noqa: E402
from app.eval import metrics as m  # noqa: E402
from app.services.rag_common import NO_ANSWER  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
QUESTION = "Каков срок возврата оплаты?"
REFERENCE = "Срок возврата оплаты — 14 дней после списания."
CONTEXTS = ["Вернуть оплату можно в течение 14 дней после списания. Деньги придут на ту же карту."]
ANSWER = "Вернуть оплату можно в течение 14 дней после списания.\nИсточники: [1]"


def settings(tmp_path: Path, **ragas: Any) -> Settings:
    return Settings(_env_file=None, llm={"openai_api_key": "test-key", "base_url": "http://127.0.0.1:11434/v1",
                                         "default_model": "llama3.2"},
                    ragas={"results_dir": tmp_path / "results", "golden_path": tmp_path / "golden.json",
                           "cache_dir": tmp_path / "cache", **ragas})


def score(judge: FakeJudge | None = None, **row: Any) -> tuple[dict[str, float], dict[str, str]]:
    metrics = m.build_metrics(judge or FakeJudge(), FakeEmbeddings())
    values = {"question": QUESTION, "reference": REFERENCE, "response": ANSWER, "contexts": CONTEXTS, **row}
    return asyncio.run(m.score_row(metrics, **values))


def load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- метрики одной строки
def test_score_row_good_answer_has_all_five_metrics():
    judge = FakeJudge()
    scores, errors = score(judge)
    assert errors == {}
    assert scores["faithfulness"] == scores["context_recall"] == scores["has_citation"] == 1.0
    assert scores["context_precision"] == pytest.approx(1.0) and scores["answer_relevancy"] == pytest.approx(1.0)
    assert list(scores) == list(m.METRICS)
    schemas = [name for name, _ in judge.calls]
    assert schemas == ["StatementGeneratorOutput", "NLIStatementOutput", "AnswerRelevanceOutput",   # strictness=1
                       "ContextPrecisionOutput", "ContextRecallOutput", "CitationVerdict"]


def test_faithfulness_and_relevancy_see_answer_without_sources_line():
    """«Источники: [1]» — разметка, а не утверждение: faithfulness и answer_relevancy получают ответ без
    неё, has_citation — как есть."""
    judge = FakeJudge()
    score(judge)
    by_schema = {name: prompt for name, prompt in reversed(judge.calls)}
    assert "Источники" not in by_schema["StatementGeneratorOutput"]
    assert "Источники" not in by_schema["AnswerRelevanceOutput"]
    assert "Источники: [1]" in by_schema["CitationVerdict"]


def test_score_guard_refusal_does_not_crash():
    """Отказ score-guard: фрагментов нет — faithfulness проверять не по чему (NaN), поиск не нашёл ответ,
    который есть в базе, — precision и recall 0; ответ «не нашёл» — не по вопросу и без ссылки."""
    scores, errors = score(response=NO_ANSWER, contexts=[])
    assert math.isnan(scores["faithfulness"]) and "score-guard" in errors["faithfulness"]
    assert scores["context_precision"] == scores["context_recall"] == 0.0
    assert scores["answer_relevancy"] == 0.0 and scores["has_citation"] == 0.0


def test_empty_response_gives_nan_without_judge_calls():
    judge = FakeJudge()
    scores, errors = score(judge, response="   ")
    assert all(math.isnan(v) for v in scores.values()) and set(errors) == set(m.METRICS)
    assert judge.calls == []


def test_no_reference_skips_context_metrics():
    scores, errors = score(reference="")
    assert math.isnan(scores["context_precision"]) and math.isnan(scores["context_recall"])
    assert errors == {"context_precision": "нет reference", "context_recall": "нет reference"}
    assert scores["faithfulness"] == 1.0


def test_judge_failure_on_one_metric_keeps_the_rest():
    scores, errors = score(FakeJudge(fail_on="ContextRecallOutput"))
    assert math.isnan(scores["context_recall"]) and errors["context_recall"].startswith("ValueError: судья ответил")
    assert set(errors) == {"context_recall"} and scores["faithfulness"] == 1.0


def test_has_citation_judge_failure_is_nan_not_no():
    """@discrete_metric глотает исключение судьи и отдаёт value=None — это NaN с ошибкой, а не «no»."""
    scores, errors = score(FakeJudge(fail_on="CitationVerdict"))
    assert math.isnan(scores["has_citation"]) and errors["has_citation"].startswith("нет значения:")
    assert "судья ответил не по схеме" in errors["has_citation"]


@pytest.mark.parametrize(("response", "value"), [
    ("Срок — 14 дней [2].", "yes"),
    ("Срок — 14 дней.\nИсточники: [1], [3]", "yes"),
    ("Срок — 14 дней.", "no"),
])
def test_has_citation_discrete_metric(response, value):
    judge = FakeJudge()
    metric = m.make_has_citation(judge)
    assert metric.name == "has_citation" and set(metric.allowed_values) == {"yes", "no"}
    assert asyncio.run(metric.ascore(response=response)).value == value
    (schema, prompt), = judge.calls
    assert schema == "CitationVerdict" and prompt.startswith("Содержит ли ответ ссылку на источник")
    assert prompt.endswith(response) and "'согласно …'" in prompt


def test_eval_row_asks_rag_and_scores():
    class Rag:
        def evaluate_inputs(self, question: str) -> dict[str, Any]:
            return {"answer": ANSWER, "retrieved_contexts": CONTEXTS, "latency_ms": 1500, "top_score": 0.8,
                    "confident": True, "sources": [], "retrieved_files": ["billing/refunds_v2.docx"]}

    row = {"id": "q001", "user_input": QUESTION, "reference": REFERENCE}
    result = asyncio.run(m.eval_row(Rag(), row, m.build_metrics(FakeJudge(), FakeEmbeddings())))
    assert result["id"] == "q001" and result["response"] == ANSWER and result["latency_ms"] == 1500
    assert result["faithfulness"] == 1.0 and result["has_citation"] == 1.0 and result["errors"] == {}


# ---------------------------------------------------------------- судья из настроек
def test_judge_defaults_to_qwen_in_service_ollama(tmp_path):
    cfg = settings(tmp_path)
    judge = m.build_judge(cfg)
    assert judge.model == "qwen3:4b-instruct" and judge.provider == "openai"
    assert judge.model_args["max_tokens"] == 2048 and "extra_body" not in judge.model_args
    client = m.judge_client(cfg)
    assert str(client.base_url) == "http://127.0.0.1:11434/v1/" and client.timeout == 600.0
    assert m._http_client(cfg, cfg.llm.base_url) is None            # Ollama — без прокси и сертификатов ОС
    assert m.build_embeddings(cfg).model == "bge-m3"
    assert m.judge_info(cfg) == {"judge_model": "qwen3:4b-instruct", "judge_host": "127.0.0.1",
                                 "judge_max_tokens": 2048, "judge_embeddings": "bge-m3"}


def test_openrouter_judge_requires_json_parameters_and_hides_key(tmp_path):
    cfg = settings(tmp_path, judge_model="nvidia/nemotron-3-super-120b-a12b:free",
                   judge_base_url="https://openrouter.ai/api/v1", judge_api_key="test-openrouter-key",
                   judge_max_tokens=4000, embeddings_model="openai/text-embedding-3-small")
    judge = m.build_judge(cfg)
    assert judge.model_args["extra_body"] == {"provider": {"require_parameters": True}}
    client = m.judge_client(cfg)
    assert client.api_key == "test-openrouter-key" and "X-Title" in client.default_headers
    info = m.judge_info(cfg)
    assert info["judge_host"] == "openrouter.ai" and "test-openrouter-key" not in json.dumps(info)
    assert m.build_embeddings(cfg).model == "openai/text-embedding-3-small"


def test_judge_cache_folder_per_model(tmp_path):
    assert Path(m.judge_cache(settings(tmp_path)).cache.directory) == tmp_path / "cache" / "qwen3_4b-instruct"
    other = m.judge_cache(settings(tmp_path, judge_model="nvidia/nemotron-3-super-120b-a12b:free"))
    assert Path(other.cache.directory).name == "nvidia_nemotron-3-super-120b-a12b_free"
    assert m.judge_cache(settings(tmp_path, cache_enabled=False)) is None


def test_ragas_settings_paths_from_project_root(monkeypatch):
    cfg = RagasSettings(golden_path="tests/eval/other.json", results_dir="out/results")
    assert cfg.golden_path == ROOT / "tests" / "eval" / "other.json" and cfg.results_dir == ROOT / "out" / "results"
    monkeypatch.setenv("RAGAS__JUDGE_MODEL", "gemma3:4b")
    monkeypatch.setenv("RAGAS__CONCURRENCY", "4")
    loaded = Settings(_env_file=None, llm={"openai_api_key": "test-key"})
    assert (loaded.ragas.judge_model, loaded.ragas.concurrency) == ("gemma3:4b", 4)


# ---------------------------------------------------------------- scripts/run_eval.py
class FakeEvalRag:
    """RAGService для run_eval: ответ по базе, отказ score-guard на «борщ», сбой на «сломай»."""

    instances: ClassVar[list[FakeEvalRag]] = []

    def __init__(self, cfg: Settings, collection: str) -> None:
        self.cfg, self.collection, self.questions = cfg, collection, []
        FakeEvalRag.instances.append(self)

    def ingestion(self) -> Any:
        return SimpleNamespace(run=lambda: SimpleNamespace(summary=lambda: "0 new, 66 unchanged"))

    def build(self, create: bool = True) -> FakeEvalRag:
        return self

    def close(self) -> None:
        pass

    def evaluate_inputs(self, question: str) -> dict[str, Any]:
        self.questions.append(question)
        if "сломай" in question:
            raise RuntimeError("модель вернула пустой ответ")
        if "борщ" in question:
            return {"answer": NO_ANSWER, "retrieved_contexts": [], "retrieved_files": [], "sources": [],
                    "top_score": 0.31, "confident": False, "latency_ms": 40}
        return {"answer": ANSWER, "retrieved_contexts": CONTEXTS, "retrieved_files": ["billing/refunds_v2.docx"],
                "sources": [], "top_score": 0.82, "confident": True, "latency_ms": 2400}


@pytest.fixture
def eval_env(tmp_path, monkeypatch):
    """run_eval без Qdrant, модели и судьи: настройки во временной папке, подставные RAG и судья."""
    golden = [
        {"id": "g01", "user_input": QUESTION, "reference": REFERENCE, "reference_sources": ["billing/refunds_v2.docx"]},
        {"id": "g02", "user_input": "Как сварить борщ?", "reference": ""},
        {"id": "g03", "user_input": "А теперь сломай сервис", "reference": REFERENCE},
        {"id": "g04", "user_input": "  ", "reference": "вопроса нет — строка пропускается"},
    ]
    (tmp_path / "golden.json").write_text(json.dumps(golden, ensure_ascii=False), encoding="utf-8")
    cfg = settings(tmp_path)
    judges: list[FakeJudge] = []

    def build_judge(cfg: Settings, **kwargs: Any) -> FakeJudge:
        judges.append(FakeJudge())
        return judges[-1]

    FakeEvalRag.instances = []
    monkeypatch.setattr("app.core.config.get_settings", lambda: cfg)
    monkeypatch.setattr("app.services.rag.RAGService", FakeEvalRag)
    monkeypatch.setattr(m, "build_judge", build_judge)
    monkeypatch.setattr(m, "build_embeddings", lambda cfg, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(m, "judge_cache", lambda cfg: None)
    return SimpleNamespace(cfg=cfg, results=tmp_path / "results", judges=judges, script=load_script("run_eval"))


def test_run_eval_writes_csv_and_json(eval_env, capsys):
    assert eval_env.script.main(["--label", "unit"]) == 0
    (csv_path,) = eval_env.results.glob("*_unit.csv")
    assert not list(eval_env.results.glob("*.partial.jsonl"))          # журнал удалён после успеха
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == list(eval_env.script.CSV_COLUMNS) and [r["id"] for r in rows] == ["g01", "g02", "g03"]
    good, guard, broken = rows
    assert float(good["faithfulness"]) == 1.0 and float(good["has_citation"]) == 1.0 and good["errors"] == ""
    assert json.loads(good["retrieved_contexts"]) == CONTEXTS and good["latency_ms"] == "2400"
    assert guard["faithfulness"] == guard["context_precision"] == "" and float(guard["has_citation"]) == 0.0
    assert "RuntimeError" in json.loads(broken["errors"])["answer"] and broken["response"] == ""

    summary = json.loads(csv_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert summary["label"] == "unit" and summary["rows"] == 3
    assert summary["config"]["collection"] == "kb_docs" and summary["config"]["chunk_size"] == 512
    assert summary["judge"]["judge_model"] == "qwen3:4b-instruct" and summary["golden"]["items"] == 3
    assert summary["metrics"]["has_citation"] == {"mean": 0.5, "n": 2, "nan": 1}
    assert summary["metrics"]["faithfulness"] == {"mean": 1.0, "n": 1, "nan": 2}
    assert summary["refusals"] == {"score_guard": 1, "model": 0} and summary["rag_errors"] == 1
    assert summary["latency_ms"]["p50"] == 1220 and set(summary["versions"]) >= {"ragas", "llama-index-core"}
    assert FakeEvalRag.instances[0].questions == [QUESTION, "Как сварить борщ?", "А теперь сломай сервис"]
    assert "Худшие по faithfulness" in capsys.readouterr().out


def test_run_eval_answers_only_then_resume_with_journal_config(eval_env):
    """Ответы и оценки — раздельно: после --answers-only судья не нужен, --resume берёт конфигурацию
    (нарезку, коллекцию) из журнала и не спрашивает RAG второй раз."""
    script = eval_env.script
    assert script.main(["--label", "chunk_128", "--chunk-size", "128", "--answers-only"]) == 0
    (partial,) = eval_env.results.glob("*_chunk_128.partial.jsonl")
    assert eval_env.judges == [] and not list(eval_env.results.glob("*.csv"))
    first = FakeEvalRag.instances[0]
    assert first.collection == "kb_docs_recursive_128_64" and first.cfg.rag.chunk_size == 128

    assert script.main(["--resume", str(partial)]) == 0
    resumed = FakeEvalRag.instances[1]
    assert resumed.questions == [] and resumed.collection == "kb_docs_recursive_128_64"
    assert resumed.cfg.rag.chunk_size == 128 and len(eval_env.judges) == 1
    (csv_path,) = eval_env.results.glob("*_chunk_128.csv")
    assert csv_path.name.removesuffix(".csv") == partial.name.removesuffix(".partial.jsonl")
    assert json.loads(csv_path.with_suffix(".json").read_text(encoding="utf-8"))["config"]["chunk_size"] == 128


def test_run_eval_resume_skips_scored_rows(eval_env):
    script = eval_env.script
    assert script.main(["--label", "part", "--answers-only"]) == 0
    (partial,) = eval_env.results.glob("*.partial.jsonl")
    journal = script.Journal(partial)
    journal.write("score", {"id": "g01", "scores": dict.fromkeys(m.METRICS, 0.25), "errors": {}, "judge_seconds": 1.0})
    assert script.main(["--resume", str(partial)]) == 0
    with next(eval_env.results.glob("*_part.csv")).open(encoding="utf-8") as fh:
        rows = {r["id"]: r for r in csv.DictReader(fh)}
    assert float(rows["g01"]["faithfulness"]) == 0.25                    # оценка из журнала, судья не звался
    assert not any(QUESTION in prompt for _, prompt in eval_env.judges[0].calls)


def test_configure_names_collection_by_chunking(eval_env):
    script = eval_env.script
    base = eval_env.cfg
    same, collection = script.configure(base, script.parse_args(["--top-k", "3", "--candidates", "10"]))
    assert collection == "kb_docs" and same.rag.similarity_top_k == 3
    _, collection = script.configure(base, script.parse_args(["--chunking", "fixed", "--chunk-overlap", "32"]))
    assert collection == "kb_docs_fixed_512_32"
    rerank, _ = script.configure(base, script.parse_args(["--rerank", "on"]))
    assert rerank.rerank.enabled is True and base.rerank.enabled is False
    with pytest.raises(ValueError):
        script.configure(base, script.parse_args(["--chunk-size", "64", "--chunk-overlap", "64"]))


@pytest.mark.parametrize("label", ["../baseline", "with space", ""])
def test_run_eval_label_is_part_of_file_name(eval_env, label):
    with pytest.raises(SystemExit):
        eval_env.script.parse_args(["--label", label])


def test_run_eval_without_questions_exits_2(eval_env, tmp_path):
    (tmp_path / "golden.json").write_text("[]", encoding="utf-8")
    assert eval_env.script.main(["--label", "empty"]) == 2


# ---------------------------------------------------------------- scripts/generate_testset.py
@pytest.fixture
def small_corpus(tmp_path):
    """Шесть статей раздела api корпуса — граф и вопросы на подставной модели за секунды."""
    data = tmp_path / "data"
    (data / "api").mkdir(parents=True)
    for name in ("api-keys.md", "api-limits.md", "webhooks.md", "api-versions.md", "api-sandbox.md",
                 "api-error-401.md"):
        shutil.copy(ROOT / "data" / "api" / name, data / "api" / name)
    (data / "uploads").mkdir()
    (data / "uploads" / "draft.md").write_text("# Черновик\n\nЗагруженный пользователем файл — не в golden.",
                                               encoding="utf-8")
    return data


def generator_env(tmp_path: Path, small_corpus: Path, monkeypatch: Any, judge: Any) -> Any:
    """generate_testset без сети: настройки с маленьким корпусом, модель — подставной судья."""
    script = load_script("generate_testset")
    cfg = Settings(_env_file=None, llm={"openai_api_key": "test-key"},
                   rag={"data_dir": small_corpus, "docstore_dir": tmp_path / "ingest"})
    monkeypatch.setattr("app.core.config.get_settings", lambda: cfg)
    monkeypatch.setattr(m, "build_judge", lambda cfg, **kwargs: judge)
    monkeypatch.setattr(m, "judge_cache", lambda cfg: None)
    monkeypatch.setattr("app.services.embeddings.default_service", lambda: TrigramEmbeddings())
    return script


def test_generate_testset_with_fake_model(tmp_path, small_corpus, monkeypatch, capsys):
    """Модель — тот же судья llm_factory, что у метрик (JSON по схеме); граф — сущности и связи по ним."""
    fake = make_generator_judge()
    script = generator_env(tmp_path, small_corpus, monkeypatch, fake)
    out, kg = tmp_path / "raw.csv", tmp_path / "kg.json"

    assert script.main(["--size", "4", "--out", str(out), "--kg", str(kg)]) == 0
    assert kg.is_file()
    err = capsys.readouterr().err
    assert "6 узлов, сущности у 6, краткие содержания у 0; связи: entities_overlap" in err
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and [r["id"] for r in rows] == [f"raw_{n:03d}" for n in range(1, len(rows) + 1)]
    assert list(rows[0]) == ["id", "user_input", "reference", "reference_contexts", "reference_sources",
                             "synthesizer_name"]
    assert {r["synthesizer_name"] for r in rows} <= {"single_hop_specific_query_synthesizer",
                                                    "multi_hop_specific_query_synthesizer"}
    for row in rows:
        assert row["user_input"] and row["reference"] and json.loads(row["reference_contexts"])
        sources = json.loads(row["reference_sources"])
        assert sources and all(s.startswith("api/") for s in sources)          # путь в корпусе, без uploads/
    titles = [p for p in fake.prompts if '"title": "NEROutput"' in p]
    assert len(titles) == 6 and all(script.RUSSIAN_EXTRACT.strip() in p for p in titles)   # вызов на документ
    assert not [p for p in fake.prompts if '"title": "StringIO"' in p]                  # кратких содержаний нет
    generation = [p for p in fake.prompts if '"title": "GeneratedQueryAnswer"' in p]
    assert generation and all(script.RUSSIAN.strip() in p for p in generation)

    calls = len(fake.prompts)                 # граф сохранён: второй запуск строит только вопросы
    assert script.main(["--size", "2", "--out", str(out), "--kg", str(kg)]) == 0
    assert not [p for p in fake.prompts[calls:] if '"title": "NEROutput"' in p]


def test_generate_testset_model_without_json_fails_without_saving_graph(tmp_path, small_corpus, monkeypatch, capsys):
    """Windows, первая версия: модель отвечала обычным текстом, все вызовы извлечения падали — граф без
    сущностей. Такой граф не сохраняется, скрипт выходит с кодом 2 и понятной причиной."""
    script = generator_env(tmp_path, small_corpus, monkeypatch, make_generator_judge(plain_text=True))
    kg = tmp_path / "kg.json"
    assert script.main(["--size", "4", "--out", str(tmp_path / "raw.csv"), "--kg", str(kg)]) == 2
    err = capsys.readouterr().err
    assert not kg.exists() and "не вернула сущности" in err
    assert "Сбоев модели при извлечении: 6" in err and "NERExtractor · api/" in err and "Invalid json output" in err


def test_generate_testset_one_failed_document_does_not_break_graph(tmp_path, small_corpus, monkeypatch, capsys):
    """RAGAS дожидается всех задач и поднимает первое исключение: без tolerant() один сбой модели из сотни
    обрывал бы построение графа. Документ со сбоем — без сущностей, остальные — в графе."""
    fake = make_generator_judge()
    real = fake.agenerate
    calls = {"ner": 0}

    async def flaky(prompt: str, response_model: Any) -> Any:
        if response_model.__name__ == "NEROutput":
            calls["ner"] += 1
            if calls["ner"] == 2:
                raise TimeoutError("модель не ответила за 600 с")
        return await real(prompt, response_model)

    fake.agenerate = flaky
    script = generator_env(tmp_path, small_corpus, monkeypatch, fake)
    assert script.main(["--size", "3", "--out", str(tmp_path / "raw.csv"), "--kg", str(tmp_path / "kg.json")]) == 0
    err = capsys.readouterr().err
    assert "сущности у 5" in err and "Сбоев модели при извлечении: 1" in err and "TimeoutError" in err


def test_generate_testset_rebuilds_unusable_saved_graph(tmp_path, small_corpus, monkeypatch, capsys):
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType

    kg = tmp_path / "kg.json"
    KnowledgeGraph(nodes=[Node(type=NodeType.DOCUMENT, properties={"page_content": "текст"})] * 3).save(str(kg))
    fake = make_generator_judge()
    script = generator_env(tmp_path, small_corpus, monkeypatch, fake)
    assert script.main(["--size", "2", "--out", str(tmp_path / "raw.csv"), "--kg", str(kg)]) == 0
    assert "не годится" in capsys.readouterr().err
    assert [p for p in fake.prompts if '"title": "NEROutput"' in p]           # построен заново
    assert script.graph_usable(script.graph_stats(KnowledgeGraph.load(str(kg))), abstract=False)


def test_generate_testset_abstract_adds_summaries_and_themes(tmp_path, small_corpus, monkeypatch):
    fake = make_generator_judge()
    script = generator_env(tmp_path, small_corpus, monkeypatch, fake)
    assert script.main(["--size", "4", "--abstract", "--out", str(tmp_path / "raw.csv"),
                        "--kg", str(tmp_path / "kg.json")]) == 0
    titles = {t for p in fake.prompts for t in ("StringIO", "ThemesAndConcepts", "NEROutput") if f'"title": "{t}"' in p}
    assert titles == {"StringIO", "ThemesAndConcepts", "NEROutput"}


def test_balanced_graph_round_robin_by_category_and_exclusion():
    """Single-hop RAGAS берёт первые n узлов: на Windows 28 вопросов пришлись на account, api и billing —
    первые файлы корпуса. balanced() ставит узлы по кругу по категориям и убирает уже взятые документы."""
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship

    script = load_script("generate_testset")
    cats = ["account"] * 3 + ["api"] * 2 + ["team"]
    nodes = [Node(type=NodeType.DOCUMENT, properties={"page_content": f"текст {i}",
                                                      "document_metadata": {"source": f"{cat}/doc{i}.md"}})
             for i, cat in enumerate(cats)]
    graph = KnowledgeGraph(nodes=nodes, relationships=[
        Relationship(source=nodes[0], target=nodes[3], type="entities_overlap", properties={}),
        Relationship(source=nodes[1], target=nodes[5], type="entities_overlap", properties={})])
    ordered = script.balanced(graph, seed=1)
    assert [script.category_of(n) for n in ordered.nodes] == ["account", "api", "team", "account", "api", "account"]
    assert {n.id for n in ordered.nodes} == {n.id for n in nodes} and len(ordered.relationships) == 2
    assert [n.id for n in script.balanced(graph, seed=1).nodes] == [n.id for n in ordered.nodes]   # seed
    without = script.balanced(graph, seed=1, exclude_sources={"account/doc0.md"})
    assert len(without.nodes) == 5 and [r.target.id for r in without.relationships] == [nodes[5].id]


def test_generate_testset_append_continues_ids_on_new_documents(tmp_path, small_corpus, monkeypatch, capsys):
    script = generator_env(tmp_path, small_corpus, monkeypatch, make_generator_judge())
    out = tmp_path / "raw.csv"
    first = [{"id": f"raw_{n:03d}", "user_input": f"Вопрос {n}?", "reference": "Ответ.",
              "reference_contexts": json.dumps(["Контекст."], ensure_ascii=False),
              "reference_sources": json.dumps([source]), "synthesizer_name": "single_hop_specific_query_synthesizer"}
             for n, source in ((1, "api/api-keys.md"), (2, "api/webhooks.md"))]
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(first[0]))
        writer.writeheader()
        writer.writerows(first)

    assert script.main(["--append", "--size", "3", "--single-hop-only", "--out", str(out),
                        "--kg", str(tmp_path / "kg.json")]) == 0
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[:2] == first and [r["id"] for r in rows[2:]] == [f"raw_{n:03d}" for n in range(3, len(rows) + 1)]
    new_sources = {s for r in rows[2:] for s in json.loads(r["reference_sources"])}
    assert new_sources and not new_sources & {"api/api-keys.md", "api/webhooks.md"}
    assert {r["synthesizer_name"] for r in rows[2:]} == {"single_hop_specific_query_synthesizer"}
    err = capsys.readouterr().err
    assert "в нём 2 пар по 2 файлам" in err and "осталось 4 документов" in err and f"(всего {len(rows)})" in err


def test_testset_documents_and_transforms(small_corpus):
    script = load_script("generate_testset")
    docs = script.corpus_documents(small_corpus)
    assert docs and {d.metadata["source"].split("/")[0] for d in docs} == {"api"}
    from ragas.testset.transforms import OverlapScoreBuilder, Parallel
    from ragas.testset.transforms.extractors.llm_based import NERExtractor, NERPrompt
    from ragas.testset.transforms.filters import CustomNodeFilter

    llm = make_generator_judge()
    ner, overlap = script.transforms_for(llm, FakeEmbeddings())
    assert isinstance(ner, NERExtractor) and isinstance(overlap, OverlapScoreBuilder) and ner.max_num_entities == 6
    assert ner.prompt.instruction.endswith(script.RUSSIAN_EXTRACT)
    assert script.RUSSIAN_EXTRACT not in NERPrompt().instruction                  # промпт класса не тронут
    assert script.RUSSIAN_EXTRACT not in NERExtractor(llm=llm).prompt.instruction
    full = script.transforms_for(llm, FakeEmbeddings(), abstract=True)
    flat = [t for item in full for t in (item.transformations if isinstance(item, Parallel) else [item])]
    assert len(flat) == 6 and not any(isinstance(t, CustomNodeFilter) for t in flat)


def test_testset_sources_of_multi_hop_contexts(small_corpus):
    script = load_script("generate_testset")
    docs = script.corpus_documents(small_corpus)
    first, second = docs[0], docs[-1]
    contexts = [f"<1-hop>\n\n{first.text}", f"<2-hop>\n\n{second.text}"]
    assert script.sources_of(contexts, docs) == [first.metadata["source"], second.metadata["source"]]
    assert script.sources_of(["текст, которого нет в корпусе"], docs) == []


def test_russian_instruction_added_once():
    from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer

    script = load_script("generate_testset")
    synthesizer = SingleHopSpecificQuerySynthesizer(llm=make_generator_judge())
    script.russian_synthesizer(script.russian_synthesizer(synthesizer))
    instruction = synthesizer.get_prompts()["query_answer_generation_prompt"].instruction
    assert instruction.count(script.RUSSIAN.strip()) == 1


# ---------------------------------------------------------------- scripts/verify_eval.py
def test_verify_eval_imports_and_live_call(monkeypatch, capsys):
    pytest.importorskip("phoenix.evals", reason="группа eval целиком")
    script = load_script("verify_eval")
    monkeypatch.setattr(m, "build_judge", lambda cfg, **kwargs: FakeJudge())
    monkeypatch.setattr(m, "build_embeddings", lambda cfg, **kwargs: FakeEmbeddings())
    assert script.main(["--live"]) == 0
    out = capsys.readouterr().out
    assert "ragas==0.4." in out and "[ragas] OK" in out and "[app.eval.metrics] OK" in out
    assert "[phoenix] OK" in out and "[судья из .env] OK" in out
    assert "[live faithfulness] OK: 1.0" in out and "[live has_citation] OK: yes" in out
    assert "[live эмбеддинги] OK: размерность 64" in out and out.rstrip().endswith("Всё на месте.")


def test_verify_eval_reports_broken_judge(monkeypatch, capsys):
    pytest.importorskip("phoenix.evals", reason="группа eval целиком")
    script = load_script("verify_eval")
    monkeypatch.setattr(m, "build_judge", lambda cfg, **kwargs: FakeJudge(fail_on="CitationVerdict"))
    monkeypatch.setattr(m, "build_embeddings", lambda cfg, **kwargs: FakeEmbeddings())
    assert script.main(["--live"]) == 1
    assert "[live has_citation] ОШИБКА:" in capsys.readouterr().out
