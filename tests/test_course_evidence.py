import json
import runpy
from pathlib import Path

import pytest

from retrieval_lab.common import read_json, sha256, write_json

EVIDENCE = runpy.run_path(str(Path(__file__).parents[1] / "scripts/enrich_course_evidence.py"))


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def negative_metadata(strategy="hard"):
    return {
        "negative_strategy": strategy,
        "query_ids": ["train-a", "train-b"],
        "train_queries": 2,
        "positive_pairs": 3,
        "pair_weighting": "uniform positive pairs",
        "negatives_per_positive": 1,
        "document_ids": [f"d{i:03}" for i in range(110)],
        "duplicate_group_count": 109,
        "duplicate_groups": [["d005", "d006"]],
        "blocked_document_ids_by_query": {"train-a": ["d000"], "train-b": ["d003"]},
        "in_batch_fallback": "uniform eligible group",
        "hard_mining": {
            "top_k": 100,
            "rank_min": 2,
            "rank_max": 100,
            "candidates_by_query": {
                "train-a": [
                    {"document_id": "d007", "full_corpus_rank": 2, "cosine": 0.8},
                    {"document_id": "d008", "full_corpus_rank": 10, "cosine": 0.7},
                ],
                "train-b": [{"document_id": "d009", "full_corpus_rank": 99, "cosine": 0.2}],
            },
        },
    }


def make_evidence_fixture(tmp_path, *, short_pca=False):
    data = tmp_path / "data" / "fixture"
    (data / "qrels").mkdir(parents=True)
    jsonl(
        data / "queries.jsonl",
        [{"_id": qid, "text": f"Query text for {qid}"} for qid in ("qgood", "qbad")],
    )
    jsonl(
        data / "corpus.jsonl",
        [
            {
                "_id": f"d{i:03}",
                "title": f"Document {i}",
                "text": " ".join(f"excerpt{word:02}" for word in range(1, 41)),
            }
            for i in range(110)
        ],
    )
    (data / "qrels/dev.tsv").write_text(
        "query-id\tcorpus-id\tscore\nqgood\td000\t1\nqgood\td105\t1\nqbad\td000\t1\n",
        encoding="utf-8",
    )
    # A dev evidence run must neither parse nor hash the held-out qrels.
    (data / "qrels/test.tsv").write_text("DO NOT READ TEST LABELS", encoding="utf-8")
    study = tmp_path / "study"
    study.mkdir()
    metadata = {
        "study_id": "study-fixture",
        "config": {
            "data": {"directory": str(data.parent), "name": "fixture"},
            "experiment": {"split": "dev"},
        },
        "dataset": {
            "name": "fixture",
            "files": {
                # Windows relative manifest keys are intentionally included.
                name.replace("/", "\\"): sha256(data / name)
                for name in ("corpus.jsonl", "queries.jsonl", "qrels/dev.tsv")
            },
        },
        "embedding_files": {"documents.npy": "fixed", "queries.npy": "fixed"},
        "query_ids": ["qgood", "qbad"],
    }
    write_json(study / "study.json", metadata)
    groups, checked = [], {}
    for method in ("original", "pca", "rae", "qrae"):
        dimension = 8 if method == "original" else 4
        seed = 0 if method in {"original", "pca"} else 42
        name = f"{method}-{dimension}d-seed{seed}"
        folder = study / name
        folder.mkdir()
        rank_rows = []
        for qid in ("qgood", "qbad"):
            good = (qid == "qgood") == (method == "qrae")
            docs = (
                ["d000"] + [f"d{i:03}" for i in range(1, 100)]
                if good
                else [f"d{i:03}" for i in range(1, 101)]
            )
            if short_pca and method == "pca":
                docs = docs[:5]
            rank_rows.append({"query_id": qid, "document_ids": docs})
        jsonl(folder / "ranking.jsonl", rank_rows)
        files = {"ranking.jsonl": sha256(folder / "ranking.jsonl")}
        if method == "qrae":
            write_json(folder / "supervision.json", negative_metadata())
            files["supervision.json"] = sha256(folder / "supervision.json")
        result = {
            "study_id": "study-fixture",
            "method": method,
            "dimension": dimension,
            "seed": seed,
            "files": files,
        }
        write_json(folder / "result.json", result)
        checked[f"{name}/result.json"] = sha256(folder / "result.json")
        groups.append(
            {
                "group_id": method,
                "study_id": "study-fixture",
                "study_key": "fixture-key",
                "dataset": "fixture",
                "split": "dev",
                "method": method,
                "dimension": dimension,
                "run_names": [name],
                "label": f"{method}, fixture configuration",
            }
        )
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    write_json(analysis / "summary.json", groups)
    write_json(
        analysis / "analysis-manifest.json",
        {
            "studies": [
                {
                    "study_key": "fixture-key",
                    "split": "dev",
                    "source": str(study),
                    "manifest_sha256": sha256(study / "study.json"),
                    "checked_inputs": checked,
                }
            ]
        },
    )
    write_json(
        analysis / "case-selection.json",
        [
            {
                "comparison_id": "qrae-vs-rae",
                "candidate_group": "qrae",
                "reference_group": "rae",
                "candidate_label": "QRAE fixture",
                "reference_label": "RAE fixture",
                "dataset": "fixture",
                "split": "dev",
                "dimension": 4,
                "metric": "ndcg@10",
                "largest_improvements": [{"query_id": "qgood", "delta": 0.3}],
                "largest_regressions": [{"query_id": "qbad", "delta": -0.2}],
            }
        ],
    )
    return analysis, study, data


def test_case_text_ranks_negative_metadata_and_report_excerpt(tmp_path):
    analysis, _, data = make_evidence_fixture(tmp_path)
    output = EVIDENCE["enrich"](analysis, tmp_path / "evidence", split="dev")
    cases = read_json(output / "enriched-cases.json")
    assert {case["direction"] for case in cases} == {"improvement", "regression"}
    assert cases[0]["query_text"] == "Query text for qgood"
    context = {row["method"]: row for row in cases[0]["ranking_context"]}
    assert context["qrae"]["seed"] == 42
    assert context["qrae"]["relevant_ranks"]["d000"]["rank"] == 1
    assert context["rae"]["relevant_ranks"]["d000"]["display"] == ">100"
    assert context["original"]["seed"] == context["pca"]["seed"] == 0
    assert not context["rank_only"]["available"]
    assert context["qrae"]["relevant_ranks"]["d105"]["display"] == ">100"
    document = cases[0]["relevant_documents"][0]
    assert len(document["report_excerpt"].split()) == 20
    assert "excerpt21" in document["snippet"]
    assert "excerpt21" not in (output / "case-evidence.md").read_text(encoding="utf-8")
    negatives = read_json(output / "negative-quality.json")
    assert len(negatives) == 1
    assert negatives[0]["positive_pairs"] == 3
    assert negatives[0]["hard_pool_sizes"]["mean"] == 1.5
    assert negatives[0]["hard_full_corpus_ranks"]["max"] == 99
    assert negatives[0]["hard_teacher_cosines"]["mean"] == pytest.approx(1.7 / 3)
    assert negatives[0]["documents_in_nontrivial_duplicate_groups"] == 2
    assert negatives[0]["hard_candidates_blocked_by_train_metadata"] == 0
    manifest = read_json(output / "evidence-manifest.json")
    assert str((data / "qrels/test.tsv").resolve()) not in manifest["consumed_file_sha256"]
    assert manifest["negative_cleaning_uses_held_out_qrels"] is False


def test_short_saved_rankings_and_missing_seed_are_explicit(tmp_path):
    analysis, _, _ = make_evidence_fixture(tmp_path, short_pca=True)
    output = EVIDENCE["enrich"](analysis, tmp_path / "evidence", context_seed=17)
    context = {
        row["method"]: row
        for row in read_json(output / "enriched-cases.json")[0]["ranking_context"]
    }
    assert context["pca"]["relevant_ranks"]["d105"]["display"] == ">5 (only 5 saved)"
    assert not context["qrae"]["available"]
    assert "no substitute seed" in context["qrae"]["reason"]


@pytest.mark.parametrize("target", ["ranking", "corpus", "result"])
def test_changed_evidence_inputs_are_rejected(tmp_path, target):
    analysis, study, data = make_evidence_fixture(tmp_path)
    path = {
        "ranking": study / "qrae-4d-seed42/ranking.jsonl",
        "corpus": data / "corpus.jsonl",
        "result": study / "qrae-4d-seed42/result.json",
    }[target]
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum differs"):
        EVIDENCE["enrich"](analysis, tmp_path / "evidence")


def test_fallback_and_false_negative_rates_remain_unmeasured():
    summary = EVIDENCE["summarize_negatives"](negative_metadata("in_batch"), {})
    assert summary["observed_fallback_count"] is None
    assert summary["observed_fallback_rate"] is None
    assert summary["false_negative_rate"] is None
    assert summary["fallback_measurement_status"] == "not logged; no rate inferred"
    assert summary["hard_pool_sizes"] is None


def test_query_overlap_is_descriptive_and_whitespace_sensitive(tmp_path):
    folder = tmp_path / "fixture"
    (folder / "qrels").mkdir(parents=True)
    rows = [
        {"_id": "a", "text": "same query"},
        {"_id": "b", "text": "  same   query "},
        {"_id": "c", "text": "same query"},
        {"_id": "d", "text": "Same query"},
    ]
    jsonl(folder / "queries.jsonl", rows)
    for split, ids in {"train": ["a"], "dev": ["b"], "test": ["c", "d"]}.items():
        (folder / "qrels" / f"{split}.tsv").write_text(
            "query-id\tcorpus-id\tscore\n" + "".join(f"{qid}\tdoc\tUNKNOWN\n" for qid in ids),
            encoding="utf-8",
        )
    before = {str(path): sha256(path) for path in folder.rglob("*") if path.is_file()}
    audit = EVIDENCE["audit_query_text_overlap"](folder, {"name": "fixture"}, {})
    assert audit["exact"]["train__dev"]["shared_text_count"] == 0
    assert audit["exact"]["train__test"]["shared_text_count"] == 1
    assert audit["normalized_whitespace"]["dev__test"]["query_id_pair_count"] == 1
    assert audit["normalized_whitespace"]["train__test"]["id_overlap_count"] == 0
    assert audit["relevance_scores_used"] is False
    assert audit["primary_results_changed"] is False
    assert before == {str(path): sha256(path) for path in folder.rglob("*") if path.is_file()}


def test_query_overlap_requires_explicit_test_evidence(tmp_path):
    with pytest.raises(ValueError, match="post-freeze test"):
        EVIDENCE["enrich"](tmp_path / "analysis", tmp_path / "out", audit_query_overlap=True)
