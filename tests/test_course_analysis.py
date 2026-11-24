import copy
import runpy
from pathlib import Path

import numpy as np
import pytest

from retrieval_lab.common import read_json, sha256, write_json

ANALYSIS = runpy.run_path(str(Path(__file__).parents[1] / "scripts/analyze_course.py"))


def make_study(root, study_id="study-a", *, actual_negative="random", other_query_ids=False):
    root.mkdir()
    qids = ["q0", "q1", "q2", "q3"]
    if other_query_ids:
        qids = [f"other-{qid}" for qid in qids]
    config = {
        "experiment": {
            "split": "dev",
            "methods": ["pca", "ae", "rae", "qrae"],
            "dimensions": [4],
            "seeds": [17, 42],
            "ks": [10],
        },
        "embedding": {"model": "fixture", "revision": "fixed"},
        "ae": {"steps": 2},
        "neural": {"steps": 2, "ranking_weight": 1.0, "regularization_weight": 0.1},
        "supervision": {"negative_strategy": "random"},
    }
    write_json(
        root / "study.json",
        {
            "study_id": study_id,
            "config": config,
            "dataset": {"name": "fixture"},
            "embedding_files": {"documents.npy": "same-docs", "queries.npy": "same-queries"},
            "query_ids": qids,
            "protocol": "fixture train labels only",
        },
    )
    base = {
        "qrae": [0.8, 0.4, 0.5, 0.2],
        "rae": [0.6, 0.6, 0.5, 0.2],
        "ae": [0.5, 0.5, 0.4, 0.2],
        "pca": [0.55, 0.5, 0.4, 0.2],
    }
    for method in config["experiment"]["methods"]:
        for seed in [0] if method == "pca" else [17, 42]:
            folder = root / f"{method}-4d-seed{seed}"
            folder.mkdir()
            scores = np.array(base[method]) + (0.01 if seed == 42 else 0)
            per_query = {
                qid: {"ndcg@10": float(score)} for qid, score in zip(qids, scores, strict=True)
            }
            info = {"method": method, "dimension": 4, "seed": seed}
            if method != "pca":
                info.update(
                    {
                        "training": copy.deepcopy(
                            config["neural"] if method in {"rae", "qrae"} else config["ae"]
                        ),
                        "actual_steps": 2,
                        "history": [
                            {
                                "step": 1,
                                "total_loss": 0.5,
                                "corpus_probe_reconstruction": 0.4,
                                "train_probe_ranking": 0.1,
                                "gradient_norm": 2.0,
                                "latent_norm_min": 0.2,
                                "active_triplet_fraction": 0.5,
                            },
                            {
                                "step": 2,
                                "total_loss": 0.4,
                                "corpus_probe_reconstruction": 0.3,
                                "train_probe_ranking": 0.1,
                                "gradient_norm": 1.0,
                                "latent_norm_min": 0.3,
                                "active_triplet_fraction": 0.4,
                            },
                        ],
                    }
                )
            if method == "qrae":
                info["training"]["negative_strategy"] = actual_negative
            write_json(folder / "per_query.json", per_query)
            write_json(folder / "compressor.json", info)
            write_json(
                folder / "result.json",
                {
                    "dataset": "fixture",
                    "split": "dev",
                    "study_id": study_id,
                    "method": method,
                    "dimension": 4,
                    "seed": seed,
                    "queries": len(qids),
                    "metrics": {"ndcg@10": float(scores.mean())},
                    "query_doc_overlap@10": 0.6,
                    "doc_doc_overlap@10": 0.7,
                    "storage": {
                        "vector_payload_bytes": 1600,
                        "index_file_bytes": 1645,
                        "compressor_file_bytes": 256,
                        "training_decoder_file_bytes": 128 if method != "pca" else 0,
                    },
                    "timing": {
                        "search_p50_ms": 0.1,
                        "search_p95_ms": 0.2,
                        "fit_seconds": 1.0,
                        "scope": "search only; no encoding/projection",
                    },
                    "files": {
                        name: sha256(folder / name)
                        for name in ("per_query.json", "compressor.json")
                    },
                },
            )
    return root


def test_crossed_bootstrap_retains_seed_uncertainty():
    result = ANALYSIS["seed_query_bootstrap"](
        [[0, 0, 0], [1, 1, 1]], [[0, 0, 0]], [17, 42], [0], samples=2000
    )
    assert result["delta"] == 0.5
    assert result["seed_query_bootstrap_95ci"] == [0, 1]
    assert result["query_count"] == 3


def test_crossed_bootstrap_pairs_query_and_matching_seed_ids():
    baseline = np.array([[0.1, 0.5, 0.8], [0.2, 0.4, 0.7]])
    result = ANALYSIS["seed_query_bootstrap"](
        baseline[::-1] + 0.02, baseline, [42, 17], [17, 42], samples=500
    )
    np.testing.assert_allclose(result["seed_query_bootstrap_95ci"], [0.02, 0.02])
    assert result["seed_resampling"] == "paired matching seed IDs"


def test_analysis_preserves_sources_and_uses_actual_parameters(tmp_path):
    source = make_study(tmp_path / "source", actual_negative="hard")
    before = {str(p.relative_to(source)): sha256(p) for p in source.rglob("*") if p.is_file()}
    output = ANALYSIS["analyze"]([source], tmp_path / "analysis", samples=200, plots=False)
    rows = read_json(output / "summary.json")
    qrae = next(row for row in rows if row["method"] == "qrae")
    assert qrae["parameters"]["negative_strategy"] == "hard"
    assert "neg=hard" in qrae["label"]
    assert qrae["n_seeds"] == 2
    assert qrae["deployment_bytes_mean"] == 1645 + 256
    assert qrae["deployment_bytes_mean"] != 1645 + 256 + 128
    cases = read_json(output / "case-selection.json")
    case = next(
        row
        for row in cases
        if row["candidate_label"].startswith("qrae") and row["reference_label"].startswith("rae")
    )
    assert case["largest_improvements"][0]["query_id"] == "q0"
    assert case["largest_regressions"][0]["query_id"] == "q1"
    assert all(row["delta"] > 0 for row in case["largest_improvements"])
    assert all(row["delta"] < 0 for row in case["largest_regressions"])
    assert before == {
        str(p.relative_to(source)): sha256(p) for p in source.rglob("*") if p.is_file()
    }


def test_different_studies_and_actual_lambdas_are_never_pooled(tmp_path):
    first = make_study(tmp_path / "first")
    second = make_study(tmp_path / "second", "study-b", actual_negative="hard")
    folder = second / "qrae-4d-seed42"
    info = read_json(folder / "compressor.json")
    info["training"]["ranking_weight"] = 2.0
    write_json(folder / "compressor.json", info)
    result = read_json(folder / "result.json")
    result["files"]["compressor.json"] = sha256(folder / "compressor.json")
    write_json(folder / "result.json", result)
    output = ANALYSIS["analyze"]([first, second], tmp_path / "analysis", samples=100, plots=False)
    qrae = [row for row in read_json(output / "summary.json") if row["method"] == "qrae"]
    assert sorted(row["n_seeds"] for row in qrae) == [1, 1, 2]
    assert len({row["group_id"] for row in qrae}) == 3
    assert len({row["study_id"] for row in qrae}) == 2


def test_cross_study_comparisons_require_identical_queries(tmp_path):
    first = make_study(tmp_path / "first")
    second = make_study(tmp_path / "second", "study-b", other_query_ids=True)
    output = ANALYSIS["analyze"](
        [first, second], tmp_path / "analysis", samples=100, plots=False, across_studies=True
    )
    groups = {row["group_id"]: row for row in read_json(output / "summary.json")}
    for row in read_json(output / "comparisons.json"):
        assert (
            groups[row["candidate_group"]]["study_id"] == groups[row["reference_group"]]["study_id"]
        )


def test_rejects_incomplete_studies_and_changed_scores(tmp_path):
    source = make_study(tmp_path / "source")
    path = source / "qrae-4d-seed42" / "result.json"
    data = path.read_bytes()
    path.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        ANALYSIS["load_study"](source)
    path.write_bytes(data)
    result = read_json(path)
    result["metrics"]["ndcg@10"] += 0.1
    write_json(path, result)
    with pytest.raises(ValueError, match="aggregate"):
        ANALYSIS["load_study"](source)


def test_figures_and_training_diagnostics_are_generated(tmp_path):
    source = make_study(tmp_path / "source")
    output = ANALYSIS["analyze"]([source], tmp_path / "analysis", samples=100)
    manifest = read_json(output / "analysis-manifest.json")
    assert len(manifest["figures"]) == 4
    assert all((output / filename).stat().st_size > 1000 for filename in manifest["figures"])
    assert len(read_json(output / "training-summary.json")) == 6
    with pytest.raises(ValueError, match="outside"):
        ANALYSIS["analyze"]([source], source / "analysis", plots=False)


def test_existing_public_snapshot_reproduces_aggregate_and_seed_count():
    source = Path(__file__).parents[1] / "docs/results/scifact-dev"
    study = ANALYSIS["load_study"](source)
    rows = [ANALYSIS["summarize"](group) for group in study["groups"]]
    assert len(rows) == 13
    original = next(row for row in rows if row["method"] == "original")
    assert original["ndcg@10_mean"] == pytest.approx(0.7520122495804374)
    assert original["queries"] == 162
    ae128 = next(row for row in rows if row["method"] == "ae" and row["dimension"] == 128)
    assert ae128["n_seeds"] == 3
    assert ae128["ndcg@10_mean"] == pytest.approx(0.6883797887159222)
    assert ae128["ndcg@10_std"] == pytest.approx(0.009010681728691702)


def test_every_registered_artifact_is_checked_not_only_scores(tmp_path):
    source = make_study(tmp_path / "source")
    folder = source / "qrae-4d-seed42"
    artifact = folder / "training_decoder.npz"
    artifact.write_bytes(b"saved decoder")
    result = read_json(folder / "result.json")
    result["files"][artifact.name] = sha256(artifact)
    write_json(folder / "result.json", result)
    study = ANALYSIS["load_study"](source)
    assert "qrae-4d-seed42/training_decoder.npz" in study["checked_inputs"]
    artifact.write_bytes(b"changed decoder")
    with pytest.raises(ValueError, match="checksum differs"):
        ANALYSIS["load_study"](source)


def test_three_datasets_remain_separate_even_with_matching_query_ids(tmp_path):
    sources = []
    for dataset in ["scifact", "nfcorpus", "fiqa"]:
        source = make_study(tmp_path / dataset, study_id=f"fixture-{dataset}")
        study = read_json(source / "study.json")
        study["dataset"]["name"] = dataset
        write_json(source / "study.json", study)
        for path in source.glob("*/result.json"):
            result = read_json(path)
            result["dataset"] = dataset
            write_json(path, result)
        sources.append(source)
    output = ANALYSIS["analyze"](
        sources, tmp_path / "analysis", samples=100, plots=False, across_studies=True
    )
    rows = read_json(output / "summary.json")
    assert {row["dataset"] for row in rows} == {"scifact", "nfcorpus", "fiqa"}
    groups = {row["group_id"]: row for row in rows}
    assert len(rows) == 12
    for comparison in read_json(output / "comparisons.json"):
        assert (
            groups[comparison["candidate_group"]]["dataset"]
            == groups[comparison["reference_group"]]["dataset"]
        )
