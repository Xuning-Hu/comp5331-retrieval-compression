import copy
import itertools
import runpy
from pathlib import Path

import numpy as np
import pytest

from retrieval_lab.common import read_json, sha256, write_json

SCRIPT = Path(__file__).parents[1] / "scripts/analyze_supplementary.py"
SUPPLEMENT = runpy.run_path(str(SCRIPT))
SEEDS = (17, 42, 2026)


def snapshot(root, dataset="scifact", encoder="e5", *, weights=None, seeds=SEEDS):
    """Small portable score snapshot using the production export schema."""
    root.mkdir()
    model, revision = SUPPLEMENT["MODELS"][encoder]
    methods = list(SUPPLEMENT["METHODS"]) if weights is None else ["qrae"]
    weight, alpha = weights or (1e-5, 10.0)
    training = {
        "steps": 3000,
        "center": True,
        "lr": 0.001,
        "min_lr": 1e-5,
        "batch_size": 128,
        "regularization_weight": weight,
        "ranking_weight": alpha,
        "negative_strategy": "random",
        "margin": 0.1,
    }
    config = {
        "experiment": {
            "methods": methods,
            "dimensions": [128],
            "seeds": list(seeds),
            "split": "dev" if weights is not None else "test",
            "ks": [10],
        },
        "embedding": {
            "model": model,
            "revision": revision,
            "max_length": 384 if encoder == "mpnet" else 512,
            "query_prefix": "" if encoder == "mpnet" else "query: ",
            "document_prefix": "" if encoder == "mpnet" else "passage: ",
        },
        "ae": {"steps": 3000},
        "neural": training,
        "supervision": {"negative_strategy": "random"},
    }
    qids = ["q0", "q1", "q2"]
    write_json(
        root / "study.json",
        {
            "study_id": root.name,
            "config": config,
            "dataset": {"name": dataset, "archive": "fixed"},
            "query_ids": qids,
            "document_ids_sha256": "same-ordered-ids",
            "embedding_files": {
                "documents.npy": f"{encoder}-{dataset}-docs",
                "queries.npy": f"{encoder}-{dataset}-queries",
            },
        },
    )
    results, scores, infos = {}, {}, {}
    for method in methods:
        dimension = 768 if method == "original" else 128
        for seed in (0,) if method in {"original", "pca"} else seeds:
            name = f"{method}-{dimension}d-seed{seed}"
            base = {
                "original": 0.60,
                "pca": 0.52,
                "rp": 0.45,
                "ae": 0.50,
                "rae": 0.55,
                "qrae": 0.62,
            }[method]
            if encoder == "mpnet":
                base += 0.04 if method == "qrae" else 0.02
            if weights is not None:
                base = 0.55 + (0 if alpha == 0 else 0.05 - alpha * 0.0005)
                base += {0.0: 0.0, 1e-6: 0.002, 1e-5: 0.01, 1e-4: -0.01}[weight]
            base += 0.0 if seed == 0 else (SEEDS.index(seed) - 1) * 0.01
            query_scores = np.array([base + 0.02, base - 0.03, base + 0.01])
            scores[name] = {
                qid: {"ndcg@10": float(value)}
                for qid, value in zip(qids, query_scores, strict=True)
            }
            info = {"method": method, "dimension": dimension, "seed": seed}
            if method in {"ae", "rae", "qrae"}:
                info["training"] = copy.deepcopy(training if method != "ae" else config["ae"])
            if method in {"rae", "qrae"}:
                info["regularization_weight"] = weight
                info["ranking_weight"] = alpha if method == "qrae" else 0.0
            infos[name] = info
            results[name] = {
                "dataset": dataset,
                "split": config["experiment"]["split"],
                "study_id": root.name,
                "method": method,
                "dimension": dimension,
                "seed": seed,
                "queries": len(qids),
                "metrics": {"ndcg@10": float(query_scores.mean())},
                "timing": {"fit_seconds": 1.0, "scope": "search only"},
                "storage": {
                    "vector_payload_bytes": 200,
                    "index_file_bytes": 210,
                    "compressor_file_bytes": 20,
                },
            }
    for filename, content in (
        ("run-results.json", results),
        ("per-query.json", scores),
        ("compressor-metadata.json", infos),
    ):
        write_json(root / filename, content)
    write_json(
        root / "snapshot.json",
        {
            "files": {
                name: sha256(root / name)
                for name in (
                    "study.json",
                    "run-results.json",
                    "per-query.json",
                    "compressor-metadata.json",
                )
            }
        },
    )
    return root


def test_cross_encoder_analysis_preserves_all_seeds_queries_and_paired_differences(tmp_path):
    e5 = [snapshot(tmp_path / f"e5-{ds}", ds) for ds in ("scifact", "nfcorpus", "fiqa")]
    mpnet = [snapshot(tmp_path / f"mp-{ds}", ds, "mpnet") for ds in ("scifact", "nfcorpus", "fiqa")]
    before = {p: sha256(p) for root in e5 + mpnet for p in root.iterdir()}
    output = SUPPLEMENT["analyze"](mpnet, e5, [], tmp_path / "out", samples=50, plots=True)
    manifest = read_json(output / "analysis-manifest.json")
    assert manifest["condition_count"] == 36
    assert manifest["run_count"] == 84
    assert len(read_json(output / "per-query.json")) == 36
    assert (output / "encoder-128d.png").is_file()
    comparisons = read_json(output / "comparisons.json")
    changes = [row for row in comparisons if row["kind"] == "change_in_qrae_minus_rae"]
    assert len(changes) == 3
    assert all(row["delta"]["mean"] == pytest.approx(0.02) for row in changes)
    assert all(row["seeds"] == list(SEEDS) for row in changes)
    original = next(
        row for row in read_json(output / "summary.json") if row["method"] == "original"
    )
    assert original["metrics"]["ndcg@10"]["sample_sd"] is None
    assert all(sha256(p) == checksum for p, checksum in before.items())
    for name, checksum in manifest["files"].items():
        assert sha256(output / name) == checksum


def test_exported_metadata_tampering_is_rejected(tmp_path):
    source = snapshot(tmp_path / "source", encoder="mpnet")
    metadata = source / "compressor-metadata.json"
    data = read_json(metadata)
    data["qrae-128d-seed17"]["ranking_weight"] = 100.0
    write_json(metadata, data)
    with pytest.raises(ValueError, match="checksum"):
        SUPPLEMENT["analyze"]([source], [], [], tmp_path / "out", plots=False)
    assert not (tmp_path / "out").exists()


def test_cross_encoder_different_query_identity_is_rejected(tmp_path):
    source = snapshot(tmp_path / "source", encoder="mpnet")
    reference = snapshot(tmp_path / "reference")
    loaded = SUPPLEMENT["_load"](source, "ndcg@10")
    other = SUPPLEMENT["_load"](reference, "ndcg@10")
    conditions = SUPPLEMENT["encoder_conditions"]([("mpnet", loaded), ("e5", other)])
    changed = copy.deepcopy(conditions)
    for condition in changed:
        if condition["encoder"]["label"] == "e5" and condition["method"] == "qrae":
            condition["parameters"]["training"]["steps"] = 1000
    with pytest.raises(ValueError, match="changes compressor parameters"):
        SUPPLEMENT["comparisons"](changed, "ndcg@10")
    for condition in conditions:
        if condition["encoder"]["label"] == "e5":
            condition["data_identity"] = "different-query-split"
    with pytest.raises(ValueError, match="identical data/split/query"):
        SUPPLEMENT["comparisons"](conditions, "ndcg@10")


def test_grid_aggregates_single_seed_studies_and_retains_negative_effect(tmp_path):
    paths = []
    for dataset, weight, alpha, seed in itertools.product(
        ("scifact", "fiqa"), SUPPLEMENT["LAMBDAS"], SUPPLEMENT["ALPHAS"], SEEDS
    ):
        paths.append(
            snapshot(
                tmp_path / f"{dataset}-w{weight}-a{alpha}-s{seed}",
                dataset,
                weights=(weight, alpha),
                seeds=(seed,),
            )
        )
    grid_map = tmp_path / "grid-map.json"
    write_json(
        grid_map,
        {"status": "complete", "path_base": "manifest", "studies": [path.name for path in paths]},
    )
    output = SUPPLEMENT["analyze"]([], [], [], tmp_path / "out", plots=True, grid_manifest=grid_map)
    manifest = read_json(output / "analysis-manifest.json")
    assert manifest["condition_count"] == 32
    assert manifest["run_count"] == 96
    assert manifest["figures"] == ["grid-fiqa.png", "grid-scifact.png"]
    assert manifest["grid_manifest"]["relative_path"] == "../grid-map.json"
    assert all(
        (output / row["source_relative_path"]).is_dir() for row in manifest["source_studies"]
    )
    summaries = read_json(output / "summary.json")
    zero = next(row for row in summaries if row["lambda"] == row["alpha"] == 0)
    assert zero["method"] == "qrae"
    assert zero["metrics"]["ndcg@10"]["mean"] == pytest.approx(0.55)
    assert zero["metrics"]["ndcg@10"]["sample_sd"] == pytest.approx(0.01)
    assert len(zero["source_groups"]) == 3
    effects = [
        row
        for row in read_json(output / "comparisons.json")
        if row["kind"] == "lambda_minus_zero_at_fixed_alpha"
    ]
    assert len(effects) == 24
    assert min(row["delta"]["mean"] for row in effects) == pytest.approx(-0.01)
    assert max(row["delta"]["mean"] for row in effects) == pytest.approx(0.01)
    assert "no cell is designated" in (output / "supplemental-fragment.md").read_text()
    with pytest.raises(ValueError, match="Incomplete or duplicated seeds"):
        SUPPLEMENT["analyze"]([], [], paths[:-1], tmp_path / "missing", plots=False)
    loaded = [SUPPLEMENT["_load"](path, "ndcg@10") for path in paths]
    loaded[-1]["groups"][0]["parameters"]["training"]["margin"] = 0.2
    with pytest.raises(ValueError, match="nuisance parameters"):
        SUPPLEMENT["grid_conditions"](loaded)


def test_grid_cannot_read_test(tmp_path):
    source = snapshot(tmp_path / "source", weights=(0.0, 0.0), seeds=(17,))
    loaded = SUPPLEMENT["_load"](source, "ndcg@10")
    loaded["split"] = "test"
    with pytest.raises(ValueError, match="dev-only"):
        SUPPLEMENT["grid_conditions"]([loaded])


def test_output_cannot_overwrite_source_or_existing_analysis(tmp_path):
    source = snapshot(tmp_path / "source", encoder="mpnet")
    with pytest.raises(ValueError, match="outside"):
        SUPPLEMENT["analyze"]([source], [], [], source / "analysis", plots=False)
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("preserve")
    with pytest.raises(FileExistsError, match="nonempty"):
        SUPPLEMENT["analyze"]([source], [], [], output, plots=False)
