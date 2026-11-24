"""Offline report-asset tests. All scores/models below are synthetic placeholders."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_course_assets.py"
SPEC = importlib.util.spec_from_file_location("course_assets", SCRIPT)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


def synthetic_study(root, dataset, role="main"):
    """Complete synthetic grids make guard tests independent of experimental data."""
    path = root / f"SYNTHETIC-{dataset}-{role}"
    path.mkdir(parents=True)
    source = root / f"SYNTHETIC-{dataset}-{role}-source-dev"
    source.mkdir()
    jobs = assets.expected_jobs(role)
    config = {
        "data": {"name": dataset},
        "experiment": {
            "methods": sorted({j[0] for j in jobs}),
            "dimensions": list(assets.DIMS)
            if role == "main"
            else [64, 128, 768]
            if role == "rank_only"
            else [64, 128],
            "seeds": list(assets.SEEDS),
            "split": "test",
            "ks": [10, 100],
        },
        "ae": {"steps": 10},
        "neural": {"steps": 10, "regularization_weight": 1e-5, "ranking_weight": 10.0},
        "supervision": {"negative_strategy": role if role in {"hard", "in_batch"} else "random"},
    }
    study = {
        "study_id": path.name,
        "config": config,
        "evaluation_mode": "frozen",
        "frozen_source": {"study_id": source.name, "runs": {}},
        "source_study": str(source.resolve()),
        "query_ids": ["synthetic-1", "synthetic-2"],
        "embedding_identity": {"model": "synthetic", "dimension": 768},
        "embedding_files": {"documents.npy": f"synthetic-{dataset}"},
        "document_ids_sha256": f"synthetic-{dataset}",
        "dataset": {"name": dataset},
    }
    assets.write_json(path / "study.json", study)
    for method, dim, seed in sorted(jobs):
        name = f"{method}-{dim}d-seed{seed}"
        folder = path / name
        folder.mkdir()
        info = {
            "method": method,
            "dimension": dim,
            "seed": seed,
            "actual_steps": 10,
            "ranking_weight": 10.0 if method == "qrae" else 1.0 if method == "rank_only" else 0.0,
            "regularization_weight": 0.0 if method == "rank_only" else 1e-5,
            "training": {"negative_strategy": config["supervision"]["negative_strategy"]},
        }
        assets.write_json(folder / "compressor.json", info)
        (folder / "index.faiss").write_bytes(b"x" * (100 * dim))
        (folder / "compressor.npz").write_bytes(b"x" * (10 * dim))
        delta = {0: 0, 17: -0.01, 42: 0, 2026: 0.01}[seed]
        quality = 0.5 + 0.1 * dim / 768 + delta
        result = {
            "dataset": dataset,
            "study_id": path.name,
            "split": "test",
            "evaluation_mode": "frozen",
            "frozen_source": {"fixture": True},
            "method": method,
            "dimension": dim,
            "seed": seed,
            "documents": 100,
            "queries": 2,
            "metrics": {k: quality for k in assets.METRICS},
            "query_doc_overlap@10": min(1.0, 0.2 + 0.8 * dim / 768 + delta),
            "doc_doc_overlap@10": min(1.0, 0.3 + 0.7 * dim / 768 + delta),
            "storage": {
                "vector_payload_bytes": 100 * dim * 4,
                "index_file_bytes": 100 * dim,
                "compressor_file_bytes": 10 * dim,
                "training_decoder_file_bytes": 10 * dim,
            },
            "timing": {
                "search_p50_ms": 2.0 + delta,
                "scope": "synthetic test fixture",
                "fit_seconds": 0.0,
                "document_transform_seconds": 0.25,
                "index_build_seconds": 0.01,
            },
            "files": {"compressor.json": assets.digest(folder / "compressor.json")},
        }
        if method == "qrae" and dim == 128 and seed == 42:
            (folder / "training.csv").write_text(
                "step,corpus_probe_reconstruction,train_probe_ranking,weighted_regularizer,gradient_norm,latent_norm_min\n"
                "1,0.2,0.01,0.01,0.4,0.1\n5,0.15,0.001,0.02,0.2,0.2\n10,0.1,0,0.03,0.1,0.3\n",
                encoding="utf-8",
            )
            result["files"]["training.csv"] = assets.digest(folder / "training.csv")
        dev = copy.deepcopy(result)
        dev.update({"study_id": source.name, "split": "dev", "evaluation_mode": "fitted"})
        dev.pop("frozen_source")
        dev["timing"].update(
            {
                "fit_seconds": 12.0 + 100 * delta,
                "document_transform_seconds": 0.5 + delta,
                "index_build_seconds": 0.1 + delta,
            }
        )
        (source / name).mkdir()
        assets.write_json(source / name / "result.json", dev)
        registered = {"result_sha256": assets.digest(source / name / "result.json")}
        study["frozen_source"]["runs"][name] = registered
        result["frozen_source"] = registered
        assets.write_json(folder / "result.json", result)
    assets.write_json(path / "study.json", study)
    return path


def test_complete_grid_and_sample_sd_and_training_weight(tmp_path):
    path = synthetic_study(tmp_path, "scifact")
    consumed = {}
    item = assets.load_study(path, "main", consumed)
    assert len(item["records"]) == 53
    rows = assets.aggregate(item)
    assert len(rows) == 21
    qrae = assets.row_lookup(rows, "scifact", "qrae", 128)
    value = qrae["statistics"]["ndcg@10"]
    assert value["n"] == 3
    assert value["sd"] == pytest.approx(0.01)  # sample SD, not population SD
    assert qrae["statistics"]["deployment_mib"]["mean"] == 110 * 128 / assets.MIB
    probe = assets.training_probe(item, consumed)
    assert probe["values"]["weighted_train_probe_ranking"] == [0.1, 0.01, 0]
    assert len(consumed) == 161  # study + 53 test result/compressor/dev result + one CSV


def test_source_dev_fit_cost_is_separate_from_zero_frozen_test_fit(tmp_path):
    path = synthetic_study(tmp_path, "scifact")
    consumed = {}
    item = assets.load_study(path, "main", consumed)
    qrae = assets.row_lookup(assets.aggregate(item), "scifact", "qrae", 128)
    assert qrae["statistics"]["timing.fit_seconds"] == {"mean": 0.0, "sd": 0.0, "n": 3}
    dev_fit = {"mean": 12.0, "sd": 1.0, "n": 3}
    assert qrae["statistics"]["source_dev_timing.fit_seconds"] == dev_fit
    assert qrae["source_dev_timing"]["statistics"]["fit_seconds"] == dev_fit
    assert [x["fit_seconds"] for x in qrae["source_dev_timing"]["per_seed"]] == [11, 12, 13]
    assert "worker startup" in qrae["source_dev_timing"]["scope"]
    source_file = Path(item["study"]["source_study"]) / "qrae-128d-seed42/result.json"
    assert consumed[assets.display_path(source_file)] == assets.digest(source_file)


def test_source_dev_timing_rejects_changed_frozen_result(tmp_path):
    path = synthetic_study(tmp_path, "scifact")
    study = json.loads((path / "study.json").read_text())
    source_file = Path(study["source_study"]) / "qrae-128d-seed42/result.json"
    source_file.write_text(source_file.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum differs"):
        assets.load_study(path, "main", {})


def test_incomplete_grid_rejected(tmp_path):
    path = synthetic_study(tmp_path, "scifact")
    (path / "qrae-128d-seed2026/result.json").unlink()
    with pytest.raises(ValueError, match="Incomplete/unexpected"):
        assets.load_study(path, "main", {})


def test_fitted_or_dev_study_rejected_before_reading_scores(tmp_path):
    path = synthetic_study(tmp_path, "scifact")
    study = json.loads((path / "study.json").read_text())
    study["evaluation_mode"] = "fitted"
    assets.write_json(path / "study.json", study)
    with pytest.raises(ValueError, match="frozen test"):
        assets.load_study(path, "main", {})
    study["evaluation_mode"] = "frozen"
    study["config"]["experiment"]["split"] = "dev"
    assets.write_json(path / "study.json", study)
    with pytest.raises(ValueError, match="frozen test"):
        assets.load_study(path, "main", {})


def test_tampered_consumed_metadata_rejected(tmp_path):
    path = synthetic_study(tmp_path, "scifact")
    target = path / "qrae-128d-seed42/compressor.json"
    target.write_text(target.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum differs"):
        assets.load_study(path, "main", {})


def test_no_mixed_protocol_or_query_population(tmp_path):
    main = [assets.load_study(synthetic_study(tmp_path, d), "main", {}) for d in assets.DATASETS]
    ablations = [
        assets.load_study(synthetic_study(tmp_path, "scifact", r), "ablation", {})
        for r in ("in_batch", "hard", "rank_only")
    ]
    assets.validate_collection(main, ablations)
    changed = copy.deepcopy(main)
    changed[-1]["study"]["config"]["neural"]["ranking_weight"] = 1.0
    with pytest.raises(ValueError, match="different effective"):
        assets.validate_collection(changed, ablations)
    ablations[0]["study"]["query_ids"] = ["different-population"]
    with pytest.raises(ValueError, match="query_ids"):
        assets.validate_collection(main, ablations)


def test_full_cli_generates_five_svg_png_and_traceable_table(tmp_path):
    main = [synthetic_study(tmp_path, d) for d in assets.DATASETS]
    ablations = [synthetic_study(tmp_path, "scifact", r) for r in ("in_batch", "hard", "rank_only")]
    output = tmp_path / "SYNTHETIC-output"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--main-studies",
            *map(str, main),
            "--ablation-studies",
            *map(str, ablations),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert len(list(output.glob("*.png"))) == len(list(output.glob("*.svg"))) == 5
    assert not list(output.glob("*.pdf"))
    table = json.loads((output / "table-data.json").read_text(encoding="utf-8"))
    assert len(table["rows"]) == 70  # 3*21 main + 2+2+3 ablation groups
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert sum(x["runs"] for x in provenance["main_studies"]) == 159
    assert sum(x["runs"] for x in provenance["ablations"]) == 21
    assert len(provenance["consumed_files_sha256"]) == 549
    # Every plotted aggregate is finite, including deterministic zero SD conventions.
    assert all(
        np.isfinite(v["mean"]) and np.isfinite(v["sd"])
        for row in table["rows"]
        for v in row["statistics"].values()
    )
