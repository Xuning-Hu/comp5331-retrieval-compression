import copy
import json
import math
import runpy
from pathlib import Path

import pytest

from retrieval_lab.common import fingerprint, read_json, sha256, write_json
from retrieval_lab.metrics import evaluate

VERIFY = runpy.run_path(str(Path(__file__).parents[1] / "scripts/verify_course_results.py"))


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_studies(tmp_path):
    data = tmp_path / "data" / "fixture"
    (data / "qrels").mkdir(parents=True)
    document_ids = [f"d{i:02}" for i in range(12)]
    _jsonl(data / "corpus.jsonl", [{"_id": did} for did in document_ids])
    _jsonl(data / "queries.jsonl", [{"_id": qid} for qid in ("qtrain", "qdev", "qtest")])
    for split in ("train", "dev", "test"):
        (data / "qrels" / f"{split}.tsv").write_text(
            f"query-id\tcorpus-id\tscore\nq{split}\td00\t3\nq{split}\td03\t1\n",
            encoding="utf-8",
        )
    dataset = {
        "name": "fixture",
        "files": {
            path.relative_to(data).as_posix(): sha256(path)
            for path in data.rglob("*")
            if path.is_file()
        },
    }
    config = {
        "data": {
            "directory": str(data.parent),
            "name": "fixture",
            "split_seed": 2026,
            "dev_fraction": 0.2,
        },
        "experiment": {
            "split": "dev",
            "methods": ["original", "pca", "qrae"],
            "dimensions": [4],
            "seeds": [42],
            "ks": [3, 10],
        },
        "ae": {},
        "neural": {"ranking_weight": 10},
        "supervision": {"negative_strategy": "random"},
    }
    dev, test = tmp_path / "dev", tmp_path / "test"
    source_manifest = None
    for split, root in (("dev", dev), ("test", test)):
        root.mkdir()
        settings = copy.deepcopy(config)
        settings["experiment"]["split"] = split
        study = {
            "config": settings,
            "study_id": f"fixture-{split}",
            "query_ids": [f"q{split}"],
            "document_ids_sha256": fingerprint(document_ids),
            "dataset": dataset,
            "embedding_files": {"documents.npy": "fixture-vectors"},
            "embedding_identity": {"dimension": 8, "model": "fixture"},
            "evaluation_mode": "fitted" if split == "dev" else "frozen",
        }
        frozen_runs = {}
        for method, dim, seed in (("original", 8, 0), ("pca", 4, 0), ("qrae", 4, 42)):
            name = f"{method}-{dim}d-seed{seed}"
            folder = root / name
            folder.mkdir()
            rankings = {f"q{split}": document_ids[:10]}
            metrics, scores = evaluate({f"q{split}": {"d00": 3, "d03": 1}}, rankings, [3, 10])
            scores[f"q{split}"]["query_doc_overlap@10"] = 1.0
            write_json(folder / "per_query.json", scores)
            _jsonl(
                folder / "ranking.jsonl",
                [
                    {
                        "query_id": f"q{split}",
                        "document_ids": document_ids[:10],
                        "scores": list(range(10, 0, -1)),
                    }
                ],
            )
            write_json(
                folder / "geometry.json",
                {"anchor_document_ids": ["d00", "d01"], "doc_doc_overlap": [1.0, 0.5], "k": 10},
            )
            write_json(
                folder / "compressor.json", {"method": method, "dimension": dim, "seed": seed}
            )
            (folder / "compressor.npz").write_bytes(b"fixture saved weights")
            (folder / "index.faiss").write_bytes(b"fixture index")
            if method == "qrae":
                (folder / "training_decoder.npz").write_bytes(b"fixture training-only decoder")
                (folder / "training.csv").write_text("step,loss\n1,0.1\n", encoding="utf-8")
                write_json(
                    folder / "supervision.json",
                    {"query_ids": ["qtrain"], "train_queries": 1, "negative_strategy": "random"},
                )
            files = {path.name: sha256(path) for path in folder.iterdir()}
            result = {
                "study_id": study["study_id"],
                "dataset": "fixture",
                "split": split,
                "method": method,
                "dimension": dim,
                "seed": seed,
                "documents": 12,
                "queries": 1,
                "metrics": metrics,
                "files": files,
                "query_doc_overlap@10": 1.0,
                "doc_doc_overlap@10": 0.75,
                "timing": {"fit_seconds": 0.0 if split == "test" else 1.0},
                "storage": {
                    "vector_payload_bytes": 12 * dim * 4,
                    "index_file_bytes": (folder / "index.faiss").stat().st_size,
                    "compressor_file_bytes": (folder / "compressor.npz").stat().st_size,
                    "training_decoder_file_bytes": (folder / "training_decoder.npz").stat().st_size
                    if method == "qrae"
                    else 0,
                },
                "evaluation_mode": study["evaluation_mode"],
            }
            if split == "test":
                source_result = read_json(dev / name / "result.json")
                frozen = {
                    "result_sha256": sha256(dev / name / "result.json"),
                    "artifact_sha256": {
                        filename: value
                        for filename, value in source_result["files"].items()
                        if filename in VERIFY["DEPLOYMENT"] | VERIFY["TRAINING"]
                    },
                }
                result["frozen_source"] = frozen
                frozen_runs[name] = frozen
            write_json(folder / "result.json", result)
        if split == "test":
            study["source_study"] = str(dev)
            study["frozen_source"] = {
                "study_id": source_manifest["study_id"],
                "study_manifest_sha256": sha256(dev / "study.json"),
                "runs": frozen_runs,
            }
        write_json(root / "study.json", study)
        source_manifest = study
    return dev, test, data


def _reregister(folder, filename):
    result = read_json(folder / "result.json")
    result["files"][filename] = sha256(folder / filename)
    write_json(folder / "result.json", result)


def test_scalar_metrics_use_linear_graded_gain_and_positive_recall():
    scores = VERIFY["scalar_metrics"]({"a": 3, "b": 1, "c": 0}, ["b", "x", "a", "c"], [1, 3])
    assert scores["ndcg@1"] == pytest.approx(1 / 3)
    assert scores["ndcg@3"] == pytest.approx((1 + 3 / math.log2(4)) / (3 + 1 / math.log2(3)))
    assert scores["recall@1"] == 0.5
    assert scores["recall@3"] == scores["mrr@3"] == 1
    assert all(value == 0 for value in VERIFY["scalar_metrics"]({"x": 0}, ["x"], [1]).values())


def test_dev_audit_does_not_open_test_labels(tmp_path):
    dev, _, data = make_studies(tmp_path)
    (data / "qrels/test.tsv").write_text("MUST NOT BE READ OR HASHED", encoding="utf-8")
    output = VERIFY["verify"]([dev], tmp_path / "audit")
    report = read_json(output / "verification.json")
    assert report["run_count"] == 3
    assert report["maximum_error"] < 1e-12
    assert report["query_overlap_unreplayed_runs"] == 0
    assert str((data / "qrels/test.tsv").resolve()) not in report["consumed_file_sha256"]


def test_frozen_artifacts_and_metrics_verified_across_both_splits(tmp_path):
    dev, test, _ = make_studies(tmp_path)
    report = read_json(VERIFY["verify"]([dev, test], tmp_path / "audit") / "verification.json")
    assert report["run_count"] == 6
    test_report = next(row for row in report["studies"] if row["split"] == "test")
    assert all(row["frozen_source"]["matched_artifact_count"] >= 2 for row in test_report["runs"])
    qrae = next(row for row in test_report["runs"] if row["method"] == "qrae")
    assert "supervision.json" in qrae["frozen_source"]["matched_files"]
    assert qrae["supervision"]["only_train"]


def test_self_consistent_saved_scores_cannot_hide_wrong_rank_metrics(tmp_path):
    dev, _, _ = make_studies(tmp_path)
    folder = dev / "pca-4d-seed0"
    scores = read_json(folder / "per_query.json")
    scores["qdev"]["ndcg@3"] = 0
    write_json(folder / "per_query.json", scores)
    result = read_json(folder / "result.json")
    result["metrics"]["ndcg@3"] = 0
    result["files"]["per_query.json"] = sha256(folder / "per_query.json")
    write_json(folder / "result.json", result)
    with pytest.raises(ValueError, match="Independent metric mismatch"):
        VERIFY["verify"]([dev], tmp_path / "audit")


def test_rehashed_test_weights_still_fail_source_comparison(tmp_path):
    _, test, _ = make_studies(tmp_path)
    folder = test / "pca-4d-seed0"
    before = (folder / "compressor.npz").read_bytes()
    (folder / "compressor.npz").write_bytes(b"x" * len(before))
    _reregister(folder, "compressor.npz")
    with pytest.raises(ValueError, match="source/test artifact hashes"):
        VERIFY["verify"]([test], tmp_path / "audit")


def test_incomplete_jobs_are_rejected(tmp_path):
    dev, _, _ = make_studies(tmp_path)
    (dev / "pca-4d-seed0/result.json").unlink()
    with pytest.raises(ValueError, match="predeclared jobs"):
        VERIFY["verify"]([dev], tmp_path / "audit")


def test_supervision_cannot_include_dev_query(tmp_path):
    dev, _, _ = make_studies(tmp_path)
    folder = dev / "qrae-4d-seed42"
    write_json(
        folder / "supervision.json",
        {"query_ids": ["qdev"], "train_queries": 1, "negative_strategy": "random"},
    )
    _reregister(folder, "supervision.json")
    with pytest.raises(ValueError, match="restricted to official/rebuilt train"):
        VERIFY["verify"]([dev], tmp_path / "audit")


def test_internal_dev_is_rebuilt_from_train_ids(tmp_path):
    dev, _, data = make_studies(tmp_path)
    (data / "qrels/dev.tsv").unlink()
    (data / "qrels/train.tsv").write_text(
        "query-id\tcorpus-id\tscore\nqtrain\td00\t3\nqtrain\td03\t1\nqdev\td00\t3\nqdev\td03\t1\n",
        encoding="utf-8",
    )
    study = read_json(dev / "study.json")
    study["dataset"]["files"].pop("qrels/dev.tsv")
    study["dataset"]["files"]["qrels/train.tsv"] = sha256(data / "qrels/train.tsv")
    study["config"]["data"]["split_seed"] = 0
    write_json(dev / "study.json", study)
    report = read_json(VERIFY["verify"]([dev], tmp_path / "audit") / "verification.json")
    assert "official train IDs" in report["studies"][0]["dev_rule"]
