import copy

import faiss
import numpy as np
import pytest

from retrieval_lab.common import read_json, sha256, unit_vectors, write_json
from retrieval_lab.compressors import Compressor
from retrieval_lab.config import DEFAULTS
from retrieval_lab.data import Dataset
from retrieval_lab.experiment import evaluate_saved, run
from retrieval_lab.query import _SavedExactIndex
from retrieval_lab.search import ExactIndex


@pytest.fixture
def fitted_study(tmp_path):
    config = copy.deepcopy(DEFAULTS)
    config["experiment"].update(
        {
            "output": str(tmp_path / "runs"),
            "methods": ["original", "pca", "rp"],
            "dimensions": [4],
            "seeds": [17],
            "ks": [3, 10],
            "geometry_anchors": 5,
            "latency_queries": 2,
            "latency_repeats": 1,
            "bootstrap_samples": 20,
        }
    )
    x = unit_vectors(np.random.default_rng(42).normal(size=(30, 8)))
    doc_ids = [f"d{i:03}" for i in range(len(x))]
    qids = [f"q{i}" for i in range(6)]
    dataset = Dataset(
        "fixture",
        dict.fromkeys(doc_ids, {}),
        dict.fromkeys(qids, "query"),
        {
            split: {qids[i]: {doc_ids[i]: 1} for i in indices}
            for split, indices in {"train": [0, 1], "dev": [2, 3], "test": [4, 5]}.items()
        },
        {"name": "fixture", "seed": 42},
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "documents.npy", x)
    np.save(cache / "queries.npy", x[:6])
    write_json(cache / "doc_ids.json", doc_ids)
    write_json(cache / "query_ids.json", qids)
    write_json(
        cache / "manifest.json",
        {
            "spec": {"embedding": config["embedding"], "preprocessing": "fixture"},
            "resolved_model_revision": "frozen-fixture-revision",
            "dimension": 8,
            "files": {
                name: sha256(cache / name)
                for name in ("documents.npy", "queries.npy", "doc_ids.json", "query_ids.json")
            },
        },
    )
    source = run(config, dataset, cache)
    return config, dataset, cache, source


def _forbid_fitting(*args, **kwargs):
    raise AssertionError("Frozen evaluation must never fit a compressor")


def test_frozen_same_queries_reproduce_results_without_fitting(fitted_study, monkeypatch):
    config, dataset, cache, source = fitted_study
    before = {str(p.relative_to(source)): sha256(p) for p in source.rglob("*") if p.is_file()}
    monkeypatch.setattr(Compressor, "fit", _forbid_fitting)
    frozen = evaluate_saved(config, dataset, cache, source)
    assert frozen != source
    for result_path in source.glob("*/result.json"):
        name = result_path.parent.name
        measured = read_json(result_path)
        replayed = read_json(frozen / name / "result.json")
        assert replayed["metrics"] == measured["metrics"]
        assert replayed["timing"]["fit_seconds"] == 0
        assert replayed["evaluation_mode"] == "frozen"
        for filename in ("per_query.json", "ranking.jsonl", "geometry.json", "compressor.npz"):
            assert (frozen / name / filename).read_bytes() == (
                source / name / filename
            ).read_bytes()
    assert before == {
        str(p.relative_to(source)): sha256(p) for p in source.rglob("*") if p.is_file()
    }
    assert evaluate_saved(config, dataset, cache, source) == frozen


def test_test_split_uses_frozen_weights_and_records_source(fitted_study, monkeypatch):
    config, dataset, cache, source = fitted_study
    config["experiment"]["split"] = "test"
    monkeypatch.setattr(Compressor, "fit", _forbid_fitting)
    frozen = evaluate_saved(config, dataset, cache, source)
    study = read_json(frozen / "study.json")
    assert study["query_ids"] == ["q4", "q5"]
    assert study["frozen_source"]["study_manifest_sha256"] == sha256(source / "study.json")
    assert study["evaluation_mode"] == "frozen"
    for name, origin in study["frozen_source"]["runs"].items():
        assert origin["result_sha256"] == sha256(source / name / "result.json")
        for filename, checksum in origin["artifact_sha256"].items():
            assert sha256(source / name / filename) == checksum
            assert sha256(frozen / name / filename) == checksum


@pytest.mark.parametrize("changed", ["embeddings", "model", "dataset", "document_ids"])
def test_frozen_evaluation_rejects_mismatched_inputs(fitted_study, changed):
    config, dataset, cache, source = fitted_study
    manifest = read_json(cache / "manifest.json")
    if changed == "embeddings":
        x = np.load(cache / "documents.npy")
        x[0, 0] += 0.1
        np.save(cache / "documents.npy", x)
        manifest["files"]["documents.npy"] = sha256(cache / "documents.npy")
    elif changed == "model":
        manifest["resolved_model_revision"] = "another-model-revision"
    elif changed == "dataset":
        dataset.manifest["seed"] = 999
    else:
        write_json(cache / "doc_ids.json", list(reversed(read_json(cache / "doc_ids.json"))))
        manifest["files"]["doc_ids.json"] = sha256(cache / "doc_ids.json")
    write_json(cache / "manifest.json", manifest)
    with pytest.raises(ValueError, match="differs|differ"):
        evaluate_saved(config, dataset, cache, source)


@pytest.mark.parametrize("changed", ["weights", "untracked_training"])
def test_frozen_evaluation_rejects_corrupt_or_untracked_artifacts(fitted_study, changed):
    config, dataset, cache, source = fitted_study
    folder = source / "pca-4d-seed0"
    if changed == "weights":
        with (folder / "compressor.npz").open("ab") as file:
            file.write(b"changed")
    else:
        (folder / "training.csv").write_text("step,loss\n1,0.2\n")
    with pytest.raises(ValueError, match="checksum"):
        evaluate_saved(config, dataset, cache, source)


def test_test_queries_must_be_disjoint_from_development(fitted_study):
    config, dataset, cache, source = fitted_study
    config["experiment"]["split"] = "test"
    dataset.qrels["test"] = dataset.qrels["dev"]
    with pytest.raises(ValueError, match="overlap"):
        evaluate_saved(config, dataset, cache, source)


def test_saved_query_index_preserves_canonical_cutoff_ties(tmp_path):
    index = ExactIndex(np.ones((30, 4), dtype=np.float32), threads=1)
    path = tmp_path / "index.faiss"
    faiss.write_index(index.index, str(path))
    saved = _SavedExactIndex(path, threads=1)
    scores, rows = saved.search(np.ones((1, 4), dtype=np.float32), 3)
    np.testing.assert_array_equal(rows, [[0, 1, 2]])
    np.testing.assert_array_equal(scores, [[1, 1, 1]])
