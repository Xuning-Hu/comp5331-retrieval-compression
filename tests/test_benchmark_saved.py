"""Offline timing protocol checks; no model download, encoding, or GPU execution."""

import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from retrieval_lab.common import sha256, unit_vectors, write_json
from retrieval_lab.config import DEFAULTS
from retrieval_lab.data import load_dataset
from retrieval_lab.experiment import run

SPEC = importlib.util.spec_from_file_location(
    "benchmark_saved", Path(__file__).parents[1] / "scripts/benchmark_saved.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


@pytest.fixture
def saved_study(tmp_path):
    config = copy.deepcopy(DEFAULTS)
    config["data"]["directory"] = str(tmp_path / "data")
    config["experiment"].update(
        output=str(tmp_path / "runs"),
        methods=["original", "pca"],
        dimensions=[2],
        seeds=[42],
        ks=[1, 3],
        threads=1,
        geometry_anchors=4,
        latency_queries=1,
        latency_repeats=1,
        bootstrap_samples=4,
    )
    data_path = tmp_path / "data/scifact"
    (data_path / "qrels").mkdir(parents=True)
    ids, qids = [f"d{i:02}" for i in range(16)], ["q0", "q1"]
    for name, rows in [
        ("corpus", [{"_id": did, "text": f"Document {did}"} for did in ids]),
        ("queries", [{"_id": qid, "text": f"Query {qid}"} for qid in qids]),
    ]:
        (data_path / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
        )
    for split, qid in [("train", "q0"), ("dev", "q1")]:
        (data_path / f"qrels/{split}.tsv").write_text(
            f"query-id\tcorpus-id\tscore\n{qid}\td00\t1\n", encoding="utf-8"
        )
    dataset = load_dataset(data_path, config["data"])
    cache = tmp_path / "cache"
    cache.mkdir()
    x = unit_vectors(np.random.default_rng(42).normal(size=(16, 4)))
    np.save(cache / "documents.npy", x)
    np.save(cache / "queries.npy", x[:2])
    write_json(cache / "doc_ids.json", ids)
    write_json(cache / "query_ids.json", qids)
    write_json(
        cache / "manifest.json",
        {
            "dimension": 4,
            "resolved_model_revision": "a" * 40,
            "spec": {"embedding": {**config["embedding"], "model": "offline-fixture"}},
            "files": {
                name: sha256(cache / name)
                for name in ["documents.npy", "queries.npy", "doc_ids.json", "query_ids.json"]
            },
        },
    )
    return run(config, dataset, cache), cache


def test_query_schedule_is_balanced_randomized_and_reproducible():
    inputs = ([f"q{i}" for i in range(40)], ["original", "pca", "qrae"])
    qids, schedule = benchmark.choose_schedule(*inputs)
    assert len(qids) == len(set(qids)) == 30
    assert (qids, schedule) == benchmark.choose_schedule(*inputs)
    assert schedule != benchmark.choose_schedule(*inputs, seed=42)[1]
    counts = Counter((row["query_id"], row["run"]) for row in schedule)
    assert set(counts.values()) == {3}
    assert len(counts) == 90


def test_source_loading_is_read_only_and_saved_search_timing_runs_offline(saved_study):
    study, _ = saved_study
    before = {str(p): sha256(p) for p in study.rglob("*") if p.is_file()}
    loaded = benchmark.load_saved(study, dimension=2)
    assert set(loaded["runs"]) == {"original-4d-seed0", "pca-2d-seed0"}
    cached = loaded["cached_queries"][loaded["query_positions"]["q1"]]

    class FixtureEncoder:
        def encode(self, text, request_id):
            return {"type": "result", "id": request_id, "vector": cached.tolist(), "encode_ms": 0.0}

    model, index = loaded["runs"]["pca-2d-seed0"]
    timing = benchmark.measure_request(FixtureEncoder(), model, index, "query", "0", 3, cached)
    assert all(
        timing[field] >= 0 for field in ["encode_ms", "projection_ms", "search_ms", "total_ms"]
    )
    assert timing["total_ms"] == pytest.approx(
        sum(
            timing[field]
            for field in ["encode_ms", "projection_ms", "search_ms", "ipc_and_python_overhead_ms"]
        )
    )
    assert timing["live_cached_cosine"] == pytest.approx(1, abs=1e-6)
    assert before == {str(p): sha256(p) for p in study.rglob("*") if p.is_file()}


@pytest.mark.parametrize("target", ["weights", "cache"])
def test_source_checksum_mismatch_is_rejected(saved_study, target):
    study, cache = saved_study
    path = study / "pca-2d-seed0/compressor.npz" if target == "weights" else cache / "queries.npy"
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum|integrity"):
        benchmark.load_saved(study, dimension=2)


def test_worker_protocol_keeps_one_process_for_repeated_requests():
    code = """
import json, sys
print(json.dumps({"type": "ready"}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("type") == "stop": break
    print(json.dumps({"type":"result", "id":request["id"],
                      "vector":[1.0,0.0], "encode_ms":0.25}), flush=True)
"""
    worker = benchmark.EncoderWorker([sys.executable, "-c", code], timeout=20)
    try:
        pid = worker.process.pid
        for i in range(3):
            assert worker.encode("offline fixture", str(i))["encode_ms"] == 0.25
            assert worker.process.pid == pid
    finally:
        worker.close()
    assert worker.process.poll() == 0


def test_percentiles_are_over_all_repeated_query_measurements():
    fields = [
        "encode_ms",
        "projection_ms",
        "search_ms",
        "total_ms",
        "encode_roundtrip_ms",
        "ipc_and_python_overhead_ms",
    ]
    rows = [{"run": "fixture", **dict.fromkeys(fields, value)} for value in [1.0, 3.0]]
    result = benchmark.summarize(rows)["fixture"]
    assert result["requests"] == 2
    assert result["total_ms"]["p50"] == 2.0
    assert result["total_ms"]["p95"] == pytest.approx(2.9)


def test_timing_refuses_to_write_inside_source_study(saved_study, monkeypatch):
    study, _ = saved_study
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_saved.py",
            "--study",
            str(study),
            "--dimension",
            "2",
            "--output",
            str(study / "timing.json"),
        ],
    )
    with pytest.raises(ValueError, match="outside source"):
        benchmark.main()


def test_saved_index_supports_unicode_paths_without_changing_rankings(saved_study, tmp_path):
    study, cache = saved_study
    original = study / "original-4d-seed0/index.faiss"
    target = tmp_path / "中文路径/index.faiss"
    target.parent.mkdir()
    target.write_bytes(original.read_bytes())
    native = benchmark.load_index_view(original, 1)
    unicode = benchmark.load_index_view(target, 1)
    queries = np.load(cache / "queries.npy", allow_pickle=False)
    for actual, expected in zip(unicode.search(queries, 3), native.search(queries, 3), strict=True):
        np.testing.assert_array_equal(actual, expected)
