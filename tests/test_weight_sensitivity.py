"""Offline guards for the supplementary grid; no model loading or training."""

import copy
import csv
import runpy
from pathlib import Path

import numpy as np
import pytest

from retrieval_lab.common import fingerprint, read_json, sha256, write_json
from retrieval_lab.config import DEFAULTS

GRID = runpy.run_path(str(Path(__file__).parents[1] / "scripts/run_weight_sensitivity.py"))


def base(dataset):
    config = copy.deepcopy(DEFAULTS)
    config["data"]["name"] = dataset
    config["neural"]["device"] = "cuda"
    config["ae"]["device"] = "cuda"
    return config


def test_fixed_grid_has_all_96_cells_and_keeps_zero_axes_in_qrae():
    configs = {name: base(name) for name in ("scifact", "fiqa")}
    original = copy.deepcopy(configs)
    jobs = GRID["job_configs"](configs, "runs/new-grid")
    assert len(jobs) == len({job["job_id"] for job in jobs}) == 96
    assert {
        (job["lambda"], job["alpha"], job["seed"]) for job in jobs if job["dataset"] == "fiqa"
    } == {
        (w, alpha, seed)
        for w in (0, 1e-6, 1e-5, 1e-4)
        for alpha in (0, 1, 10, 100)
        for seed in (17, 42, 2026)
    }
    for job in jobs:
        exp = job["config"]["experiment"]
        assert exp["methods"] == ["qrae"] and exp["split"] == "dev"
        assert exp["dimensions"] == [128] and exp["seeds"] == [job["seed"]]
        assert job["config"]["supervision"]["negative_strategy"] == "random"
        assert job["config"]["neural"]["steps"] == 3000
        assert job["config"]["neural"]["ranking_weight"] == job["alpha"]
    assert configs == original


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("experiment", "split", "test"),
        ("embedding", "revision", "main"),
        ("embedding", "query_prefix", ""),
        ("neural", "steps", 1000),
        ("neural", "device", "cpu"),
        ("supervision", "negative_strategy", "hard"),
    ],
)
def test_protocol_departures_are_rejected_before_execution(section, key, value):
    config = base("scifact")
    config[section][key] = value
    with pytest.raises(ValueError):
        GRID["validate_base"](config, "scifact")


@pytest.fixture
def completed(tmp_path, monkeypatch):
    globals_ = GRID["validate_run"].__globals__
    monkeypatch.setitem(globals_, "REPO", tmp_path)
    jobs = GRID["job_configs"]({name: base(name) for name in ("scifact", "fiqa")}, "grid")
    job = jobs[0]  # lambda=alpha=0 must still have training supervision and probes.
    study = tmp_path / job["config"]["experiment"]["output"] / "dev" / "fixture"
    folder = study / "qrae-128d-seed17"
    folder.mkdir(parents=True)
    binding = {
        "dev_query_ids": ["dev-1", "dev-2"],
        "train_query_ids": ["train-only"],
        "embedding_files": {"documents.npy": "cache-identity"},
        "dataset_manifest": {"name": "fixture", "files": {}},
    }
    write_json(
        study / "study.json",
        {
            "config": job["config"],
            "source_sha256": "source-identity",
            "study_id": "fixture",
            "query_ids": binding["dev_query_ids"],
            "embedding_files": binding["embedding_files"],
            "dataset": binding["dataset_manifest"],
        },
    )
    history = []
    for step in [1, *range(100, 3001, 100)]:
        row = {key: 1.0 for key in GRID["PROBE_FIELDS"]}
        row.update({"step": step, "weighted_ranking": 0.0, "weighted_regularizer": 0.0})
        history.append(row)
    info = {
        "method": "qrae",
        "dimension": 128,
        "input_dimension": 768,
        "seed": 17,
        "regularization_weight": 0.0,
        "ranking_weight": 0.0,
        "actual_steps": 3000,
        "actual_device": "cuda",
        "history": history,
        "training": {**job["config"]["neural"], "negative_strategy": "random"},
    }
    write_json(folder / "compressor.json", info)
    with (folder / "training.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    np.savez(folder / "compressor.npz", mean=np.zeros(768), encoder=np.zeros((128, 768)))
    np.savez(folder / "training_decoder.npz", decoder=np.zeros((768, 128)))
    write_json(
        folder / "supervision.json",
        {"query_ids": ["train-only"], "negative_strategy": "random", "seed": 17},
    )
    write_json(folder / "per_query.json", {"dev-1": {"ndcg@10": 0.5}, "dev-2": {"ndcg@10": 1.0}})
    for name in ("ranking.jsonl", "index.faiss", "geometry.json"):
        (folder / name).write_bytes(b"offline artifact fixture")
    result = {
        "dataset": "scifact",
        "split": "dev",
        "method": "qrae",
        "dimension": 128,
        "seed": 17,
        "evaluation_mode": "fitted",
        "study_id": "fixture",
        "metrics": {"ndcg@10": 0.75},
        "files": {p.name: sha256(p) for p in folder.iterdir()},
    }
    write_json(folder / "result.json", result)
    return tmp_path, job, study, folder, binding


def refresh_artifact(folder, name):
    result = read_json(folder / "result.json")
    result["files"][name] = sha256(folder / name)
    write_json(folder / "result.json", result)


def test_complete_zero_weight_fit_keeps_supervised_diagnostics(completed):
    _, job, study, _, binding = completed
    verified = GRID["validate_run"](study, job, binding, "source-identity")
    assert verified["actual_regularization_weight"] == verified["actual_ranking_weight"] == 0
    assert verified["training_rows"] == 31
    assert verified["training_diagnostics_finite"]
    assert verified["metrics"]["ndcg@10"] == 0.75


@pytest.mark.parametrize(
    "change",
    [
        "test_result",
        "wrong_alpha",
        "missing_train_probe",
        "wrong_dev_ids",
        "wrong_train_ids",
        "truncated_budget",
        "bad_mean",
        "nonfinite_weights",
    ],
)
def test_semantic_corruption_is_rejected_even_after_internal_hash_refresh(completed, change):
    _, job, study, folder, binding = completed
    if change == "test_result":
        value = read_json(folder / "result.json")
        value["split"] = "test"
        write_json(folder / "result.json", value)
    elif change in {"wrong_alpha", "missing_train_probe", "truncated_budget"}:
        value = read_json(folder / "compressor.json")
        if change == "wrong_alpha":
            value["ranking_weight"] = 10
        elif change == "missing_train_probe":
            value["history"][0].pop("train_probe_ranking")
        else:
            value["actual_steps"] = 1000
        write_json(folder / "compressor.json", value)
        refresh_artifact(folder, "compressor.json")
    elif change == "wrong_dev_ids":
        write_json(folder / "per_query.json", {"test-query": {"ndcg@10": 0.75}})
        refresh_artifact(folder, "per_query.json")
    elif change == "wrong_train_ids":
        value = read_json(folder / "supervision.json")
        value["query_ids"] = ["dev-1"]
        write_json(folder / "supervision.json", value)
        refresh_artifact(folder, "supervision.json")
    elif change == "bad_mean":
        value = read_json(folder / "result.json")
        value["metrics"]["ndcg@10"] = 0.99
        write_json(folder / "result.json", value)
    else:
        np.savez(folder / "compressor.npz", mean=np.zeros(768), encoder=np.full((128, 768), np.nan))
        refresh_artifact(folder, "compressor.npz")
    with pytest.raises(ValueError):
        GRID["validate_run"](study, job, binding, "source-identity")


def test_immutable_completion_rejects_self_consistent_artifact_replacement(completed):
    _, job, study, folder, binding = completed
    checked = GRID["validate_run"](study, job, binding, "source-identity")
    plan = {
        "plan_id": "fixed-plan",
        "datasets": {"scifact": binding},
        "core_source_sha256": "source-identity",
    }
    record = {
        **checked,
        "status": "complete",
        "plan_id": "fixed-plan",
        "logs": {},
        "job": {
            key: job[key]
            for key in (
                "job_id",
                "dataset",
                "lambda",
                "alpha",
                "seed",
                "dimension",
                "method",
                "split",
            )
        },
    }
    GRID["verify_completion"](record, plan, job)
    np.savez(folder / "compressor.npz", mean=np.ones(768), encoder=np.zeros((128, 768)))
    refresh_artifact(folder, "compressor.npz")
    with pytest.raises(ValueError, match="changed"):
        GRID["verify_completion"](record, plan, job)


def test_changed_plan_rejected_before_any_core_execution(tmp_path, monkeypatch):
    globals_ = GRID["validate_plan"].__globals__
    monkeypatch.setitem(globals_, "REPO", tmp_path)
    output = tmp_path / "grid"
    output.mkdir()
    write_json(output / "plan.json", {"split": "dev"})
    (output / "plan.sha256").write_text(sha256(output / "plan.json"))
    write_json(output / "plan.json", {"split": "test"})
    with pytest.raises(ValueError, match="Immutable grid plan changed"):
        GRID["validate_plan"](output)


def test_declared_grid_mutation_rejected_even_if_plan_digest_is_recomputed(tmp_path, monkeypatch):
    globals_ = GRID["validate_plan"].__globals__
    monkeypatch.setitem(globals_, "REPO", tmp_path)
    output = tmp_path / "grid"
    output.mkdir()
    configs = {name: base(name) for name in ("scifact", "fiqa")}
    plan = {
        "output": "grid",
        "protocol_sha256": GRID["PROTOCOL_SHA"],
        "split": "dev",
        "datasets": {name: {"config": c} for name, c in configs.items()},
        "jobs": GRID["job_configs"](configs, "grid"),
    }
    plan["jobs"][0]["config"]["experiment"]["split"] = "test"
    plan["plan_id"] = fingerprint(plan)
    write_json(output / "plan.json", plan)
    (output / "plan.sha256").write_text(sha256(output / "plan.json"))
    with pytest.raises(ValueError, match="96-fit"):
        GRID["validate_plan"](output)


def test_input_hash_verification_rejects_cache_replacement(tmp_path):
    (tmp_path / "documents.npy").write_bytes(b"frozen vectors")
    expected = {"documents.npy": sha256(tmp_path / "documents.npy")}
    (tmp_path / "documents.npy").write_bytes(b"replacement vectors")
    with pytest.raises(ValueError, match="changed"):
        GRID["verify_files"](tmp_path, expected)


def test_resume_skips_sealed_success_and_failure_and_executes_only_missing(tmp_path, monkeypatch):
    globals_ = GRID["run_plan"].__globals__
    monkeypatch.setitem(globals_, "REPO", tmp_path)
    output = tmp_path / "grid"
    output.mkdir()
    write_json(output / "plan.json", {"fixture": "orchestration only"})
    plan = {"plan_id": "fixed", "jobs": [{"job_id": name} for name in ("a", "b", "c")]}
    monkeypatch.setitem(globals_, "validate_plan", lambda *args, **kwargs: plan)
    invoked, verified = [], []

    def execute(output, plan, job):
        invoked.append(job["job_id"])
        status = "failed" if job["job_id"] == "b" else "complete"
        return {"status": status, "fit_invoked": True, "study_path": job["job_id"]}

    def verify(record, plan, job):
        verified.append(job["job_id"])
        assert record["study_path"] == job["job_id"]

    monkeypatch.setitem(globals_, "execute_job", execute)
    monkeypatch.setitem(globals_, "verify_completion", verify)
    first = GRID["run_plan"](output, max_new_runs=1)
    assert first["status"] == "partial" and invoked == ["a"]
    sealed_a = (output / "completed/a.json").read_bytes()
    second = GRID["run_plan"](output, max_new_runs=1)
    assert second["status"] == "partial" and invoked == ["a", "b"]
    sealed_b = (output / "completed/b.json").read_bytes()
    final = GRID["run_plan"](output)
    assert invoked == ["a", "b", "c"]
    assert final["status"] == "complete_with_failures"
    assert final["studies"] == ["a", "c"]
    assert verified == ["a", "a", "b"]
    assert (output / "completed/a.json").read_bytes() == sealed_a
    assert (output / "completed/b.json").read_bytes() == sealed_b
