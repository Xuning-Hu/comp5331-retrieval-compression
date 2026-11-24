import runpy
import sys
from pathlib import Path

import pytest

from retrieval_lab.common import read_json, sha256, write_json

COURSE = runpy.run_path(str(Path(__file__).parents[1] / "scripts/course_experiments.py"))


@pytest.fixture
def dev_study(tmp_path, monkeypatch):
    course_root = tmp_path / "course"
    course_root.mkdir()
    selection = course_root / "selection.json"
    write_json(selection, {"regularization_weight": 0.0001, "ranking_weight": 1.0})
    globals_ = COURSE["record_dev"].__globals__
    monkeypatch.setitem(globals_, "ROOT", course_root)
    monkeypatch.setitem(globals_, "SELECTION", selection)
    study = course_root / "dev" / "fixture"
    folder = study / "pca-4d-seed0"
    folder.mkdir(parents=True)
    config = {
        "data": {"name": "fixture"},
        "experiment": {"split": "dev", "methods": ["pca"], "dimensions": [4], "seeds": [17, 42]},
    }
    write_json(study / "study.json", {"study_id": "fixture", "config": config})
    artifacts = [
        "compressor.npz",
        "compressor.json",
        "per_query.json",
        "ranking.jsonl",
        "geometry.json",
        "index.faiss",
        "training.csv",
        "supervision.json",
    ]
    for name in artifacts:
        (folder / name).write_bytes(f"fixture:{name}\n".encode())
    write_json(
        folder / "result.json",
        {
            "study_id": "fixture",
            "split": "dev",
            "method": "pca",
            "dimension": 4,
            "seed": 0,
            "files": {name: sha256(folder / name) for name in artifacts},
        },
    )
    return study, course_root


def _replace_weights_and_result(study):
    folder = study / "pca-4d-seed0"
    (folder / "compressor.npz").write_bytes(b"different but internally consistent weights")
    result = read_json(folder / "result.json")
    result["files"]["compressor.npz"] = sha256(folder / "compressor.npz")
    write_json(folder / "result.json", result)


def test_freeze_records_results_and_all_artifacts(dev_study):
    study, _ = dev_study
    target = COURSE["record_dev"](study, "fixture")
    frozen = read_json(target)
    record = frozen["runs"]["pca-4d-seed0"]
    assert record["result_sha256"] == sha256(study / "pca-4d-seed0/result.json")
    assert {"compressor.npz", "training.csv", "supervision.json"} <= set(record["artifact_sha256"])
    config, source = COURSE["validate_frozen"](target)
    assert source == study.resolve()
    assert config["experiment"]["split"] == "dev"


def test_consistent_weight_and_result_replacement_is_rejected_before_test(dev_study, monkeypatch):
    study, _ = dev_study
    target = COURSE["record_dev"](study, "fixture")
    _replace_weights_and_result(study)
    with pytest.raises(ValueError, match="changed since freezing"):
        COURSE["validate_frozen"](target)

    def forbidden_dataset_load(*args, **kwargs):
        raise AssertionError("No test data should be loaded before frozen artifacts are validated")

    monkeypatch.setitem(COURSE["main"].__globals__, "load_dataset", forbidden_dataset_load)
    monkeypatch.setattr(sys, "argv", ["course_experiments.py", "test", "--frozen", str(target)])
    with pytest.raises(ValueError, match="changed since freezing"):
        COURSE["main"]()


def test_existing_freeze_is_never_overwritten(dev_study):
    study, _ = dev_study
    target = COURSE["record_dev"](study, "fixture")
    initial_bytes, initial_mtime = target.read_bytes(), target.stat().st_mtime_ns
    assert COURSE["record_dev"](study, "fixture") == target
    assert target.read_bytes() == initial_bytes
    assert target.stat().st_mtime_ns == initial_mtime
    _replace_weights_and_result(study)
    with pytest.raises(FileExistsError, match="cannot be replaced"):
        COURSE["record_dev"](study, "fixture")
    assert target.read_bytes() == initial_bytes
    assert target.stat().st_mtime_ns == initial_mtime


@pytest.mark.parametrize("change", ["result_only", "new_artifact", "selection"])
def test_other_post_freeze_changes_are_rejected(dev_study, change):
    study, course_root = dev_study
    target = COURSE["record_dev"](study, "fixture")
    if change == "result_only":
        path = study / "pca-4d-seed0/result.json"
        result = read_json(path)
        result["new_field"] = "changed"
        write_json(path, result)
    elif change == "new_artifact":
        (study / "pca-4d-seed0/new-artifact.txt").write_text("new")
    else:
        write_json(course_root / "selection.json", {"ranking_weight": 10.0})
    with pytest.raises(ValueError, match="changed"):
        COURSE["validate_frozen"](target)


def test_incomplete_study_cannot_be_frozen(dev_study):
    study, course_root = dev_study
    manifest = read_json(study / "study.json")
    manifest["config"]["experiment"]["dimensions"] = [4, 8]
    write_json(study / "study.json", manifest)
    with pytest.raises(ValueError, match="every predeclared run"):
        COURSE["record_dev"](study, "fixture")
    assert not (course_root / "frozen-fixture.json").exists()


def test_legacy_config_only_freeze_is_rejected(dev_study):
    _, course_root = dev_study
    target = course_root / "old-freeze.json"
    write_json(target, {"source_study": "somewhere", "frozen_before_test": True})
    with pytest.raises(ValueError, match="immutable per-run"):
        COURSE["validate_frozen"](target)
