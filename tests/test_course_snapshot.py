import csv
import runpy
import shutil
from pathlib import Path

import pytest

from retrieval_lab.common import read_json, sha256, write_json

SCRIPTS = Path(__file__).parents[1] / "scripts"
EXPORT = runpy.run_path(str(SCRIPTS / "export_course_snapshot.py"))
ANALYSIS_FIXTURE = runpy.run_path(str(Path(__file__).with_name("test_course_analysis.py")))
COURSE = runpy.run_path(str(SCRIPTS / "course_experiments.py"))


def all_methods_study(tmp_path):
    # Reuse the analysis fixture's schema/aggregation contracts; extend its method set.
    source = ANALYSIS_FIXTURE["make_study"](tmp_path / "source")
    study = read_json(source / "study.json")
    study["config"]["experiment"]["methods"] += ["original", "rp", "rank_only"]
    for method, template, seeds in (
        ("original", "pca", [0]),
        ("rp", "ae", [17, 42]),
        ("rank_only", "qrae", [17, 42]),
    ):
        for seed in seeds:
            dimension = 8 if method == "original" else 4
            folder = source / f"{method}-{dimension}d-seed{seed}"
            shutil.copytree(source / f"{template}-4d-seed{seed}", folder)
            result = read_json(folder / "result.json")
            info = read_json(folder / "compressor.json")
            for row in (result, info):
                row.update({"method": method, "dimension": dimension})
            if method in {"original", "rp"}:
                info.pop("history", None)
            write_json(folder / "compressor.json", info)
            result["files"]["compressor.json"] = sha256(folder / "compressor.json")
            write_json(folder / "result.json", result)
    cache = tmp_path / "cache"
    cache.mkdir()
    write_json(cache / "manifest.json", {"files": study["embedding_files"], "dimension": 8})
    study["embedding_cache"] = str(cache)
    write_json(source / "study.json", study)
    for path in source.glob("*/result.json"):
        folder = path.parent
        result = read_json(path)
        info = read_json(folder / "compressor.json")
        for filename in ("compressor.npz", "index.faiss", "ranking.jsonl", "geometry.json"):
            (folder / filename).write_bytes(b"registered raw artifact, excluded from snapshot")
            result["files"][filename] = sha256(folder / filename)
        if info.get("history"):
            with (folder / "training.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(info["history"][0]))
                writer.writeheader()
                writer.writerows(info["history"])
            result["files"]["training.csv"] = sha256(folder / "training.csv")
        write_json(path, result)
    return source


def test_all_seven_methods_all_seeds_export_without_large_artifacts(tmp_path):
    source = all_methods_study(tmp_path)
    before = {str(path): sha256(path) for path in source.rglob("*") if path.is_file()}
    output = EXPORT["export"](source, tmp_path / "export")
    manifest = read_json(output / "snapshot.json")
    assert manifest["methods"] == ["ae", "original", "pca", "qrae", "rae", "rank_only", "rp"]
    assert manifest["run_count"] == 12
    assert len(read_json(output / "run-results.json")) == 12
    assert len(read_json(output / "compressor-metadata.json")) == 12
    assert manifest["training_rows"] == 16
    assert all(sha256(output / name) == value for name, value in manifest["files"].items())
    assert not any(
        path.suffix in {".npz", ".npy", ".faiss", ".jsonl"} for path in output.rglob("*")
    )
    assert before == {str(path): sha256(path) for path in source.rglob("*") if path.is_file()}
    # The standard score-only snapshot reader can still recompute every aggregate.
    reloaded = ANALYSIS_FIXTURE["ANALYSIS"]["load_study"](output)
    assert sum(len(group["runs"]) for group in reloaded["groups"]) == 12


def test_tampered_source_model_is_rejected_before_export(tmp_path):
    source = all_methods_study(tmp_path)
    (source / "qrae-4d-seed42/compressor.npz").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="checksum differs"):
        EXPORT["export"](source, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_incomplete_seed_grid_is_rejected(tmp_path):
    source = all_methods_study(tmp_path)
    (source / "rank_only-4d-seed42/result.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        EXPORT["export"](source, tmp_path / "export")


def test_export_refuses_overwrite_and_source_subdirectory(tmp_path):
    source = all_methods_study(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        EXPORT["export"](source, source / "export")
    output = EXPORT["export"](source, tmp_path / "export")
    with pytest.raises(FileExistsError, match="nonempty"):
        EXPORT["export"](source, output)


def test_verification_reference_must_bind_exact_current_runs(tmp_path):
    source = all_methods_study(tmp_path)
    verification = tmp_path / "verification.json"
    audit = {
        "status": "passed",
        "studies": [
            {
                "source": str(source),
                "study_id": read_json(source / "study.json")["study_id"],
                "study_manifest_sha256": sha256(source / "study.json"),
                "runs": [
                    {
                        "run": path.parent.name,
                        "result_sha256": sha256(path),
                        "registered_artifacts": read_json(path)["files"],
                        "maximum_error": 0.0,
                    }
                    for path in source.glob("*/result.json")
                ],
            }
        ],
    }
    write_json(verification, audit)
    output = EXPORT["export"](source, tmp_path / "export", verification=verification)
    assert read_json(output / "verification-reference.json")["sha256"] == sha256(verification)
    audit["studies"][0]["runs"][0]["result_sha256"] = "stale-hash"
    write_json(verification, audit)
    with pytest.raises(ValueError, match="predates changed"):
        EXPORT["export"](source, tmp_path / "rejected", verification=verification)


def test_frozen_provenance_blocks_consistent_source_replacement(tmp_path, monkeypatch):
    source = all_methods_study(tmp_path)
    selection = tmp_path / "selection.json"
    write_json(selection, {"regularization_weight": 0.1, "ranking_weight": 1.0})
    monkeypatch.setitem(COURSE["record_dev"].__globals__, "ROOT", tmp_path)
    monkeypatch.setitem(COURSE["record_dev"].__globals__, "SELECTION", selection)
    frozen = COURSE["record_dev"](source, "fixture")
    output = EXPORT["export"](source, tmp_path / "export", frozen=frozen, selection=selection)
    assert sha256(output / "frozen-manifest.json") == sha256(frozen)
    assert sha256(output / "selection.json") == sha256(selection)
    folder = source / "qrae-4d-seed42"
    (folder / "compressor.npz").write_bytes(b"different, internally rehashed weights")
    result = read_json(folder / "result.json")
    result["files"]["compressor.npz"] = sha256(folder / "compressor.npz")
    write_json(folder / "result.json", result)
    with pytest.raises(ValueError, match="provenance checksum differs"):
        EXPORT["export"](source, tmp_path / "rejected", frozen=frozen, selection=selection)


def test_snapshot_preserves_rank_only_effective_parameters_and_history(tmp_path):
    source = all_methods_study(tmp_path)
    study = read_json(source / "study.json")
    study["config"]["neural"]["ranking_weight"] = 10.0
    write_json(source / "study.json", study)
    for path in source.glob("rank_only-*/compressor.json"):
        info = read_json(path)
        info["training"]["ranking_weight"] = 10.0
        info["ranking_weight"] = 1.0
        write_json(path, info)
        result_path = path.parent / "result.json"
        result = read_json(result_path)
        result["files"][path.name] = sha256(path)
        write_json(result_path, result)
    load = ANALYSIS_FIXTURE["ANALYSIS"]["load_study"]
    original = load(source)
    exported = load(EXPORT["export"](source, tmp_path / "export"))
    a = next(group for group in original["groups"] if group["method"] == "rank_only")
    b = next(group for group in exported["groups"] if group["method"] == "rank_only")
    assert a["parameters"]["ranking_weight"] == b["parameters"]["ranking_weight"] == 1.0
    assert a["parameter_id"] == b["parameter_id"]
    assert a["group_id"] == b["group_id"]
    assert b["parameter_source"] == "compressor metadata"
    assert a["runs"][0]["info"]["history"] == b["runs"][0]["info"]["history"]
    assert "training.csv" in exported["checked_inputs"]


@pytest.mark.parametrize("changed", ["bytes", "run_keys"])
def test_exported_compressor_metadata_tampering_is_rejected(tmp_path, changed):
    output = EXPORT["export"](all_methods_study(tmp_path), tmp_path / "export")
    path = output / "compressor-metadata.json"
    metadata = read_json(path)
    if changed == "bytes":
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        error = "checksum differs"
    else:
        metadata.pop(next(iter(metadata)))
        write_json(path, metadata)
        manifest = read_json(output / "snapshot.json")
        manifest["files"][path.name] = sha256(path)
        write_json(output / "snapshot.json", manifest)
        error = "run keys disagree"
    with pytest.raises(ValueError, match=error):
        ANALYSIS_FIXTURE["ANALYSIS"]["load_study"](output)


def test_exported_history_can_be_recovered_from_merged_csv(tmp_path):
    output = EXPORT["export"](all_methods_study(tmp_path), tmp_path / "export")
    path = output / "compressor-metadata.json"
    metadata = read_json(path)
    expected = metadata["qrae-4d-seed42"].pop("history")
    write_json(path, metadata)
    manifest = read_json(output / "snapshot.json")
    manifest["files"][path.name] = sha256(path)
    write_json(output / "snapshot.json", manifest)
    loaded = ANALYSIS_FIXTURE["ANALYSIS"]["load_study"](output)
    run = next(
        run
        for group in loaded["groups"]
        for run in group["runs"]
        if run["name"] == "qrae-4d-seed42"
    )
    assert run["info"]["history"] == expected
