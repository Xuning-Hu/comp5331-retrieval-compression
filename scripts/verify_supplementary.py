"""Verify completed bounded follow-ups and export compact evidence, without fitting.

The independent scalar evaluator reads saved rankings and official judgments.
The grid has no test evaluation. Original evidence preservation checks the exact
pre-follow-up file inventory; it is not another replay of the original training.
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import course_experiments as course
import run_second_encoder as second
import run_weight_sensitivity as grid
from analyze_course import load_study
from export_course_snapshot import export
from verify_course_results import verify

from retrieval_lab.common import read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
DEST = Path("docs/results/supplementary")
AUDIT = Path("docs/verification/supplementary")


def verified_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(source) != sha256(target):
            raise ValueError(f"An earlier exported artifact differs: {target}")
    else:
        shutil.copy2(source, target)


def checked_export(source, target, **kwargs):
    if (target / "snapshot.json").exists():
        # Exports are immutable; the common reader validates registered checksums.
        loaded = load_study(target)
        raw = read_json(source / "study.json")
        snapshot = read_json(target / "snapshot.json")
        exported = snapshot["files"]
        observed = {p.relative_to(target).as_posix() for p in target.rglob("*")
                    if p.is_file() and p != target / "snapshot.json"}
        if observed != set(exported):
            raise ValueError("Existing export file inventory changed")
        for name, expected in exported.items():
            artifact = (target / name).resolve()
            if not artifact.is_relative_to(target.resolve()) or sha256(artifact) != expected:
                raise ValueError(f"Existing export artifact changed: {name}")
        if (loaded["study_id"] != raw["study_id"]
                or snapshot["source_study_manifest_sha256"] != sha256(source / "study.json")):
            raise ValueError("Existing export identifies another study")
        load_study(source)  # Validate current raw artifacts, not merely their stored names.
        required = {str((source / "study.json").resolve())}
        for result_path in source.glob("*/result.json"):
            required.add(str(result_path.resolve()))
            required.update(str((result_path.parent / name).resolve())
                            for name in read_json(result_path)["files"])
        required.update(str(Path(value).resolve()) for value in kwargs.values() if value is not None)
        consumed = snapshot["source_inputs_sha256"]
        if not required <= set(consumed):
            raise ValueError("Existing export lacks required source bindings")
        for name, expected in consumed.items():
            if not Path(name).is_file() or sha256(Path(name)) != expected:
                raise ValueError(f"Existing export source binding changed: {name}")
        if kwargs.get("verification") and snapshot["verification"]["sha256"] != sha256(
            kwargs["verification"]
        ):
            raise ValueError("Existing export verification binding changed")
        return target
    return export(source, target, **kwargs)


def audit_mpnet():
    plan = read_json(second.ROOT / "plan.json")
    second.assert_inputs(plan)
    if sha256(second.ROOT / "execution-source.zip") != plan["source_archive_sha256"]:
        raise ValueError("MPNet source archive changed")
    invocations = []
    for path in (second.ROOT / "invocations").glob("*/progress.json"):
        progress = read_json(path)
        if progress.get("status") == "complete":
            invocations.append((path, progress))
    if not invocations:
        raise ValueError("No complete MPNet invocation")
    path, progress = sorted(invocations, key=lambda row: str(row[0]))[-1]
    jobs = progress["jobs"]
    expected = [f"{dataset}-{stage}" for stage in ("embed", "dev", "test")
                for dataset in second.DATASETS]
    if [job["name"] for job in jobs] != expected or any(
        job["status"] != "complete" or job["exit_code"] != 0 for job in jobs
    ):
        raise ValueError("Incomplete or reordered MPNet stages")
    if progress["plan_sha256"] != sha256(second.ROOT / "plan.json"):
        raise ValueError("MPNet chronology has a different plan")
    if max(datetime.fromisoformat(job["finished_utc"]) for job in jobs[3:6]) > min(
        datetime.fromisoformat(job["started_utc"]) for job in jobs[6:9]
    ):
        raise ValueError("MPNet test preceded complete development freezing")
    studies, records = [], []
    for dataset in second.DATASETS:
        frozen = second.ROOT / f"frozen-{dataset}.json"
        config, dev = course.validate_frozen(frozen)
        if config != plan["configs"][dataset]:
            raise ValueError("Frozen MPNet configuration differs from the declared plan")
        pointer = read_json(second.ROOT / f"test-{dataset}.json")
        test = Path(pointer["study"])
        if (Path(pointer["source"]).resolve() != dev.resolve()
                or Path(read_json(test / "study.json")["source_study"]).resolve() != dev.resolve()):
            raise ValueError("MPNet test pointer/source differs from frozen development")
        studies += [dev, test]
        records.append((dataset, frozen, dev, test))
    verify(studies, AUDIT / "mpnet")
    audit = AUDIT / "mpnet/verification.json"
    verified = read_json(audit)
    coverage = {(row["dataset"], row["split"]): len(row["runs"]) for row in verified["studies"]}
    expected_coverage = {(dataset, split): 14 for dataset in second.DATASETS for split in ("dev", "test")}
    if verified["run_count"] != 84 or coverage != expected_coverage or len(verified["studies"]) != 6:
        raise ValueError("MPNet expected 42 dev + 42 frozen test results")
    for dataset, frozen, dev, test in records:
        for split, study in (("dev", dev), ("test", test)):
            checked_export(study, DEST / "mpnet" / f"{dataset}-{split}",
                           selection=course.SELECTION, frozen=frozen, verification=audit)
        verified_copy(frozen, DEST / "protocol/mpnet" / frozen.name)
    for source in (second.ROOT / "plan.json", second.ROOT / "execution-source.zip", path):
        verified_copy(source, DEST / "protocol/mpnet" / source.name)
    for job in jobs:
        source = path.parent / f"{job['name']}.log"
        verified_copy(source, DEST / "protocol/mpnet/logs" / source.name)
    return {"verification": str(audit), "sha256": sha256(audit),
            "dev_runs": 42, "test_runs": 42, "new_neural_fits": 27,
            "all_dev_finished_before_test": True}


def audit_grid():
    folder = Path("runs/supplementary/weight-sensitivity")
    plan = grid.validate_plan(folder)
    progress = read_json(folder / "progress.json")
    if progress["status"] != "complete" or len(progress["studies"]) != 96:
        raise ValueError("Grid not completely successful; report failures before exporting")
    if progress["test_evaluated"] is not False:
        raise ValueError("Grid must be dev-only")
    if progress["plan_sha256"] != sha256(folder / "plan.json"):
        raise ValueError("Grid progress belongs to another plan")
    if progress.get("end_input_verification") != {
        "source_protocol_cache_data_hashes_unchanged": True,
        "plan_sha256": sha256(folder / "plan.json"),
    }:
        raise ValueError("Grid end-of-run input verification is missing or inconsistent")
    if len(progress.get("jobs", [])) != 96:
        raise ValueError("Grid chronology does not cover 96 completed jobs")
    studies = []
    for job in plan["jobs"]:
        receipt = read_json(folder / "completed" / f"{job['job_id']}.json")
        grid.verify_completion(receipt, plan, job)
        if receipt["status"] != "complete":
            raise ValueError("Unexpected failed grid receipt")
        progress_job = progress["jobs"][len(studies)]
        if {k: v for k, v in progress_job.items() if k != "reused_this_invocation"} != receipt:
            raise ValueError("Grid chronology disagrees with a sealed completion")
        studies.append(Path(receipt["study_path"]))
    if [str(p).replace("\\", "/") for p in studies] != progress["studies"]:
        raise ValueError("Grid study list differs from sealed completions")
    verify(studies, AUDIT / "weight-grid")
    audit = AUDIT / "weight-grid/verification.json"
    verified = read_json(audit)
    if verified["run_count"] != 96 or any(s["split"] != "dev" for s in verified["studies"]):
        raise ValueError("Grid verification has unexpected coverage")
    snapshots = []
    for job, study in zip(plan["jobs"], studies, strict=True):
        target = DEST / "weight-grid" / job["job_id"]
        checked_export(study, target, verification=audit)
        snapshots.append(job["job_id"])
    write_json(DEST / "weight-grid/snapshot-map.json", {
        "status": "complete", "path_base": "manifest", "studies": snapshots,
        "scope": "96 dev-only saved-score snapshots; no raw model/index/corpus required for reaggregation",
        "raw_progress_sha256": sha256(folder / "progress.json"),
        "verification_sha256": sha256(audit),
    })
    for name in ("plan.json", "plan.sha256", "execution-source.zip", "progress.json"):
        verified_copy(folder / name, DEST / "protocol/weight-grid" / name)
    for source in sorted((folder / "completed").glob("*.json")):
        verified_copy(source, DEST / "protocol/weight-grid/completed" / source.name)
    for source in sorted((folder / "attempts").rglob("*")):
        if source.is_file() and source.suffix in {".json", ".log"}:
            verified_copy(source, DEST / "protocol/weight-grid/attempts" / source.relative_to(folder / "attempts"))
    return {"verification": str(audit), "sha256": sha256(audit),
            "dev_runs": 96, "test_runs": 0, "new_neural_fits": 96,
            "query_overlap_boundary": "No new Original grid run: saved geometric aggregates are checked; scalar relevance metrics are independently replayed from rankings."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("mpnet", "grid", "all"), default="all", nargs="?")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error("Run from repository root")
    AUDIT.mkdir(parents=True, exist_ok=True)
    stages = ("mpnet", "grid") if args.stage == "all" else (args.stage,)
    for name in stages:
        record = audit_mpnet() if name == "mpnet" else audit_grid()
        record.update({"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
                       "script_sha256": sha256(Path(__file__))})
        write_json(AUDIT / f"{name}-export.json", record)
        print(record, flush=True)


if __name__ == "__main__":
    main()
