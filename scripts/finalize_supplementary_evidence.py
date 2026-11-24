"""Seal already verified follow-up exports and the unchanged original evidence.

This does not run experiments, repeat scalar verification, or certify PDF layout.
Run after verify_supplementary.py, analysis and the current test/static checks.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from package_course_submission import verify_supplementary_delivery

from retrieval_lab.common import read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run from repository root")
    baseline = Path("runs/supplementary/original-evidence-before.json")
    original = read_json(baseline)
    changed = [name for name, expected in original["files"].items()
               if not Path(name).is_file() or sha256(Path(name)) != expected]
    if changed:
        raise ValueError(f"Original evidence changed: {changed}")
    preserved = Path("docs/verification/supplementary/original-evidence-preservation.json")
    write_json(preserved, {
        "status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_sha256": sha256(baseline), "file_count": len(original["files"]),
        "scope": "Exactly the pre-follow-up inventory of original frozen selections, dev/test study/result records and compact course exports. This is not a new hash replay of every large model/index or a training replay.",
        "unchanged_file_sha256": original["files"],
    })
    for stage in ("mpnet", "grid"):
        record = read_json(Path(f"docs/verification/supplementary/{stage}-export.json"))
        if record["status"] != "passed" or sha256(Path(record["verification"])) != record["sha256"]:
            raise ValueError("Supplementary scalar verification/export is missing or changed")
    protocol = Path("docs/results/supplementary/protocol")
    mpnet = read_json(protocol / "mpnet/plan.json")
    grid = read_json(protocol / "weight-grid/plan.json")
    for name, plan in (("mpnet", mpnet), ("weight-grid", grid)):
        if sha256(protocol / name / "execution-source.zip") != plan["source_archive_sha256"]:
            raise ValueError("Exported source archive differs from its plan")
        progress = read_json(protocol / name / "progress.json")
        if progress["status"] != "complete" or progress["plan_sha256"] != sha256(protocol / name / "plan.json"):
            raise ValueError("Exported complete chronology is missing or changed")
    grid_ids = [job["job_id"] for job in grid["jobs"]]
    if len(grid_ids) != 96 or len(set(grid_ids)) != 96:
        raise ValueError("Grid must retain every distinct declared job")
    if {p.stem for p in (protocol / "weight-grid/completed").glob("*.json")} != set(grid_ids):
        raise ValueError("Grid receipt export is incomplete")
    grid_fits = 0
    for job_id in grid_ids:
        receipt = read_json(protocol / "weight-grid/completed" / f"{job_id}.json")
        if receipt["status"] != "complete":
            raise ValueError("Grid has an unresolved failure")
        grid_fits += int(receipt["fit_invoked"])
    mapping = read_json(Path("docs/results/supplementary/weight-grid/snapshot-map.json"))
    if mapping["status"] != "complete" or mapping["path_base"] != "manifest" or mapping["studies"] != grid_ids:
        raise ValueError("Grid snapshot map differs from the complete plan")
    mpnet_fits = 0
    for dataset in ("scifact", "nfcorpus", "fiqa"):
        frozen = read_json(protocol / "mpnet" / f"frozen-{dataset}.json")
        if frozen["config"] != mpnet["configs"][dataset] or len(frozen["runs"]) != 14:
            raise ValueError("Exported MPNet freeze differs from the plan")
        mpnet_fits += sum(name.startswith(("ae-", "rae-", "qrae-")) for name in frozen["runs"])
    analysis = read_json(Path("docs/analysis/supplementary/analysis-manifest.json"))
    if (analysis["condition_count"], analysis["run_count"], analysis["bootstrap_samples"]) != (68, 180, 0):
        raise ValueError("Unexpected supplemental analysis coverage or inference scope")
    analysis_root = Path("docs/analysis/supplementary").resolve()
    if len(analysis["source_studies"]) != 102:
        raise ValueError("Expected three existing E5, three MPNet and 96 grid inputs")
    for source in analysis["source_studies"]:
        folder = (analysis_root / source["source_relative_path"]).resolve()
        if sha256(folder / "study.json") != source["manifest_sha256"]:
            raise ValueError("An analysis input study changed")
        if "snapshot_manifest_sha256" in source and sha256(folder / "snapshot.json") != source["snapshot_manifest_sha256"]:
            raise ValueError("An analysis input snapshot changed")
        for name, expected in source["checked_inputs"].items():
            artifact = (folder / name).resolve()
            if not artifact.is_relative_to(folder) or sha256(artifact) != expected:
                raise ValueError("An analysis input artifact changed")
    grid_map = analysis["grid_manifest"]
    if sha256((analysis_root / grid_map["relative_path"]).resolve()) != grid_map["sha256"]:
        raise ValueError("Analysis refers to another grid snapshot map")
    for name, expected in analysis["files"].items():
        if sha256(Path("docs/analysis/supplementary") / name) != expected:
            raise ValueError("Supplemental analysis artifact changed")
    checks = Path("docs/results/supplementary/checks")
    checks.mkdir(parents=True, exist_ok=True)
    for suffix in ("pytest", "ruff"):
        source = Path(f"runs/environment/revision3-final-{suffix}.log")
        (checks / source.name).write_bytes(source.read_bytes())
    test_text = (checks / "revision3-final-pytest.log").read_text(encoding="utf-8-sig")
    match = re.search(r"(\d+) passed", test_text)
    if not match or re.search(r"\d+ (?:failed|errors?)", test_text):
        raise ValueError("Full test log does not report a passing suite")
    if "All checks passed!" not in (checks / "revision3-final-ruff.log").read_text(encoding="utf-8-sig"):
        raise ValueError("Static check log does not report success")
    for source in Path("runs/supplementary/weight-sensitivity").glob("execution-console-*.log"):
        target = protocol / "weight-grid" / source.name
        if target.exists() and sha256(target) != sha256(source):
            raise ValueError("An earlier exported console log differs")
        target.write_bytes(source.read_bytes())
    paths = {Path("docs/supplementary-experiment-protocol.md")}
    for folder in ("docs/results/supplementary", "docs/analysis/supplementary", "docs/verification/supplementary"):
        paths.update(p for p in Path(folder).rglob("*") if p.is_file())
    paths.update(Path("src/retrieval_lab").glob("*.py"))
    paths.update(Path("scripts").glob("*supplementary*.py"))
    paths.update(Path("tests").glob("test_supplementary*.py"))
    paths.update(Path(name) for name in (
        "scripts/run_second_encoder.py", "scripts/run_weight_sensitivity.py",
        "scripts/package_course_submission.py", "tests/test_second_encoder.py",
        "tests/test_weight_sensitivity.py",
    ))
    record = {
        "schema_version": 1, "status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Post-hoc exploratory second encoder and dev-only joint weight grid; original 180 final-test results remain separate.",
        "counts": {"mpnet_dev": 42, "mpnet_test": 42, "weight_grid_dev": 96,
                   "weight_grid_test": 0, "new_neural_fits": mpnet_fits + grid_fits},
        "original_evidence_unchanged": True, "original_preserved_file_count": len(original["files"]),
        "passed_tests": int(match[1]), "bootstrap_samples": 0,
        "analysis_rows": {"total": 180, "existing_E5_test": 42, "new_MPNet_test": 42, "new_grid_dev": 96},
        "neural_fit_count_boundary": "Completed new configurations; interrupted/repeated attempts, if present, are separately recorded in execution logs.",
        "delivery_file_sha256": {path.as_posix(): sha256(path) for path in sorted(paths)},
    }
    write_json(Path("docs/verification/supplementary-completion.json"), record)
    verify_supplementary_delivery(ROOT)
    print(f"Supplementary evidence sealed: {record['counts']}; {len(paths)} delivery files; {match[1]} tests passed")


if __name__ == "__main__":
    main()
