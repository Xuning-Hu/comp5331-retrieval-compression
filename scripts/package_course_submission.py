"""Package the reviewed report, code, compact evidence and execution source snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_supplementary_delivery(root):
    """A declared follow-up must have complete, unchanged reviewed evidence."""
    if not (root / "docs/supplementary-experiment-protocol.md").exists():
        return
    path = root / "docs/verification/supplementary-completion.json"
    if not path.is_file():
        raise ValueError("Declared supplementary experiments lack completion verification")
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "mpnet_dev": 42, "mpnet_test": 42, "weight_grid_dev": 96,
        "weight_grid_test": 0, "new_neural_fits": 123,
    }
    if record.get("status") != "passed" or record.get("counts") != expected:
        raise ValueError("Supplementary evidence is incomplete or has unexpected scope")
    if record.get("original_evidence_unchanged") is not True:
        raise ValueError("Original evidence preservation was not verified")
    files = record.get("delivery_file_sha256", {})
    if not files:
        raise ValueError("Supplementary verification has no bound delivery files")
    required = {
        "docs/supplementary-experiment-protocol.md",
        "docs/verification/supplementary/mpnet/verification.json",
        "docs/verification/supplementary/weight-grid/verification.json",
        "docs/results/supplementary/protocol/mpnet/plan.json",
        "docs/results/supplementary/protocol/mpnet/execution-source.zip",
        "docs/results/supplementary/protocol/weight-grid/plan.json",
        "docs/results/supplementary/protocol/weight-grid/plan.sha256",
        "docs/results/supplementary/protocol/weight-grid/execution-source.zip",
        "docs/results/supplementary/weight-grid/snapshot-map.json",
        "docs/analysis/supplementary/analysis-manifest.json",
        "docs/analysis/supplementary/summary.json",
    }
    for folder in ("docs/results/supplementary", "docs/analysis/supplementary",
                   "docs/verification/supplementary"):
        required.update(p.relative_to(root).as_posix() for p in (root / folder).rglob("*") if p.is_file())
    if not required <= set(files):
        raise ValueError("Supplementary delivery inventory omits required evidence")
    if len(list((root / "docs/results/supplementary/mpnet").glob("*/snapshot.json"))) != 6 or len(
        list((root / "docs/results/supplementary/weight-grid").glob("*/snapshot.json"))
    ) != 96:
        raise ValueError("Expected all 6 MPNet and 96 grid snapshots")
    for label, expected_runs, expected_splits in (
        ("mpnet", 84, {"dev": 42, "test": 42}), ("weight-grid", 96, {"dev": 96}),
    ):
        audit = json.loads((root / f"docs/verification/supplementary/{label}/verification.json").read_text(encoding="utf-8"))
        splits = {}
        for study in audit.get("studies", []):
            splits[study["split"]] = splits.get(study["split"], 0) + len(study["runs"])
        if audit.get("status") != "passed" or audit.get("run_count") != expected_runs or splits != expected_splits:
            raise ValueError("Supplementary scalar verification coverage differs")
    for name, expected_hash in files.items():
        artifact = (root / name).resolve()
        if not artifact.is_relative_to(root.resolve()) or not artifact.is_file():
            raise ValueError(f"Missing or unsafe supplementary artifact: {name}")
        if checksum(artifact) != expected_hash:
            raise ValueError(f"Supplementary artifact changed after verification: {name}")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--execution-archive", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=root / "output/COMP5331_Group14_Submission.zip"
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(root / "output"):
        raise ValueError("Submission archives must stay inside this project's output directory")
    if output.exists() and not args.replace:
        raise FileExistsError("Retain the existing package or explicitly use --replace")
    report = args.report.resolve()
    if report.read_bytes()[:5] != b"%PDF-":
        raise ValueError("A real rendered PDF is required")
    layout = json.loads(
        (root / "docs/verification/final-report-layout.json").read_text(encoding="utf-8")
    )
    report_source_hash = checksum(root / layout["source"])
    if checksum(report) != layout["pdf_sha256"] or report_source_hash != layout["source_sha256"]:
        raise ValueError("The report differs from the source/PDF pair that passed visual review")
    if layout["checked_page_ids"] != list(range(1, layout["page_count"] + 1)) or layout.get(
        "unresolved_layout_issues"
    ):
        raise ValueError("The report has not passed complete page-by-page visual review")
    scientific_review = (root / "docs/verification/final-report-scientific-review.md").read_text(
        encoding="utf-8"
    )
    if report_source_hash not in scientific_review:
        raise ValueError("Scientific review is not bound to this report source")
    verify_supplementary_delivery(root)
    audit = json.loads(args.verification.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise ValueError("Final independent verification has not passed")
    studies = audit.get("studies", [])
    if len(studies) != 6 or sum(len(study["runs"]) for study in studies) != 180:
        raise ValueError("Expected the six complete final-test studies and all 180 runs")
    if any(study.get("split") != "test" for study in studies):
        raise ValueError("The attached independent verification must cover final test")
    execution_archive = args.execution_archive.resolve()
    progress_path = execution_archive.with_name("progress.json")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("source_archive_sha256") != checksum(execution_archive):
        raise ValueError("Pipeline chronology is not bound to this execution archive")
    jobs = progress.get("jobs", [])
    if progress.get("status") != "complete" or len(jobs) != 10:
        raise ValueError("The full ten-stage pipeline chronology is required")
    if any(job.get("status") != "complete" or job.get("exit_code") != 0 for job in jobs):
        raise ValueError("Pipeline chronology contains an incomplete or failed stage")
    development = [job for job in jobs if not job["name"].endswith("-test")]
    final_test = [job for job in jobs if job["name"].endswith("-test")]
    if len(development) != 4 or len(final_test) != 6:
        raise ValueError("Expected four development stages and six final-test stages")
    if max(datetime.fromisoformat(job["finished_utc"]) for job in development) > min(
        datetime.fromisoformat(job["started_utc"]) for job in final_test
    ):
        raise ValueError("Final testing started before all development stages completed")
    members = {}
    for folder in ("src", "scripts", "configs", "tests", "docs"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix.lower() in {
                ".py", ".toml", ".md", ".json", ".csv", ".png", ".svg", ".zip",
                ".log", ".txt", ".sha256",
            }:
                members[f"source/{path.relative_to(root).as_posix()}"] = path
    for name in (
        "README.md", "CONTRIBUTING.md", "THIRD_PARTY.md", "LICENSE", "LICENSE.md",
        "pyproject.toml", "requirements-course-windows.txt", "requirements-report.txt",
        "uv.lock", ".gitignore", ".gitattributes", ".python-version",
    ):
        path = root / name
        if path.is_file():
            members[f"source/{name}"] = path
    members["report/COMP5331_Group14_Report.pdf"] = report
    members["evidence/main-execution-source.zip"] = execution_archive
    members["evidence/pipeline-progress.json"] = progress_path
    for job in jobs:
        log = execution_archive.with_name(f"{job['name']}.log")
        if not log.is_file():
            raise ValueError(f"Missing pipeline stage log: {log.name}")
        members[f"evidence/pipeline-logs/{log.name}"] = log
    members["evidence/final-test-verification.json"] = args.verification.resolve()
    members["START_HERE_zh.md"] = root / "docs/答辩与交接说明.md"
    files = {
        name: {"sha256": checksum(path), "bytes": path.stat().st_size}
        for name, path in sorted(members.items())
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "COMP5331 Group 14 course submission and review",
        "final_test_studies": 6,
        "final_test_runs": 180,
        "files": files,
        "excluded": "large caches, raw datasets, fitted arrays, indexes, complete ranking files, "
        "virtual environments, git internals and temporary rendering files",
        "verification_boundary": "Compact per-query scores support aggregate replay; independent "
        "ranking/qrels verification is recorded but requires the retained original local runs to repeat.",
        "reproduction_limit": "Re-encoded, unit-normalized IMDb closely recovers published "
        "RAE performance (about 91.9% Euclidean and 87.8% cosine). Original-author input "
        "and protocol identity remain unconfirmed; raw/unit and search-backend differences "
        "are preserved in the report.",
        "visual_review": "The PDF must be inspected separately; packaging does not certify layout.",
    }
    readme = (
        "COMP5331 Group 14\n\n"
        "Read report/COMP5331_Group14_Report.pdf first. START_HERE_zh.md explains the project "
        "in Chinese; its repository paths are inside source/.\n\n"
        "Editable report: source/docs/report/final-report.md and its figures. "
        "Code and reproducible commands: source/README.md. "
        "Teacher requirement mapping: source/docs/course-completion.md. "
        "All main and ablation test scores: source/docs/results/course/.\n\n"
        "SUBMISSION_MANIFEST.json records packaged source, report and evidence file hashes. "
        "The original large "
        "datasets/indexes remain on the experiment host and are reproducible from the documented "
        "sources. A unit-input diagnostic closely recovers the published IMDb performance; "
        "the authors' original data/protocol identity is not claimed. AI assistance is "
        "disclosed in the report; no individual group contribution was invented.\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".zip.part")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
        for name, path in sorted(members.items()):
            archive.write(path, name)
        archive.writestr("README.txt", readme)
        archive.writestr(
            "SUBMISSION_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )
    with zipfile.ZipFile(temp) as archive:
        if archive.testzip() is not None:
            raise ValueError("Submission archive failed its CRC integrity check")
    temp.replace(output)
    summary = {
        "archive": str(output), "bytes": output.stat().st_size,
        "sha256": checksum(output), "packaged_source_files": len(files),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
