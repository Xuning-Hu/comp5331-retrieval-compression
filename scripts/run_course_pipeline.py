"""Run the already selected dev matrix, freeze it, then evaluate saved test models.

This is a sequential convenience wrapper, not a new experiment specification.
It records the actual source bytes before launching and stops on any failure.
Completed runs are reused only by the checksum-validating experiment harness.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from retrieval_lab.common import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/course/pipeline"))
    parser.add_argument("--dev-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    invocation = args.output / stamp
    invocation.mkdir(exist_ok=False)
    files = sorted(Path("src/retrieval_lab").glob("*.py")) + [
        Path("scripts/course_experiments.py"),
        Path(__file__).relative_to(Path.cwd()) if Path(__file__).is_absolute() else Path(__file__),
        Path("docs/experiment-protocol.md"),
        Path("configs/course-scifact-pilot.toml"),
        Path("runs/course/selection.json"),
    ]
    hashes = {path.as_posix(): sha256(path) for path in files}
    with zipfile.ZipFile(invocation / "execution-source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.as_posix())
    jobs = [
        ("scifact-dev", ["dev", "--dataset", "scifact"]),
        ("scifact-ablations", ["ablations"]),
        ("nfcorpus-dev", ["dev", "--dataset", "nfcorpus"]),
        ("fiqa-dev", ["dev", "--dataset", "fiqa"]),
    ]
    if not args.dev_only:
        jobs += [
            (f"{label}-test", ["test", "--frozen", f"runs/course/frozen-{label}.json"])
            for label in [
                "scifact", "scifact-in_batch", "scifact-hard", "scifact-rank-only",
                "nfcorpus", "fiqa",
            ]
        ]
    record = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": hashes,
        "source_archive_sha256": sha256(invocation / "execution-source.zip"),
        "note": "All dev stages and immutable freezes complete before any final test stage.",
        "jobs": [],
    }
    environment = {**os.environ, "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"}
    for name, arguments in jobs:
        changed = [name for name, expected in hashes.items() if sha256(Path(name)) != expected]
        if changed:
            raise RuntimeError(f"Recorded execution inputs changed during pipeline: {changed}")
        command = [sys.executable, "scripts/course_experiments.py", *arguments]
        job = {
            "name": name, "command": command,
            "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        }
        record["jobs"].append(job)
        write_json(invocation / "progress.json", record)
        print(json.dumps(job), flush=True)
        with (invocation / f"{name}.log").open("w", encoding="utf-8") as logfile:
            process = subprocess.run(command, env=environment, stdout=logfile, stderr=subprocess.STDOUT)
        job.update({
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "exit_code": process.returncode,
            "status": "complete" if process.returncode == 0 else "failed",
        })
        write_json(invocation / "progress.json", record)
        print(json.dumps(job), flush=True)
        if process.returncode:
            raise SystemExit(process.returncode)
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    record["status"] = "complete"
    write_json(invocation / "progress.json", record)
    print(f"Pipeline complete: {invocation}", flush=True)


if __name__ == "__main__":
    main()
