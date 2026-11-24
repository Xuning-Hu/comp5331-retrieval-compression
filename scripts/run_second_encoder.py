"""Bounded post-hoc MPNet transfer check; no new hyperparameter selection.

Run at repository root. All three development studies are frozen before any
MPNet test evaluation. The original E5 studies and selection are read-only.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import course_experiments as course

from retrieval_lab.common import read_json, sha256, write_json
from retrieval_lab.data import load_dataset
from retrieval_lab.embedding import cache_path
from retrieval_lab.experiment import evaluate_saved
from retrieval_lab.reporting import write_report

ROOT = Path("runs/supplementary/mpnet")
PROTOCOL = Path("docs/supplementary-experiment-protocol.md")
DATASETS = ("scifact", "nfcorpus", "fiqa")
REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"


def now():
    return datetime.now(timezone.utc).isoformat()


def config_for(dataset):
    if dataset not in DATASETS:
        raise ValueError("Only the three predeclared datasets are permitted")
    config = course.selected_config(dataset)
    expected = {"regularization_weight": 1e-5, "ranking_weight": 10.0, "margin": 0.1}
    if any(config["neural"][key] != value for key, value in expected.items()):
        raise ValueError("Original selected loss weights changed")
    config["embedding"].update({
        "model": "sentence-transformers/all-mpnet-base-v2",
        "revision": REVISION, "directory": "cache/embeddings-mpnet",
        "max_length": 384, "query_prefix": "", "document_prefix": "",
    })
    config["experiment"].update({"dimensions": [128], "output": str(ROOT / dataset)})
    return config


def inputs():
    paths = sorted(Path("src/retrieval_lab").glob("*.py")) + [
        Path("scripts/run_second_encoder.py"), Path("scripts/course_experiments.py"),
        Path("configs/course-scifact-pilot.toml"), Path("runs/course/selection.json"),
        PROTOCOL,
    ]
    # Bind dataset and all official split manifests, excluding large corpus text.
    for dataset in DATASETS:
        loaded = load_dataset(Path("data") / dataset, config_for(dataset)["data"])
        paths += [p for p in (Path("data") / dataset).rglob("*.json") if p.is_file()]
        if not loaded.manifest:
            raise ValueError("Missing dataset identity")
    return {p.as_posix(): sha256(p) for p in sorted(set(paths))}


def assert_inputs(plan):
    changed = [p for p, digest in plan["input_sha256"].items()
               if not Path(p).is_file() or sha256(Path(p)) != digest]
    if changed:
        raise ValueError(f"Frozen supplemental inputs changed: {changed}")


def create_plan():
    ROOT.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": 1,
        "status": "post-hoc exploratory follow-up; original test outcomes already examined",
        "configs": {dataset: config_for(dataset) for dataset in DATASETS},
        "input_sha256": inputs(),
        "planned_dev_runs": 42, "planned_test_runs": 42, "new_neural_fits": 27,
        "selection": "Fixed original E5 coefficients; no selection from MPNet dev or test",
    }
    path = ROOT / "plan.json"
    if path.exists():
        plan = read_json(path)
        if {k: plan[k] for k in identity} != identity:
            raise ValueError("Existing MPNet plan differs; never overwrite completed evidence")
        assert_inputs(plan)
        if sha256(ROOT / "execution-source.zip") != plan["source_archive_sha256"]:
            raise ValueError("MPNet execution source archive changed")
        return plan
    with zipfile.ZipFile(ROOT / "execution-source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for p in identity["input_sha256"]:
            archive.write(p, p)
    identity.update({"declared_utc": now(),
                     "source_archive_sha256": sha256(ROOT / "execution-source.zip")})
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(identity, ensure_ascii=False, indent=2) + "\n")
    return identity


def stage(kind, dataset):
    plan = read_json(ROOT / "plan.json")
    assert_inputs(plan)
    config = copy.deepcopy(plan["configs"][dataset])
    course.ROOT = ROOT
    if kind == "embed":
        from retrieval_lab.embedding import embed

        data = load_dataset(Path(config["data"]["directory"]) / dataset, config["data"])
        embed(config, data)
    elif kind == "dev":
        study = course.execute(config)
        course.record_dev(study, dataset)
    else:
        # Even a manually invoked test stage requires all three full freezes.
        for name in DATASETS:
            course.validate_frozen(ROOT / f"frozen-{name}.json")
        config, source = course.validate_frozen(ROOT / f"frozen-{dataset}.json")
        config["experiment"]["split"] = "test"
        data = load_dataset(Path(config["data"]["directory"]) / dataset, config["data"])
        study = evaluate_saved(config, data, cache_path(config, data), source)
        write_report(study)
        write_json(ROOT / f"test-{dataset}.json", {"study": str(study), "source": str(source)})
    assert_inputs(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["embed", "dev", "test"])
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.stage:
        if not args.dataset:
            parser.error("--stage requires --dataset")
        stage(args.stage, args.dataset)
        return
    plan = create_plan()
    if args.plan_only:
        print(ROOT / "plan.json")
        return
    invocation = ROOT / "invocations" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    invocation.mkdir(parents=True, exist_ok=False)
    record = {"started_utc": now(), "plan_sha256": sha256(ROOT / "plan.json"), "jobs": []}
    environment = {**os.environ, "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"}
    for kind in ("embed", "dev", "test"):
        for dataset in DATASETS:
            assert_inputs(plan)
            command = [sys.executable, "scripts/run_second_encoder.py",
                       "--stage", kind, "--dataset", dataset]
            name = f"{dataset}-{kind}"
            job = {"name": name, "started_utc": now(), "status": "running", "command": command}
            record["jobs"].append(job)
            write_json(invocation / "progress.json", record)
            print(json.dumps(job), flush=True)
            with (invocation / f"{name}.log").open("w", encoding="utf-8") as logfile:
                process = subprocess.run(command, env=environment, stdout=logfile, stderr=subprocess.STDOUT)
            job.update({"finished_utc": now(), "exit_code": process.returncode,
                        "status": "complete" if process.returncode == 0 else "failed"})
            write_json(invocation / "progress.json", record)
            print(json.dumps(job), flush=True)
            if process.returncode:
                raise SystemExit(process.returncode)
    assert_inputs(plan)
    record.update({"status": "complete", "finished_utc": now()})
    write_json(invocation / "progress.json", record)
    print(f"MPNet follow-up complete: {invocation}", flush=True)


if __name__ == "__main__":
    main()
