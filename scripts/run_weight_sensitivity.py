"""Bounded E5 coefficient grid: plan once, then resume 96 dev-only QRAE fits.

Run from the repository root. Planning validates existing inputs but never trains.
All zero-weight cells retain QRAE's triplet/probe schedule. No test evaluator is used.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import math
import os
import shutil
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from retrieval_lab.common import (
    environment,
    fingerprint,
    read_json,
    sha256,
    source_digest,
    write_json,
)
from retrieval_lab.config import validate_config

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = Path("docs/supplementary-experiment-protocol.md")
PROTOCOL_SHA = "ae550f3f951852489f831e3f15bdf5b9cdbb21206ad284d1ff169a70e6a08c07"
DATASETS = ("scifact", "fiqa")
LAMBDAS = (0.0, 1e-6, 1e-5, 1e-4)
ALPHAS = (0.0, 1.0, 10.0, 100.0)
SEEDS = (17, 42, 2026)
DIMENSION = 128
EXPECTED_FITS = 96
REQUIRED_ARTIFACTS = {
    "compressor.npz",
    "compressor.json",
    "training_decoder.npz",
    "training.csv",
    "supervision.json",
    "result.json",
    "per_query.json",
    "ranking.jsonl",
    "geometry.json",
    "index.faiss",
}
PROBE_FIELDS = {
    "step",
    "total_loss",
    "batch_reconstruction",
    "regularizer",
    "weighted_regularizer",
    "batch_ranking",
    "weighted_ranking",
    "active_triplet_fraction",
    "corpus_probe_reconstruction",
    "train_probe_ranking",
    "gradient_norm",
    "latent_norm_min",
    "latent_norm_mean",
    "learning_rate",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def relative(path):
    """ASCII-relative outputs also avoid FAISS's Windows Unicode filename limitation."""
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(REPO):
        raise ValueError("Grid paths must stay inside this repository")
    return resolved.relative_to(REPO).as_posix()


def exclusive_json(path, value):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def file_hashes(folder):
    return {
        p.relative_to(folder).as_posix(): sha256(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def verify_files(folder, expected, *, exact=False):
    for name, digest in expected.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            raise ValueError(f"Missing/unsafe bound artifact: {folder}/{name}")
        if sha256(path) != digest:
            raise ValueError(f"Bound artifact changed: {folder}/{name}")
    if exact and file_hashes(folder) != expected:
        raise ValueError(f"Bound directory contents changed: {folder}")


def execution_environment():
    observed = environment()
    return {key: observed[key] for key in ("python", "platform", "machine", "hardware", "packages")}


def validate_base(config, dataset):
    validate_config(config)
    if config["data"]["name"] != dataset or config["experiment"]["split"] != "dev":
        raise ValueError("Grid references must be the designated dev datasets")
    embedding = config["embedding"]
    if (
        embedding["model"],
        embedding["revision"],
        embedding["max_length"],
        embedding["query_prefix"],
        embedding["document_prefix"],
    ) != (
        "intfloat/e5-base-v2",
        "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
        512,
        "query: ",
        "passage: ",
    ):
        raise ValueError("Grid requires the unchanged pinned E5 preprocessing")
    fixed = {
        "steps": 3000,
        "batch_size": 128,
        "lr": 0.001,
        "min_lr": 0.00001,
        "center": True,
        "margin": 0.1,
        "device": "cuda",
        "threads": 4,
        "log_every": 100,
    }
    if any(config["neural"][key] != value for key, value in fixed.items()):
        raise ValueError("Original 3000-step centered CUDA training settings are required")
    if config["supervision"]["negative_strategy"] != "random":
        raise ValueError("Grid permits only random negatives")
    exp = config["experiment"]
    if exp["ks"] != [10, 100] or exp["threads"] != 4 or exp["geometry_anchors"] != 256:
        raise ValueError("Original metric/search settings are required")


def job_configs(configs, output):
    jobs = []
    for dataset in DATASETS:
        validate_base(configs[dataset], dataset)
        for coefficient in LAMBDAS:
            for alpha in ALPHAS:
                for seed in SEEDS:
                    name = f"{dataset}-lambda{coefficient:g}-alpha{alpha:g}-seed{seed}"
                    config = copy.deepcopy(configs[dataset])
                    config["experiment"].update(
                        {
                            "output": f"{output}/jobs/{name}",
                            "methods": ["qrae"],
                            "dimensions": [DIMENSION],
                            "seeds": [seed],
                            "split": "dev",
                        }
                    )
                    config["neural"].update(
                        {
                            "regularization_weight": coefficient,
                            "ranking_weight": alpha,
                        }
                    )
                    jobs.append(
                        {
                            "job_id": name,
                            "dataset": dataset,
                            "lambda": coefficient,
                            "alpha": alpha,
                            "seed": seed,
                            "dimension": DIMENSION,
                            "method": "qrae",
                            "split": "dev",
                            "config": config,
                        }
                    )
    return jobs


def bind_dataset(dataset, frozen_path):
    from retrieval_lab.data import load_dataset
    from retrieval_lab.embedding import validate_cache

    frozen_path = Path(frozen_path).resolve()
    frozen = read_json(frozen_path)
    if frozen.get("schema_version") != 2:
        raise ValueError("A frozen original dev configuration is required")
    config = frozen["config"]
    validate_base(config, dataset)
    source = Path(frozen["source_study"]) / "study.json"
    if sha256(source) != frozen["study_manifest_sha256"]:
        raise ValueError("Original dev study manifest differs from its frozen record")
    manifest = read_json(source)
    if manifest["config"] != config or fingerprint(config) != frozen["config_fingerprint"]:
        raise ValueError("Frozen configuration identity differs")
    data_folder = Path(config["data"]["directory"]) / dataset
    data = load_dataset(data_folder, config["data"])
    if data.manifest != manifest["dataset"]:
        raise ValueError("Original dataset identity differs")
    cache = Path(manifest["embedding_cache"])
    cache_manifest = validate_cache(cache)
    if cache_manifest["files"] != manifest["embedding_files"] or cache_manifest["dimension"] != 768:
        raise ValueError("Original embedding cache differs")
    if cache_manifest["resolved_model_revision"] != config["embedding"]["revision"]:
        raise ValueError("Cached E5 revision differs")
    if sorted(data.qrels["dev"]) != manifest["query_ids"]:
        raise ValueError("Original development query IDs differ")
    data_files = {**data.manifest["files"], "manifest.json": sha256(data_folder / "manifest.json")}
    if (data_folder / "query_splits.json").exists():
        data_files["query_splits.json"] = sha256(data_folder / "query_splits.json")
    return {
        "config": config,
        "frozen_reference": relative(frozen_path),
        "frozen_reference_sha256": sha256(frozen_path),
        "source_study_manifest": relative(source),
        "source_study_manifest_sha256": sha256(source),
        "dataset_directory": relative(data_folder),
        "dataset_files": data_files,
        "dataset_manifest": data.manifest,
        "cache_directory": relative(cache),
        "cache_files": {
            **cache_manifest["files"],
            "manifest.json": sha256(cache / "manifest.json"),
        },
        "embedding_files": cache_manifest["files"],
        "dev_query_ids": sorted(data.qrels["dev"]),
        "train_query_ids": sorted(
            qid for qid, rels in data.qrels["train"].items() if any(v > 0 for v in rels.values())
        ),
        "test_labels_used_for_selection_or_evaluation": False,
        "loader_boundary": "Standard loader verifies the existing complete dataset; only train/dev qrels are retained in the view passed to the experiment core.",
    }


def create_plan(output, references):
    output = Path(output).resolve()
    output_name = relative(output)
    if not output_name.isascii() or output == REPO:
        raise ValueError("Use a dedicated ASCII-named output namespace inside the repository")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Plan output must be empty; existing plans are never replaced")
    if sha256(REPO / PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Supplementary protocol differs from the authorized fixed version")
    datasets = {name: bind_dataset(name, references[name]) for name in DATASETS}
    files = sorted((REPO / "src/retrieval_lab").glob("*.py")) + [
        Path(__file__).resolve(),
        REPO / PROTOCOL,
        REPO / "docs/experiment-protocol.md",
    ]
    source_files = {relative(path): sha256(path) for path in files}
    plan = {
        "schema_version": 1,
        "created_utc": now(),
        "output": output_name,
        "scope": "Post-hoc bounded development sensitivity; no test evaluation, best-cell promotion or replacement of original studies",
        "protocol": PROTOCOL.as_posix(),
        "protocol_sha256": PROTOCOL_SHA,
        "source_files": source_files,
        "core_source_sha256": source_digest(),
        "execution_environment": execution_environment(),
        "datasets": datasets,
        "lambda_values": list(LAMBDAS),
        "alpha_values": list(ALPHAS),
        "seeds": list(SEEDS),
        "dimension": DIMENSION,
        "expected_fits": EXPECTED_FITS,
        "split": "dev",
        "zero_axis_policy": "Every cell uses QRAE, including alpha=0: identical initialization and independent reconstruction/ranking RNG streams, triplet draws and probes at each fixed seed. Zero coefficients remove gradient contributions, not sampling or diagnostics. No claim of bitwise identity to separately executed AE/RAE.",
        "resume_policy": "Validate immutable completion records; retain numerical failures without automatic retries. A genuinely interrupted missing run restarts from initialization after archiving its partial artifacts; no optimizer-state continuation.",
        "jobs": job_configs({name: row["config"] for name, row in datasets.items()}, output_name),
    }
    plan["plan_id"] = fingerprint(plan)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output / "execution-source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in source_files:
            archive.write(REPO / name, name)
    plan["source_archive_sha256"] = sha256(output / "execution-source.zip")
    exclusive_json(output / "plan.json", plan)
    (output / "plan.sha256").write_text(sha256(output / "plan.json") + "\n", encoding="ascii")
    return plan


def validate_plan(output, *, full_inputs=True):
    output = Path(output).resolve()
    if sha256(output / "plan.json") != (output / "plan.sha256").read_text().strip():
        raise ValueError("Immutable grid plan changed")
    plan = read_json(output / "plan.json")
    payload = {k: v for k, v in plan.items() if k not in {"plan_id", "source_archive_sha256"}}
    if fingerprint(payload) != plan["plan_id"] or plan["output"] != relative(output):
        raise ValueError("Grid plan identity differs")
    if plan["protocol_sha256"] != PROTOCOL_SHA or plan["split"] != "dev":
        raise ValueError("Only the fixed dev-only supplementary protocol is permitted")
    expected = job_configs(
        {name: plan["datasets"][name]["config"] for name in DATASETS}, plan["output"]
    )
    if plan["jobs"] != expected or len(expected) != EXPECTED_FITS:
        raise ValueError("Grid jobs differ from the complete 96-fit declaration")
    verify_files(REPO, plan["source_files"])
    if source_digest() != plan["core_source_sha256"]:
        raise ValueError("Core source set changed")
    if sha256(output / "execution-source.zip") != plan["source_archive_sha256"]:
        raise ValueError("Execution source archive changed")
    if execution_environment() != plan["execution_environment"]:
        raise ValueError("Execution environment changed; preserve this plan and create a new one")
    for record in plan["datasets"].values():
        for key in ("frozen_reference", "source_study_manifest"):
            if sha256(REPO / record[key]) != record[f"{key}_sha256"]:
                raise ValueError("Original configuration reference changed")
        for folder_key, hashes_key in (
            ("cache_directory", "cache_files"),
            ("dataset_directory", "dataset_files"),
        ):
            hashes = record[hashes_key]
            if not full_inputs:
                hashes = {name: value for name, value in hashes.items() if name.endswith(".json")}
            verify_files(REPO / record[folder_key], hashes)
    return plan


def validate_run(study, job, binding, core_digest):
    """Validate semantic identity and complete diagnostics before sealing a run."""
    from retrieval_lab.experiment import _verified_run

    study = Path(study).resolve()
    if not study.is_relative_to((REPO / job["config"]["experiment"]["output"]).resolve()):
        raise ValueError("Completed study is outside its declared job namespace")
    manifest = read_json(study / "study.json")
    if manifest["config"] != job["config"] or manifest["source_sha256"] != core_digest:
        raise ValueError("Completed study configuration/source differs")
    if (
        manifest["query_ids"] != binding["dev_query_ids"]
        or manifest["embedding_files"] != binding["embedding_files"]
    ):
        raise ValueError("Completed study query/cache identity differs")
    if manifest["dataset"] != binding["dataset_manifest"]:
        raise ValueError("Completed dataset identity differs")
    folder = study / f"qrae-{DIMENSION}d-seed{job['seed']}"
    if len(list(study.glob("*/result.json"))) != 1:
        raise ValueError("Each grid job must contain exactly one declared fit")
    result = _verified_run(folder)
    if REQUIRED_ARTIFACTS - {p.name for p in folder.iterdir() if p.is_file()}:
        raise ValueError("Missing required training/evaluation artifact")
    if (
        result["dataset"],
        result["split"],
        result["method"],
        result["dimension"],
        result["seed"],
        result["evaluation_mode"],
    ) != (job["dataset"], "dev", "qrae", DIMENSION, job["seed"], "fitted"):
        raise ValueError("Result is not the designated dev-only QRAE fit")
    if result["study_id"] != manifest["study_id"]:
        raise ValueError("Result study identity differs")
    info = read_json(folder / "compressor.json")
    training = {**job["config"]["neural"], "negative_strategy": "random"}
    if info["training"] != training or info["actual_steps"] != 3000:
        raise ValueError("Actual training settings/budget differ")
    if info["regularization_weight"] != job["lambda"] or info["ranking_weight"] != job["alpha"]:
        raise ValueError("Actual loss coefficients differ")
    if (info["method"], info["dimension"], info["input_dimension"], info["seed"]) != (
        "qrae",
        DIMENSION,
        768,
        job["seed"],
    ):
        raise ValueError("Compressor identity differs")
    if info["actual_device"] != "cuda":
        raise ValueError("Declared GPU fit silently changed device")
    history = info["history"]
    if [row["step"] for row in history] != [1, *range(100, 3001, 100)]:
        raise ValueError("Incomplete training diagnostic history")
    for row in history:
        if PROBE_FIELDS - set(row) or not all(math.isfinite(float(v)) for v in row.values()):
            raise ValueError("Missing or nonfinite training probes")
        if row["latent_norm_min"] < 1e-12:
            raise ValueError("Collapsed latent diagnostic")
        if not math.isclose(
            row["weighted_ranking"],
            job["alpha"] * row["batch_ranking"],
            abs_tol=1e-10,
            rel_tol=1e-10,
        ) or not math.isclose(
            row["weighted_regularizer"],
            job["lambda"] * row["regularizer"],
            abs_tol=1e-10,
            rel_tol=1e-10,
        ):
            raise ValueError("Diagnostic coefficient application differs")
    with (folder / "training.csv").open(encoding="utf-8", newline="") as stream:
        csv_history = list(csv.DictReader(stream))
    if len(csv_history) != len(history) or any(
        {k: float(v) for k, v in row.items()} != expected
        for row, expected in zip(csv_history, history, strict=True)
    ):
        raise ValueError("CSV and checkpoint training diagnostics differ")
    audit = read_json(folder / "supervision.json")
    if (
        audit["query_ids"] != binding["train_query_ids"]
        or audit["negative_strategy"] != "random"
        or audit["seed"] != job["seed"]
    ):
        raise ValueError("Supervision is not bound to the designated training queries")
    per_query = read_json(folder / "per_query.json")
    if sorted(per_query) != binding["dev_query_ids"]:
        raise ValueError("Per-query records include missing or non-dev queries")
    for metric, value in result["metrics"].items():
        values = [float(row[metric]) for row in per_query.values()]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values) or not math.isclose(
            value, math.fsum(values) / len(values), abs_tol=1e-12
        ):
            raise ValueError("Invalid or inconsistent per-query metrics")
    for name in ("compressor.npz", "training_decoder.npz"):
        with np.load(folder / name, allow_pickle=False) as arrays:
            if not all(np.isfinite(arrays[key]).all() for key in arrays.files):
                raise ValueError("Nonfinite checkpoint arrays")
            shapes = {key: arrays[key].shape for key in arrays.files}
            expected_shapes = (
                {"mean": (768,), "encoder": (DIMENSION, 768)}
                if name == "compressor.npz"
                else {"decoder": (768, DIMENSION)}
            )
            if shapes != expected_shapes:
                raise ValueError("Checkpoint architecture differs")
    return {
        "study_path": relative(study),
        "run_path": relative(folder),
        "study_sha256": sha256(study / "study.json"),
        "result_sha256": sha256(folder / "result.json"),
        "artifact_sha256": file_hashes(study),
        "metrics": result["metrics"],
        "actual_regularization_weight": info["regularization_weight"],
        "actual_ranking_weight": info["ranking_weight"],
        "actual_steps": info["actual_steps"],
        "training_diagnostics_finite": True,
        "training_rows": len(history),
    }


def verify_completion(record, plan, job):
    if record["plan_id"] != plan["plan_id"] or record["job"] != {
        k: job[k]
        for k in ("job_id", "dataset", "lambda", "alpha", "seed", "dimension", "method", "split")
    }:
        raise ValueError("Completion record belongs to another plan/job")
    for name, digest in record["logs"].items():
        if sha256(REPO / name) != digest:
            raise ValueError("Retained job log changed")
    if record["status"] == "complete":
        verify_files(REPO / record["study_path"], record["artifact_sha256"], exact=True)
        checked = validate_run(
            REPO / record["study_path"],
            job,
            plan["datasets"][job["dataset"]],
            plan["core_source_sha256"],
        )
        if any(record[k] != value for k, value in checked.items()):
            raise ValueError("Sealed completion contents changed")
    elif record["status"] == "failed":
        verify_files(
            REPO / job["config"]["experiment"]["output"],
            record["partial_artifact_sha256"],
            exact=True,
        )
    else:
        raise ValueError("Unknown completion status")


@contextlib.contextmanager
def output_lock(output):
    """OS advisory lock releases on process death, so interruption does not wedge resume."""
    lock = (output / "runner.lock").open("a+b")
    if lock.seek(0, os.SEEK_END) == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        lock.close()


def run_plan(output, *, max_new_runs=None):
    output = Path(output).resolve()
    plan = validate_plan(output)
    with output_lock(output):
        started = now()
        progress = {
            "plan_id": plan["plan_id"],
            "plan_sha256": sha256(output / "plan.json"),
            "started_utc": started,
            "status": "running",
            "jobs": [],
            "studies": [],
            "expected_fits": EXPECTED_FITS,
            "new_fits": 0,
            "resumed_complete": 0,
            "test_evaluated": False,
        }
        write_json(output / "progress.json", progress)
        for job in plan["jobs"]:
            validate_plan(output, full_inputs=False)
            receipt = output / "completed" / f"{job['job_id']}.json"
            if receipt.exists():
                record = read_json(receipt)
                verify_completion(record, plan, job)
                record = {**record, "reused_this_invocation": True}
                progress["resumed_complete"] += int(record["status"] == "complete")
            elif max_new_runs is not None and progress["new_fits"] >= max_new_runs:
                progress["jobs"].append({"job_id": job["job_id"], "status": "pending"})
                continue
            else:
                record = execute_job(output, plan, job)
                exclusive_json(receipt, record)
                progress["new_fits"] += int(record.get("fit_invoked", False))
                record = {**record, "reused_this_invocation": False}
            progress["jobs"].append(record)
            if record["status"] == "complete":
                progress["studies"].append(record["study_path"])
            write_json(output / "progress.json", progress)
            print(f"{job['job_id']}: {record['status']}", flush=True)
        validate_plan(output)
        progress["finished_utc"] = now()
        statuses = [item["status"] for item in progress["jobs"]]
        progress["status"] = (
            "partial"
            if "pending" in statuses
            else ("complete_with_failures" if "failed" in statuses else "complete")
        )
        progress["end_input_verification"] = {
            "source_protocol_cache_data_hashes_unchanged": True,
            "plan_sha256": sha256(output / "plan.json"),
        }
        write_json(output / "progress.json", progress)
        return progress


def execute_job(output, plan, job):
    from retrieval_lab.data import Dataset, load_dataset
    from retrieval_lab.experiment import run

    binding = plan["datasets"][job["dataset"]]
    root = REPO / job["config"]["experiment"]["output"]
    attempts = output / "attempts" / job["job_id"]
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = attempts / f"{len(list(attempts.iterdir())) + 1:04}"
    attempt.mkdir(exist_ok=False)
    record = {
        "plan_id": plan["plan_id"],
        "job": {
            k: job[k]
            for k in (
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
        "started_utc": now(),
        "fit_invoked": False,
        "start_source_protocol_metadata_verified": True,
    }
    exclusive_json(
        attempt / "launch.json",
        {
            **record,
            "config": job["config"],
            "source_files": plan["source_files"],
            "cache_files": binding["cache_files"],
            "dataset_files": binding["dataset_files"],
            "source_archive_sha256": plan["source_archive_sha256"],
        },
    )
    existing = sorted(root.glob("dev/*/*/result.json"))
    log = attempt / "execution.log"
    try:
        with (
            log.open("w", encoding="utf-8") as stream,
            contextlib.redirect_stdout(stream),
            contextlib.redirect_stderr(stream),
        ):
            if existing:
                if len(existing) != 1:
                    raise ValueError("Unexpected multiple completed studies in one job namespace")
                study = existing[0].parents[1]
                record["recovered_unsealed_complete"] = True
            else:
                if root.exists() and any(root.rglob("*")):
                    shutil.copytree(root, attempt / "interrupted-artifacts")
                    record["restart"] = (
                        "Previous incomplete artifacts copied before restarting from initialization"
                    )
                data = load_dataset(REPO / binding["dataset_directory"], job["config"]["data"])
                view = Dataset(
                    data.name,
                    data.corpus,
                    data.queries,
                    {name: data.qrels[name] for name in ("train", "dev")},
                    data.manifest,
                )
                if sorted(view.qrels["dev"]) != binding["dev_query_ids"]:
                    raise ValueError("Development query split changed")
                record["fit_invoked"] = True
                study = run(job["config"], view, REPO / binding["cache_directory"])
            record.update(validate_run(study, job, binding, plan["core_source_sha256"]))
            record["status"] = "complete"
    except Exception as error:
        with log.open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        record.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "retry_policy": "Retained terminal failure; no automatic score-driven retry",
                "partial_artifact_sha256": file_hashes(root) if root.exists() else {},
            }
        )
    validate_plan(output, full_inputs=False)
    record.update(
        {
            "finished_utc": now(),
            "end_source_protocol_metadata_verified": True,
            "logs": {relative(p): sha256(p) for p in sorted(attempt.rglob("*")) if p.is_file()},
        }
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "run", "verify"))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/supplementary/weight-sensitivity")
    )
    parser.add_argument(
        "--scifact-reference", type=Path, default=Path("runs/course/frozen-scifact.json")
    )
    parser.add_argument("--fiqa-reference", type=Path, default=Path("runs/course/frozen-fiqa.json"))
    parser.add_argument(
        "--max-new-runs", type=int, help="Bound this invocation; remaining cells stay pending"
    )
    args = parser.parse_args()
    if Path.cwd().resolve() != REPO:
        parser.error("Run from the repository root")
    if args.max_new_runs is not None and args.max_new_runs < 1:
        parser.error("--max-new-runs must be positive")
    if args.stage == "plan":
        plan = create_plan(
            args.output, {"scifact": args.scifact_reference, "fiqa": args.fiqa_reference}
        )
        print(
            f"Fixed plan {plan['plan_id']}: {len(plan['jobs'])} dev-only fits; no training launched"
        )
    elif args.stage == "run":
        progress = run_plan(args.output, max_new_runs=args.max_new_runs)
        print(
            f"{progress['status']}: {len(progress['studies'])}/{EXPECTED_FITS} successful studies"
        )
    else:
        plan = validate_plan(args.output)
        checked = 0
        for job in plan["jobs"]:
            receipt = args.output / "completed" / f"{job['job_id']}.json"
            if receipt.exists():
                verify_completion(read_json(receipt), plan, job)
                checked += 1
        print(f"Plan/input hashes valid; {checked} sealed success/failure records verified")


if __name__ == "__main__":
    main()
