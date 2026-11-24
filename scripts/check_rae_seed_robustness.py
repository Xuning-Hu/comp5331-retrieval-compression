"""Bounded, post-hoc RAE training-seed check on the existing IMDb geometry split.

The saved train/held-out row order and train-probe positions are immutable. Only
initialization/minibatch seeds change. Four existing 2026 fits are reused and
their metrics recomputed as anchors; exactly eight new CPU fits are requested.
This is not preregistration, an untouched test, or published-number replication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import runpy
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VECTOR_SHA = "48edd54702b747d879788c465c7fa1ce39b1c028bfeab0a782ea14f35add60f1"
SPLIT_SHA = "7731cbba1456c18feda61d7515e3e0731efc57e59ec70ffe5878e3540d98bc88"
HELPER_SHA = "327838dd1bf77568c3bb320c32a7cc3478d73cdf29bf106dbf00a1fe4363df41"
SETTINGS = {
    "dimension": 256,
    "steps": 3000,
    "batch_size": 128,
    "lr": 0.001,
    "min_lr": 0.00001,
    "device": "cpu",
    "threads": 2,
}
SEEDS = [17, 42, 2026]
COEFFICIENTS = [0.0, 3e-7]
MODES = ["formula", "released"]
KS = [1, 5, 10, 20, 50]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def csv_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def checked(path, expected):
    if not Path(path).is_file() or sha(path) != expected:
        raise ValueError(f"Pinned robustness input differs: {path}")


def load_fixed_split(path, row_count, expected_sha):
    checked(path, expected_sha)
    saved = read(path)
    if set(saved) != {"train_rows", "heldout_rows", "train_probe_positions"}:
        raise ValueError("Unexpected saved split fields")
    for key, values in saved.items():
        if (
            not values
            or any(type(value) is not int for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"Invalid or duplicate saved split identifiers: {key}")
    train, heldout, anchors = (
        np.array(saved[key], dtype=np.int64)
        for key in ("train_rows", "heldout_rows", "train_probe_positions")
    )
    if set(train) & set(heldout) or sorted([*train, *heldout]) != list(range(row_count)):
        raise ValueError("Saved split does not give disjoint complete vector coverage")
    if np.any(anchors < 0) or np.any(anchors >= len(train)):
        raise ValueError("Saved train-probe position is out of range")
    return train, heldout, anchors


def job_plan():
    # Check reused anchors before performing any new training.
    return [
        {
            "mode": mode,
            "training_seed": seed,
            "nominal_adam_decay": coefficient,
            "reused": seed == 2026,
            "name": f"{mode}-d256-s{seed}-w{coefficient:g}",
        }
        for seed in [2026, 17, 42]
        for mode in MODES
        for coefficient in COEFFICIENTS
    ]


def fit_new_job(helper, base, job):
    if job["reused"] or job["training_seed"] not in {17, 42}:
        raise ValueError("Only the eight declared new fits may call training")
    if job["mode"] not in MODES or job["nominal_adam_decay"] not in COEFFICIENTS:
        raise ValueError("Undeclared training configuration")
    return helper["train_rae"](
        base,
        256,
        job["training_seed"],
        job["mode"],
        job["nominal_adam_decay"],
        3000,
        128,
        0.001,
        0.00001,
        "cpu",
        2,
    )


def evaluate_with_queries(helper, arrays, base, heldout, anchors, references, ks):
    """Same preprocessing/search helpers as the original independent experiment."""
    aggregate, individual, neighbors = {}, {}, {}
    for metric in ("cosine", "euclidean"):
        base_input, heldout_input = (
            (helper["normalize"](base), helper["normalize"](heldout))
            if metric == "cosine"
            else (base, heldout)
        )
        projected_base = helper["project"](base_input, arrays)
        projected_queries = helper["project"](heldout_input, arrays)
        aggregate[metric], individual[metric] = {}, {}
        for label, queries, self_positions in (
            ("heldout", projected_queries, None),
            ("train_probe", projected_base[anchors], anchors),
        ):
            reduced = helper["nearest_neighbors"](
                projected_base, queries, max(ks), metric, self_positions
            )
            scores = helper["overlaps"](references[metric][label], reduced, ks)
            aggregate[metric].update({f"{label}_{key}": value for key, value in scores.items()})
            individual[metric][label] = [
                len(set(a[:5]) & set(b[:5])) / 5
                for a, b in zip(references[metric][label], reduced, strict=True)
            ]
            if not math.isclose(
                np.mean(individual[metric][label]), scores["nn_overlap@5"], abs_tol=1e-12, rel_tol=0
            ):
                raise ValueError("Per-query and aggregate overlap disagree")
            neighbors[f"{metric}_{label}"] = reduced
    return aggregate, individual, neighbors


def source_inputs(root, vectors, helper_path):
    coarse, weak = (
        root / "runs/rae-geometry-imdb-reencoded-hf",
        root / "runs/rae-geometry-imdb-figure1-check",
    )
    protocols = {"coarse": read(coarse / "protocol.json"), "weak": read(weak / "protocol.json")}
    checked(vectors, VECTOR_SHA)
    checked(helper_path, HELPER_SHA)
    inputs = {str(vectors.resolve()): VECTOR_SHA, str(helper_path.resolve()): HELPER_SHA}
    for directory, protocol in ((coarse, protocols["coarse"]), (weak, protocols["weak"])):
        expected = {
            "source_vectors_sha256": VECTOR_SHA,
            "source_shape": [20000, 768],
            "script_sha256": HELPER_SHA,
            "dimensions": [256],
            "seeds": [2026],
            "steps": 3000,
            "batch_size": 128,
            "lr": 0.001,
            "min_lr": 0.00001,
            "device": "cpu",
            "threads": 2,
            "train_fraction": 0.9,
            "train_probes": 256,
            "ks": KS,
            "normalize_training": False,
        }
        if any(protocol.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing source protocol differs from the fixed robustness settings")
        checked(directory / "split-2026.json", SPLIT_SHA)
        checked(directory / "experiment_script.py", HELPER_SHA)
        for name in ("protocol.json", "results.json", "split-2026.json", "experiment_script.py"):
            inputs[str((directory / name).resolve())] = sha(directory / name)
    anchors = {}
    for job in job_plan():
        if not job["reused"]:
            continue
        directory = coarse if job["nominal_adam_decay"] == 0 else weak
        folder = directory / job["name"]
        info = read(folder / "training.json")
        expected = {
            "seed": 2026,
            "mode": job["mode"],
            "dimension": 256,
            "nominal_adam_decay": job["nominal_adam_decay"],
            "requested_steps": 3000,
            "actual_steps": 3000 if job["mode"] == "formula" else 2961,
            "batch_size": 128,
            "lr": 0.001,
            "min_lr": 0.00001,
            "centering": "none",
        }
        if any(info.get(key) != value for key, value in expected.items()):
            raise ValueError("Reused fit metadata differs from its declared identity")
        rows = [
            row
            for row in read(directory / "results.json")
            if row["method"] == job["mode"]
            and row["seed"] == 2026
            and row["dimension"] == 256
            and row["nominal_adam_decay"] == job["nominal_adam_decay"]
        ]
        if {row["metric"] for row in rows} != {"cosine", "euclidean"} or len(rows) != 2:
            raise ValueError("Reused anchor metrics are incomplete or duplicated")
        for filename in ("model.npz", "training.json"):
            inputs[str((folder / filename).resolve())] = sha(folder / filename)
        anchors[job["name"]] = {"folder": str(folder.resolve()), "rows": rows}
    pca = [row for row in read(coarse / "results.json") if row["method"] == "pca"]
    if pca != [row for row in read(weak / "results.json") if row["method"] == "pca"]:
        raise ValueError("Existing PCA reference records disagree")
    return coarse / "split-2026.json", inputs, anchors, pca


def summarize(output):
    protocol = read(output / "protocol.json")
    records = [read(output / job["name"] / "result.json") for job in protocol["jobs"]]
    rows, paired, query_changes, histories = [], [], [], []
    for record in records:
        folder = output / record["run"]
        for filename, checksum in record["files"].items():
            checked(folder / filename, checksum)
        info = read(folder / "training.json")
        for row in info["history"]:
            histories.append(
                {
                    "run": record["run"],
                    "reused": record["reused"],
                    "mode": record["mode"],
                    "training_seed": record["training_seed"],
                    "nominal_adam_decay": record["nominal_adam_decay"],
                    **row,
                }
            )
        for metric, values in record["metrics"].items():
            rows.append(
                {
                    "run": record["run"],
                    "mode": record["mode"],
                    "training_seed": record["training_seed"],
                    "split_seed": 2026,
                    "nominal_adam_decay": record["nominal_adam_decay"],
                    "reused": record["reused"],
                    "metric": metric,
                    "actual_steps": info["actual_steps"],
                    "condition_number_nonzero_spectrum": info["condition_number_nonzero_spectrum"],
                    **values,
                }
            )
    lookup = {
        (record["mode"], record["training_seed"], record["nominal_adam_decay"]): record
        for record in records
    }
    for mode in MODES:
        for seed in SEEDS:
            baseline, weak = lookup[mode, seed, 0.0], lookup[mode, seed, 3e-7]
            per_base, per_weak = (
                read(output / baseline["run"] / "per-query.json"),
                read(output / weak["run"] / "per-query.json"),
            )
            if (
                per_base["heldout_source_rows"] != per_weak["heldout_source_rows"]
                or per_base["train_probe_source_rows"] != per_weak["train_probe_source_rows"]
            ):
                raise ValueError("Paired query identities differ")
            for metric in ("cosine", "euclidean"):
                changes = np.array(per_weak["overlap@5"][metric]["heldout"]) - np.array(
                    per_base["overlap@5"][metric]["heldout"]
                )
                delta = (
                    weak["metrics"][metric]["heldout_nn_overlap@5"]
                    - baseline["metrics"][metric]["heldout_nn_overlap@5"]
                )
                if not math.isclose(float(changes.mean()), delta, abs_tol=1e-12, rel_tol=0):
                    raise ValueError(
                        "Paired query changes do not aggregate to the score difference"
                    )
                row = {
                    "mode": mode,
                    "metric": metric,
                    "training_seed": seed,
                    "split_seed": 2026,
                    "reused_pair": seed == 2026,
                    "w0_overlap@5": baseline["metrics"][metric]["heldout_nn_overlap@5"],
                    "w3e-7_overlap@5": weak["metrics"][metric]["heldout_nn_overlap@5"],
                    "delta": delta,
                    "delta_percentage_points": delta * 100,
                    "improved_queries": int((changes > 0).sum()),
                    "regressed_queries": int((changes < 0).sum()),
                    "tied_queries": int((changes == 0).sum()),
                }
                paired.append(row)
                query_changes.append(
                    {
                        **row,
                        "heldout_source_rows": per_base["heldout_source_rows"],
                        "per_query_delta": changes.tolist(),
                    }
                )
    summaries = []
    for mode in MODES:
        for metric in ("cosine", "euclidean"):
            pairs = [row for row in paired if row["mode"] == mode and row["metric"] == metric]
            summary = {
                "mode": mode,
                "metric": metric,
                "training_seeds": SEEDS,
                "split_seed": 2026,
                "n_training_seeds": 3,
                "new_pairs": 2,
                "reused_pairs": 1,
            }
            for column in ("w0_overlap@5", "w3e-7_overlap@5", "delta", "delta_percentage_points"):
                values = [row[column] for row in pairs]
                summary[column + "_mean"] = float(np.mean(values))
                summary[column + "_sample_sd"] = float(np.std(values, ddof=1))
            summaries.append(summary)
    for name, value in (
        ("results", rows),
        ("paired-seed-differences", paired),
        ("summary", summaries),
    ):
        write(output / (name + ".json"), value)
        csv_rows(output / (name + ".csv"), value)
    write(output / "paired-query-differences.json", query_changes)
    csv_rows(output / "training.csv", histories)
    return summaries


def export_small(output, destination):
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Retain the previous robustness export; choose an empty destination")
    destination.mkdir(parents=True, exist_ok=True)
    copied = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".csv", ".py", ".md"}:
            relative = path.relative_to(output)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied[relative.as_posix()] = {"source_sha256": sha(path), "export_sha256": sha(target)}
    write(
        destination / "export-manifest.json",
        {
            "source": str(output.resolve()),
            "files": copied,
            "excluded": "raw vectors, model/checkpoint arrays and saved neighbor arrays",
            "scope": "Saved per-query overlap reaggregation and paired training-seed comparisons; geometry replay requires the local vectors and models. Exactly 8 new fits and 4 reused 2026 fits; no model/seed selection.",
        },
    )


def execute(root, output, destination):
    root, output, destination = root.resolve(), output.resolve(), destination.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Robustness output must be empty; original experiments are immutable")
    if destination == output or destination.is_relative_to(output):
        raise ValueError("Compact export must be outside the raw output")
    helper_path = Path(__file__).with_name("reproduce_rae.py").resolve()
    vectors_path = root / "cache/imdb-rae-reencoded-hf/raw_embeddings.npy"
    split_path, inputs, reused, pca = source_inputs(root, vectors_path, helper_path)
    vectors = np.load(vectors_path, allow_pickle=False)
    if (
        vectors.shape != (20000, 768)
        or vectors.dtype != np.float32
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Pinned vectors have invalid shape/type/values")
    train_rows, heldout_rows, anchors = load_fixed_split(split_path, len(vectors), SPLIT_SHA)
    if (len(train_rows), len(heldout_rows), len(anchors)) != (18000, 2000, 256):
        raise ValueError("Pinned split sizes differ")
    import torch
    from threadpoolctl import threadpool_limits

    helper = runpy.run_path(str(helper_path))
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "source-snapshot"
    source_dir.mkdir()
    for path in (Path(__file__), helper_path):
        shutil.copy2(path, source_dir / path.name)
    shutil.copy2(split_path, output / "fixed-split.json")
    protocol = {
        "schema_version": 1,
        "recorded_before_new_fits_utc": timestamp(),
        "scope": "Post-hoc fixed-split training-seed robustness check; NOT preregistered, NOT an untouched test, NOT exact published-number replication",
        "split_seed": 2026,
        "training_seeds": SEEDS,
        "nominal_adam_decay_values": COEFFICIENTS,
        "modes": MODES,
        "settings": SETTINGS,
        "ks": KS,
        "source_vectors_sha256": VECTOR_SHA,
        "source_split_sha256": SPLIT_SHA,
        "source_shape": [20000, 768],
        "train_rows": 18000,
        "heldout_rows": 2000,
        "train_probe_rows": 256,
        "fixed_split_rule": "Load existing ordered train_rows/heldout_rows/probe_positions byte-for-byte; never invoke split_vectors with a training seed",
        "candidate_order": "existing train_rows order; exact ties use ascending training row position; train probes exclude their exact base row",
        "training_input": "raw uncentered float32 MPNet vectors; normalize_training=false",
        "cosine_evaluation": "normalize original inputs before encoder; normalize projected vectors for cosine exact search, as in the original independent helper",
        "lambda_scale": "nominal coupled-Adam w; formula explicit coefficient768*w/2; released uses coupled Adam decay and biases",
        "budgets": {
            "formula_actual_updates": 3000,
            "released_actual_updates": 2961,
            "released_complete_epochs": 21,
        },
        "new_fit_count": 8,
        "reused_fit_count": 4,
        "reused_2026": reused,
        "pca_reference_scope": "Existing matching PCA records copied as context; PCA is not refitted in this supplement",
        "jobs": job_plan(),
        "source_files_sha256": inputs,
        "source_snapshots_sha256": {path.name: sha(path) for path in source_dir.iterdir()},
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "analysis_plan": "Retain all seeds and both w values; report per-seed paired overlap@5 changes and seed mean/sample SD; no additional lambda search or best-seed selection; all other k values retained",
    }
    write(output / "protocol.json", protocol)
    write(
        output / "pca-reference.json",
        {
            "reused": True,
            "rows": pca,
            "source": str((root / "runs/rae-geometry-imdb-reencoded-hf/results.json").resolve()),
        },
    )
    print(f"Protocol frozen before new fits: {sha(output / 'protocol.json')}", flush=True)
    started = time.perf_counter()
    progress = {
        "status": "running",
        "started_utc": timestamp(),
        "jobs": [],
        "new_fits_completed": 0,
        "reused_anchors_completed": 0,
    }
    write(output / "progress.json", progress)
    with threadpool_limits(limits=2):
        base, heldout = vectors[train_rows], vectors[heldout_rows]
        references = {}
        print("Computing fixed original-neighbor references", flush=True)
        for metric in ("cosine", "euclidean"):
            references[metric] = {
                "heldout": helper["nearest_neighbors"](base, heldout, 50, metric),
                "train_probe": helper["nearest_neighbors"](
                    base, base[anchors], 50, metric, anchors
                ),
            }
        np.savez(
            output / "original-neighbors.npz",
            **{
                f"{metric}_{label}": array
                for metric, values in references.items()
                for label, array in values.items()
            },
        )
        for job in protocol["jobs"]:
            for filename, expected in protocol["source_snapshots_sha256"].items():
                checked(source_dir / filename, expected)
            checked(helper_path, HELPER_SHA)
            folder = output / job["name"]
            folder.mkdir()
            print(f"{'Reusing' if job['reused'] else 'Training'} {job['name']}", flush=True)
            entry = {**job, "started_utc": timestamp(), "status": "running"}
            progress["jobs"].append(entry)
            write(output / "progress.json", progress)
            if job["reused"]:
                origin = Path(reused[job["name"]]["folder"])
                for filename in ("model.npz", "training.json"):
                    checked(origin / filename, inputs[str((origin / filename).resolve())])
                    shutil.copy2(origin / filename, folder / filename)
                with np.load(folder / "model.npz", allow_pickle=False) as archive:
                    arrays = {key: archive[key].copy() for key in archive.files}
                info = read(folder / "training.json")
            else:
                arrays, info = fit_new_job(helper, base, job)
                np.savez(folder / "model.npz", **arrays)
                write(folder / "training.json", info)
                origin = None
            singular = np.linalg.svd(arrays["encoder"].astype(np.float64), compute_uv=False)
            if (
                not np.isfinite(singular).all()
                or not math.isclose(
                    float(singular[0]), info["singular_max"], rel_tol=1e-10, abs_tol=1e-12
                )
                or not math.isclose(
                    float(singular[-1]), info["singular_min"], rel_tol=1e-10, abs_tol=1e-12
                )
            ):
                raise ValueError("Stored training spectrum disagrees with model weights")
            write(
                folder / "spectrum.json",
                {
                    "encoder_singular_values": singular.tolist(),
                    "condition_number_nonzero_spectrum": float(singular[0] / singular[-1]),
                    "scope": "Nonzero row-space encoder spectrum; no full-space distance guarantee",
                },
            )
            csv_rows(folder / "training.csv", info["history"])
            evaluation_started = time.perf_counter()
            metrics, individual, neighbors = evaluate_with_queries(
                helper, arrays, base, heldout, anchors, references, KS
            )
            anchor_error = None
            if job["reused"]:
                errors = [
                    abs(metrics[row["metric"]][key] - row[key])
                    for row in reused[job["name"]]["rows"]
                    for key in metrics[row["metric"]]
                ]
                anchor_error = max(errors)
                if anchor_error > 1e-12:
                    raise ValueError(
                        f"Reused 2026 geometry anchor changed: maximum error {anchor_error}"
                    )
            write(
                folder / "per-query.json",
                {
                    "heldout_source_rows": heldout_rows.tolist(),
                    "train_probe_source_rows": train_rows[anchors].tolist(),
                    "split_sha256": SPLIT_SHA,
                    "overlap@5": individual,
                },
            )
            np.savez(folder / "neighbors.npz", **neighbors)
            record = {
                "run": job["name"],
                **job,
                "split_seed": 2026,
                "dimension": 256,
                "fit_performed_now": not job["reused"],
                "reused_source": str(origin) if origin else None,
                "training_fit_seconds_scope": "Historical source fit"
                if origin
                else "New CPU fit with 2 threads",
                "metrics": metrics,
                "anchor_maximum_error": anchor_error,
                "evaluation_seconds": time.perf_counter() - evaluation_started,
                "files": {path.name: sha(path) for path in folder.iterdir() if path.is_file()},
            }
            write(folder / "result.json", record)
            progress["reused_anchors_completed" if job["reused"] else "new_fits_completed"] += 1
            entry.update(
                {
                    "status": "complete",
                    "finished_utc": timestamp(),
                    "result_sha256": sha(folder / "result.json"),
                }
            )
            progress["elapsed_seconds"] = time.perf_counter() - started
            write(output / "progress.json", progress)
            print(
                f"Completed {job['name']}: cosine@5={metrics['cosine']['heldout_nn_overlap@5']:.5f}, euclidean@5={metrics['euclidean']['heldout_nn_overlap@5']:.5f}; elapsed={progress['elapsed_seconds']:.1f}s",
                flush=True,
            )
    if (progress["new_fits_completed"], progress["reused_anchors_completed"]) != (8, 4):
        raise ValueError("Unexpected number of new/reused fits")
    summaries = summarize(output)
    progress.update(
        {
            "status": "complete",
            "finished_utc": timestamp(),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    write(output / "progress.json", progress)
    text = [
        "# Post-hoc fixed-split RAE training-seed check",
        "",
        "This supplement varies initialization and minibatch order while reusing the exact existing 18000/2000 split and 256 train-probe positions. It is exploratory: the geometry held-out set was already inspected, the weak point was source-driven, and unavailable author vectors prevent published-number matching.",
        "",
        "Exactly 8 new CPU fits with 2 threads use seeds 17/42. Four seed 2026 models and original training records are copied byte-identically, clearly marked reused; their geometry metrics are recomputed and checked against their earlier values. Formula uses 3000 updates; released-style uses 2961 updates (21 complete epochs). No extra coefficient or best-seed search is performed.",
        "",
        "All scores below are held-out NNOverlap@5. SD is the sample SD over three training seeds, conditional on one fixed data split; it is not a confidence interval or split-uncertainty estimate.",
        "",
        "| Track | Metric | w0 mean ± SD | w3e-7 mean ± SD | Paired change, percentage points ± SD |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summaries:
        text.append(
            f"| {row['mode']} | {row['metric']} | {row['w0_overlap@5_mean']:.5f} ± {row['w0_overlap@5_sample_sd']:.5f} | {row['w3e-7_overlap@5_mean']:.5f} ± {row['w3e-7_overlap@5_sample_sd']:.5f} | {row['delta_percentage_points_mean']:+.3f} ± {row['delta_percentage_points_sample_sd']:.3f} |"
        )
    text += [
        "",
        "The compact export retains ordered query-row IDs, per-query overlap, every seed difference, training histories, full spectra and source snapshots. Vectors, fitted arrays and neighbor arrays remain only in the raw local run; hash references to them do not mean they are distributed.",
    ]
    (output / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    write(
        output / "artifact-manifest.json",
        {
            "protocol_sha256": sha(output / "protocol.json"),
            "files": {
                path.relative_to(output).as_posix(): sha(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
            "new_fits": 8,
            "reused_fits": 4,
            "geometry_anchor_maximum_error": max(
                read(output / job["name"] / "result.json")["anchor_maximum_error"]
                for job in protocol["jobs"]
                if job["reused"]
            ),
        },
    )
    export_small(output, destination)
    print(
        f"Completed fixed-split robustness; export={destination}; elapsed={progress['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/rae-geometry-imdb-fixed-split-seeds")
    )
    parser.add_argument(
        "--export", type=Path, default=Path("docs/results/course/rae-fixed-split-seeds")
    )
    args = parser.parse_args()
    execute(Path(__file__).resolve().parents[1], args.output, args.export)
