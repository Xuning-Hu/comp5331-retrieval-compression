"""Read completed studies into traceable course-report tables and diagnostics.

This script never fits models, selects checkpoints, or chooses test configurations.
Each study/configuration remains a separate series. Raw studies and exported public
snapshots are supported; the manifest states the resulting verification boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from retrieval_lab.common import fingerprint, read_json, sha256, write_json

COMPARISONS = (
    ("qrae", "rae"),
    ("qrae", "ae"),
    ("qrae", "pca"),
    ("rae", "ae"),
    ("qrae", "rank_only"),
    ("ae", "pca"),
)
NOTES = [
    "All values are measured inputs; no fitting, checkpoint selection or test tuning is performed.",
    "Seeds are averaged only within the same study, method, dimension and effective parameters.",
    "Seed SD is not a confidence interval; deterministic fits with SD=0 are not uncertainty-free.",
    "Bootstrap resamples shared query columns and training-seed rows, not seed-query cells.",
    "Intervals are exploratory percentile intervals, without multiple-comparison adjustment; "
    "few seeds only coarsely describe training uncertainty. A CI crossing zero does not prove equality.",
    "Vector payload is not process memory. Deployment bytes here equal index plus compressor NPZ; "
    "they exclude the common embedding model, metadata, corpus text and runtime overhead.",
    "Training decoder bytes are training-only. Search timing excludes embedding and projection; "
    "p50 single-query latency and batch QPS are separate measurements, not end-to-end speedup.",
    "Loss/probe curves are training diagnostics, not held-out relevance validation. Copied frozen "
    "training histories describe source fitting, not additional test training.",
    "Improvement/regression cases use a symmetric numeric rule on seed-mean query deltas; "
    "they identify examples, not causal explanations.",
    "This analysis verifies saved-score aggregation and checksums, not embedding generation, "
    "rankings against original qrels, or training reproducibility.",
]


def seed_query_bootstrap(
    candidate, reference, candidate_seeds, reference_seeds, samples, seed=2026
):
    """Crossed resampling; matched seeds stay paired and all rows share query draws."""
    a, b = np.asarray(candidate, dtype=float), np.asarray(reference, dtype=float)
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != b.shape[1]
        or min(*a.shape, *b.shape) < 1
        or len(candidate_seeds) != a.shape[0]
        or len(reference_seeds) != b.shape[0]
        or len(set(candidate_seeds)) != a.shape[0]
        or len(set(reference_seeds)) != b.shape[0]
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or samples < 1
    ):
        raise ValueError("Bootstrap requires finite seed-by-query matrices and unique seed IDs")
    a = a[np.argsort(candidate_seeds)]
    b = b[np.argsort(reference_seeds)]
    paired_seeds = sorted(candidate_seeds) == sorted(reference_seeds)
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for offset in range(0, samples, 128):
        count = min(128, samples - offset)
        rows_a = rng.integers(len(a), size=(count, len(a), 1))
        rows_b = rows_a if paired_seeds else rng.integers(len(b), size=(count, len(b), 1))
        columns = rng.integers(a.shape[1], size=(count, 1, a.shape[1]))
        means[offset : offset + count] = a[rows_a, columns].mean(axis=(1, 2)) - b[
            rows_b, columns
        ].mean(axis=(1, 2))
    return {
        "delta": float(a.mean() - b.mean()),
        "seed_query_bootstrap_95ci": np.quantile(means, [0.025, 0.975]).tolist(),
        "candidate_seed_count": len(a),
        "reference_seed_count": len(b),
        "query_count": a.shape[1],
        "seed_resampling": "paired matching seed IDs" if paired_seeds else "independent seed rows",
        "shared_query_resampling": True,
        "samples": samples,
        "bootstrap_seed": seed,
        "interpretation": "Exploratory; observed seeds only; no multiplicity correction",
    }


def _verify_file(path, expected, *, exported_text=False):
    if not path.is_file():
        raise ValueError(f"Missing analysis input: {path}")
    if sha256(path) == expected:
        return
    # Public Git text snapshots may be checked out with CRLF on Windows.
    if exported_text and path.suffix in {".json", ".csv", ".md"}:
        normalized = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if normalized == expected:
            return
    raise ValueError(f"Analysis input checksum differs: {path}")


def _parameters(method, info, config):
    """Compressor metadata takes precedence over requested configuration."""
    section = "neural" if method in {"rae", "qrae", "rank_only"} else "ae"
    training = (
        info.get("training", config.get(section, {}))
        if method not in {"original", "pca", "rp"}
        else {}
    )
    result = {"training": training}
    for key in ("architecture", "centering", "definition", "regularizer", "ranking_loss"):
        if key in info:
            result[key] = info[key]
    if method in {"rae", "qrae", "rank_only"}:
        result["regularization_weight"] = info.get(
            "regularization_weight",
            0.0 if method == "rank_only" else training.get("regularization_weight"),
        )
        result["ranking_weight"] = info.get(
            "ranking_weight",
            training.get("ranking_weight") if method in {"qrae", "rank_only"} else 0.0,
        )
    if method in {"qrae", "rank_only"}:
        result["negative_strategy"] = training.get(
            "negative_strategy", config.get("supervision", {}).get("negative_strategy", "unknown")
        )
        result["supervision_config"] = config.get("supervision", {})
    return result


def _label(method, study_key, parameter_id, params):
    parts = [f"{method} [{study_key[:6]}/{parameter_id[:6]}]"]
    for key, short in (
        ("negative_strategy", "neg"),
        ("regularization_weight", "reg"),
        ("ranking_weight", "rank"),
    ):
        if key in params:
            parts.append(f"{short}={params[key]}")
    return " ".join(parts)


def _exported_training_history(path, records, infos):
    """Restore sparse merged CSV rows without mixing runs or inventing parameters."""
    identities = {"run_directory", "study_id", "dataset", "split", "method", "dimension", "seed"}
    histories = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:  # A deterministic-only study has no training rows.
            if any(info.get("history") for info in infos.values()):
                raise ValueError("Exported training CSV omits recorded compressor histories")
            return
        if not identities <= set(reader.fieldnames):
            raise ValueError("Exported training CSV lacks run identity columns")
        for row in reader:
            name = row["run_directory"]
            if name not in records:
                raise ValueError("Exported training CSV contains an unknown run")
            result = records[name]
            for key in identities - {"run_directory"}:
                value = int(row[key]) if key in {"dimension", "seed"} else row[key]
                if value != result[key]:
                    raise ValueError("Exported training CSV disagrees with run identity")
            numeric = {}
            for key, value in row.items():
                if key in identities or value == "":
                    continue
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("Non-finite exported training diagnostic")
                numeric[key] = number
            step = numeric.get("step")
            if step is None or step < 1 or not step.is_integer():
                raise ValueError("Invalid exported training step")
            numeric["step"] = int(step)
            if histories[name] and histories[name][-1]["step"] >= numeric["step"]:
                raise ValueError("Exported training steps are duplicated or unordered")
            histories[name].append(numeric)
    for name, rows in histories.items():
        saved = infos[name].get("history")
        if saved and saved != rows:
            raise ValueError("Exported training CSV disagrees with compressor history")
        infos[name]["history"] = rows
    if any(info.get("history") and name not in histories for name, info in infos.items()):
        raise ValueError("Exported training CSV omits recorded compressor histories")


def load_study(root: Path, metric="ndcg@10", allow_incomplete=False):
    root = root.resolve()
    study = read_json(root / "study.json")
    manifest_sha = sha256(root / "study.json")
    study_key = fingerprint({"study_id": study["study_id"], "manifest_sha256": manifest_sha})[:12]
    config = study["config"]
    raw = sorted(root.glob("*/result.json"))
    records, per_query, infos = {}, {}, {}
    checked = {}
    if raw:
        for path in raw:
            name = path.parent.name
            result = read_json(path)
            checksums = result.get("files", {})
            if {"per_query.json", "compressor.json"} - set(checksums):
                raise ValueError(f"Completed run lacks score/model metadata checksums: {path}")
            for filename, expected in checksums.items():
                artifact = (path.parent / filename).resolve()
                if not artifact.is_relative_to(path.parent.resolve()):
                    raise ValueError(f"Registered artifact escapes its run directory: {filename}")
                _verify_file(artifact, expected)
                checked[f"{name}/{filename}"] = expected
            records[name] = result
            per_query[name] = read_json(path.parent / "per_query.json")
            infos[name] = read_json(path.parent / "compressor.json")
            checked[f"{name}/result.json"] = sha256(path)
        boundary = (
            "raw run result and every registered artifact checksum checked; "
            "saved-score aggregation checked; no index/model replay or independent freeze validation"
        )
    else:
        records = read_json(root / "run-results.json")
        per_query = read_json(root / "per-query.json")
        snapshot = read_json(root / "snapshot.json")
        for filename in ("study.json", "run-results.json", "per-query.json"):
            expected = snapshot["files"][filename]
            _verify_file(root / filename, expected, exported_text=True)
            checked[filename] = expected
        metadata_name = "compressor-metadata.json"
        if metadata_name in snapshot["files"]:
            expected = snapshot["files"][metadata_name]
            _verify_file(root / metadata_name, expected, exported_text=True)
            checked[metadata_name] = expected
            infos = read_json(root / metadata_name)
            if set(infos) != set(records):
                raise ValueError("Exported compressor metadata run keys disagree with scores")
            for name, info in infos.items():
                if any(
                    info.get(key) != records[name][key] for key in ("method", "dimension", "seed")
                ):
                    raise ValueError("Exported compressor metadata disagrees with run identity")
            history_note = "histories read from compressor metadata"
            if "training.csv" in snapshot["files"]:
                expected = snapshot["files"]["training.csv"]
                _verify_file(root / "training.csv", expected, exported_text=True)
                checked["training.csv"] = expected
                _exported_training_history(root / "training.csv", records, infos)
                history_note = "histories checked/recovered against merged training CSV"
            boundary = (
                "public exported score snapshot; effective parameters from hash-verified "
                f"compressor metadata; {history_note}; model/index/ranking replay unavailable"
            )
        else:
            boundary = "public exported score snapshot; effective parameters from study config fallback, histories unavailable"
    if not records or set(records) != set(per_query):
        raise ValueError(f"Empty or inconsistent run/per-query inputs: {root}")
    first = next(iter(records.values()))
    dataset, split = first["dataset"], first["split"]
    qids = sorted(study["query_ids"])
    if split != config["experiment"]["split"] or not qids or len(set(qids)) != len(qids):
        raise ValueError("Study split/query identity is invalid")
    expected_jobs = set()
    for method in config["experiment"]["methods"]:
        if method == "original":
            original = [r for r in records.values() if r["method"] == "original"]
            if len(original) != 1:
                raise ValueError(
                    "A requested original baseline must have exactly one completed run"
                )
            dims = [original[0]["dimension"]]
        else:
            dims = config["experiment"]["dimensions"]
        seeds = [0] if method in {"original", "pca"} else config["experiment"]["seeds"]
        expected_jobs.update((method, dim, seed) for dim in dims for seed in seeds)
    actual_jobs = {(r["method"], r["dimension"], r["seed"]) for r in records.values()}
    if not allow_incomplete and actual_jobs != expected_jobs:
        raise ValueError(f"Study is incomplete or has unexpected jobs: {root}")
    groups = {}
    for name, result in records.items():
        if (result["dataset"], result["split"], result["study_id"]) != (
            dataset,
            split,
            study["study_id"],
        ):
            raise ValueError(f"Run identity disagrees with study: {name}")
        if sorted(per_query[name]) != qids or len(qids) != result["queries"]:
            raise ValueError(f"Run query IDs disagree with study: {name}")
        for key, reported in result["metrics"].items():
            values = np.array([per_query[name][qid][key] for qid in qids], dtype=float)
            if not np.isfinite(values).all() or not np.isclose(
                values.mean(), reported, atol=1e-12, rtol=0
            ):
                raise ValueError(
                    f"Saved metric does not aggregate from per-query values: {name}/{key}"
                )
        if metric not in result["metrics"]:
            raise ValueError(f"Requested metric missing: {name}/{metric}")
        info = infos.get(name, {})
        params = _parameters(result["method"], info, config)
        parameter_id = fingerprint(params)[:12]
        key = (result["method"], result["dimension"], parameter_id)
        if key not in groups:
            groups[key] = {
                "group_id": fingerprint([study_key, *key])[:16],
                "study_key": study_key,
                "study_id": study["study_id"],
                "config_id": fingerprint(config)[:12],
                "parameter_id": parameter_id,
                "parameters": params,
                "parameter_source": "compressor metadata"
                if info
                else "study configuration fallback",
                "protocol": study.get("protocol", "unspecified"),
                "evaluation_mode": study.get("evaluation_mode", "fitted"),
                "dataset": dataset,
                "split": split,
                "method": result["method"],
                "dimension": result["dimension"],
                "label": _label(result["method"], study_key, parameter_id, params),
                "qids": qids,
                "runs": [],
            }
        groups[key]["runs"].append(
            {"name": name, "result": result, "scores": per_query[name], "info": info}
        )
    identity = {
        "dataset": dataset,
        "split": split,
        "dataset_manifest": study["dataset"],
        "embedding_files": study["embedding_files"],
        "embedding_identity": study.get("embedding_identity", config.get("embedding", {})),
        "query_ids": qids,
        "ks": config["experiment"]["ks"],
    }
    for group in groups.values():
        group["runs"].sort(key=lambda r: r["result"]["seed"])
        seeds = [r["result"]["seed"] for r in group["runs"]]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate seed within the same study/method/dimension/parameters")
        group["comparison_identity"] = fingerprint(identity)
    return {
        "source": str(root),
        "study_key": study_key,
        "study_id": study["study_id"],
        "manifest_sha256": manifest_sha,
        "config": config,
        "dataset": dataset,
        "split": split,
        "verification_boundary": boundary,
        "checked_inputs": checked,
        "missing_jobs": [list(row) for row in sorted(expected_jobs - actual_jobs)],
        "groups": list(groups.values()),
    }


def summarize(group):
    runs = group["runs"]
    result = {
        key: value
        for key, value in group.items()
        if key not in {"runs", "qids", "comparison_identity"}
    }
    result.update(
        {
            "n_seeds": len(runs),
            "seeds": [r["result"]["seed"] for r in runs],
            "queries": len(group["qids"]),
            "run_names": [r["name"] for r in runs],
        }
    )
    metrics = list(runs[0]["result"]["metrics"]) + ["query_doc_overlap@10", "doc_doc_overlap@10"]
    for metric in metrics:
        values = [r["result"]["metrics"].get(metric, r["result"].get(metric)) for r in runs]
        result[f"{metric}_mean"] = float(np.mean(values))
        result[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    for field in (
        "vector_payload_bytes",
        "index_file_bytes",
        "compressor_file_bytes",
        "training_decoder_file_bytes",
    ):
        result[f"{field}_mean"] = float(
            np.mean([r["result"]["storage"].get(field, 0) for r in runs])
        )
    result["deployment_bytes_mean"] = (
        result["index_file_bytes_mean"] + result["compressor_file_bytes_mean"]
    )
    for field in (
        "search_p50_ms",
        "search_p95_ms",
        "batch_search_queries_per_second",
        "fit_seconds",
    ):
        values = [r["result"]["timing"][field] for r in runs if field in r["result"]["timing"]]
        result[f"{field}_mean"] = float(np.mean(values)) if values else None
    result["search_timing_scopes"] = sorted(
        {r["result"]["timing"].get("scope", "unspecified") for r in runs}
    )
    return result


def compare_groups(groups, metric, samples, seed, across_studies=False, case_count=3):
    comparisons, cases, skipped = [], [], []
    for candidate_method, reference_method in COMPARISONS:
        candidates = [g for g in groups if g["method"] == candidate_method]
        for a in candidates:
            references = [
                g
                for g in groups
                if g["method"] == reference_method
                and g["dimension"] == a["dimension"]
                and (across_studies or g["study_key"] == a["study_key"])
            ]
            compatible = [
                b for b in references if b["comparison_identity"] == a["comparison_identity"]
            ]
            if not compatible:
                skipped.append(
                    {
                        "candidate_group": a["group_id"],
                        "reference_method": reference_method,
                        "reason": "No matching same-dimension, same-data/model/query protocol reference",
                    }
                )
            for b in compatible:
                ids = a["qids"]
                am = np.array([[r["scores"][qid][metric] for qid in ids] for r in a["runs"]])
                bm = np.array([[r["scores"][qid][metric] for qid in ids] for r in b["runs"]])
                identity = {
                    "comparison_id": fingerprint([a["group_id"], b["group_id"], metric])[:16],
                    "candidate_group": a["group_id"],
                    "reference_group": b["group_id"],
                    "candidate_label": a["label"],
                    "reference_label": b["label"],
                    "dataset": a["dataset"],
                    "split": a["split"],
                    "dimension": a["dimension"],
                    "metric": metric,
                }
                comparisons.append(
                    {
                        **identity,
                        **seed_query_bootstrap(
                            am,
                            bm,
                            [r["result"]["seed"] for r in a["runs"]],
                            [r["result"]["seed"] for r in b["runs"]],
                            samples,
                            seed,
                        ),
                    }
                )
                deltas = am.mean(axis=0) - bm.mean(axis=0)
                rows = [
                    {
                        "query_id": qid,
                        "delta": float(deltas[i]),
                        "candidate_seed_mean": float(am[:, i].mean()),
                        "reference_seed_mean": float(bm[:, i].mean()),
                    }
                    for i, qid in enumerate(ids)
                ]
                regressions = sorted(
                    (r for r in rows if r["delta"] < 0), key=lambda r: (r["delta"], r["query_id"])
                )[:case_count]
                improvements = sorted(
                    (r for r in rows if r["delta"] > 0), key=lambda r: (-r["delta"], r["query_id"])
                )[:case_count]
                cases.append(
                    {
                        **identity,
                        "largest_improvements": improvements,
                        "largest_regressions": regressions,
                        "improved_queries": int((deltas > 0).sum()),
                        "regressed_queries": int((deltas < 0).sum()),
                        "tied_queries": int((deltas == 0).sum()),
                    }
                )
    return comparisons, cases, skipped


def _csv(path, rows):
    import json

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v, ensure_ascii=False, sort_keys=True)
                if isinstance(v, (list, dict))
                else v
                for k, v in row.items()
            }
            for row in rows
        )


def _training_summary(groups):
    summaries = []
    for group in groups:
        for run in group["runs"]:
            history = run["info"].get("history", [])
            if not history:
                continue
            summary = {
                "group_id": group["group_id"],
                "study_id": group["study_id"],
                "run": run["name"],
                "label": group["label"],
                "dimension": group["dimension"],
                "seed": run["result"]["seed"],
                "actual_steps": run["info"].get("actual_steps"),
                "recorded_steps": [row["step"] for row in history],
                "probe_scope": run["info"].get(
                    "probe_scope", "training corpus probe; not validation"
                ),
                "evaluation_mode": group["evaluation_mode"],
            }
            for key in sorted({key for row in history for key in row if key != "step"}):
                values = [row[key] for row in history if isinstance(row.get(key), (int, float))]
                if values:
                    summary[key] = {
                        "first": values[0],
                        "last": values[-1],
                        "min": min(values),
                        "max": max(values),
                        "all_finite": all(math.isfinite(v) for v in values),
                    }
            summaries.append(summary)
    return summaries


def _plots(output, summaries, groups, metric):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs = []
    by_dataset = defaultdict(list)
    for row in summaries:
        by_dataset[(row["dataset"], row["split"])].append(row)
    for (dataset, split), rows in sorted(by_dataset.items()):
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", f"{dataset}-{split}")
        folder = output / "figures" / f"{slug}-{fingerprint([dataset, split])[:6]}"
        folder.mkdir(parents=True, exist_ok=True)
        series = defaultdict(list)
        for row in rows:
            series[(row["study_key"], row["method"], row["parameter_id"])].append(row)
        for filename, xfield, xlabel in (
            ("quality_dimension.png", "dimension", "Embedding dimension"),
            ("quality_storage.png", "deployment_bytes_mean", "Index + compressor NPZ (MiB)"),
        ):
            fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
            for values in series.values():
                values.sort(key=lambda r: r["dimension"])
                x = [v[xfield] / (2**20 if "bytes" in xfield else 1) for v in values]
                ax.errorbar(
                    x,
                    [v[f"{metric}_mean"] for v in values],
                    yerr=[v[f"{metric}_std"] for v in values],
                    marker="o",
                    capsize=3,
                    label=values[0]["label"],
                )
            ax.set(xlabel=xlabel, ylabel=metric, title=f"{dataset} / {split}; mean and seed SD")
            ax.grid(alpha=0.2)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=7, ncol=2)
            fig.savefig(folder / filename, dpi=160, bbox_inches="tight")
            plt.close(fig)
            outputs.append(str((folder / filename).relative_to(output)))
        dimensions = sorted({r["dimension"] for r in rows if r["method"] != "original"})
        if dimensions:
            fig, axes = plt.subplots(
                math.ceil(len(dimensions) / 2),
                2,
                figsize=(12, 4 * math.ceil(len(dimensions) / 2)),
                squeeze=False,
                layout="constrained",
            )
            for ax, dim in zip(axes.flat, dimensions):
                for row in (r for r in rows if r["dimension"] == dim and r["method"] != "original"):
                    ax.errorbar(
                        row["query_doc_overlap@10_mean"],
                        row[f"{metric}_mean"],
                        xerr=row["query_doc_overlap@10_std"],
                        yerr=row[f"{metric}_std"],
                        marker="o",
                        capsize=2,
                        label=row["label"],
                    )
                ax.set(
                    xlabel="Query-document overlap@10",
                    ylabel=metric,
                    title=f"{dim} dimensions; mean and seed SD",
                )
                ax.grid(alpha=0.2)
                ax.legend(fontsize=6)
            for ax in list(axes.flat)[len(dimensions) :]:
                ax.set_visible(False)
            fig.suptitle(f"{dataset} / {split}; geometry compared within dimension")
            filename = folder / "same_dimension_geometry.png"
            fig.savefig(filename, dpi=160, bbox_inches="tight")
            plt.close(fig)
            outputs.append(str(filename.relative_to(output)))
        matching = [g for g in groups if (g["dataset"], g["split"]) == (dataset, split)]
        for study_key, dim in sorted(
            {
                (g["study_key"], g["dimension"])
                for g in matching
                if any(r["info"].get("history") for r in g["runs"])
            }
        ):
            selected = [
                g for g in matching if g["study_key"] == study_key and g["dimension"] == dim
            ]
            fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
            for ax, field in zip(
                axes.flat,
                (
                    "total_loss",
                    "corpus_probe_reconstruction",
                    "train_probe_ranking",
                    "gradient_norm",
                    "latent_norm_min",
                    "active_triplet_fraction",
                ),
            ):
                for group in selected:
                    histories = [r["info"].get("history", []) for r in group["runs"]]
                    available = [
                        {
                            row["step"]: row[field] if field in row else row["batch_reconstruction"]
                            for row in h
                            if field in row
                            or (
                                field == "total_loss"
                                and group["method"] == "ae"
                                and "batch_reconstruction" in row
                            )
                        }
                        for h in histories
                    ]
                    available = [h for h in available if h]
                    if not available:
                        continue
                    steps = sorted(set.intersection(*(set(h) for h in available)))
                    if not steps:
                        continue
                    values = np.array([[h[s] for s in steps] for h in available])
                    mean = values.mean(axis=0)
                    sd = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(len(steps))
                    ax.plot(steps, mean, label=group["label"])
                    ax.fill_between(steps, mean - sd, mean + sd, alpha=0.15)
                ax.set(xlabel="Training update", ylabel=field.replace("_", " "))
                ax.grid(alpha=0.2)
                if ax.lines:
                    ax.legend(fontsize=6)
                else:
                    ax.text(0.5, 0.5, "No recorded diagnostic", transform=ax.transAxes, ha="center")
            fig.suptitle(
                f"{dataset} / source training {study_key} / {dim}d; mean and seed SD, not validation"
            )
            filename = folder / f"training_{study_key}_{dim}d.png"
            fig.savefig(filename, dpi=160, bbox_inches="tight")
            plt.close(fig)
            outputs.append(str(filename.relative_to(output)))
    return outputs


def analyze(
    study_paths,
    output,
    *,
    metric="ndcg@10",
    samples=10000,
    seed=2026,
    across_studies=False,
    allow_incomplete=False,
    plots=True,
):
    output = Path(output).resolve()
    paths = sorted({Path(path).resolve() for path in study_paths})
    if not paths:
        raise ValueError("At least one completed study is required")
    if samples < 1 or seed < 0:
        raise ValueError("Bootstrap sample count must be positive and seed nonnegative")
    if any(output == path or output.is_relative_to(path) for path in paths):
        raise ValueError("Analysis output must be outside all input studies")
    studies = [load_study(path, metric, allow_incomplete) for path in paths]
    keys = [study["study_key"] for study in studies]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate study manifests supplied from different paths")
    groups = [group for study in studies for group in study["groups"]]
    summaries = [summarize(group) for group in groups]
    comparisons, cases, skipped = compare_groups(groups, metric, samples, seed, across_studies)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("summary", summaries),
        ("comparisons", comparisons),
        ("case-selection", cases),
        ("training-summary", _training_summary(groups)),
    ):
        write_json(output / f"{name}.json", rows)
        _csv(output / f"{name}.csv", rows)
    figures = _plots(output, summaries, groups, metric) if plots else []
    manifest = {
        "schema_version": 1,
        "metric": metric,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "compare_across_studies": across_studies,
        "allow_incomplete": allow_incomplete,
        "studies": [
            {key: value for key, value in study.items() if key != "groups"} for study in studies
        ],
        "skipped_comparisons": skipped,
        "figures": figures,
        "notes": NOTES,
        "source_sha256": sha256(Path(__file__)),
    }
    write_json(output / "analysis-manifest.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--compare-across-studies", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(
        analyze(
            args.studies,
            args.output,
            metric=args.metric,
            samples=args.bootstrap_samples,
            seed=args.seed,
            across_studies=args.compare_across_studies,
            allow_incomplete=args.allow_incomplete,
            plots=not args.no_plots,
        )
    )


if __name__ == "__main__":
    main()
