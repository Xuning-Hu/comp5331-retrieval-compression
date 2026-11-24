"""Describe fixed exploratory encoder and loss-weight follow-ups, without fitting.

Inputs are completed standard studies or exported course snapshots. The original
test has already been inspected: these outputs are post-hoc descriptive evidence,
not a new untouched confirmation, hyperparameter selection, or replacement result.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import os
import runpy
from collections import defaultdict
from pathlib import Path

import numpy as np

from retrieval_lab.common import fingerprint, read_json, sha256, write_json

HELPER_PATH = Path(__file__).with_name("analyze_course.py")
HELPERS = runpy.run_path(str(HELPER_PATH))
SEEDS = (17, 42, 2026)
LAMBDAS = (0.0, 1e-6, 1e-5, 1e-4)
ALPHAS = (0.0, 1.0, 10.0, 100.0)
METHODS = ("original", "pca", "rp", "ae", "rae", "qrae")
MODELS = {
    "e5": ("intfloat/e5-base-v2", "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"),
    "mpnet": (
        "sentence-transformers/all-mpnet-base-v2",
        "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    ),
}
NOTES = [
    "Post-hoc exploratory follow-up after the original final test was inspected; "
    "no new untouched confirmation or optimum is declared.",
    "All declared conditions and seeds are retained; sample SD describes training seeds, "
    "not query uncertainty or a confidence interval. Deterministic baselines have one run.",
    "The MPNet comparison uses transferred E5 loss weights; it is not independently tuned "
    "encoder performance. Model-specific prefixes/token limits differ, so this is a "
    "comparison of encoding pipelines, not a causal architecture-only intervention.",
    "The 4x4 loss-weight grid is dev-only. Its zero-weight axes are explicitly trained "
    "QRAE configurations, not assumed byte-identical copies of previous AE/RAE fits.",
    "Positive-lambda minus zero-lambda contrasts hold alpha, split and training seed fixed. "
    "They describe a conditional effect for this grid, not global geometric guarantees.",
    "Optional bootstrap intervals resample matching training seeds and common query IDs, "
    "not seed-query cells independently; they are exploratory without multiplicity correction.",
    "Inputs undergo registered-artifact checksum and saved-score aggregation checks through "
    "analyze_course.load_study. This script does not independently search an index, verify "
    "rankings against qrels, encode documents, or replay training.",
    "Vector/index/compressor sizes and search timing retain their original scope. They are "
    "not overall RAM or end-to-end text-query latency measurements.",
]


def _stats(values):
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite values")
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)) if values.size > 1 else None,
        "count": int(values.size),
    }


def _load(path, metric):
    loaded = HELPERS["load_study"](Path(path), metric)
    manifest = read_json(Path(path) / "study.json")
    loaded["experiment_source_sha256"] = manifest.get("source_sha256", "unspecified")
    loaded["evaluation_mode"] = manifest.get("evaluation_mode", "unspecified")
    if (Path(path) / "snapshot.json").is_file():
        loaded["snapshot_manifest_sha256"] = sha256(Path(path) / "snapshot.json")
    loaded["data_identity"] = fingerprint(
        {
            "dataset": manifest["dataset"],
            "split": loaded["split"],
            "query_ids": sorted(manifest["query_ids"]),
            "document_ids_sha256": manifest.get("document_ids_sha256"),
            "ks": loaded["config"]["experiment"]["ks"],
        }
    )
    return loaded


def _relative_reference(path, output):
    try:
        return Path(os.path.relpath(path, output)).as_posix()
    except ValueError:  # Different Windows drives have no relative path.
        return str(Path(path).resolve())


def _encoder(study, expected):
    settings = study["config"]["embedding"]
    if (settings.get("model"), settings.get("revision")) != MODELS[expected]:
        raise ValueError(f"Expected pinned {expected} model: {study['source']}")
    convention = (384, "", "") if expected == "mpnet" else (512, "query: ", "passage: ")
    if (
        tuple(settings.get(k) for k in ("max_length", "query_prefix", "document_prefix"))
        != convention
    ):
        raise ValueError(f"Unexpected {expected} token limit/prefix convention")
    return {
        "label": expected,
        "model": settings["model"],
        "revision": settings["revision"],
        "max_length": settings.get("max_length"),
        "query_prefix": settings.get("query_prefix"),
        "document_prefix": settings.get("document_prefix"),
    }


def _check_seeds(group):
    actual = sorted(run["result"]["seed"] for run in group["runs"])
    expected = [0] if group["method"] in {"original", "pca"} else list(SEEDS)
    if actual != expected:
        raise ValueError(f"Incomplete or duplicated seeds for {group['group_id']}: {actual}")


def _condition(group, study, scope, encoder, *, weights=None):
    group = copy.deepcopy(group)
    _check_seeds(group)
    identity = {
        "scope": scope,
        "encoder": encoder,
        "dataset": group["dataset"],
        "split": group["split"],
        "dimension": group["dimension"],
        "method": group["method"],
        "parameters": group["parameters"],
        "source_groups": group.get("source_groups", [group["group_id"]]),
    }
    if weights is not None:
        identity.update(weights)
    return {
        **identity,
        "condition_id": fingerprint(identity)[:16],
        "data_identity": study["data_identity"],
        "qids": group["qids"],
        "runs": group["runs"],
    }


def encoder_conditions(studies):
    conditions, seen = [], set()
    for label, study in studies:
        encoder = _encoder(study, label)
        key = (label, study["dataset"], study["split"])
        if key in seen:
            raise ValueError("Multiple same-encoder/dataset/split studies cannot be mixed")
        seen.add(key)
        groups = [
            group
            for group in study["groups"]
            if group["dimension"] == 128 or group["method"] == "original"
        ]
        if sorted(group["method"] for group in groups) != sorted(METHODS):
            raise ValueError("Encoder comparison requires exactly six methods at 128d")
        for group in groups:
            if group["method"] == "original" and group["dimension"] != 768:
                raise ValueError("Expected the full-dimensional 768d Original")
            params = group["parameters"]
            if group["method"] in {"rae", "qrae"}:
                if params.get("regularization_weight") != 1e-5:
                    raise ValueError("Encoder follow-up must retain the fixed lambda=1e-5")
            if group["method"] == "qrae" and (
                params.get("ranking_weight") != 10.0 or params.get("negative_strategy") != "random"
            ):
                raise ValueError("Encoder follow-up must retain alpha=10 and random negatives")
            conditions.append(_condition(group, study, "encoder", encoder))
    return conditions


def grid_conditions(studies):
    cells = defaultdict(list)
    datasets = {}
    nuisance_parameters = None
    for study in studies:
        _encoder(study, "e5")
        if study["split"] != "dev":
            raise ValueError("The exploratory weight grid must remain dev-only")
        dataset = study["dataset"]
        if dataset not in {"scifact", "fiqa"}:
            raise ValueError("Unexpected dataset in the fixed grid")
        if dataset in datasets and datasets[dataset]["data_identity"] != study["data_identity"]:
            raise ValueError("Grid dataset/query protocol changed between cells")
        datasets[dataset] = study
        for group in study["groups"]:
            params = group["parameters"]
            weight = params.get("regularization_weight")
            alpha = params.get("ranking_weight")
            if (
                group["method"] != "qrae"
                or group["dimension"] != 128
                or weight not in LAMBDAS
                or alpha not in ALPHAS
                or params.get("negative_strategy") != "random"
            ):
                raise ValueError("Grid has an unexpected method/dimension/weight/negative strategy")
            nuisance = copy.deepcopy(params)
            for key in ("regularization_weight", "ranking_weight"):
                nuisance.pop(key, None)
                nuisance["training"].pop(key, None)
            if nuisance_parameters is not None and nuisance != nuisance_parameters:
                raise ValueError("Grid changes nuisance parameters beyond lambda and alpha")
            nuisance_parameters = nuisance
            cells[(dataset, weight, alpha)].append((study, group))
    conditions = []
    if studies and set(datasets) != {"scifact", "fiqa"}:
        raise ValueError("The complete declared grid requires both SciFact and FiQA")
    expected = set(itertools.product(datasets, LAMBDAS, ALPHAS))
    if set(cells) != expected:
        raise ValueError("Incomplete 4x4 grid: every planned weight pair must be present")
    for (dataset, weight, alpha), items in sorted(cells.items()):
        first_study, first = items[0]
        merged = copy.deepcopy(first)
        merged["runs"], merged["source_groups"] = [], []
        for study, group in items:
            if (
                group["comparison_identity"] != first["comparison_identity"]
                or group["parameters"] != first["parameters"]
            ):
                raise ValueError("A grid cell mixes parameters, caches or query identities")
            for run in group["runs"]:
                run = copy.deepcopy(run)
                run["source_study_id"] = study["study_id"]
                run["source_study"] = study["source"]
                merged["runs"].append(run)
            merged["source_groups"].append(group["group_id"])
        merged["runs"].sort(key=lambda run: run["result"]["seed"])
        merged["source_groups"].sort()
        conditions.append(
            _condition(
                merged,
                first_study,
                "grid",
                _encoder(first_study, "e5"),
                weights={"lambda": weight, "alpha": alpha},
            )
        )
    return conditions


def _matrix(condition, metric):
    return np.array(
        [[run["scores"][qid][metric] for qid in condition["qids"]] for run in condition["runs"]],
        dtype=float,
    )


def _paired(candidate, reference, metric):
    if (
        candidate["data_identity"] != reference["data_identity"]
        or candidate["qids"] != reference["qids"]
    ):
        raise ValueError("Comparisons require identical data/split/query identities")
    if (
        candidate["encoder"]["label"] != reference["encoder"]["label"]
        and candidate["method"] == reference["method"]
        and candidate["parameters"] != reference["parameters"]
    ):
        raise ValueError("Cross-encoder comparison changes compressor parameters")
    a, b = _matrix(candidate, metric), _matrix(reference, metric)
    sa = [run["result"]["seed"] for run in candidate["runs"]]
    sb = [run["result"]["seed"] for run in reference["runs"]]
    if sa == sb:
        return a - b, sa
    if len(sb) == 1 and reference["method"] in {"original", "pca"}:
        return a - b, sa
    if len(sa) == 1 and candidate["method"] in {"original", "pca"}:
        return a - b, sb
    raise ValueError("Unmatched training seeds in a paired comparison")


def _contrast_record(kind, candidate, reference, values, seeds, metric, samples):
    identity = {
        "kind": kind,
        "candidate": candidate["condition_id"],
        "reference": reference["condition_id"],
        "dataset": candidate["dataset"],
        "split": candidate["split"],
        "metric": metric,
    }
    result = {
        **identity,
        "comparison_id": fingerprint(identity)[:16],
        "candidate_encoder": candidate["encoder"]["label"],
        "reference_encoder": reference["encoder"]["label"],
        "candidate_method": candidate["method"],
        "reference_method": reference["method"],
        "delta": _stats(values.mean(axis=1)),
        "seeds": seeds,
        "per_seed_delta": dict(zip(map(str, seeds), values.mean(axis=1).tolist(), strict=True)),
        "query_count": len(candidate["qids"]),
        "improved_query_count": int((values.mean(axis=0) > 0).sum()),
        "regressed_query_count": int((values.mean(axis=0) < 0).sum()),
        "tied_query_count": int((values.mean(axis=0) == 0).sum()),
    }
    for key in ("lambda", "alpha"):
        if key in candidate:
            result[f"candidate_{key}"] = candidate[key]
            result[f"reference_{key}"] = reference[key]
    if samples:
        result["exploratory_bootstrap"] = HELPERS["seed_query_bootstrap"](
            values, np.zeros_like(values), seeds, seeds, samples, seed=2026
        )
    detail = {
        **identity,
        "comparison_id": result["comparison_id"],
        "query_ids": candidate["qids"],
        "seeds": seeds,
        "seed_mean_query_delta": values.mean(axis=0).tolist(),
        "paired_seed_query_delta": values.tolist(),
    }
    return result, detail


def comparisons(conditions, metric, samples=0):
    rows, details = [], []
    encoders = {
        (c["dataset"], c["split"], c["encoder"]["label"], c["method"]): c
        for c in conditions
        if c["scope"] == "encoder"
    }

    def add(kind, a, b):
        values, seeds = _paired(a, b, metric)
        row, detail = _contrast_record(kind, a, b, values, seeds, metric, samples)
        rows.append(row)
        details.append(detail)
        return values, seeds

    for (dataset, split, encoder, method), candidate in sorted(encoders.items()):
        if method == "qrae":
            for reference in ("rae", "original"):
                add("within_encoder", candidate, encoders[(dataset, split, encoder, reference)])
        if encoder == "mpnet" and (dataset, split, "e5", method) in encoders:
            add("mpnet_minus_e5", candidate, encoders[(dataset, split, "e5", method)])
        if encoder != "mpnet" or method != "qrae":
            continue
        for reference in ("rae", "original"):
            keys = [
                (dataset, split, label, method_name)
                for label in ("mpnet", "e5")
                for method_name in ("qrae", reference)
            ]
            if not all(key in encoders for key in keys):
                continue
            aq, ar, bq, br = [encoders[key] for key in keys]
            av, seeds = _paired(aq, ar, metric)
            bv, other_seeds = _paired(bq, br, metric)
            _paired(aq, bq, metric)  # Check cross-encoder data/seed identity too.
            if seeds != other_seeds:
                raise ValueError("Cross-encoder improvement contrasts have different seeds")
            row, detail = _contrast_record(
                f"change_in_qrae_minus_{reference}", aq, bq, av - bv, seeds, metric, samples
            )
            row["within_encoder_reference_method"] = reference
            row["component_conditions"] = [c["condition_id"] for c in (aq, ar, bq, br)]
            detail["component_conditions"] = row["component_conditions"]
            rows.append(row)
            details.append(detail)
    grid = {(c["dataset"], c["lambda"], c["alpha"]): c for c in conditions if c["scope"] == "grid"}
    for (dataset, weight, alpha), candidate in sorted(grid.items()):
        if weight > 0:
            add("lambda_minus_zero_at_fixed_alpha", candidate, grid[(dataset, 0.0, alpha)])
        if alpha > 0:
            add("alpha_minus_zero_at_fixed_lambda", candidate, grid[(dataset, weight, 0.0)])
    return rows, details


def _tables(conditions):
    summaries, per_seed, per_query = [], [], []
    for condition in conditions:
        identity = {k: v for k, v in condition.items() if k not in {"runs", "qids"}}
        metrics = list(condition["runs"][0]["result"]["metrics"])
        summary = {**identity, "queries": len(condition["qids"]), "metrics": {}}
        for metric in metrics:
            summary["metrics"][metric] = _stats(
                [run["result"]["metrics"][metric] for run in condition["runs"]]
            )
        summary["seeds"] = [run["result"]["seed"] for run in condition["runs"]]
        summaries.append(summary)
        for run in condition["runs"]:
            per_seed.append(
                {
                    **identity,
                    "seed": run["result"]["seed"],
                    "run": run["name"],
                    "study_id": run["result"]["study_id"],
                    "metrics": run["result"]["metrics"],
                    "geometry": {
                        k: run["result"].get(k)
                        for k in ("query_doc_overlap@10", "doc_doc_overlap@10")
                    },
                    "timing": run["result"]["timing"],
                    "storage": run["result"]["storage"],
                }
            )
        per_query.append(
            {
                "condition_id": condition["condition_id"],
                "query_ids": condition["qids"],
                "runs": [
                    {
                        "seed": run["result"]["seed"],
                        "run": run["name"],
                        "study_id": run["result"]["study_id"],
                        "scores": run["scores"],
                    }
                    for run in condition["runs"]
                ],
            }
        )
    return summaries, per_seed, per_query


def _flat_table(rows, table):
    result = []
    for row in rows:
        flat = {
            key: value
            for key, value in row.items()
            if key not in {"parameters", "metrics", "encoder", "data_identity"}
        }
        if "encoder" in row:
            flat["encoder"] = row["encoder"]["label"]
        for metric, value in row.get("metrics", {}).items():
            if table == "summary":
                flat[f"{metric}_mean"] = value["mean"]
                flat[f"{metric}_sample_sd"] = value["sample_sd"]
                flat["n_seeds"] = value["count"]
            else:
                flat[metric] = value
        result.append(flat)
    return result


def _plots(output, summaries, metric):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    encoders = [row for row in summaries if row["scope"] == "encoder"]
    datasets = sorted({(row["dataset"], row["split"]) for row in encoders})
    if datasets:
        fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4.4), squeeze=False)
        for ax, (dataset, split) in zip(axes[0], datasets, strict=True):
            for label, offset, color in (("e5", -0.12, "#2472A3"), ("mpnet", 0.12, "#C66A28")):
                selected = {
                    row["method"]: row
                    for row in encoders
                    if (row["dataset"], row["split"], row["encoder"]["label"])
                    == (dataset, split, label)
                }
                if not selected:
                    continue
                ax.errorbar(
                    np.arange(len(METHODS)) + offset,
                    [selected[m]["metrics"][metric]["mean"] for m in METHODS],
                    yerr=[selected[m]["metrics"][metric]["sample_sd"] or 0 for m in METHODS],
                    fmt="o",
                    capsize=3,
                    label=label.upper(),
                    color=color,
                )
            ax.set_xticks(range(len(METHODS)), ["Orig.\n768d", "PCA", "RP", "AE", "RAE", "QRAE"])
            ax.set_title(f"{dataset} / {split}", fontsize=15)
            ax.set_ylabel(metric, fontsize=14)
            ax.tick_params(labelsize=13)
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle("128d encoder follow-up: mean and training-seed SD (exploratory)", fontsize=15)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.90),
            ncol=2,
            fontsize=12,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.83))
        path = output / "encoder-128d.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        figures.append(path.name)
    grid = [row for row in summaries if row["scope"] == "grid"]
    for dataset in sorted({row["dataset"] for row in grid}):
        selected = {(row["lambda"], row["alpha"]): row for row in grid if row["dataset"] == dataset}
        values = np.array(
            [[selected[(w, a)]["metrics"][metric]["mean"] for a in ALPHAS] for w in LAMBDAS]
        )
        fig, ax = plt.subplots(figsize=(6.3, 5.4))
        color = ax.imshow(
            values, cmap="viridis", vmin=float(values.min()), vmax=float(values.max())
        )
        for i, w in enumerate(LAMBDAS):
            for j, a in enumerate(ALPHAS):
                stat = selected[(w, a)]["metrics"][metric]
                text_color = (
                    "black" if values[i, j] > values.min() + 0.6 * np.ptp(values) else "white"
                )
                ax.text(
                    j,
                    i,
                    f"{stat['mean']:.4f}\n±{stat['sample_sd']:.4f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=13,
                )
        ax.set_xticks(range(len(ALPHAS)), [f"{v:g}" for v in ALPHAS])
        ax.set_yticks(range(len(LAMBDAS)), [f"{v:g}" for v in LAMBDAS])
        ax.set_xlabel("Ranking weight α", fontsize=14)
        ax.set_ylabel("Regularization weight λ", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_title(
            f"{dataset} dev, E5 128d: {metric}\nExploratory: mean ± training-seed SD",
            fontsize=14,
        )
        colorbar = fig.colorbar(color, ax=ax)
        colorbar.set_label(metric, fontsize=12)
        colorbar.ax.tick_params(labelsize=11)
        fig.tight_layout()
        path = output / f"grid-{dataset}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        figures.append(path.name)
    return figures


def _fragment(summaries, contrasts, metric):
    lines = ["# Exploratory supplementary results", "", NOTES[0], "", NOTES[1], ""]
    encoder_rows = [row for row in summaries if row["scope"] == "encoder"]
    if encoder_rows:
        lines += [
            "## Second encoding pipeline at 128 dimensions",
            "",
            NOTES[2],
            "",
            f"| Dataset / split | Encoder | Method | {metric} mean | Seed SD |",
            "|---|---|---|---:|---:|",
        ]
        for row in encoder_rows:
            stat = row["metrics"][metric]
            sd = (
                "— (one deterministic run)"
                if stat["sample_sd"] is None
                else f"{stat['sample_sd']:.5f}"
            )
            lines.append(
                f"| {row['dataset']} / {row['split']} | {row['encoder']['label']} | "
                f"{row['method']} | {stat['mean']:.5f} | {sd} |"
            )
        lines += ["", "Within each encoder, the paired three-seed mean differences are:", ""]
        for row in contrasts:
            if row["kind"] == "within_encoder":
                lines.append(
                    f"- {row['dataset']} / {row['split']} / {row['candidate_encoder']}: "
                    f"QRAE − {row['reference_method']} = {row['delta']['mean']:+.5f} "
                    f"(paired seed SD {row['delta']['sample_sd']:.5f})."
                )
    grid_rows = [row for row in summaries if row["scope"] == "grid"]
    if grid_rows:
        lines += [
            "",
            "## Joint loss-weight grid on dev",
            "",
            NOTES[3],
            "",
            NOTES[4],
            "",
            "Every cell is listed in summary.json and the heatmaps; no cell is designated "
            "as a new test-optimal configuration. The following ranges retain both "
            "improvements and regressions over the zero-regularization reference:",
            "",
            "| Dataset | Fixed alpha | Positive-lambda minus zero-lambda: min to max |",
            "|---|---:|---:|",
        ]
        by_id = {row["condition_id"]: row for row in summaries}
        buckets = defaultdict(list)
        for row in contrasts:
            if row["kind"] == "lambda_minus_zero_at_fixed_alpha":
                candidate = by_id[row["candidate"]]
                buckets[(row["dataset"], candidate["alpha"])].append(row["delta"]["mean"])
        for (dataset, alpha), values in sorted(buckets.items()):
            lines.append(f"| {dataset} | {alpha:g} | {min(values):+.5f} to {max(values):+.5f} |")
    lines += ["", "## Interpretation and provenance", ""]
    lines += [f"- {note}" for note in NOTES[5:]]
    lines += [
        "",
        "Sources and their checked hashes are retained in analysis-manifest.json. "
        "Per-seed metrics, every saved per-query score, and paired contrast matrices "
        "are available beside this fragment. Original course results remain unchanged.",
        "",
    ]
    return "\n".join(lines)


def analyze(
    mpnet_paths,
    e5_paths,
    grid_paths,
    output,
    *,
    metric="ndcg@10",
    samples=0,
    plots=True,
    grid_manifest=None,
):
    output = Path(output).resolve()
    if samples < 0:
        raise ValueError(
            "Bootstrap samples must be nonnegative; zero requests descriptive statistics"
        )
    if grid_manifest is not None:
        grid_manifest = Path(grid_manifest).resolve()
        mapping = read_json(grid_manifest)
        if mapping.get("status") != "complete" or not mapping.get("studies"):
            raise ValueError("Grid manifest must be complete and list all study paths")
        path_base = mapping.get("path_base", "repository")
        if path_base not in {"manifest", "repository"}:
            raise ValueError("Grid path_base must be manifest or repository")
        base = grid_manifest.parent if path_base == "manifest" else Path.cwd()
        grid_paths = list(grid_paths) + [base / Path(path) for path in mapping["studies"]]
    paths = [
        (label, Path(p).resolve())
        for label, values in (("mpnet", mpnet_paths), ("e5", e5_paths), ("grid", grid_paths))
        for p in values
    ]
    if not paths:
        raise ValueError("At least one completed supplementary input is required")
    if len({p for _, p in paths}) != len(paths):
        raise ValueError("A source study was supplied more than once")
    if any(output == p or output.is_relative_to(p) for _, p in paths):
        raise ValueError("Output must remain outside all source studies")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Analysis output is nonempty; choose a new output directory")
    loaded = [(label, _load(path, metric)) for label, path in paths]
    conditions = encoder_conditions([(label, study) for label, study in loaded if label != "grid"])
    conditions += grid_conditions([study for label, study in loaded if label == "grid"])
    conditions.sort(
        key=lambda c: (
            c["scope"],
            c["dataset"],
            c["split"],
            c["encoder"]["label"],
            c.get("lambda", -1),
            c.get("alpha", -1),
            METHODS.index(c["method"]),
        )
    )
    summaries, per_seed, per_query = _tables(conditions)
    contrasts, contrast_details = comparisons(conditions, metric, samples)
    output.mkdir(parents=True)
    for name, rows in (("summary", summaries), ("per-seed", per_seed), ("comparisons", contrasts)):
        write_json(output / f"{name}.json", rows)
        HELPERS["_csv"](output / f"{name}.csv", _flat_table(rows, name))
    write_json(output / "per-query.json", per_query)
    write_json(output / "comparison-per-query.json", contrast_details)
    (output / "supplemental-fragment.md").write_text(
        _fragment(summaries, contrasts, metric), encoding="utf-8"
    )
    figures = _plots(output, summaries, metric) if plots else []
    sources = [
        {
            "role": label,
            "source_relative_path": _relative_reference(study["source"], output),
            **{key: value for key, value in study.items() if key != "groups"},
        }
        for label, study in loaded
    ]
    manifest = {
        "schema_version": 1,
        "scope": NOTES[0],
        "metric": metric,
        "condition_count": len(conditions),
        "run_count": len(per_seed),
        "bootstrap_samples": samples,
        "bootstrap_seed": 2026 if samples else None,
        "grid_axes": {"lambda": list(LAMBDAS), "alpha": list(ALPHAS), "seeds": list(SEEDS)},
        "source_studies": sources,
        "figures": figures,
        "notes": NOTES,
        "script_sha256": sha256(Path(__file__)),
        "analysis_helper_sha256": sha256(HELPER_PATH),
        "grid_manifest": (
            {
                "path": str(grid_manifest),
                "relative_path": _relative_reference(grid_manifest, output),
                "sha256": sha256(grid_manifest),
            }
            if grid_manifest is not None
            else None
        ),
        "files": {
            str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()
        },
    }
    write_json(output / "analysis-manifest.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpnet-studies", nargs="*", type=Path, default=[])
    parser.add_argument("--e5-studies", nargs="*", type=Path, default=[])
    parser.add_argument("--grid-studies", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--grid-manifest",
        type=Path,
        help="Complete grid map; path_base='manifest' supports portable relative snapshot paths",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(
        analyze(
            args.mpnet_studies,
            args.e5_studies,
            args.grid_studies,
            args.output,
            metric=args.metric,
            samples=args.bootstrap_samples,
            plots=not args.no_plots,
            grid_manifest=args.grid_manifest,
        )
    )


if __name__ == "__main__":
    main()
