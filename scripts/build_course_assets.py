"""Build report figures from complete, frozen test studies without fitting or selection.

Inputs are raw study directories, never exported/public score snapshots. This tool
checks the prespecified experiment grid and hashes every consumed file. It does
not replay retrieval or replace the separate experimental audit. All plots show
all prespecified seeds; none select a best seed. PNG and SVG outputs only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("original", "pca", "rp", "ae", "rae", "qrae")
COMPRESSED = METHODS[1:]
DIMS = (64, 128, 256, 384)
SEEDS = (17, 42, 2026)
DATASETS = ("scifact", "nfcorpus", "fiqa")
LABELS = {
    "scifact": "SciFact",
    "nfcorpus": "NFCorpus",
    "fiqa": "FiQA",
    "original": "Original",
    "pca": "PCA",
    "rp": "RP",
    "ae": "AE",
    "rae": "RAE",
    "qrae": "QRAE",
    "rank_only": "Rank-only",
}
COLORS = {
    "original": "#666666",
    "pca": "#0072B2",
    "rp": "#E69F00",
    "ae": "#009E73",
    "rae": "#CC79A7",
    "qrae": "#D55E00",
    "rank_only": "#56B4E9",
}
MARKERS = dict(zip((*METHODS, "rank_only"), ("X", "o", "s", "^", "D", "v", "P")))
METRICS = ("ndcg@10", "ndcg@100", "recall@10", "recall@100", "mrr@10", "mrr@100")
GEOMETRY = ("query_doc_overlap@10", "doc_doc_overlap@10")
STORAGE = (
    "vector_payload_bytes",
    "index_file_bytes",
    "compressor_file_bytes",
    "training_decoder_file_bytes",
)
SOURCE_DEV_TIMING_FIELDS = ("fit_seconds", "document_transform_seconds", "index_build_seconds")
SOURCE_DEV_TIMING_SCOPE = (
    "source-dev descriptive timing; may have concurrent background work; "
    "worker-based fit includes worker startup"
)
MIB = 1024**2
NOTES = [
    "Frozen test runs only. No fitting, checkpoint selection, test tuning, or best-seed selection.",
    "Mean and sample SD (ddof=1) across the three prescribed training seeds; SD is not a CI.",
    "Original and PCA each have one deterministic fit, shown with SD=0 by plotting convention; "
    "this does not estimate their uncertainty.",
    "RAE denotes the controlled RAE-style BEIR model, not the separate paper reproduction.",
    "Deployment storage equals serialized index plus compressor NPZ, excluding the training "
    "decoder, common E5 model, corpus text, audit metadata and runtime overhead; it is not RAM.",
    "Study search/transform timings are descriptive measurements that may share concurrent "
    "sweep resources. They are not end-to-end latency or evidence of production speedup.",
    "source_dev_timing records the original fitted-dev compressor fit, document projection, "
    "and index construction costs, checked against the frozen result checksum. It is kept "
    "separate from frozen-test timing: test fit_seconds=0 means no refitting, not free training. "
    "Source-dev timings may have concurrent background work; worker-based fit includes startup.",
    "Training diagnostics use a fixed corpus reconstruction probe and train-label ranking "
    "probe. They are not held-out validation. Frozen runs copy their source training histories.",
    "Geometry comparisons use 128d compressed models and the 768d Original reference only; "
    "no correlation is pooled across dimensions.",
    "Rank-only at 768d is an uncompressed supervised reference, not a compression result.",
    "Consumed JSON/CSV hashes and registered file sizes are checked; this rendering step does "
    "not rehash or replay every index/model, independently recompute qrels metrics, or validate "
    "the original chronological freeze. Use the separate audit for those checks.",
]


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def display_path(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def read_checked(path, consumed, expected=None):
    path = Path(path)
    value = path.read_bytes()
    sha = hashlib.sha256(value).hexdigest()
    if expected is not None and sha != expected:
        raise ValueError(f"Consumed artifact checksum differs: {path}")
    consumed[display_path(path)] = sha
    return value


def read_json(path, consumed, expected=None):
    return json.loads(read_checked(path, consumed, expected).decode("utf-8-sig"))


def finite(value, context, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric value for {context}: {value!r}")
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"Invalid numeric value for {context}: {value}")
    if maximum is not None and value > maximum + 1e-12:
        raise ValueError(f"Out-of-range value for {context}: {value}")
    return float(value)


def expected_jobs(kind):
    if kind == "main":
        return (
            {("original", 768, 0)}
            | {("pca", d, 0) for d in DIMS}
            | {(m, d, s) for m in ("rp", "ae", "rae", "qrae") for d in DIMS for s in SEEDS}
        )
    if kind in {"in_batch", "hard"}:
        return {("qrae", d, s) for d in (64, 128) for s in SEEDS}
    if kind == "rank_only":
        return {("rank_only", d, s) for d in (64, 128, 768) for s in SEEDS}
    raise ValueError(f"Unknown study role: {kind}")


def infer_role(study):
    config = study["config"]
    methods = set(config["experiment"]["methods"])
    if methods == {"rank_only"}:
        return "rank_only"
    strategy = config["supervision"]["negative_strategy"]
    if methods == {"qrae"} and strategy in {"in_batch", "hard"}:
        return strategy
    raise ValueError("Ablations must be QRAE in_batch, QRAE hard, or rank_only studies")


def protocol_recipe(study, *, ablation=False):
    config = study["config"]
    exp = {
        k: v
        for k, v in config["experiment"].items()
        if k not in {"output", "methods", "dimensions", "seeds"}
    }
    supervision = dict(config["supervision"])
    if ablation:
        supervision.pop("negative_strategy", None)
    return {
        "embedding": study["embedding_identity"],
        "ae": config["ae"],
        "neural": config["neural"],
        "supervision": supervision,
        "experiment": exp,
        "source_sha256": study.get("source_sha256"),
    }


def source_dev_timing(study, name, test_result, consumed):
    """Read only the hash-bound source result; never infer fit cost from frozen test."""
    source = Path(study["source_study"])
    if not source.is_absolute():
        source = ROOT / source
    frozen = study["frozen_source"]
    registered = frozen["runs"][name]
    if test_result["frozen_source"] != registered:
        raise ValueError(f"Frozen source run identity differs between study and result: {name}")
    result = read_json(source / name / "result.json", consumed, registered["result_sha256"])
    if (
        result["split"] != "dev"
        or result.get("evaluation_mode") != "fitted"
        or result["study_id"] != frozen["study_id"]
        or result["dataset"] != test_result["dataset"]
        or (result["method"], result["dimension"], result["seed"])
        != (test_result["method"], test_result["dimension"], test_result["seed"])
        or result["documents"] != test_result["documents"]
    ):
        raise ValueError(f"Source-dev result identity disagrees with frozen run: {name}")
    return {
        field: finite(result["timing"][field], f"source-dev/{name}/{field}", 0)
        for field in SOURCE_DEV_TIMING_FIELDS
    }


def load_study(path, kind, consumed):
    path = Path(path).resolve()
    study = read_json(path / "study.json", consumed)
    kind = infer_role(study) if kind == "ablation" else kind
    config, exp = study["config"], study["config"]["experiment"]
    dataset = config["data"]["name"]
    if dataset not in DATASETS or (kind != "main" and dataset != "scifact"):
        raise ValueError(f"Unexpected dataset for {kind}: {dataset}")
    if exp["split"] != "test" or study.get("evaluation_mode") != "frozen":
        raise ValueError(f"Assets require frozen test studies: {path}")
    if not study.get("frozen_source"):
        raise ValueError(f"Missing frozen source identity: {path}")
    expected = expected_jobs(kind)
    methods = set(METHODS) if kind == "main" else {"rank_only" if kind == "rank_only" else "qrae"}
    dims = set(DIMS) if kind == "main" else {64, 128, 768} if kind == "rank_only" else {64, 128}
    if (
        set(exp["methods"]) != methods
        or set(exp["dimensions"]) != dims
        or sorted(exp["seeds"]) != list(SEEDS)
    ):
        raise ValueError(f"Study configuration does not match prescribed {kind} grid: {path}")
    strategy = config["supervision"]["negative_strategy"]
    if strategy != (kind if kind in {"in_batch", "hard"} else "random"):
        raise ValueError(f"Unexpected negative strategy in {kind}: {strategy}")
    qids = study["query_ids"]
    if not qids or len(qids) != len(set(qids)):
        raise ValueError(f"Empty/duplicated query IDs: {path}")
    records, actual = [], set()
    for result_path in sorted(path.glob("*/result.json")):
        result = read_json(result_path, consumed)
        job = (result["method"], result["dimension"], result["seed"])
        name = f"{job[0]}-{job[1]}d-seed{job[2]}"
        if result_path.parent.name != name or job in actual:
            raise ValueError(f"Duplicate or mislabeled run: {result_path}")
        actual.add(job)
        if (
            result["dataset"] != dataset
            or result["study_id"] != study["study_id"]
            or result["split"] != "test"
            or result.get("evaluation_mode") != "frozen"
            or result["queries"] != len(qids)
            or not result.get("frozen_source")
        ):
            raise ValueError(f"Run identity/frozen evaluation disagrees with study: {result_path}")
        for metric in METRICS:
            finite(result["metrics"][metric], f"{name}/{metric}", 0, 1)
        for metric in GEOMETRY:
            finite(result[metric], f"{name}/{metric}", 0, 1)
        for field in STORAGE:
            finite(result["storage"][field], f"{name}/{field}", 0)
        for filename, field in (
            ("index.faiss", "index_file_bytes"),
            ("compressor.npz", "compressor_file_bytes"),
        ):
            if (result_path.parent / filename).stat().st_size != result["storage"][field]:
                raise ValueError(f"Registered deployed artifact size differs: {name}/{filename}")
        info = read_json(
            result_path.parent / "compressor.json", consumed, result["files"]["compressor.json"]
        )
        if (info["method"], info["dimension"], info["seed"]) != job:
            raise ValueError(f"Compressor identity differs: {name}")
        if job[0] in {"rae", "qrae", "rank_only"}:
            if info["actual_steps"] != config["neural"]["steps"]:
                raise ValueError(f"Actual training budget differs: {name}")
            expected_alpha = (
                1.0
                if job[0] == "rank_only"
                else config["neural"]["ranking_weight"]
                if job[0] == "qrae"
                else 0.0
            )
            expected_reg = (
                0.0 if job[0] == "rank_only" else config["neural"]["regularization_weight"]
            )
            if (
                info["ranking_weight"] != expected_alpha
                or info["regularization_weight"] != expected_reg
            ):
                raise ValueError(f"Effective coefficients disagree with configuration: {name}")
            if (
                job[0] in {"qrae", "rank_only"}
                and info["training"]["negative_strategy"] != strategy
            ):
                raise ValueError(f"Effective negative strategy differs: {name}")
        records.append(
            {
                "name": name,
                "result": result,
                "info": info,
                "folder": result_path.parent,
                "source_dev_timing": source_dev_timing(study, name, result, consumed),
            }
        )
    if actual != expected or len(records) != len(expected):
        raise ValueError(
            f"Incomplete/unexpected {kind} grid: {path}; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if len({r["result"]["documents"] for r in records}) != 1:
        raise ValueError("Document counts disagree within study")
    return {"path": path, "study": study, "kind": kind, "dataset": dataset, "records": records}


def validate_collection(main, ablations):
    if len(main) != 3 or {x["dataset"] for x in main} != set(DATASETS):
        raise ValueError("Exactly one main study for SciFact, NFCorpus and FiQA is required")
    base = next(x for x in main if x["dataset"] == "scifact")
    if any(protocol_recipe(x["study"]) != protocol_recipe(base["study"]) for x in main):
        raise ValueError(
            "Main studies have different effective model/training/evaluation protocols"
        )
    if len(ablations) != 3 or {x["kind"] for x in ablations} != {"in_batch", "hard", "rank_only"}:
        raise ValueError("Exactly one in_batch, hard and rank_only SciFact ablation is required")
    for item in ablations:
        if protocol_recipe(item["study"], ablation=True) != protocol_recipe(
            base["study"], ablation=True
        ):
            raise ValueError(f"Ablation changes the shared protocol: {item['kind']}")
        for field in ("query_ids", "embedding_files", "document_ids_sha256", "dataset"):
            if item["study"].get(field) != base["study"].get(field):
                raise ValueError(f"Ablation differs from SciFact main in {field}")


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Aggregation requires a nonempty set of finite observations")
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def aggregate(item):
    groups = defaultdict(list)
    source_timings = {record["name"]: record["source_dev_timing"] for record in item["records"]}
    for record in item["records"]:
        result = record["result"]
        groups[result["method"], result["dimension"]].append(result)
    rows = []
    for (method, dim), results in sorted(groups.items()):
        results.sort(key=lambda x: x["seed"])
        values = {}
        for result in results:
            fields = dict(result["metrics"])
            fields.update({field: result[field] for field in GEOMETRY})
            fields.update(result["storage"])
            fields["deployment_mib"] = (
                result["storage"]["index_file_bytes"] + result["storage"]["compressor_file_bytes"]
            ) / MIB
            fields.update(
                {
                    f"timing.{key}": finite(value, key, 0)
                    for key, value in result["timing"].items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
            )
            name = f"{method}-{dim}d-seed{result['seed']}"
            fields.update(
                {f"source_dev_timing.{key}": value for key, value in source_timings[name].items()}
            )
            values[result["seed"]] = fields
        if len({tuple(sorted(x)) for x in values.values()}) != 1:
            raise ValueError("Seeds have inconsistent reported fields")
        stats = {
            field: mean_sd([value[field] for value in values.values()])
            for field in next(iter(values.values()))
        }
        rows.append(
            {
                "dataset": item["dataset"],
                "role": item["kind"],
                "study_id": item["study"]["study_id"],
                "method": method,
                "dimension": dim,
                "seeds": [r["seed"] for r in results],
                "queries": results[0]["queries"],
                "documents": results[0]["documents"],
                "statistics": stats,
                "source_dev_timing": {
                    "scope": SOURCE_DEV_TIMING_SCOPE,
                    "statistics": {
                        field: stats[f"source_dev_timing.{field}"]
                        for field in SOURCE_DEV_TIMING_FIELDS
                    },
                    "per_seed": [
                        {"seed": r["seed"], **source_timings[f"{method}-{dim}d-seed{r['seed']}"]}
                        for r in results
                    ],
                },
                "negative_strategy": item["study"]["config"]["supervision"]["negative_strategy"],
                "run_names": [f"{method}-{dim}d-seed{r['seed']}" for r in results],
            }
        )
    return rows


def training_probe(item, consumed):
    record = next(r for r in item["records"] if r["name"] == "qrae-128d-seed42")
    raw = read_checked(
        record["folder"] / "training.csv", consumed, record["result"]["files"]["training.csv"]
    )
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    fields = (
        "step",
        "corpus_probe_reconstruction",
        "train_probe_ranking",
        "weighted_regularizer",
        "gradient_norm",
        "latent_norm_min",
    )
    if not rows or any(set(fields) - set(row) for row in rows):
        raise ValueError("QRAE training CSV lacks required diagnostic columns")
    values = {field: [finite(float(row[field]), field, 0) for row in rows] for field in fields}
    if any(a >= b for a, b in zip(values["step"], values["step"][1:])):
        raise ValueError("Training steps must be strictly increasing")
    if values["step"][-1] != record["info"]["actual_steps"]:
        raise ValueError("Training diagnostic does not include the final checkpoint")
    alpha = record["info"]["ranking_weight"]
    values["weighted_train_probe_ranking"] = [alpha * x for x in values["train_probe_ranking"]]
    return {
        "dataset": item["dataset"],
        "run": record["name"],
        "ranking_weight": alpha,
        "scope": "training probes; copied source fit; not validation",
        "values": values,
    }


def load_benchmarks(paths, main, consumed):
    output = []
    seen = set()
    for path in paths:
        benchmark = read_json(path, consumed)
        source = Path(benchmark["source_study"])
        if not source.is_absolute():
            source = ROOT / source
        manifest = read_json(
            source / "study.json", consumed, benchmark["source_hashes"]["study_manifest"]
        )
        dataset = manifest["config"]["data"]["name"]
        if dataset in seen or dataset not in DATASETS:
            raise ValueError("Expected at most one independent benchmark for each main dataset")
        seen.add(dataset)
        target = next(item for item in main if item["dataset"] == dataset)
        expected_names = {"original-768d-seed0", "pca-128d-seed0"} | {
            f"{method}-128d-seed42" for method in ("rp", "ae", "rae", "qrae")
        }
        if set(benchmark["summary"]) != expected_names:
            raise ValueError("Benchmark must contain Original 768d plus all five 128d methods")
        config = benchmark["configuration"]
        if (
            config["seed"] != 2026
            or config["repeats"] != 3
            or config["warmup"] < 1
            or benchmark["effective_queries"] != 30
        ):
            raise ValueError("Benchmark must use 30 queries, seed 2026, three repeats and warmup")
        query_ids = benchmark["query_ids"]
        if (
            len(query_ids) != 30
            or len(set(query_ids)) != 30
            or not set(query_ids) <= set(manifest["query_ids"])
        ):
            raise ValueError("Benchmark query IDs disagree with its source study")
        if set(r["run"] for r in benchmark["measurements"]) != expected_names:
            raise ValueError("Benchmark raw measurements contain missing or extra runs")
        if (
            manifest["embedding_identity"] != target["study"]["embedding_identity"]
            or manifest["embedding_files"] != target["study"]["embedding_files"]
        ):
            raise ValueError("Benchmark embedding identity differs from plotted study")
        # Allow the matching source-dev study: timing is independent of test relevance labels.
        recorded_hashes = benchmark["source_hashes"]["run_artifacts"]
        for name in benchmark["summary"]:
            record = next((x for x in target["records"] if x["name"] == name), None)
            if record is None:
                raise ValueError(f"Unexpected benchmark run: {name}")
            hashes = recorded_hashes[name]
            if hashes.get("compressor.npz") != record["result"]["files"]["compressor.npz"]:
                raise ValueError(f"Benchmark uses different fitted compressor: {name}")
        for name, summary in benchmark["summary"].items():
            measurements = [r for r in benchmark["measurements"] if r["run"] == name]
            if len(measurements) != summary["requests"] or not measurements:
                raise ValueError("Benchmark measurement count differs from summary")
            requests = [(r["query_id"], r["repeat"]) for r in measurements]
            if len(requests) != 90 or set(requests) != {
                (qid, repeat) for qid in query_ids for repeat in range(3)
            }:
                raise ValueError("Benchmark lacks one request per query/repeat/method")
            method, dimseed = name.split("-", 1)
            dim = int(dimseed.split("d-seed")[0])
            row = {
                "dataset": dataset,
                "run": name,
                "method": method,
                "dimension": dim,
                "requests": len(measurements),
                "queries": benchmark["effective_queries"],
                "scope": benchmark["scope"],
                "source_split": manifest["config"]["experiment"]["split"],
                "environment": benchmark["environment"],
                "encoder": benchmark["encoder"],
                "configuration": benchmark["configuration"],
            }
            for field in ("encode_ms", "projection_ms", "search_ms", "total_ms"):
                vals = [finite(r[field], f"benchmark/{field}", 0) for r in measurements]
                derived = {"p50": float(np.median(vals)), "p95": float(np.quantile(vals, 0.95))}
                for stat, value in derived.items():
                    if not np.isclose(summary[field][stat], value, rtol=1e-10, atol=1e-10):
                        raise ValueError(f"Benchmark percentile disagrees with raw values: {name}")
                row[field] = derived
            output.append(row)
    return output


def row_lookup(rows, dataset, method, dim, role="main"):
    return next(
        r
        for r in rows
        if (r["dataset"], r["method"], r["dimension"], r["role"]) == (dataset, method, dim, role)
    )


def stat(row, field):
    value = row["statistics"][field]
    return value["mean"], value["sd"]


def configure_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.35,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#777777",
            "axes.linewidth": 0.7,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.5,
            "svg.fonttype": "none",
            "savefig.dpi": 320,
        }
    )
    return plt


def clean_axes(ax, dataset, xlabel, ylabel):
    ax.set_title(LABELS[dataset], pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def save_figure(plt, fig, out, name, caption, figures, consumed):
    files = {}
    for suffix in ("png", "svg"):
        target = out / f"{name}.{suffix}"
        fig.savefig(target, bbox_inches="tight", facecolor="white")
        files[suffix] = {"path": target.name, "sha256": digest(target)}
    figures.append({"name": name, "caption": caption, "files": files})
    plt.close(fig)


def plot_main(rows, out, figures, consumed):
    plt = configure_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75))
    for ax, dataset in zip(axes, DATASETS):
        original = row_lookup(rows, dataset, "original", 768)
        ax.axhline(
            stat(original, "ndcg@10")[0], color=COLORS["original"], ls="--", label="Original (768d)"
        )
        for method in COMPRESSED:
            values = [stat(row_lookup(rows, dataset, method, dim), "ndcg@10") for dim in DIMS]
            ax.errorbar(
                DIMS,
                [v[0] for v in values],
                yerr=[v[1] for v in values],
                color=COLORS[method],
                marker=MARKERS[method],
                ms=3.8,
                capsize=2,
                label=LABELS[method],
            )
        clean_axes(ax, dataset, "Compressed dimension", "nDCG@10")
        ax.set_xticks(DIMS)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.1)
    save_figure(
        plt,
        fig,
        out,
        "01-effectiveness",
        "Frozen-test nDCG@10 by compressed dimension. "
        "Points are all-seed means and bars are sample SD; Original and PCA use one fit. "
        "Original is a 768d reference. RAE is the controlled RAE-style model.",
        figures,
        consumed,
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75))
    for ax, dataset in zip(axes, DATASETS):
        for method in METHODS:
            row = row_lookup(rows, dataset, method, 768 if method == "original" else 128)
            x, sx = stat(row, "query_doc_overlap@10")
            y, sy = stat(row, "ndcg@10")
            ax.errorbar(
                x,
                y,
                xerr=sx,
                yerr=sy,
                color=COLORS[method],
                marker=MARKERS[method],
                ms=5,
                linestyle="none",
                capsize=2,
                label=LABELS[method],
                zorder=3,
            )
            if method in {"original", "qrae"}:
                offset = (-3, 7) if method == "original" else (4, -11)
                ax.annotate(
                    LABELS[method],
                    (x, y),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=7.5,
                    ha="right" if method == "original" else "left",
                )
        clean_axes(ax, dataset, "Query–document overlap@10", "nDCG@10")
        ax.margins(x=0.14, y=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, handlelength=1.3)
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.1)
    save_figure(
        plt,
        fig,
        out,
        "02-geometry",
        "Query–document top-10 overlap with Original versus "
        "nDCG@10 at 128d, plus Original at 768d. Horizontal and vertical bars are seed "
        "sample SD. No correlations pool different dimensions; overlap is not relevance.",
        figures,
        consumed,
    )

    from matplotlib.ticker import LogLocator, ScalarFormatter

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75))
    for ax, dataset in zip(axes, DATASETS):
        for method in METHODS:
            dims = (768,) if method == "original" else DIMS
            rs = [row_lookup(rows, dataset, method, dim) for dim in dims]
            xs = [stat(r, "deployment_mib")[0] for r in rs]
            ys = [stat(r, "ndcg@10") for r in rs]
            ax.errorbar(
                xs,
                [v[0] for v in ys],
                yerr=[v[1] for v in ys],
                color=COLORS[method],
                marker=MARKERS[method],
                ms=4,
                capsize=2,
                linestyle="none" if method == "original" else "-",
                label=LABELS[method],
            )
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(LogLocator(base=2, numticks=4))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        clean_axes(ax, dataset, "Index + compressor (MiB)", "nDCG@10")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, handlelength=1.3)
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.1)
    save_figure(
        plt,
        fig,
        out,
        "03-storage",
        "Serialized index plus deployed compressor against "
        "nDCG@10 (64, 128, 256 and 384d; Original at 768d; logarithmic x-axis). "
        "Bars are seed sample SD. "
        "The training decoder and common E5 model are excluded; bytes are not runtime RAM.",
        figures,
        consumed,
    )


def plot_ablations(rows, out, figures, consumed):
    plt = configure_plotting()
    fig, ax = plt.subplots(figsize=(7.4, 2.65))
    variants = (
        ("main", "qrae", "QRAE · random", "#D55E00", "v"),
        ("in_batch", "qrae", "QRAE · in-batch", "#0072B2", "s"),
        ("hard", "qrae", "QRAE · static hard", "#009E73", "D"),
        ("rank_only", "rank_only", "Rank-only", "#CC79A7", "P"),
    )
    for i, (role, method, label, color, marker) in enumerate(variants):
        values = [
            stat(row_lookup(rows, "scifact", method, dim, role), "ndcg@10") for dim in (64, 128)
        ]
        ax.errorbar(
            np.array([0, 1]) + (i - 1.5) * 0.055,
            [v[0] for v in values],
            yerr=[v[1] for v in values],
            color=color,
            marker=marker,
            ms=5,
            capsize=3,
            label=label,
        )
    y, sd = stat(row_lookup(rows, "scifact", "rank_only", 768, "rank_only"), "ndcg@10")
    ax.axhspan(y - sd, y + sd, color="#666666", alpha=0.12, linewidth=0)
    ax.axhline(y, color="#666666", ls="--", label="Rank-only 768d (uncompressed)")
    ax.set_xticks([0, 1], ["64", "128"])
    ax.set_xlim(-0.3, 1.3)
    clean_axes(
        ax, "scifact", "Compressed dimension (horizontal offsets for readability)", "nDCG@10"
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    save_figure(
        plt,
        fig,
        out,
        "04-ablations",
        "SciFact frozen-test negative-sampling and objective "
        "ablations. Points and bars are means and sample SD over seeds 17, 42 and 2026. "
        "Rank-only 768d (dashed mean and shaded SD) is an uncompressed reference. "
        "Sampling strategies use one negative per positive; Rank-only uses unit ranking weight.",
        figures,
        consumed,
    )


def plot_training(probes, out, figures, consumed):
    plt = configure_plotting()
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.7))
    lower_handles = None
    for col, dataset in enumerate(DATASETS):
        probe = next(p for p in probes if p["dataset"] == dataset)
        values = probe["values"]
        ax = axes[0, col]
        for field, label, color in (
            ("corpus_probe_reconstruction", "Corpus reconstruction probe", "#0072B2"),
            ("weighted_train_probe_ranking", "α × train ranking probe", "#D55E00"),
            ("weighted_regularizer", "λ × weight regularizer", "#009E73"),
        ):
            ax.plot(values["step"], values[field], label=label, color=color)
        ax.set_yscale("symlog", linthresh=1e-4)
        clean_axes(ax, dataset, "Training step", "Loss (symlog)")
        ax = axes[1, col]
        grad = ax.plot(
            values["step"], values["gradient_norm"], color="#0072B2", label="Gradient norm"
        )
        right = ax.twinx()
        latent = right.plot(
            values["step"],
            values["latent_norm_min"],
            color="#D55E00",
            ls="--",
            label="Minimum raw latent norm",
        )
        ax.set_xlabel("Training step")
        ax.set_ylabel("Gradient norm", color="#0072B2")
        right.set_ylabel("Latent minimum", color="#D55E00")
        right.spines["right"].set_visible(True)
        ax.tick_params(axis="y", labelcolor="#0072B2")
        right.tick_params(axis="y", labelcolor="#D55E00")
        ax.grid(axis="y")
        ax.set_ylim(bottom=0)
        right.set_ylim(bottom=0)
        lower_handles = grad + latent
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False
    )
    fig.legend(
        lower_handles,
        [h.get_label() for h in lower_handles],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96), h_pad=1.3, w_pad=0.8)
    save_figure(
        plt,
        fig,
        out,
        "05-training-diagnostics",
        "Prespecified QRAE 128d seed-42 source "
        "training diagnostics; seed 42 is fixed in advance, not selected by test score. "
        "Top: fixed corpus reconstruction probe, weighted train-label ranking probe and "
        "logged weighted regularizer (symlog permits zero). These curves are not validation "
        "and their sum is not the stochastic training loss. Bottom: logged gradient norm "
        "and minimum raw latent norm, with separate labeled axes.",
        figures,
        consumed,
    )


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_tables(out, rows, benchmarks, probes):
    write_json(
        out / "table-data.json",
        {
            "schema_version": 1,
            "notes": NOTES,
            "rows": rows,
            "benchmarks": benchmarks,
            "training_probes": probes,
        },
    )
    fields = (
        "dataset",
        "role",
        "study_id",
        "method",
        "dimension",
        "seeds",
        "queries",
        "documents",
        "negative_strategy",
        "measure",
        "mean",
        "sample_sd",
        "n",
        "interpretation",
    )
    with (out / "table-data.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            identity = {key: row[key] for key in fields[:9]}
            identity["seeds"] = ",".join(map(str, row["seeds"]))
            for field, stats in row["statistics"].items():
                writer.writerow(
                    {
                        **identity,
                        "measure": field,
                        "mean": stats["mean"],
                        "sample_sd": stats["sd"],
                        "n": stats["n"],
                        "interpretation": SOURCE_DEV_TIMING_SCOPE
                        if field.startswith("source_dev_timing.")
                        else "frozen-test evaluation: no refitting; zero fit is not training cost; "
                        "descriptive sweep timing, no speedup claim"
                        if field.startswith("timing.")
                        else "one deterministic fit; SD=0 convention"
                        if stats["n"] == 1
                        else "mean and sample SD across prescribed training seeds",
                    }
                )
    if benchmarks:
        with (out / "benchmark-table.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "dataset",
                    "method",
                    "dimension",
                    "queries",
                    "requests",
                    "measure",
                    "p50_ms",
                    "p95_ms",
                    "scope",
                ]
            )
            for row in benchmarks:
                for field in ("encode_ms", "projection_ms", "search_ms", "total_ms"):
                    writer.writerow(
                        [row[k] for k in ("dataset", "method", "dimension", "queries", "requests")]
                        + [
                            field,
                            row[field]["p50"],
                            row[field]["p95"],
                            "isolated prototype; total includes IPC; not production serving",
                        ]
                    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-studies",
        type=Path,
        nargs=3,
        required=True,
        metavar="STUDY",
        help="Completed frozen-test SciFact/NFCorpus/FiQA studies",
    )
    parser.add_argument(
        "--ablation-studies",
        type=Path,
        nargs=3,
        required=True,
        metavar="STUDY",
        help="Completed frozen-test SciFact in-batch/hard/Rank-only studies",
    )
    parser.add_argument("--benchmark", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "docs/report/figures")
    args = parser.parse_args()
    out = args.output.resolve()
    sources = [path.resolve() for path in args.main_studies + args.ablation_studies]
    if any(out == path or out.is_relative_to(path) or path.is_relative_to(out) for path in sources):
        raise ValueError("Output must be separate from every input study")
    consumed = {}
    main_studies = [load_study(path, "main", consumed) for path in sources[:3]]
    ablations = [load_study(path, "ablation", consumed) for path in sources[3:]]
    validate_collection(main_studies, ablations)
    rows = [row for item in main_studies + ablations for row in aggregate(item)]
    probes = [training_probe(item, consumed) for item in main_studies]
    benchmarks = load_benchmarks(args.benchmark, main_studies, consumed)
    out.mkdir(parents=True, exist_ok=True)
    figures = []
    plot_main(rows, out, figures, consumed)
    plot_ablations(rows, out, figures, consumed)
    plot_training(probes, out, figures, consumed)
    write_tables(out, rows, benchmarks, probes)
    write_json(out / "figure-captions.json", figures)
    write_json(
        out / "provenance.json",
        {
            "schema_version": 1,
            "script_sha256": digest(Path(__file__)),
            "notes": NOTES,
            "main_studies": [
                {
                    "dataset": item["dataset"],
                    "path": display_path(item["path"]),
                    "study_id": item["study"]["study_id"],
                    "runs": len(item["records"]),
                }
                for item in main_studies
            ],
            "ablations": [
                {
                    "role": item["kind"],
                    "path": display_path(item["path"]),
                    "study_id": item["study"]["study_id"],
                    "runs": len(item["records"]),
                }
                for item in ablations
            ],
            "consumed_files_sha256": consumed,
            "outputs_sha256": {
                p.name: digest(p)
                for p in sorted(out.iterdir())
                if p.is_file() and p.name != "provenance.json"
            },
            "seed_aggregation": "all prescribed seeds; sample SD ddof=1; deterministic SD=0 convention",
            "plot_metrics": "nDCG@10; all six requested IR metrics remain in table-data",
            "benchmark_scope": "optional independently measured prototype IPC-inclusive latency",
        },
    )
    print(
        json.dumps(
            {
                "output": str(out),
                "figures": len(figures),
                "table_rows": len(rows),
                "main_runs": sum(len(x["records"]) for x in main_studies),
                "ablation_runs": sum(len(x["records"]) for x in ablations),
                "input_files_hashed": len(consumed),
            }
        )
    )


if __name__ == "__main__":
    main()
