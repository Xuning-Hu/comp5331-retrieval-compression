"""Verify public numeric export hashes and replay saved-score aggregation offline."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def replay(root: Path) -> dict:
    manifest = json.loads((root / "results/manifest.json").read_text(encoding="utf-8"))
    counts = Counter()
    table = {}
    max_error = 0.0
    checks = 0
    directories = set()
    for study in manifest["studies"]:
        label = study["directory"]
        folder = (root / label).resolve()
        if not folder.is_relative_to((root / "results").resolve()) or label in directories:
            raise ValueError(f"Unsafe or repeated study directory: {label}")
        directories.add(label)
        for name, expected in study["files"].items():
            path = (folder / name).resolve()
            if not path.is_relative_to(folder):
                raise ValueError("File escapes numeric study directory")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {label}/{name}")
        scores = json.loads((folder / "per-query.json").read_text(encoding="utf-8"))
        summaries = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if len(scores) != study["runs"]:
            raise ValueError(f"Run count differs: {label}")
        counts[study["family"]] += len(scores)
        seen = set()
        for group in summaries:
            names = group["run_names"]
            if len(names) != group["n_seeds"] or len(set(names)) != len(names):
                raise ValueError(f"Seed count differs: {label}")
            values = {}
            query_ids = None
            for name in names:
                if name in seen:
                    raise ValueError(f"Run appears in multiple groups: {label}/{name}")
                seen.add(name)
                queries = scores[name]
                if len(queries) != group["queries"]:
                    raise ValueError(f"Query count differs: {label}/{name}")
                if query_ids is None:
                    query_ids = set(queries)
                elif query_ids != set(queries):
                    raise ValueError(f"Query identities differ across seeds: {label}")
                metrics = set(next(iter(queries.values())))
                if any(set(row) != metrics for row in queries.values()):
                    raise ValueError(f"Metric coverage differs across queries: {label}/{name}")
                for metric in metrics:
                    rows = [row[metric] for row in queries.values()]
                    if not all(math.isfinite(v) for v in rows):
                        raise ValueError(f"Nonfinite saved score: {label}/{name}")
                    values.setdefault(metric, []).append(statistics.fmean(rows))
            for metric, means in values.items():
                for suffix, actual in (
                    ("mean", statistics.fmean(means)),
                    ("std", statistics.stdev(means) if len(means) > 1 else 0.0),
                ):
                    expected = group[f"{metric}_{suffix}"]
                    error = abs(actual - expected)
                    max_error = max(max_error, error)
                    checks += 1
                    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                        raise ValueError(f"Aggregate differs: {label}/{metric}_{suffix}")
            if study["family"] == "main" and (
                group["dimension"] == 128 or group["method"] == "original"
            ):
                table.setdefault(group["method"], {})[group["dataset"]] = statistics.fmean(
                    values["ndcg@10"]
                )
        if seen != set(scores):
            raise ValueError(f"Unreported or missing runs: {label}")
    expected_counts = {
        "main": 159, "ablations": 21, "mpnet-dev": 42, "mpnet-test": 42, "weight-grid": 96
    }
    if dict(counts) != expected_counts:
        raise ValueError(f"Unexpected study coverage: {dict(counts)}")
    return {
        "status": "passed",
        "studies": len(directories),
        "runs": sum(counts.values()),
        "runs_by_family": dict(counts),
        "aggregate_checks": checks,
        "maximum_absolute_aggregation_error": max_error,
        "main_128d_ndcg10": table,
        "boundary": "Saved-score aggregation only; no ranking/qrels replay or model training.",
    }


def main() -> None:
    report = replay(Path(__file__).resolve().parents[1])
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
