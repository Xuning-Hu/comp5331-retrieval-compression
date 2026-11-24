"""Fixed descriptive ablation contrasts; no new test configuration selection.

Compare hard/in-batch QRAE with the same configured random-negative QRAE, and
the uncompressed RankOnly projection with Original at its original dimension.
"""

from __future__ import annotations

import argparse
import copy
import runpy
from pathlib import Path

import numpy as np

from retrieval_lab.common import fingerprint, read_json, sha256, write_json


def without_negative_strategy(parameters):
    parameters = copy.deepcopy(parameters)
    parameters.pop("negative_strategy", None)
    for key in ("training", "supervision_config"):
        parameters.get(key, {}).pop("negative_strategy", None)
    return parameters


def compare(analysis_dir, *, samples=10000, seed=2026):
    root = Path(analysis_dir).resolve()
    helpers_path = Path(__file__).with_name("analyze_course.py")
    helpers = runpy.run_path(str(helpers_path))
    manifest = read_json(root / "analysis-manifest.json")
    groups, source_hashes = [], {}
    for source in manifest["studies"]:
        study = helpers["load_study"](Path(source["source"]))
        if (
            study["manifest_sha256"] != source["manifest_sha256"]
            or study["checked_inputs"] != source["checked_inputs"]
        ):
            raise ValueError("Study inputs changed since primary ablation analysis")
        groups.extend(study["groups"])
        source_hashes[study["study_key"]] = study["checked_inputs"]
    pairs = []
    for candidate in groups:
        method = candidate["method"]
        strategy = candidate["parameters"].get("negative_strategy")
        if method == "qrae" and strategy in {"hard", "in_batch"}:
            references = [
                group
                for group in groups
                if group["method"] == "qrae"
                and group["parameters"].get("negative_strategy") == "random"
                and group["dimension"] == candidate["dimension"]
                and group["comparison_identity"] == candidate["comparison_identity"]
                and without_negative_strategy(group["parameters"])
                == without_negative_strategy(candidate["parameters"])
            ]
            kind = f"qrae_{strategy}_minus_random"
        elif method == "rank_only":
            references = [
                group
                for group in groups
                if group["method"] == "original"
                and group["dimension"] == candidate["dimension"]
                and group["comparison_identity"] == candidate["comparison_identity"]
            ]
            kind = "uncompressed_rank_only_minus_original"
        else:
            continue
        if len(references) > 1:
            raise ValueError("Ablation reference is ambiguous; supply separate study analyses")
        if references:
            pairs.append((kind, candidate, references[0]))
    output = []
    for kind, candidate, reference in pairs:
        ids = candidate["qids"]
        a = np.array([[run["scores"][qid]["ndcg@10"] for qid in ids] for run in candidate["runs"]])
        b = np.array([[run["scores"][qid]["ndcg@10"] for qid in ids] for run in reference["runs"]])
        output.append(
            {
                "comparison_id": fingerprint([kind, candidate["group_id"], reference["group_id"]])[
                    :16
                ],
                "contrast": kind,
                "dataset": candidate["dataset"],
                "split": candidate["split"],
                "dimension": candidate["dimension"],
                "metric": "ndcg@10",
                "candidate_group": candidate["group_id"],
                "reference_group": reference["group_id"],
                "candidate_label": candidate["label"],
                "reference_label": reference["label"],
                "candidate_seed_mean": float(a.mean()),
                "reference_seed_mean": float(b.mean()),
                **helpers["seed_query_bootstrap"](
                    a,
                    b,
                    [run["result"]["seed"] for run in candidate["runs"]],
                    [run["result"]["seed"] for run in reference["runs"]],
                    samples,
                    seed,
                ),
            }
        )
    if not output:
        raise ValueError("No compatible declared ablation contrasts found")
    write_json(root / "supplementary-comparisons.json", output)
    helpers["_csv"](root / "supplementary-comparisons.csv", output)
    write_json(
        root / "supplementary-comparisons-manifest.json",
        {
            "analysis_manifest_sha256": sha256(root / "analysis-manifest.json"),
            "script_sha256": sha256(Path(__file__)),
            "analysis_helper_sha256": sha256(helpers_path),
            "source_checked_inputs": source_hashes,
            "output_sha256": {
                name: sha256(root / name)
                for name in ("supplementary-comparisons.json", "supplementary-comparisons.csv")
            },
            "scope": "Descriptive contrasts of already prespecified variants; no parameter/model selection; unadjusted exploratory crossed seed/query percentile intervals",
        },
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    for row in compare(args.analysis, samples=args.bootstrap_samples, seed=args.seed):
        print(row["contrast"], row["dimension"], row["delta"], row["seed_query_bootstrap_95ci"])
