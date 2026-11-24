"""Execute the prespecified course stages; test is an explicit separate command.

Run from the repository root. The selection manifest retains every dev candidate.
No test score enters selection. Existing studies are never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from retrieval_lab.common import fingerprint, read_json, sha256, write_json
from retrieval_lab.config import load_config, validate_config
from retrieval_lab.data import load_dataset
from retrieval_lab.embedding import cache_path
from retrieval_lab.experiment import evaluate_saved, run
from retrieval_lab.reporting import write_report

ROOT = Path("runs/course")
SELECTION = ROOT / "selection.json"


def base_config(dataset="scifact"):
    config = load_config("configs/course-scifact-pilot.toml")
    config["data"]["name"] = dataset
    config["ae"]["steps"] = 3000
    config["neural"]["steps"] = 3000
    # Both AE and RAE/QRAE use the faster device measured before dev selection.
    config["ae"]["device"] = "cuda"
    config["neural"]["device"] = "cuda"
    config["experiment"].update(
        {
            "output": str(ROOT / dataset),
            "methods": ["original", "pca", "rp", "ae", "rae", "qrae"],
            "dimensions": [64, 128, 256, 384],
            "seeds": [17, 42, 2026],
            "split": "dev",
        }
    )
    return config


def execute(config):
    validate_config(config)
    dataset = load_dataset(
        Path(config["data"]["directory"]) / config["data"]["name"], config["data"]
    )
    study = run(config, dataset, cache_path(config, dataset))
    write_report(study)
    return study


def tune():
    if SELECTION.exists():
        raise FileExistsError(
            "Selection already exists; retain it and use a new protocol if changed"
        )
    config = base_config()
    config["experiment"].update({"dimensions": [128], "seeds": [42]})
    candidates = []
    for value in [1e-5, 1e-4, 1e-3]:
        candidate = copy.deepcopy(config)
        candidate["experiment"].update({"methods": ["rae"], "output": str(ROOT / "tuning-rae")})
        candidate["neural"]["regularization_weight"] = value
        study = execute(candidate)
        result = read_json(study / "rae-128d-seed42/result.json")
        candidates.append(
            {
                "method": "rae",
                "value": value,
                "ndcg@10": result["metrics"]["ndcg@10"],
                "study": str(study),
            }
        )
        write_json(ROOT / "tuning-progress.json", candidates)
    chosen_lambda = max(candidates, key=lambda item: (item["ndcg@10"], -item["value"]))["value"]
    config["neural"]["regularization_weight"] = chosen_lambda
    for value in [0.1, 1.0, 10.0]:
        candidate = copy.deepcopy(config)
        candidate["experiment"].update({"methods": ["qrae"], "output": str(ROOT / "tuning-qrae")})
        candidate["neural"]["ranking_weight"] = value
        study = execute(candidate)
        result = read_json(study / "qrae-128d-seed42/result.json")
        candidates.append(
            {
                "method": "qrae",
                "value": value,
                "ndcg@10": result["metrics"]["ndcg@10"],
                "study": str(study),
            }
        )
        write_json(ROOT / "tuning-progress.json", candidates)
    chosen_alpha = max(
        [item for item in candidates if item["method"] == "qrae"],
        key=lambda item: (item["ndcg@10"], -item["value"]),
    )["value"]
    write_json(
        SELECTION,
        {
            "protocol": "docs/experiment-protocol.md",
            "protocol_sha256": sha256(Path("docs/experiment-protocol.md")),
            "orchestrator_sha256": sha256(Path(__file__)),
            "selected_on": "SciFact internal dev only,128d,seed42",
            "test_accessed": False,
            "regularization_weight": chosen_lambda,
            "ranking_weight": chosen_alpha,
            "margin": 0.1,
            "negative_strategy": "random",
            "candidates": candidates,
            "tie_rule": "highest dev ndcg@10, then smaller coefficient",
        },
    )
    print(f"Frozen hyperparameter selection: {SELECTION}", flush=True)


def selected_config(dataset):
    chosen = read_json(SELECTION)
    config = base_config(dataset)
    for key in ["regularization_weight", "ranking_weight", "margin"]:
        config["neural"][key] = chosen[key]
    return config


def _frozen_run_hashes(study, manifest):
    """Capture complete dev runs and every file, not just self-consistent weights."""
    if manifest["config"]["experiment"]["split"] != "dev":
        raise ValueError("Only completed dev studies may be frozen for final test")
    paths = sorted(study.glob("*/result.json"))
    results = {path.parent.name: read_json(path) for path in paths}
    exp = manifest["config"]["experiment"]
    expected_names = set()
    for method in exp["methods"]:
        if method == "original":
            originals = [r for r in results.values() if r["method"] == "original"]
            if len(originals) != 1:
                raise ValueError("Freezing requires the completed requested original baseline")
            dimensions = [originals[0]["dimension"]]
        else:
            dimensions = exp["dimensions"]
        seeds = [0] if method in {"original", "pca"} else exp["seeds"]
        expected_names.update(f"{method}-{dim}d-seed{seed}" for dim in dimensions for seed in seeds)
    if not expected_names or set(results) != expected_names:
        raise ValueError("Freezing requires every predeclared run and no unexpected runs")
    required = {
        "compressor.npz",
        "compressor.json",
        "per_query.json",
        "ranking.jsonl",
        "geometry.json",
        "index.faiss",
    }
    snapshot = {}
    for name, result in results.items():
        folder = study / name
        if result["study_id"] != manifest["study_id"] or result["split"] != "dev":
            raise ValueError(f"Run does not belong to the frozen dev study: {name}")
        if name != f"{result['method']}-{result['dimension']}d-seed{result['seed']}":
            raise ValueError(f"Run directory/metadata identity differs: {name}")
        if required - set(result["files"]):
            raise ValueError(f"Completed run lacks required artifact checksums: {name}")
        for filename, expected in result["files"].items():
            artifact = (folder / filename).resolve()
            if not artifact.is_relative_to(folder.resolve()) or not artifact.is_file():
                raise ValueError(f"Missing or invalid dev artifact: {name}/{filename}")
            if sha256(artifact) != expected:
                raise ValueError(
                    f"Dev artifact checksum differs before freezing: {name}/{filename}"
                )
        snapshot[name] = {
            "result_sha256": sha256(folder / "result.json"),
            "artifact_sha256": {
                path.relative_to(folder).as_posix(): sha256(path)
                for path in sorted(folder.rglob("*"))
                if path.is_file() and path != folder / "result.json"
            },
        }
    return snapshot


def record_dev(study, label):
    study = Path(study).resolve()
    manifest = read_json(study / "study.json")
    snapshot = {
        "schema_version": 2,
        "source_study": str(study),
        "study_manifest_sha256": sha256(study / "study.json"),
        "selection_sha256": sha256(SELECTION),
        "config": manifest["config"],
        "config_fingerprint": fingerprint(manifest["config"]),
        "runs": _frozen_run_hashes(study, manifest),
        "frozen_before_test": True,
        "freeze_policy": "immutable snapshot required before the managed final-test command",
    }
    target = ROOT / f"frozen-{label}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if read_json(target) == snapshot:
            return target  # Idempotent reuse preserves the original bytes and timestamp.
        raise FileExistsError(f"Frozen record already exists and cannot be replaced: {target}")
    serialized = json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # Exclusive creation prevents another process from overwriting a simultaneous freeze.
    with target.open("x", encoding="utf-8") as file:
        file.write(serialized)
    return target


def validate_frozen(frozen_path):
    """Reject changed results/weights even if their current internal hashes agree."""
    frozen = read_json(Path(frozen_path))
    if frozen.get("schema_version") != 2 or not frozen.get("runs"):
        raise ValueError("Frozen record lacks immutable per-run artifact hashes; freeze dev first")
    source = Path(frozen["source_study"])
    if sha256(source / "study.json") != frozen["study_manifest_sha256"]:
        raise ValueError("Frozen source study manifest has changed")
    if sha256(SELECTION) != frozen["selection_sha256"]:
        raise ValueError("Dev selection has changed since freezing")
    config = copy.deepcopy(frozen["config"])
    if fingerprint(config) != frozen["config_fingerprint"]:
        raise ValueError("Frozen configuration has changed")
    manifest = read_json(source / "study.json")
    if config != manifest["config"]:
        raise ValueError("Frozen configuration differs from its source study")
    if _frozen_run_hashes(source, manifest) != frozen["runs"]:
        raise ValueError("Frozen run results or artifacts have changed since freezing")
    return config, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["tune", "dev", "ablations", "test"])
    parser.add_argument("--dataset", choices=["scifact", "nfcorpus", "fiqa"], default="scifact")
    parser.add_argument("--frozen", help="Frozen dev manifest for explicit test evaluation")
    args = parser.parse_args()
    if args.stage == "tune":
        tune()
    elif args.stage == "dev":
        study = execute(selected_config(args.dataset))
        record_dev(study, args.dataset)
        print(study, flush=True)
    elif args.stage == "ablations":
        if args.dataset != "scifact":
            raise ValueError("Prespecified focused ablations are on SciFact only")
        config = selected_config("scifact")
        config["experiment"].update({"dimensions": [64, 128], "methods": ["qrae"]})
        for strategy in ["in_batch", "hard"]:
            candidate = copy.deepcopy(config)
            candidate["supervision"]["negative_strategy"] = strategy
            candidate["experiment"]["output"] = str(ROOT / f"scifact-{strategy}")
            study = execute(candidate)
            record_dev(study, f"scifact-{strategy}")
        config["experiment"].update(
            {
                "methods": ["rank_only"],
                "dimensions": [64, 128, 768],
                "output": str(ROOT / "scifact-rank-only"),
            }
        )
        study = execute(config)
        record_dev(study, "scifact-rank-only")
    else:
        if not args.frozen:
            raise ValueError("test requires an explicit --frozen manifest")
        config, source = validate_frozen(Path(args.frozen))
        config["experiment"]["split"] = "test"
        dataset = load_dataset(
            Path(config["data"]["directory"]) / config["data"]["name"], config["data"]
        )
        result = evaluate_saved(config, dataset, cache_path(config, dataset), source)
        write_report(result)
        print(result, flush=True)


if __name__ == "__main__":
    main()
