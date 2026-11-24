"""Export every seed's small course evidence, excluding vectors/indexes/rankings.

Source runs must be complete and their registered artifacts intact. An export
supports saved-score aggregation; full-ranking verification requires local raw
runs. Selection, frozen-dev records and independent verification remain explicit.
"""

from __future__ import annotations

import argparse
import csv
import os
import runpy
import shutil
from pathlib import Path

from retrieval_lab.common import fingerprint, read_json, sha256, write_json

ANALYSIS_PATH = Path(__file__).with_name("analyze_course.py")


def _relative_reference(path, output):
    try:
        return Path(os.path.relpath(path, output)).as_posix()
    except ValueError:
        return str(path)


def export(study_path, output, *, selection=None, frozen=None, verification=None):
    root, output = Path(study_path).resolve(), Path(output).resolve()
    if root == output or output.is_relative_to(root):
        raise ValueError("Snapshot output must be outside the raw source study")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Snapshot output is nonempty; retain it and choose a new label")
    helpers = runpy.run_path(str(ANALYSIS_PATH))
    loaded = helpers["load_study"](root)
    if not list(root.glob("*/result.json")):
        raise ValueError("Course export requires complete original raw studies")
    study = read_json(root / "study.json")
    cache_manifest_path = Path(study["embedding_cache"]) / "manifest.json"
    cache_manifest = read_json(cache_manifest_path)
    if cache_manifest["files"] != study["embedding_files"]:
        raise ValueError("Embedding manifest disagrees with the source study")
    consumed = {
        str(root / "study.json"): sha256(root / "study.json"),
        str(cache_manifest_path.resolve()): sha256(cache_manifest_path),
    }
    consumed.update(
        {
            str((root / relative).resolve()): checksum
            for relative, checksum in loaded["checked_inputs"].items()
        }
    )
    runs = {run["name"]: run for group in loaded["groups"] for run in group["runs"]}
    results = {name: row["result"] for name, row in sorted(runs.items())}
    per_query = {name: row["scores"] for name, row in sorted(runs.items())}
    compressor_info = {name: row["info"] for name, row in sorted(runs.items())}

    def checked(path, expected=None):
        path = Path(path).resolve()
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Snapshot provenance checksum differs: {path}")
        consumed[str(path)] = actual
        return actual

    if frozen is not None:
        frozen = Path(frozen).resolve()
        record = read_json(frozen)
        if record.get("schema_version") != 2 or not record.get("frozen_before_test"):
            raise ValueError("Expected an immutable course frozen-dev record")
        source = Path(record["source_study"]).resolve()
        expected_source = (
            root if loaded["split"] == "dev" else Path(study["source_study"]).resolve()
        )
        if source != expected_source:
            raise ValueError("Frozen record belongs to a different source study")
        checked(source / "study.json", record["study_manifest_sha256"])
        if (
            fingerprint(record["config"]) != record["config_fingerprint"]
            or record["config"] != read_json(source / "study.json")["config"]
        ):
            raise ValueError("Frozen configuration identity differs")
        source_names = {path.parent.name for path in source.glob("*/result.json")}
        if source_names != set(record["runs"]):
            raise ValueError("Frozen record does not cover every source run")
        for name, registered in record["runs"].items():
            folder = source / name
            checked(folder / "result.json", registered["result_sha256"])
            observed_files = {
                path.relative_to(folder).as_posix()
                for path in folder.rglob("*")
                if path.is_file() and path.name != "result.json"
            }
            if observed_files != set(registered["artifact_sha256"]):
                raise ValueError("Frozen source file inventory has changed")
            for filename, expected in registered["artifact_sha256"].items():
                path = (folder / filename).resolve()
                if not path.is_relative_to(folder):
                    raise ValueError("Frozen artifact escapes source directory")
                checked(path, expected)
        checked(frozen)
    if selection is not None:
        selection = Path(selection).resolve()
        checked(selection, read_json(frozen)["selection_sha256"] if frozen else None)

    verification_reference = None
    if verification is not None:
        verification = Path(verification).resolve()
        audit = read_json(verification)
        matching = [
            row
            for row in audit.get("studies", [])
            if Path(row["source"]).resolve() == root and row["study_id"] == study["study_id"]
        ]
        if audit.get("status") != "passed" or len(matching) != 1:
            raise ValueError(
                "Independent verification does not identify this complete source study"
            )
        verified = matching[0]
        if verified["study_manifest_sha256"] != sha256(root / "study.json"):
            raise ValueError("Independent verification predates a changed study manifest")
        verified_runs = {row["run"]: row for row in verified["runs"]}
        if set(verified_runs) != set(runs):
            raise ValueError("Independent verification does not cover all exported runs")
        for name, row in verified_runs.items():
            if (
                row["result_sha256"] != sha256(root / name / "result.json")
                or row["registered_artifacts"] != results[name]["files"]
            ):
                raise ValueError("Independent verification predates changed run artifacts")
        verification_reference = {
            "path": str(verification),
            "relative_path": _relative_reference(verification, output),
            "sha256": checked(verification),
            "study_id": study["study_id"],
            "run_count": len(verified_runs),
            "maximum_error": max(row["maximum_error"] for row in verified_runs.values()),
            "scope": "Independent metrics were computed from local official qrels and full saved rankings; rankings are not included in this snapshot",
        }

    histories = []
    for name, run in sorted(runs.items()):
        path = root / name / "training.csv"
        if not path.exists():
            continue
        if "training.csv" not in run["result"]["files"]:
            raise ValueError("Training CSV exists without a registered source hash")
        identity = {
            "run_directory": name,
            "study_id": study["study_id"],
            "dataset": loaded["dataset"],
            "split": loaded["split"],
            "method": run["result"]["method"],
            "dimension": run["result"]["dimension"],
            "seed": run["result"]["seed"],
        }
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if set(identity) & set(row):
                    raise ValueError("Training columns conflict with export identity fields")
                histories.append({**identity, **row})

    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "study.json", output / "study.json")
    shutil.copy2(cache_manifest_path, output / "embedding-manifest.json")
    write_json(output / "run-results.json", results)
    write_json(output / "per-query.json", per_query)
    write_json(output / "compressor-metadata.json", compressor_info)
    summaries = [helpers["summarize"](group) for group in loaded["groups"]]
    write_json(output / "summary.json", summaries)
    helpers["_csv"](output / "summary.csv", summaries)
    run_rows = []
    for name, result in results.items():
        run_rows.append(
            {
                "run_directory": name,
                "study_id": study["study_id"],
                "dataset": loaded["dataset"],
                "split": loaded["split"],
                "method": result["method"],
                "dimension": result["dimension"],
                "seed": result["seed"],
                **result["metrics"],
                "query_doc_overlap@10": result["query_doc_overlap@10"],
                "doc_doc_overlap@10": result["doc_doc_overlap@10"],
                **result["storage"],
                **result["timing"],
            }
        )
    helpers["_csv"](output / "runs.csv", run_rows)
    helpers["_csv"](output / "training.csv", histories)
    for source, name in ((selection, "selection.json"), (frozen, "frozen-manifest.json")):
        if source is not None:
            shutil.copy2(source, output / name)
    if verification_reference:
        write_json(output / "verification-reference.json", verification_reference)
    lines = [
        "# Course evidence snapshot",
        "",
        f"{loaded['dataset']} / {loaded['split']} / study `{study['study_id']}`: all {len(runs)} declared runs, including every recorded seed. No best-seed selection.",
        "",
        "This export supports independent saved-score aggregation and inspection of compressor metadata/training traces. It excludes model arrays, indexes, the original corpus, and full rankings. Recomputing metrics from rankings requires the local raw study and official qrels.",
        "",
        "Summary groups use actual compressor metadata and keep distinct configurations separate. Search timing covers the recorded search-only scope; index plus compressor storage is distinct from training-only decoder storage and total RAM.",
        "",
        "All registered raw artifacts were checksum-checked at export. Checksums establish identity at export, not historical execution order.",
        "",
    ]
    if verification_reference:
        lines.append(
            f"[Independent saved-ranking verification]({verification_reference['relative_path']}) (SHA-256 `{verification_reference['sha256']}`)."
        )
    else:
        lines.append(
            "No independent full-ranking verification artifact was attached to this export."
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output / "snapshot.json",
        {
            "schema_version": 2,
            "source_study": str(root),
            "source_sha256": study.get("source_sha256"),
            "source_study_manifest_sha256": sha256(root / "study.json"),
            "scope": f"{loaded['dataset']} / {loaded['split']}; all predeclared jobs and seeds",
            "run_count": len(runs),
            "methods": sorted({result["method"] for result in results.values()}),
            "source_config_fingerprint": fingerprint(study["config"]),
            "effective_parameters": "compressor-metadata.json plus summary group parameters; do not infer effective losses solely from requested config",
            "verification_boundary": "saved-score reaggregation available; raw rankings/index/model replay unavailable from export alone",
            "raw_artifact_validation": loaded["verification_boundary"],
            "training_rows": len(histories),
            "verification": verification_reference,
            "source_inputs_sha256": consumed,
            "files": {
                path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()
            },
            "script_sha256": sha256(Path(__file__)),
            "analysis_helper_sha256": sha256(ANALYSIS_PATH),
        },
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--verification", type=Path)
    args = parser.parse_args()
    print(
        export(
            args.study,
            args.output,
            selection=args.selection,
            frozen=args.frozen,
            verification=args.verification,
        )
    )


if __name__ == "__main__":
    main()
