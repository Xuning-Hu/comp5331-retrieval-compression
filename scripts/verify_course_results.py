"""Independently recompute saved-ranking metrics without retrieval_lab.metrics.

Only the requested evaluation split and development qrels are opened. SciFact's
internal dev split is reconstructed from official train IDs. This is a saved-rank
audit, not a model/index replay or proof of the historical execution environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from retrieval_lab.common import fingerprint, read_json, sha256, write_json

DEPLOYMENT = {"compressor.npz", "compressor.json"}
TRAINING = {"training_decoder.npz", "training.csv", "supervision.json"}
REQUIRED = DEPLOYMENT | {"per_query.json", "ranking.jsonl", "geometry.json", "index.faiss"}
TOLERANCE = 1e-12


def scalar_metrics(relevance, documents, cutoffs):
    """Linear relevance gain; binary positive recall; first-positive reciprocal rank."""
    if len(documents) != len(set(documents)):
        raise ValueError("Duplicate document in saved ranking")
    positive_grades = sorted((grade for grade in relevance.values() if grade > 0), reverse=True)
    output = {}
    for cutoff in cutoffs:
        numerator = 0.0
        hits = 0
        first = 0
        for rank, document in enumerate(documents[:cutoff], start=1):
            grade = relevance.get(document, 0)
            if grade > 0:
                numerator += grade / math.log2(rank + 1)
                hits += 1
                if not first:
                    first = rank
        denominator = sum(
            grade / math.log2(rank + 1)
            for rank, grade in enumerate(positive_grades[:cutoff], start=1)
        )
        output[f"ndcg@{cutoff}"] = numerator / denominator if denominator else 0.0
        output[f"recall@{cutoff}"] = hits / len(positive_grades) if positive_grades else 0.0
        output[f"mrr@{cutoff}"] = 1.0 / first if first else 0.0
    return output


class Auditor:
    def __init__(self, repo_root):
        self.repo_root = Path(repo_root).resolve()
        self.checked = {}
        self._hash_cache = {}
        self._dataset_cache = {}

    def verify_file(self, path, expected=None):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f"Missing audit input: {path}")
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self._hash_cache:
            self._hash_cache[key] = sha256(path)
        actual = self._hash_cache[key]
        if expected is not None and actual != expected:
            raise ValueError(f"Artifact checksum differs: {path}")
        self.checked[str(path)] = actual
        return actual

    def artifacts(self, folder, result):
        required = set(REQUIRED)
        method = result["method"]
        if method in {"ae", "rae", "qrae"}:
            required.add("training_decoder.npz")
        if method in {"ae", "rae", "qrae", "rank_only"}:
            required.add("training.csv")
        if method in {"qrae", "rank_only"}:
            required.add("supervision.json")
        files = result["files"]
        if required - set(files):
            raise ValueError(
                f"Missing registered artifacts: {folder}: {sorted(required - set(files))}"
            )
        for name, expected in files.items():
            path = (folder / name).resolve()
            if not path.is_relative_to(folder.resolve()):
                raise ValueError(f"Registered artifact escapes run directory: {name}")
            self.verify_file(path, expected)
        return self.verify_file(folder / "result.json")

    def dataset(self, study):
        config = study["config"]["data"]
        root = Path(config["directory"])
        if not root.is_absolute():
            root = self.repo_root / root
        root = root / config["name"]
        split = study["config"]["experiment"]["split"]
        key = fingerprint([str(root.resolve()), config, study["dataset"], split])
        if key in self._dataset_cache:
            return self._dataset_cache[key]
        hashes = {
            name.replace("\\", "/"): value for name, value in study["dataset"]["files"].items()
        }

        def registered(name):
            if name not in hashes:
                raise ValueError(f"Dataset input lacks recorded checksum: {name}")
            self.verify_file(root / name, hashes[name])
            return root / name

        def ids(name):
            output = set()
            with registered(name).open(encoding="utf-8") as file:
                for line in file:
                    document = str(json.loads(line)["_id"])
                    if document in output:
                        raise ValueError(f"Duplicate ID in {name}: {document}")
                    output.add(document)
            return output

        document_ids, query_ids = ids("corpus.jsonl"), ids("queries.jsonl")

        def qrels(name):
            output = {}
            with registered(f"qrels/{name}.tsv").open(encoding="utf-8") as file:
                for row in csv.DictReader(file, delimiter="\t"):
                    qid, did, grade = row["query-id"], row["corpus-id"], int(row["score"])
                    if qid not in query_ids or did not in document_ids or grade < 0:
                        raise ValueError("Invalid official relevance judgment")
                    relevance = output.setdefault(qid, {})
                    if did in relevance and relevance[did] != grade:
                        raise ValueError("Conflicting duplicate official relevance judgment")
                    relevance[did] = grade
            return output

        splits = {"train": qrels("train")}
        dev_rule = "official dev.tsv"
        if (root / "qrels/dev.tsv").exists():
            splits["dev"] = qrels("dev")
        else:
            train_ids = sorted(splits["train"])
            np.random.default_rng(config["split_seed"]).shuffle(train_ids)
            count = max(1, round(len(train_ids) * config["dev_fraction"]))
            selected = set(train_ids[:count])
            splits["dev"] = {qid: splits["train"][qid] for qid in selected}
            splits["train"] = {
                qid: values for qid, values in splits["train"].items() if qid not in selected
            }
            dev_rule = f"official train IDs; default_rng({config['split_seed']}); fraction={config['dev_fraction']}"
        if split == "test":
            splits["test"] = qrels("test")
        if split not in {"dev", "test"}:
            raise ValueError("Only explicit dev/test studies are supported")
        names = list(splits)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                if set(splits[left]) & set(splits[right]):
                    raise ValueError("Official/rebuilt split query IDs overlap")
        output = {"documents": document_ids, "splits": splits, "dev_rule": dev_rule}
        self._dataset_cache[key] = output
        return output

    def frozen_source(self, root, study, result, name):
        if study.get("evaluation_mode") != "frozen" or result.get("evaluation_mode") != "frozen":
            raise ValueError("Final test requires frozen study and run metadata")
        source_root = Path(study["source_study"])
        source_record = study["frozen_source"]
        self.verify_file(source_root / "study.json", source_record["study_manifest_sha256"])
        source = read_json(source_root / "study.json")
        if (
            source["study_id"] != source_record["study_id"]
            or source["config"]["experiment"]["split"] != "dev"
        ):
            raise ValueError("Frozen source is not the recorded dev study")
        for field in ("dataset", "embedding_files", "embedding_identity", "document_ids_sha256"):
            if source[field] != study[field]:
                raise ValueError(f"Frozen source/test identity differs: {field}")
        for section in ("ae", "neural", "supervision"):
            if source["config"].get(section) != study["config"].get(section):
                raise ValueError(f"Frozen training configuration differs: {section}")
        if set(source["query_ids"]) & set(study["query_ids"]):
            raise ValueError("Frozen test overlaps source dev query IDs")
        record = source_record["runs"][name]
        if result.get("frozen_source") != record:
            raise ValueError("Run/study frozen source records disagree")
        self.verify_file(source_root / name / "result.json", record["result_sha256"])
        source_result = read_json(source_root / name / "result.json")
        if (
            source_result["study_id"],
            source_result["method"],
            source_result["dimension"],
            source_result["seed"],
        ) != (source["study_id"], result["method"], result["dimension"], result["seed"]):
            raise ValueError("Source run identity differs")
        expected_files = (DEPLOYMENT | TRAINING) & set(source_result["files"])
        if set(record["artifact_sha256"]) != expected_files or not DEPLOYMENT <= expected_files:
            raise ValueError("Frozen source artifact list is incomplete")
        for filename in expected_files:
            expected = record["artifact_sha256"][filename]
            if (
                source_result["files"][filename] != expected
                or result["files"].get(filename) != expected
            ):
                raise ValueError("Frozen source/test artifact hashes disagree")
            self.verify_file(source_root / name / filename, expected)
            self.verify_file(root / name / filename, expected)
        if result["timing"]["fit_seconds"] != 0:
            raise ValueError("Frozen evaluation records nonzero fitting time")
        return {
            "source_study": str(source_root),
            "matched_artifact_count": len(expected_files),
            "matched_files": sorted(expected_files),
        }


def _close(actual, expected, context):
    if (
        not math.isfinite(actual)
        or not math.isfinite(expected)
        or abs(actual - expected) > TOLERANCE
    ):
        raise ValueError(f"Independent metric mismatch at {context}: {actual} versus {expected}")
    return abs(actual - expected)


def _rankings(path, query_ids, document_ids, depth):
    output = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            qid, documents = row["query_id"], row["document_ids"]
            if qid in output or len(documents) != depth or len(set(documents)) != depth:
                raise ValueError("Duplicate query/document or incorrect ranking depth")
            if set(documents) - document_ids:
                raise ValueError("Saved ranking contains document absent from official corpus")
            if "scores" in row:
                scores = row["scores"]
                if len(scores) != depth or not all(math.isfinite(score) for score in scores):
                    raise ValueError("Invalid saved ranking scores")
                if any(a < b for a, b in zip(scores, scores[1:])):
                    raise ValueError("Saved ranking scores are not descending")
                if any(
                    a == b and left > right
                    for a, b, left, right in zip(scores, scores[1:], documents, documents[1:])
                ):
                    raise ValueError("Saved exact-score ties violate document-ID order")
            output[qid] = documents
    if set(output) != set(query_ids):
        raise ValueError("Saved ranking query IDs disagree with official/rebuilt split")
    return output


def verify(studies, output, *, repo_root=None):
    paths = [Path(path).resolve() for path in studies]
    if len(set(paths)) != len(paths) or not paths:
        raise ValueError("Supply distinct, nonempty study paths")
    output = Path(output).resolve()
    if any(output == path or output.is_relative_to(path) for path in paths):
        raise ValueError("Audit output must be outside source studies")
    auditor = Auditor(repo_root or Path(__file__).resolve().parents[1])
    loaded, references, shared_queries = [], {}, {}
    for root in paths:
        auditor.verify_file(root / "study.json")
        study = read_json(root / "study.json")
        exp = study["config"]["experiment"]
        data = auditor.dataset(study)
        qids = sorted(data["splits"][exp["split"]])
        if study["query_ids"] != qids or not qids:
            raise ValueError("Study query IDs disagree with official/rebuilt split")
        if study["document_ids_sha256"] != fingerprint(sorted(data["documents"])):
            raise ValueError("Study document IDs disagree with official corpus")
        shared_key = (study["config"]["data"]["name"], exp["split"])
        query_hash = fingerprint(qids)
        if shared_key in shared_queries and shared_queries[shared_key] != query_hash:
            raise ValueError("Same dataset/split studies have different query IDs")
        shared_queries[shared_key] = query_hash
        identity = fingerprint(
            {
                "dataset": study["dataset"],
                "embedding_files": study["embedding_files"],
                "embedding_identity": study["embedding_identity"],
                "split": exp["split"],
                "query_ids": qids,
            }
        )
        runs = {path.parent.name: read_json(path) for path in sorted(root.glob("*/result.json"))}
        expected = set()
        for method in exp["methods"]:
            dimensions = (
                [study["embedding_identity"]["dimension"]]
                if method == "original"
                else exp["dimensions"]
            )
            seeds = [0] if method in {"original", "pca"} else exp["seeds"]
            expected.update(f"{method}-{dim}d-seed{seed}" for dim in dimensions for seed in seeds)
        if not expected or set(runs) != expected:
            raise ValueError(
                f"Incomplete or unexpected predeclared jobs: {root}; missing={sorted(expected - set(runs))}; extra={sorted(set(runs) - expected)}"
            )
        depth = max(10, max(exp["ks"]))
        if depth > len(data["documents"]):
            raise ValueError("Ranking depth exceeds official corpus")
        for name, result in runs.items():
            if name != f"{result['method']}-{result['dimension']}d-seed{result['seed']}":
                raise ValueError("Run directory/metadata identity differs")
            if (
                result["study_id"],
                result["dataset"],
                result["split"],
                result["queries"],
                result["documents"],
            ) != (
                study["study_id"],
                shared_key[0],
                exp["split"],
                len(qids),
                len(data["documents"]),
            ):
                raise ValueError("Run identity/counts disagree with study/official data")
            auditor.artifacts(root / name, result)
            if result["method"] == "original":
                ranks = _rankings(root / name / "ranking.jsonl", qids, data["documents"], depth)
                if identity in references and references[identity] != ranks:
                    raise ValueError(
                        "Supplied Original rankings disagree for the same evaluation identity"
                    )
                references[identity] = ranks
        loaded.append((root, study, data, runs, identity))

    reports = []
    for root, study, data, runs, identity in loaded:
        exp = study["config"]["experiment"]
        qrels = data["splits"][exp["split"]]
        qids = sorted(qrels)
        run_reports = []
        for name, result in runs.items():
            folder = root / name
            rankings = _rankings(
                folder / "ranking.jsonl", qids, data["documents"], max(10, max(exp["ks"]))
            )
            saved = read_json(folder / "per_query.json")
            if set(saved) != set(qids):
                raise ValueError("Per-query metric IDs disagree with official split")
            keys = {
                f"{metric}@{cutoff}" for cutoff in exp["ks"] for metric in ("ndcg", "recall", "mrr")
            }
            if set(result["metrics"]) != keys:
                raise ValueError("Saved metric keys disagree with predeclared cutoffs")
            totals = dict.fromkeys(keys, 0.0)
            error = 0.0
            overlap_values = []
            for qid in qids:
                if set(saved[qid]) != keys | {"query_doc_overlap@10"}:
                    raise ValueError("Unexpected/missing per-query metric keys")
                recomputed = scalar_metrics(qrels[qid], rankings[qid], exp["ks"])
                for key, value in recomputed.items():
                    error = max(error, _close(value, saved[qid][key], f"{name}/{qid}/{key}"))
                    totals[key] += value
                overlap = saved[qid]["query_doc_overlap@10"]
                if not math.isfinite(overlap) or not 0 <= overlap <= 1:
                    raise ValueError("Invalid saved query overlap")
                overlap_values.append(overlap)
                if identity in references:
                    independent = (
                        len(set(rankings[qid][:10]) & set(references[identity][qid][:10])) / 10
                    )
                    error = max(error, _close(independent, overlap, f"{name}/{qid}/overlap"))
            aggregate = {key: value / len(qids) for key, value in totals.items()}
            for key, value in aggregate.items():
                error = max(error, _close(value, result["metrics"][key], f"{name}/{key}"))
            error = max(
                error,
                _close(
                    sum(overlap_values) / len(qids),
                    result["query_doc_overlap@10"],
                    f"{name}/mean_query_overlap",
                ),
            )
            geometry = read_json(folder / "geometry.json")
            values = geometry["doc_doc_overlap"]
            if (
                not values
                or len(values) != len(geometry["anchor_document_ids"])
                or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values)
            ):
                raise ValueError("Invalid saved geometry diagnostics")
            error = max(
                error,
                _close(
                    sum(values) / len(values),
                    result["doc_doc_overlap@10"],
                    f"{name}/mean_doc_overlap",
                ),
            )
            storage = result["storage"]
            if storage["vector_payload_bytes"] != len(data["documents"]) * result["dimension"] * 4:
                raise ValueError("Vector payload bytes do not match float32 corpus dimensions")
            for field, filename in (
                ("index_file_bytes", "index.faiss"),
                ("compressor_file_bytes", "compressor.npz"),
                ("training_decoder_file_bytes", "training_decoder.npz"),
            ):
                actual = (folder / filename).stat().st_size if (folder / filename).is_file() else 0
                if storage[field] != actual:
                    raise ValueError(f"Saved storage size differs: {name}/{field}")
            supervision = None
            if result["method"] in {"qrae", "rank_only"}:
                metadata = read_json(folder / "supervision.json")
                ids = metadata["query_ids"]
                if (
                    len(ids) != len(set(ids))
                    or len(ids) != metadata["train_queries"]
                    or set(ids) - set(data["splits"]["train"])
                ):
                    raise ValueError(
                        "Training supervision query IDs are not restricted to official/rebuilt train"
                    )
                supervision = {
                    "query_count": len(ids),
                    "only_train": True,
                    "negative_strategy": metadata["negative_strategy"],
                }
            frozen = (
                auditor.frozen_source(root, study, result, name) if exp["split"] == "test" else None
            )
            run_reports.append(
                {
                    "run": name,
                    "method": result["method"],
                    "dimension": result["dimension"],
                    "seed": result["seed"],
                    "metrics": aggregate,
                    "maximum_error": error,
                    "result_sha256": auditor.checked[str((folder / "result.json").resolve())],
                    "registered_artifacts": result["files"],
                    "query_overlap_independently_recomputed": identity in references,
                    "geometry_check": "saved per-anchor aggregation only; neighbors not replayed",
                    "supervision": supervision,
                    "frozen_source": frozen,
                }
            )
        reports.append(
            {
                "source": str(root),
                "study_id": study["study_id"],
                "study_manifest_sha256": auditor.checked[str(root / "study.json")],
                "dataset": study["config"]["data"]["name"],
                "split": exp["split"],
                "query_count": len(qids),
                "query_ids_sha256": fingerprint(qids),
                "document_count": len(data["documents"]),
                "dev_rule": data["dev_rule"],
                "predeclared_jobs": len(runs),
                "runs": run_reports,
            }
        )
        print(
            f"Verified {study['config']['data']['name']} {exp['split']}: {len(runs)} runs, {len(qids)} queries",
            flush=True,
        )
    report = {
        "schema_version": 1,
        "status": "passed",
        "study_count": len(reports),
        "run_count": sum(len(row["runs"]) for row in reports),
        "maximum_error": max(run["maximum_error"] for row in reports for run in row["runs"]),
        "tolerance": TOLERANCE,
        "metric_definition": "linear-gain nDCG; positive-grade Recall and MRR; macro mean over official/rebuilt query IDs",
        "implementation": "Python scalar math.log2/sums; no retrieval_lab.metrics.evaluate, sklearn dense label matrix, training or search replay",
        "historical_freeze_boundary": "Checks recorded frozen-source hashes and identical dev/test artifacts; historical freeze ordering remains supported by orchestrator/pipeline records, not established by this audit alone",
        "query_overlap_unreplayed_runs": sum(
            not run["query_overlap_independently_recomputed"]
            for row in reports
            for run in row["runs"]
        ),
        "studies": reports,
        "consumed_file_sha256": auditor.checked,
        "script_sha256": sha256(Path(__file__)),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "verification.json", report)
    lines = [
        "# Independent saved-ranking verification",
        "",
        f"Status: passed. Studies: {report['study_count']}; runs: {report['run_count']}; maximum absolute metric/aggregation error: {report['maximum_error']:.3g}.",
        "",
        report["implementation"],
        "",
        report["historical_freeze_boundary"],
        "",
        f"Runs without a supplied matching Original ranking for independent query-overlap recomputation: {report['query_overlap_unreplayed_runs']}. Saved geometry values were aggregated; geometry neighbors were not regenerated.",
        "",
        "| Dataset | Split | Runs | Queries |",
        "|---|---|---:|---:|",
    ]
    lines += [
        f"| {row['dataset']} | {row['split']} | {row['predeclared_jobs']} | {row['query_count']} |"
        for row in reports
    ]
    (output / "verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(verify(args.studies, args.output))


if __name__ == "__main__":
    main()
