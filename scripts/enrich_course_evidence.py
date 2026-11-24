"""Attach observed query/document/rank evidence to symmetric numeric cases.

Read only the explicitly requested split. Negative-quality summaries use saved
train-only supervision metadata and never use held-out qrels to clean negatives.
This script reports observations, not inferred causes or confirmed negative labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from retrieval_lab.common import fingerprint, read_json, sha256, write_json

CONTEXT_METHODS = ("original", "pca", "ae", "rae", "qrae", "rank_only")


def _verified(path, expected, consumed):
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"Evidence input checksum differs or file is missing: {path}")
    consumed[str(path.resolve())] = expected
    return path


def _text_rows(path, ids, consumed):
    wanted = set(ids)
    rows = {}
    if wanted:
        with path.open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                key = str(row["_id"])
                if key in wanted:
                    if key in rows:
                        raise ValueError(f"Duplicate requested text ID: {key}")
                    rows[key] = row
        if set(rows) != wanted:
            raise ValueError(f"Missing requested text IDs in {path}: {sorted(wanted - set(rows))}")
        consumed[str(path.resolve())] = sha256(path)
    return rows


def _positive_labels(folder, split, query_ids, consumed):
    path = folder / "qrels" / f"{split}.tsv"
    if split == "dev" and not path.exists():
        path = folder / "qrels" / "train.tsv"  # SciFact internal dev IDs were already fixed.
    labels = {qid: {} for qid in query_ids}
    with path.open(encoding="utf-8") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            if row["query-id"] in labels and int(row["score"]) > 0:
                labels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    if any(not row for row in labels.values()):
        raise ValueError(f"Case query lacks positive relevance judgments in {path}")
    consumed[str(path.resolve())] = sha256(path)
    return labels


def _distribution(values):
    if not values:
        return None
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite supervision diagnostic")
    return {
        "count": len(values),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def audit_query_text_overlap(folder, data_config, consumed, expected_files=None):
    """Post-freeze descriptive audit; never alter splits, labels or primary results."""
    split_ids = {}
    for split in ("train", "dev", "test"):
        path = folder / "qrels" / f"{split}.tsv"
        if not path.exists():
            continue
        if expected_files is not None:
            expected = expected_files.get(f"qrels/{split}.tsv")
            if expected is None:
                raise ValueError("Query-overlap input is absent from the frozen dataset manifest")
            _verified(path, expected, consumed)
        with path.open(encoding="utf-8") as file:
            split_ids[split] = {row["query-id"] for row in csv.DictReader(file, delimiter="\t")}
        consumed[str(path.resolve())] = sha256(path)
    if "dev" not in split_ids:
        ids = sorted(split_ids["train"])
        np.random.default_rng(data_config["split_seed"]).shuffle(ids)
        count = max(1, round(len(ids) * data_config["dev_fraction"]))
        split_ids["dev"] = set(ids[:count])
        split_ids["train"] = set(ids[count:])
    if set(split_ids) != {"train", "dev", "test"}:
        raise ValueError("Query-overlap audit requires complete train/dev/test ID sets")
    texts = _text_rows(folder / "queries.jsonl", set.union(*split_ids.values()), consumed)
    output = {
        "dataset": data_config["name"],
        "split_counts": {key: len(ids) for key, ids in split_ids.items()},
        "normalization": "strip and collapse whitespace; case and punctuation retained",
        "primary_splits_changed": False,
        "primary_results_changed": False,
        "relevance_scores_used": False,
    }
    for label, normalize in (
        ("exact", lambda text: text),
        ("normalized_whitespace", lambda text: " ".join(text.strip().split())),
    ):
        grouped = {}
        for split, ids in split_ids.items():
            mapping = defaultdict(list)
            for qid in sorted(ids):
                mapping[normalize(texts[qid]["text"])].append(qid)
            grouped[split] = mapping
        pairs = {}
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            shared = sorted(set(grouped[left]) & set(grouped[right]))
            pairs[f"{left}__{right}"] = {
                "id_overlap_count": len(split_ids[left] & split_ids[right]),
                "shared_text_count": len(shared),
                "left_queries": sum(len(grouped[left][text]) for text in shared),
                "right_queries": sum(len(grouped[right][text]) for text in shared),
                "query_id_pair_count": sum(
                    len(grouped[left][text]) * len(grouped[right][text]) for text in shared
                ),
                "groups": [
                    {
                        "text_sha256": fingerprint(text),
                        "left_query_ids": grouped[left][text],
                        "right_query_ids": grouped[right][text],
                    }
                    for text in shared
                ],
            }
        output[label] = pairs
    output["interpretation"] = (
        "Cross-split text duplication is an interpretation limit, not permission to revise frozen splits, clean negatives or retune on test."
    )
    return output


def summarize_negatives(metadata, identity):
    strategy = metadata["negative_strategy"]
    qids = metadata["query_ids"]
    hard = metadata.get("hard_mining", {})
    candidates = hard.get("candidates_by_query", {})
    is_hard = strategy == "hard"
    sizes = [len(candidates.get(qid, [])) for qid in qids] if is_hard else []
    flat = [candidate for qid in qids for candidate in candidates.get(qid, [])] if is_hard else []
    blocked = metadata.get("blocked_document_ids_by_query", {})
    blocked_hits = (
        sum(
            candidate["document_id"] in set(blocked.get(qid, []))
            for qid in qids
            for candidate in candidates.get(qid, [])
        )
        if is_hard
        else None
    )
    return {
        **identity,
        "negative_strategy": strategy,
        "train_queries": metadata["train_queries"],
        "positive_pairs": metadata["positive_pairs"],
        "pair_weighting": metadata.get("pair_weighting"),
        "negatives_per_positive": metadata.get("negatives_per_positive"),
        "document_count": len(metadata.get("document_ids", [])),
        "duplicate_group_count": metadata.get("duplicate_group_count"),
        "nontrivial_duplicate_groups": len(metadata.get("duplicate_groups", [])),
        "documents_in_nontrivial_duplicate_groups": sum(
            len(group) for group in metadata.get("duplicate_groups", [])
        ),
        "duplicate_rule": metadata.get("duplicate_rule"),
        "hard_pool_sizes": _distribution(sizes),
        "hard_full_corpus_ranks": _distribution([row["full_corpus_rank"] for row in flat]),
        "hard_teacher_cosines": _distribution([row["cosine"] for row in flat]),
        "empty_hard_pools": sum(size == 0 for size in sizes) if is_hard else None,
        "hard_candidates_blocked_by_train_metadata": blocked_hits,
        "hard_top_k": hard.get("top_k") if is_hard else None,
        "hard_rank_min": hard.get("rank_min") if is_hard else None,
        "hard_rank_max": hard.get("rank_max") if is_hard else None,
        "in_batch_fallback_policy": metadata.get("in_batch_fallback")
        if strategy == "in_batch"
        else None,
        "observed_fallback_count": metadata.get("in_batch_fallback_count")
        if strategy == "in_batch"
        else None,
        "observed_fallback_rate": metadata.get("in_batch_fallback_rate")
        if strategy == "in_batch"
        else None,
        "fallback_measurement_status": "not logged; no rate inferred"
        if strategy == "in_batch" and "in_batch_fallback_count" not in metadata
        else "not applicable"
        if strategy != "in_batch"
        else "recorded metadata",
        "false_negative_rate": None,
        "quality_interpretation": "Pool sizes/ranks/cosines are construction diagnostics, not confirmed irrelevance. No held-out qrels used for negative cleaning.",
    }


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        if not fields:
            return
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _markdown(path, cases):
    def clean(text):
        return str(text).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    lines = [
        "# Observed course retrieval cases",
        "",
        "Cases were selected symmetrically from seed-mean metric differences. Text and rank changes are observations, not causal explanations. Original/PCA use deterministic seed0; learned-method context uses the requested seed. Missing saved ranks are censored, not exact ranks.",
        "",
    ]
    for case in cases:
        lines += [
            f"## {case['dataset']} / {case['split']} / {case['dimension']}d / query {case['query_id']} / {case['direction']}",
            "",
            f"Comparison: {clean(case['candidate_label'])} minus {clean(case['reference_label'])}.",
            "",
            f"Seed-mean {case['metric']} delta: {case['delta']:+.6f}.",
            "",
            f"Query: {clean(case['query_text'])}",
            "",
            f"Known relevant documents: {case['known_relevant_document_count']}; displaying {len(case['relevant_documents'])} by the stated preview rule.",
            "",
        ]
        for document in case["relevant_documents"]:
            lines += [
                f"- {clean(document['document_id'])}, grade {document['relevance_grade']}: **{clean(document['title'])}**. {clean(document['report_excerpt'])}"
            ]
        lines += [
            "",
            "| Method / configuration | Seed | Saved relevant-document ranks |",
            "|---|---:|---|",
        ]
        preview_ids = [document["document_id"] for document in case["relevant_documents"]]
        for context in case["ranking_context"]:
            if not context["available"]:
                lines.append(
                    f"| {clean(context.get('label', context['method']))} | {context['requested_seed']} | {clean(context['reason'])} |"
                )
                continue
            ranks = "; ".join(
                f"{did}: {context['relevant_ranks'][did]['display']}" for did in preview_ids
            )
            lines.append(f"| {clean(context['label'])} | {context['seed']} | {clean(ranks)} |")
        lines += [
            "",
            "Interpretation boundary: this table documents relevance/rank changes. It does not identify why the model changed them.",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def enrich(
    analysis_dir,
    output,
    *,
    split="dev",
    context_seed=42,
    max_cases=3,
    snippet_chars=300,
    max_documents=10,
    repo_root=None,
    audit_query_overlap=False,
):
    if split not in {"dev", "test"} or min(max_cases, snippet_chars, max_documents) < 1:
        raise ValueError("Choose dev/test and positive evidence limits")
    if audit_query_overlap and split != "test":
        raise ValueError(
            "Cross-split query audit is reserved for explicit post-freeze test evidence"
        )
    analysis_dir, output = Path(analysis_dir).resolve(), Path(output).resolve()
    repo_root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[1]
    manifest_path = analysis_dir / "analysis-manifest.json"
    numeric_path = analysis_dir / "case-selection.json"
    summary_path = analysis_dir / "summary.json"
    manifest, numeric, summaries = (
        read_json(manifest_path),
        read_json(numeric_path),
        read_json(summary_path),
    )
    consumed = {str(path): sha256(path) for path in (manifest_path, numeric_path, summary_path)}
    groups = {row["group_id"]: row for row in summaries if row["split"] == split}
    source_entries = {row["study_key"]: row for row in manifest["studies"] if row["split"] == split}
    sources = {}
    for key, entry in source_entries.items():
        root = Path(entry["source"]).resolve()
        if output == root or output.is_relative_to(root) or output == analysis_dir:
            raise ValueError(
                "Evidence output must not overwrite a source study or numeric analysis"
            )
        metadata_path = _verified(root / "study.json", entry["manifest_sha256"], consumed)
        study = read_json(metadata_path)
        if study["config"]["experiment"]["split"] != split:
            raise ValueError("Requested evidence split disagrees with its source study")
        folder = Path(study["config"]["data"]["directory"])
        if not folder.is_absolute():
            folder = repo_root / folder
        folder = folder / study["config"]["data"]["name"]
        data_hashes = {
            name.replace("\\", "/"): value
            for name, value in study["dataset"].get("files", {}).items()
        }
        qrels_name = f"qrels/{split}.tsv"
        if split == "dev" and not (folder / qrels_name).exists():
            qrels_name = "qrels/train.tsv"
        for filename in ("corpus.jsonl", "queries.jsonl", qrels_name):
            if filename not in data_hashes:
                raise ValueError(f"Text evidence requires a recorded dataset checksum: {filename}")
            _verified(folder / filename, data_hashes[filename], consumed)
        comparison_identity = fingerprint(
            {
                "dataset": study["dataset"],
                "embedding_files": study["embedding_files"],
                "query_ids": study["query_ids"],
                "split": split,
            }
        )
        sources[key] = {
            "root": root,
            "study": study,
            "folder": folder,
            "identity": comparison_identity,
            "checked": entry["checked_inputs"],
            "data_hashes": data_hashes,
        }
    run_cache, ranking_cache = {}, {}

    def read_run(group, name):
        key = (group["study_key"], name)
        if key not in run_cache:
            source = sources[group["study_key"]]
            path = source["root"] / name / "result.json"
            expected = source["checked"].get(f"{name}/result.json")
            if expected is None:
                raise ValueError(
                    "Text/rank evidence requires original raw runs, not score-only snapshots"
                )
            result = read_json(_verified(path, expected, consumed))
            if (result["method"], result["dimension"], result["study_id"]) != (
                group["method"],
                group["dimension"],
                group["study_id"],
            ):
                raise ValueError("Evidence run disagrees with numeric summary identity")
            run_cache[key] = result
        return run_cache[key]

    def ranking_for(group, name):
        key = (group["study_key"], name)
        if key not in ranking_cache:
            result = read_run(group, name)
            path = sources[group["study_key"]]["root"] / name / "ranking.jsonl"
            _verified(path, result["files"]["ranking.jsonl"], consumed)
            rankings = {}
            with path.open(encoding="utf-8") as file:
                for line in file:
                    row = json.loads(line)
                    if row["query_id"] in rankings:
                        raise ValueError("Duplicate query in saved ranking")
                    rankings[row["query_id"]] = row["document_ids"]
            ranking_cache[key] = rankings
        return ranking_cache[key]

    selected = []
    requested = defaultdict(set)
    for comparison in numeric:
        if comparison["split"] != split:
            continue
        if (
            comparison["candidate_group"] not in groups
            or comparison["reference_group"] not in groups
        ):
            raise ValueError("Case-selection group is absent from numeric summary")
        for direction, field in (
            ("improvement", "largest_improvements"),
            ("regression", "largest_regressions"),
        ):
            for row in comparison[field][:max_cases]:
                if (direction == "improvement" and row["delta"] <= 0) or (
                    direction == "regression" and row["delta"] >= 0
                ):
                    raise ValueError("Numeric case direction disagrees with its delta")
                group = groups[comparison["candidate_group"]]
                source = sources[group["study_key"]]
                if row["query_id"] not in source["study"]["query_ids"]:
                    raise ValueError("Case query does not belong to its saved evaluation split")
                selected.append(
                    {
                        **{
                            key: value
                            for key, value in comparison.items()
                            if key not in {"largest_improvements", "largest_regressions"}
                        },
                        **row,
                        "direction": direction,
                        "source_key": group["study_key"],
                    }
                )
                requested[source["folder"]].add(row["query_id"])
    labels_by_folder = {
        folder: _positive_labels(folder, split, ids, consumed) for folder, ids in requested.items()
    }
    queries_by_folder = {
        folder: _text_rows(folder / "queries.jsonl", ids, consumed)
        for folder, ids in requested.items()
    }
    documents_wanted = defaultdict(set)
    cases = []
    for selected_case in selected:
        source = sources[selected_case["source_key"]]
        qid, dimension = selected_case["query_id"], selected_case["dimension"]
        positives = labels_by_folder[source["folder"]][qid]
        contexts = []
        for method in CONTEXT_METHODS:
            candidates = [
                group
                for group in groups.values()
                if group["method"] == method
                and (method == "original" or group["dimension"] == dimension)
                and sources[group["study_key"]]["identity"] == source["identity"]
            ]
            requested_seed = 0 if method in {"original", "pca"} else context_seed
            if not candidates:
                contexts.append(
                    {
                        "method": method,
                        "available": False,
                        "requested_seed": requested_seed,
                        "reason": "No matching method/dimension study supplied",
                    }
                )
            for group in candidates:
                matching = [
                    (name, read_run(group, name))
                    for name in group["run_names"]
                    if read_run(group, name)["seed"] == requested_seed
                ]
                if not matching:
                    contexts.append(
                        {
                            "method": method,
                            "label": group["label"],
                            "available": False,
                            "requested_seed": requested_seed,
                            "reason": "Requested seed run is unavailable; no substitute seed chosen",
                        }
                    )
                    continue
                if len(matching) != 1:
                    raise ValueError("Multiple matching context runs for one group/seed")
                name, result = matching[0]
                ranks = ranking_for(group, name)
                if qid not in ranks:
                    raise ValueError("Case query missing from a matched saved ranking")
                docs = ranks[qid]
                rank_by_id = {did: rank for rank, did in enumerate(docs[:100], 1)}
                depth = min(100, len(docs))
                contexts.append(
                    {
                        "method": method,
                        "label": group["label"],
                        "group_id": group["group_id"],
                        "study_id": group["study_id"],
                        "run": name,
                        "available": True,
                        "seed": result["seed"],
                        "saved_depth": len(docs),
                        "reported_depth": depth,
                        "relevant_ranks": {
                            did: {
                                "rank": rank_by_id.get(did),
                                "display": str(rank_by_id[did])
                                if did in rank_by_id
                                else f">{depth}"
                                if depth == 100
                                else f">{depth} (only {depth} saved)",
                                "censored": did not in rank_by_id,
                            }
                            for did in positives
                        },
                    }
                )

        def best_rank(did):
            ranks = [
                context["relevant_ranks"][did]["rank"]
                for context in contexts
                if context["available"] and context["relevant_ranks"][did]["rank"] is not None
            ]
            return min(ranks) if ranks else 10**9

        preview = sorted(positives, key=lambda did: (best_rank(did), -positives[did], did))[
            :max_documents
        ]
        documents_wanted[source["folder"]].update(preview)
        cases.append(
            {
                **{key: value for key, value in selected_case.items() if key != "source_key"},
                "source_key": selected_case["source_key"],
                "query_text": queries_by_folder[source["folder"]][qid]["text"],
                "known_relevant_document_count": len(positives),
                "known_relevance_grades": positives,
                "preview_document_ids": preview,
                "preview_rule": "smallest observed relevant rank across shown contexts, then relevance grade descending, then ID; all known grades/ranks retained in JSON",
                "ranking_context": contexts,
                "interpretation": "Observed score/rank/text evidence only; no causal mechanism inferred",
            }
        )
    texts_by_folder = {
        folder: _text_rows(folder / "corpus.jsonl", ids, consumed)
        for folder, ids in documents_wanted.items()
    }
    for case in cases:
        folder = sources[case.pop("source_key")]["folder"]
        case["relevant_documents"] = [
            {
                "document_id": did,
                "relevance_grade": case["known_relevance_grades"][did],
                "title": texts_by_folder[folder][did].get("title", ""),
                "snippet": texts_by_folder[folder][did].get("text", "")[:snippet_chars],
                "report_excerpt": " ".join(
                    texts_by_folder[folder][did].get("text", "").split()[:20]
                ),
                "report_excerpt_scope": "At most 20 whitespace-delimited words; observation only",
                "snippet_scope": f"First {snippet_chars} characters of stored corpus text; not the exact model tokenizer window",
            }
            for did in case.pop("preview_document_ids")
        ]
    negatives = []
    for group in groups.values():
        if group["method"] not in {"qrae", "rank_only"}:
            continue
        for name in group["run_names"]:
            result = read_run(group, name)
            path = sources[group["study_key"]]["root"] / name / "supervision.json"
            if "supervision.json" not in result["files"]:
                raise ValueError(f"Supervised run lacks registered training metadata: {name}")
            metadata = read_json(_verified(path, result["files"]["supervision.json"], consumed))
            negatives.append(
                summarize_negatives(
                    metadata,
                    {
                        "dataset": group["dataset"],
                        "evaluation_split": split,
                        "group_id": group["group_id"],
                        "label": group["label"],
                        "study_id": group["study_id"],
                        "run": name,
                        "seed": result["seed"],
                        "dimension": group["dimension"],
                        "source_metadata_sha256": result["files"]["supervision.json"],
                    },
                )
            )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "enriched-cases.json", cases)
    _markdown(output / "case-evidence.md", cases)
    write_json(output / "negative-quality.json", negatives)
    _csv(output / "negative-quality.csv", negatives)
    query_overlap = []
    if audit_query_overlap:
        configs_by_folder = {
            source["folder"]: (source["study"]["config"]["data"], source["data_hashes"])
            for source in sources.values()
        }
        query_overlap = [
            audit_query_text_overlap(folder, config, consumed, files)
            for folder, (config, files) in configs_by_folder.items()
        ]
        write_json(output / "query-text-overlap.json", query_overlap)
    write_json(
        output / "evidence-manifest.json",
        {
            "schema_version": 1,
            "split": split,
            "context_seed": context_seed,
            "max_cases_per_direction": max_cases,
            "max_document_previews": max_documents,
            "snippet_characters": snippet_chars,
            "case_count": len(cases),
            "negative_metadata_count": len(negatives),
            "cross_split_query_text_audit": audit_query_overlap,
            "consumed_file_sha256": consumed,
            "negative_cleaning_uses_held_out_qrels": False,
            "script_sha256": sha256(Path(__file__)),
            "notes": [
                "Raw ranking evidence requires original study artifacts; exported score-only snapshots are insufficient.",
                "Case delta is a seed mean; contextual learned-method ranks use the requested single seed and are not that mean.",
                "All supplied matching variants retain their configuration labels; missing methods/seeds are explicit.",
                "Absent documents are censored at 100 only when at least 100 saved ranks exist.",
                "Fallback occurrence/rate and true false-negative rate are not inferred when not logged.",
                "No causal explanations or new model/parameter selection are performed.",
            ],
        },
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--context-seed", type=int, default=42)
    parser.add_argument("--cases-per-direction", type=int, default=3)
    parser.add_argument("--snippet-characters", type=int, default=300)
    parser.add_argument("--max-document-previews", type=int, default=10)
    parser.add_argument(
        "--audit-query-overlap",
        action="store_true",
        help="Post-freeze test evidence only: describe cross-split text overlap without changing results",
    )
    args = parser.parse_args()
    print(
        enrich(
            args.analysis,
            args.output,
            split=args.split,
            context_seed=args.context_seed,
            max_cases=args.cases_per_direction,
            snippet_chars=args.snippet_characters,
            max_documents=args.max_document_previews,
            audit_query_overlap=args.audit_query_overlap,
        )
    )


if __name__ == "__main__":
    main()
