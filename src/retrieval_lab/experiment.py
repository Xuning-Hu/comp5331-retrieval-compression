"""Shared corpus-only fitting, exact retrieval, metrics and run provenance."""

from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path

import faiss
import numpy as np
from threadpoolctl import threadpool_limits

from .common import (
    environment,
    fingerprint,
    read_json,
    sha256,
    source_digest,
    unit_vectors,
    write_json,
)
from .compressors import Compressor
from .data import Dataset
from .embedding import validate_cache
from .metrics import evaluate, overlap, paired_bootstrap
from .search import ExactIndex

_DEPLOYMENT_FILES = ("compressor.npz", "compressor.json")
_TRAINING_FILES = ("training_decoder.npz", "training.csv", "supervision.json")
_EVALUATION_FILES = ("per_query.json", "ranking.jsonl", "geometry.json", "index.faiss")


def _embedding_identity(manifest: dict) -> dict:
    spec = manifest.get("spec", {})
    settings = spec.get("embedding", {})
    return {
        "model": settings.get("model"),
        "resolved_model_revision": manifest.get("resolved_model_revision"),
        "dimension": manifest.get("dimension"),
        "max_length": settings.get("max_length"),
        "query_prefix": settings.get("query_prefix"),
        "document_prefix": settings.get("document_prefix"),
        "preprocessing": spec.get("preprocessing"),
    }


def _verified_run(folder: Path) -> dict:
    """Check all recorded artifacts, including training diagnostics when present."""
    result = read_json(folder / "result.json")
    checksums = result["files"]
    required = set((*_DEPLOYMENT_FILES, *_EVALUATION_FILES))
    required.update(name for name in _TRAINING_FILES if (folder / name).exists())
    if required - set(checksums):
        raise ValueError(f"Run has artifacts without recorded checksums: {folder}")
    for name, expected in checksums.items():
        if Path(name).name != name or not (folder / name).is_file():
            raise ValueError(f"Missing or invalid run artifact: {folder / name}")
        if sha256(folder / name) != expected:
            raise ValueError(f"Existing run failed checksum validation: {folder / name}")
    return result


def _fit_compressor(method, dim, seed, x, config, training_data=None, training_metadata=None):
    compressor = Compressor(method, dim, seed)
    if method in {"rae", "qrae", "rank_only"}:
        return compressor.fit(
            x,
            config["ae"],
            training_data=training_data,
            neural_config={
                **config.get("neural", {}),
                "negative_strategy": config.get("supervision", {}).get(
                    "negative_strategy", "random"
                ),
            },
            training_metadata=training_metadata,
        )
    return compressor.fit(x, config["ae"])


def run(config: dict, dataset: Dataset, cache: Path) -> Path:
    """Fit requested compressors, then evaluate using the shared retrieval path."""
    return _execute(config, dataset, cache)


def evaluate_saved(config: dict, dataset: Dataset, cache: Path, source_study: Path) -> Path:
    """Evaluate immutable dev artifacts on another split without any fitting."""
    return _execute(config, dataset, cache, source_study=Path(source_study))


def _execute(config, dataset, cache, *, source_study=None) -> Path:
    embedding_manifest = validate_cache(cache)
    x = unit_vectors(np.load(cache / "documents.npy", allow_pickle=False))
    all_queries = np.load(cache / "queries.npy", allow_pickle=False)
    doc_ids = read_json(cache / "doc_ids.json")
    cached_qids = read_json(cache / "query_ids.json")
    if doc_ids != sorted(dataset.corpus) or len(x) != len(doc_ids):
        raise ValueError("Cached document order/count differs from corpus")
    exp = config["experiment"]
    qrels = dataset.qrels[exp["split"]]
    qids = sorted(qrels)
    if not qids:
        raise ValueError("Cannot evaluate an empty query split")
    if len(set(cached_qids)) != len(cached_qids) or len(all_queries) != len(cached_qids):
        raise ValueError("Cached query IDs/count are invalid")
    if set(qids) - set(cached_qids):
        raise ValueError("Evaluation query IDs are missing from the embedding cache")
    if all_queries.ndim != 2 or all_queries.shape[1] != x.shape[1]:
        raise ValueError("Cached query/document embedding dimensions differ")
    positions = {qid: i for i, qid in enumerate(cached_qids)}
    q = unit_vectors(all_queries[[positions[qid] for qid in qids]])
    if max(exp["dimensions"]) > x.shape[1]:
        raise ValueError("Requested compression dimension exceeds model output dimension")
    if max(exp["ks"]) > len(x):
        raise ValueError("Metric cutoff exceeds corpus size; use smaller ks")
    if len(x) < 11:
        raise ValueError("The overlap@10 protocol requires at least 11 corpus documents")
    jobs = []
    for method in exp["methods"]:
        dims = [x.shape[1]] if method == "original" else exp["dimensions"]
        seeds = [0] if method in {"original", "pca"} else exp["seeds"]
        jobs.extend((method, dim, seed) for dim in dims for seed in seeds)
    frozen_sources = {}
    source_record = None
    if source_study is not None:
        source = read_json(source_study / "study.json")
        if source["config"]["experiment"]["split"] != "dev":
            raise ValueError("Frozen evaluation requires a dev source study")
        for section in ["ae", "neural", "supervision"]:
            if source["config"].get(section) != config.get(section):
                raise ValueError(f"Frozen source training configuration differs: {section}")
        if source["dataset"] != dataset.manifest:
            raise ValueError("Frozen source dataset manifest differs from evaluation dataset")
        if source["embedding_files"] != embedding_manifest["files"]:
            raise ValueError("Frozen source embedding cache differs from evaluation cache")
        if source["document_ids_sha256"] != fingerprint(doc_ids):
            raise ValueError("Frozen source document IDs differ from evaluation corpus")
        source_identity = source.get("embedding_identity")
        if source_identity is None:
            source_identity = _embedding_identity(validate_cache(Path(source["embedding_cache"])))
        if source_identity != _embedding_identity(embedding_manifest):
            raise ValueError("Frozen source model/preprocessing identity differs")
        if exp["split"] == "test":
            development_ids = set(source["query_ids"]) | set(dataset.qrels.get("train", {}))
            if set(qids) & development_ids:
                raise ValueError("Test query IDs overlap the source dev or training queries")
        for method, dim, seed in jobs:
            name = f"{method}-{dim}d-seed{seed}"
            folder = source_study / name
            result = _verified_run(folder)
            if (result["method"], result["dimension"], result["seed"]) != (method, dim, seed):
                raise ValueError(f"Frozen source run identity differs: {name}")
            if result["study_id"] != source["study_id"]:
                raise ValueError(f"Frozen run does not belong to its source study: {name}")
            frozen_sources[name] = {
                "result_sha256": sha256(folder / "result.json"),
                "artifact_sha256": {
                    filename: result["files"][filename]
                    for filename in (*_DEPLOYMENT_FILES, *_TRAINING_FILES)
                    if filename in result["files"]
                },
            }
        source_record = {
            "study_id": source["study_id"],
            "study_manifest_sha256": sha256(source_study / "study.json"),
            "runs": frozen_sources,
        }
    observed_environment = environment()
    execution = {
        key: observed_environment[key]
        for key in ["python", "platform", "machine", "hardware", "packages"]
    }
    spec = {
        "config": config,
        "embedding_files": embedding_manifest["files"],
        "embedding_identity": _embedding_identity(embedding_manifest),
        "dataset": dataset.manifest,
        "source_sha256": source_digest(),
        "execution_environment": execution,
    }
    if source_record is not None:
        spec.update({"evaluation_mode": "frozen", "frozen_source": source_record})
    study_id = fingerprint(spec)[:12]
    root = Path(exp["output"]) / exp["split"] / study_id
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "study.json").exists():
        write_json(
            root / "study.json",
            {
                **spec,
                "environment": observed_environment,
                "study_id": study_id,
                "embedding_cache": str(cache.resolve()),
                "query_ids": qids,
                "document_ids_sha256": fingerprint(doc_ids),
                "protocol": "corpus-adaptive; training supervision documented per compressor; "
                "evaluation labels are not passed to training workers",
                "query_document_same_string_id": "retained; independent ID namespaces",
                **({"source_study": str(source_study.resolve())} if source_study else {}),
            },
        )
    threads = exp["threads"]
    search_depth = max(10, max(exp["ks"]))
    with threadpool_limits(limits=threads, user_api="blas"):
        original_index = ExactIndex(x, threads)
        original_scores, original_rows = original_index.search(q, search_depth)
        rankings = {qid: [doc_ids[row] for row in original_rows[i]] for i, qid in enumerate(qids)}
        _, reference = evaluate(qrels, rankings, exp["ks"])
        anchors = np.sort(
            np.random.default_rng(2026).choice(
                len(x), min(exp["geometry_anchors"], len(x)), replace=False
            )
        )
        geometry_k = min(10, len(x) - 1)
        _, original_doc_rows = original_index.search(x[anchors], geometry_k, anchors)
        supervision_by_seed = {}
        for method, dim, seed in jobs:
            folder = root / f"{method}-{dim}d-seed{seed}"
            if (folder / "result.json").exists():
                _verified_run(folder)
                print(f"Reusing complete run {folder.name}", flush=True)
                continue
            print(f"Running {method} {x.shape[1]}->{dim}, seed={seed}", flush=True)
            if source_study is None:
                training_data, training_metadata = None, None
                if method in {"qrae", "rank_only"}:
                    from .supervision import build_supervision

                    if seed not in supervision_by_seed:
                        supervision_by_seed[seed] = build_supervision(
                            dataset,
                            doc_ids,
                            cached_qids,
                            all_queries,
                            x,
                            config["supervision"],
                            seed,
                        )
                    training_data, training_metadata = supervision_by_seed[seed]
                compressor = _fit_compressor(
                    method, dim, seed, x, config, training_data, training_metadata
                )
                compressor.save(folder)
            else:
                source_folder = source_study / folder.name
                compressor = Compressor.load(source_folder)
                if (compressor.method, compressor.dimension, compressor.seed) != (
                    method,
                    dim,
                    seed,
                ):
                    raise ValueError("Frozen compressor metadata differs from requested run")
                if compressor.info["input_dimension"] != x.shape[1]:
                    raise ValueError("Frozen compressor input dimension differs from cache")
                folder.mkdir(parents=True, exist_ok=True)
                for name, checksum in frozen_sources[folder.name]["artifact_sha256"].items():
                    shutil.copy2(source_folder / name, folder / name)
                    if sha256(folder / name) != checksum:
                        raise ValueError(f"Frozen artifact changed during copying: {name}")
            start = time.perf_counter()
            xz = unit_vectors(compressor.transform(x))
            doc_transform = time.perf_counter() - start
            start = time.perf_counter()
            qz = unit_vectors(compressor.transform(q))
            query_transform = time.perf_counter() - start
            start = time.perf_counter()
            index = ExactIndex(xz, threads)
            index_seconds = time.perf_counter() - start
            scores, rows = index.search(qz, search_depth)
            rankings = {qid: [doc_ids[row] for row in rows[i]] for i, qid in enumerate(qids)}
            metrics, per_query = evaluate(qrels, rankings, exp["ks"])
            query_overlap = overlap(original_rows, rows, 10)
            _, doc_rows = index.search(xz[anchors], geometry_k, anchors)
            doc_overlap = overlap(original_doc_rows, doc_rows, geometry_k)
            for i, qid in enumerate(qids):
                per_query[qid]["query_doc_overlap@10"] = float(query_overlap[i])
            write_json(folder / "per_query.json", per_query)
            with (folder / "ranking.jsonl").open("w") as file:
                for i, qid in enumerate(qids):
                    file.write(
                        json.dumps(
                            {
                                "query_id": qid,
                                "document_ids": rankings[qid],
                                "scores": scores[i].tolist(),
                            }
                        )
                        + "\n"
                    )
            write_json(
                folder / "geometry.json",
                {
                    "anchor_document_ids": [doc_ids[row] for row in anchors],
                    "doc_doc_overlap": doc_overlap.tolist(),
                    "k": geometry_k,
                    "self_match": "excluded by row ID, not assumed to be first neighbor",
                },
            )
            faiss.write_index(index.index, str(folder / "index.faiss"))
            timing = index.benchmark(
                qz, max(exp["ks"]), exp["latency_queries"], exp["latency_repeats"]
            )
            timing.update(
                {
                    "fit_seconds": compressor.info["fit_seconds"] if source_study is None else 0.0,
                    "document_transform_seconds": doc_transform,
                    "query_batch_transform_seconds": query_transform,
                    "query_batch_transform_ms_per_query": query_transform / len(q) * 1000,
                    "index_build_seconds": index_seconds,
                }
            )
            result = {
                "dataset": dataset.name,
                "split": exp["split"],
                "method": method,
                "dimension": dim,
                "seed": seed,
                "documents": len(x),
                "queries": len(q),
                "metrics": metrics,
                "timing": timing,
                "query_doc_overlap@10": float(query_overlap.mean()),
                "doc_doc_overlap@10": float(doc_overlap.mean()),
                "storage": {
                    "vector_payload_bytes": int(xz.nbytes),
                    "index_file_bytes": (folder / "index.faiss").stat().st_size,
                    "compressor_file_bytes": (folder / "compressor.npz").stat().st_size,
                    "training_decoder_file_bytes": (
                        (folder / "training_decoder.npz").stat().st_size
                        if (folder / "training_decoder.npz").exists()
                        else 0
                    ),
                },
                "paired_vs_original": paired_bootstrap(
                    reference,
                    per_query,
                    "ndcg@10" if 10 in exp["ks"] else f"ndcg@{exp['ks'][0]}",
                    exp["bootstrap_samples"],
                ),
                "study_id": study_id,
                "evaluation_mode": "fitted" if source_study is None else "frozen",
                **(
                    {"frozen_source": frozen_sources[folder.name]}
                    if source_study is not None
                    else {}
                ),
                "files": {
                    name: sha256(folder / name)
                    for name in [
                        "compressor.npz",
                        "compressor.json",
                        "per_query.json",
                        "ranking.jsonl",
                        "geometry.json",
                        "index.faiss",
                        *[name for name in _TRAINING_FILES if (folder / name).exists()],
                    ]
                },
            }
            write_json(folder / "result.json", result)
            print(f"  {metrics}", flush=True)
        # Baseline reference is used even when not requested as a saved run.
        del original_index, original_scores
    return root


def demo(output: str) -> Path:
    """Synthetic executable integration check; never evidence of BEIR performance."""
    from .config import DEFAULTS

    config = copy.deepcopy(DEFAULTS)
    config["experiment"].update(
        {
            "output": output,
            "dimensions": [16, 8],
            "seeds": [17, 42],
            "ks": [10, 20],
            "geometry_anchors": 32,
            "latency_queries": 10,
            "latency_repeats": 1,
            "bootstrap_samples": 200,
        }
    )
    config["ae"].update({"steps": 200, "batch_size": 64, "log_every": 100})
    rng = np.random.default_rng(2026)
    basis = rng.normal(size=(8, 32))
    latent = rng.normal(size=(160, 8))
    x = unit_vectors(latent @ basis + 0.02 * rng.normal(size=(160, 32)))
    q = unit_vectors(x[:24] + 0.01 * rng.normal(size=(24, 32)))
    doc_ids = [f"doc-{i:04}" for i in range(len(x))]
    qids = [f"query-{i:04}" for i in range(len(q))]
    dataset = Dataset(
        "SYNTHETIC",
        {did: {} for did in doc_ids},
        dict.fromkeys(qids, "synthetic"),
        {"dev": {qid: {doc_ids[i]: 1} for i, qid in enumerate(qids)}},
        {"name": "SYNTHETIC", "seed": 2026, "not_a_real_benchmark": True},
    )
    cache = Path(output) / "synthetic-cache"
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "documents.npy", x)
    np.save(cache / "queries.npy", q)
    write_json(cache / "doc_ids.json", doc_ids)
    write_json(cache / "query_ids.json", qids)
    write_json(
        cache / "manifest.json",
        {
            "files": {
                name: sha256(cache / name)
                for name in ["documents.npy", "queries.npy", "doc_ids.json", "query_ids.json"]
            }
        },
    )
    return run(config, dataset, cache)
