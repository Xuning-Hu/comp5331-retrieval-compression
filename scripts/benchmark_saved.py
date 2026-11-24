"""Time real single-query requests against immutable saved retrieval artifacts.

Run only after training has finished. A persistent Torch-only encoder process
keeps model loading out of timing and isolates Torch from the FAISS parent.
Total latency includes this prototype's JSON/pipe IPC, not production serving.
No training, relevance scoring, parameter selection, or source writes occur.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from retrieval_lab.common import (
    environment,
    fingerprint,
    read_json,
    sha256,
    unit_vectors,
    write_json,
)


def encode_worker(metadata_path: Path, device: str, threads: int) -> None:
    """Stdout is exclusively a line-delimited JSON protocol; logs go to stderr."""
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from sentence_transformers import SentenceTransformer

        from retrieval_lab.compressors import select_device

        metadata = read_json(metadata_path)
        settings = metadata["spec"]["embedding"]
        resolved_device = select_device(device)
        torch.set_num_threads(threads)
        start = time.perf_counter()
        model = SentenceTransformer(
            settings["model"],
            revision=metadata["resolved_model_revision"],
            device=resolved_device,
            trust_remote_code=False,
            local_files_only=True,
        )
        model.max_seq_length = settings["max_length"]
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        hardware = {"device": resolved_device, "torch_version": torch.__version__}
        if resolved_device.startswith("cuda"):
            properties = torch.cuda.get_device_properties(torch.device(resolved_device))
            hardware.update(
                gpu_name=properties.name, gpu_total_memory_bytes=properties.total_memory
            )
        ready = {
            "type": "ready",
            "hardware": hardware,
            "threads": threads,
            "model": settings["model"],
            "revision": metadata["resolved_model_revision"],
            "model_load_seconds_excluded": time.perf_counter() - start,
            "environment": environment(),
        }
    print(json.dumps(ready, allow_nan=False), file=protocol, flush=True)

    def synchronize():
        if resolved_device.startswith("cuda"):
            torch.cuda.synchronize()
        elif resolved_device == "mps":
            torch.mps.synchronize()

    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if request.get("type") == "stop":
                break
            if not isinstance(request.get("text"), str):
                raise ValueError("Each encode request must contain text")
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                synchronize()
                start = time.perf_counter()
                vector = model.encode(
                    [settings["query_prefix"] + request["text"]],
                    batch_size=1,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                synchronize()
                duration = (time.perf_counter() - start) * 1000
            response = {
                "type": "result",
                "id": request["id"],
                "vector": np.asarray(vector[0], dtype=np.float32).tolist(),
                "encode_ms": duration,
            }
        except Exception as error:
            response = {"type": "error", "id": request.get("id"), "error": str(error)}
        print(json.dumps(response, allow_nan=False), file=protocol, flush=True)


class EncoderWorker:
    """Persistent request/response worker with bounded waits and safe cleanup."""

    def __init__(self, command: list[str], timeout: float = 180):
        self.timeout = timeout
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=dict(os.environ, PYTHONUTF8="1", HF_HUB_OFFLINE="1"),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.messages = queue.Queue()

        def read_output():
            try:
                for line in self.process.stdout:
                    self.messages.put(line)
            finally:
                self.messages.put(None)

        threading.Thread(target=read_output, daemon=True).start()
        try:
            self.ready = self._receive()
            if self.ready.get("type") != "ready":
                raise RuntimeError(f"Encoder failed to initialize: {self.ready}")
        except Exception:
            self.close()
            raise

    def _receive(self):
        try:
            line = self.messages.get(timeout=self.timeout)
        except queue.Empty as error:
            raise TimeoutError("Timed out waiting for the encoder worker") from error
        if line is None:
            raise RuntimeError(f"Encoder closed its output; exit={self.process.poll()}")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError("Encoder stdout was not valid protocol JSON") from error
        if response.get("type") == "error":
            raise RuntimeError(f"Encoder request failed: {response.get('error')}")
        return response

    def encode(self, text: str, request_id: str):
        self.process.stdin.write(json.dumps({"id": request_id, "text": text}) + "\n")
        self.process.stdin.flush()
        response = self._receive()
        if response.get("type") != "result" or response.get("id") != request_id:
            raise RuntimeError("Encoder response does not match the request")
        return response

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"type":"stop"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()


def choose_schedule(qids, run_names, query_count=30, repeats=3, seed=2026):
    """Same selected queries per method; randomized method order per query/repeat."""
    if query_count < 1 or repeats < 1 or not qids or not run_names:
        raise ValueError("Queries, runs, counts and repeats must be nonempty/positive")
    rng = np.random.default_rng(seed)
    selected = rng.choice(sorted(qids), size=min(query_count, len(qids)), replace=False).tolist()
    schedule = []
    for repeat in range(repeats):
        for qid in rng.permutation(selected).tolist():
            for name in rng.permutation(sorted(run_names)).tolist():
                schedule.append({"repeat": repeat, "query_id": qid, "run": name})
    return selected, schedule


def load_index_view(path: Path, threads: int):
    """Keep ExactIndex tie handling while supporting Unicode paths on Windows."""
    import faiss

    from retrieval_lab.search import ExactIndex

    class SavedIndex(ExactIndex):
        def __init__(self):
            faiss.omp_set_num_threads(threads)
            # FAISS's narrow-character fopen cannot read some Windows Unicode paths.
            self.index = faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype=np.uint8))
            self.vectors = self.index.reconstruct_n(0, self.index.ntotal)

    return SavedIndex()


def load_saved(study_path: Path, dimension=128, run_seed=42, names=None):
    """Validate immutable source identities before loading any timed artifacts."""
    from retrieval_lab.compressors import Compressor
    from retrieval_lab.data import load_dataset
    from retrieval_lab.embedding import validate_cache
    from retrieval_lab.experiment import _verified_run

    study_path = study_path.resolve()
    study = read_json(study_path / "study.json")
    cache = Path(study["embedding_cache"]).resolve()
    metadata = validate_cache(cache)
    if metadata["files"] != study["embedding_files"]:
        raise ValueError("Embedding cache differs from saved study")
    if "spec" not in metadata or not metadata.get("resolved_model_revision"):
        raise ValueError("Live text timing requires a real frozen-model embedding cache")
    ids = read_json(cache / "doc_ids.json")
    if fingerprint(ids) != study["document_ids_sha256"]:
        raise ValueError("Document ID ordering differs from saved study")
    settings = study["config"]["data"]
    data_path = (Path(settings["directory"]) / settings["name"]).resolve()
    dataset = load_dataset(data_path, settings)
    if dataset.manifest != study["dataset"] or ids != sorted(dataset.corpus):
        raise ValueError("Source data differs from saved study")
    if set(study["query_ids"]) - set(dataset.queries):
        raise ValueError("Saved study queries are missing from source data")
    width = metadata["dimension"]
    config = study["config"]["experiment"]
    expected = [
        f"{method}-{dim}d-seed{seed}"
        for method in config["methods"]
        for dim in ([width] if method == "original" else config["dimensions"])
        for seed in ([0] if method in {"original", "pca"} else config["seeds"])
    ]
    if any(not (study_path / name / "result.json").is_file() for name in expected):
        raise ValueError("Source study is incomplete; finish training before timing")
    if names is None:
        names = [
            f"{method}-{width if method == 'original' else dimension}d-"
            f"seed{0 if method in {'original', 'pca'} else run_seed}"
            for method in config["methods"]
        ]
        if "original" not in config["methods"]:
            raise ValueError("Default comparison requires an Original baseline; specify --runs")
    if len(set(names)) != len(names) or not names:
        raise ValueError("Selected run names must be nonempty and unique")
    loaded, hashes = {}, {}
    for name in names:
        if Path(name).name != name or name not in expected:
            raise ValueError(f"Selected run is not in this study: {name}")
        folder = study_path / name
        result = _verified_run(folder)
        if result["study_id"] != study["study_id"]:
            raise ValueError(f"Run belongs to another study: {name}")
        compressor = Compressor.load(folder)
        index = load_index_view(folder / "index.faiss", config["threads"])
        if index.index.ntotal != len(ids) or index.index.d != compressor.dimension:
            raise ValueError(f"Saved index shape disagrees with compressor/corpus: {name}")
        loaded[name] = (compressor, index)
        hashes[name] = {"result.json": sha256(folder / "result.json"), **result["files"]}
    cached_qids = read_json(cache / "query_ids.json")
    cached_queries = np.load(cache / "queries.npy", allow_pickle=False)
    if len(cached_qids) != len(cached_queries) or len(set(cached_qids)) != len(cached_qids):
        raise ValueError("Invalid cached query ID/vector alignment")
    lookup = {qid: i for i, qid in enumerate(cached_qids)}
    if set(study["query_ids"]) - set(lookup):
        raise ValueError("Study query is missing from the immutable embedding cache")
    return {
        "study_path": study_path,
        "study": study,
        "cache": cache,
        "metadata": metadata,
        "data_path": data_path,
        "dataset": dataset,
        "runs": loaded,
        "cached_queries": cached_queries,
        "query_positions": lookup,
        "artifact_hashes": hashes,
        "study_manifest_sha256": sha256(study_path / "study.json"),
        "cache_manifest_sha256": sha256(cache / "manifest.json"),
    }


def measure_request(worker, compressor, index, text, request_id, k, cached_query):
    start = time.perf_counter()
    response = worker.encode(text, request_id)
    vector = np.asarray(response["vector"], dtype=np.float32)[None, :]
    if vector.shape != (1, compressor.info["input_dimension"]):
        raise ValueError("Live model output dimension differs from saved compressor")
    projection_start = time.perf_counter()
    projected = unit_vectors(compressor.transform(unit_vectors(vector)))
    search_start = time.perf_counter()
    _, rows = index.search(projected, k)
    end = time.perf_counter()
    total = (end - start) * 1000
    projection = (search_start - projection_start) * 1000
    search = (end - search_start) * 1000
    encoding = float(response["encode_ms"])
    if not np.isfinite(encoding) or encoding < 0:
        raise ValueError("Encoder returned invalid duration")
    # Numeric diagnostics are outside the timed request and do not select parameters.
    actual, expected = unit_vectors(vector)[0], unit_vectors(cached_query[None, :])[0]
    return {
        "encode_ms": encoding,
        "projection_ms": projection,
        "search_ms": search,
        "total_ms": total,
        "encode_roundtrip_ms": (projection_start - start) * 1000,
        "ipc_and_python_overhead_ms": total - encoding - projection - search,
        "live_cached_cosine": float(actual @ expected),
        "live_cached_max_abs_difference": float(np.max(np.abs(actual - expected))),
        "ranking_rows_sha256": fingerprint(rows[0].tolist()),
    }


def summarize(measurements):
    fields = [
        "encode_ms",
        "projection_ms",
        "search_ms",
        "total_ms",
        "encode_roundtrip_ms",
        "ipc_and_python_overhead_ms",
    ]
    summary = {}
    for name in sorted({row["run"] for row in measurements}):
        rows = [row for row in measurements if row["run"] == name]
        summary[name] = {"requests": len(rows)}
        for field in fields:
            values = np.asarray([row[field] for row in rows])
            summary[name][field] = {
                "p50": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3, help="Untimed requests per method")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encoder-threads", type=int, default=4)
    parser.add_argument("--worker-timeout", type=float, default=180)
    parser.add_argument("--encode-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--metadata", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.encode_worker:
        if args.metadata is None:
            parser.error("--encode-worker requires --metadata")
        encode_worker(args.metadata, args.device, args.encoder_threads)
        return
    if args.study is None or args.output is None:
        parser.error("--study and --output are required")
    if min(args.queries, args.repeats, args.top_k, args.encoder_threads) < 1 or args.warmup < 1:
        parser.error("Counts, threads, top-k and warmup must be positive")
    if args.seed < 0 or args.worker_timeout <= 0:
        parser.error("Seed must be nonnegative and worker timeout positive")
    loaded = load_saved(args.study, args.dimension, args.run_seed, args.runs)
    output = args.output.resolve()
    for protected in (loaded["study_path"], loaded["cache"], loaded["data_path"]):
        if output.is_relative_to(protected):
            raise ValueError(
                "Timing output must be outside source study, cache and data directories"
            )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing timing artifact: {output}")
    qids, schedule = choose_schedule(
        loaded["study"]["query_ids"], loaded["runs"], args.queries, args.repeats, args.seed
    )
    k = min(args.top_k, next(iter(loaded["runs"].values()))[1].index.ntotal)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--encode-worker",
        "--metadata",
        str(loaded["cache"] / "manifest.json"),
        "--device",
        args.device,
        "--encoder-threads",
        str(args.encoder_threads),
    ]
    worker = EncoderWorker(command, args.worker_timeout)
    measurements = []
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(
            limits=loaded["study"]["config"]["experiment"]["threads"], user_api="blas"
        ):
            warm_rng = np.random.default_rng(args.seed + 1)
            for cycle in range(args.warmup):
                for name in warm_rng.permutation(sorted(loaded["runs"])):
                    qid = qids[cycle % len(qids)]
                    model, index = loaded["runs"][name]
                    measure_request(
                        worker,
                        model,
                        index,
                        loaded["dataset"].queries[qid],
                        f"warmup-{cycle}-{name}",
                        k,
                        loaded["cached_queries"][loaded["query_positions"][qid]],
                    )
            for order, item in enumerate(schedule):
                qid, name = item["query_id"], item["run"]
                model, index = loaded["runs"][name]
                timing = measure_request(
                    worker,
                    model,
                    index,
                    loaded["dataset"].queries[qid],
                    str(order),
                    k,
                    loaded["cached_queries"][loaded["query_positions"][qid]],
                )
                measurements.append({"order": order, **item, **timing})
    finally:
        worker.close()
    # The output is independent from all source result files and never changes scores.
    result = {
        "schema_version": 1,
        "source_study": str(loaded["study_path"]),
        "study_id": loaded["study"]["study_id"],
        "environment": environment(),
        "encoder": worker.ready,
        "query_ids": qids,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"metadata", "encode_worker"}
        },
        "effective_queries": len(qids),
        "effective_top_k": k,
        "source_hashes": {
            "study_manifest": loaded["study_manifest_sha256"],
            "cache_manifest": loaded["cache_manifest_sha256"],
            "embedding_files": loaded["metadata"]["files"],
            "run_artifacts": loaded["artifact_hashes"],
            "benchmark_script": sha256(Path(__file__)),
        },
        "scope": {
            "total": "single-query request serialization + pipe IPC + fresh frozen-model encoding "
            "+ vector decoding + projection/normalization + CPU exact search/canonical ties",
            "encoding": "worker tokenization/model/pooling/normalization/CPU transfer; synchronized",
            "excluded": "model/index loading, warmup, corpus ingestion, returned-document rendering",
            "interpretation": "prototype IPC-inclusive latency, not production serving latency; "
            "no concurrent training; descriptive p50/p95 over repeated fixed queries",
            "methods": "all indices resident; randomized method order for each query/repeat",
        },
        "summary": summarize(measurements),
        "measurements": measurements,
    }
    write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
