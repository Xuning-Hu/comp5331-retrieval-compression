"""Prepare an explicitly documented replacement IMDb/MPNet reproduction dataset.

This does not recover the RAE authors' original embeddings. Their arXiv v1 names
MPNet but does not identify an immutable checkpoint or the sampled review IDs.
Only standard archive/network tools and the existing embedding dependencies are
used; no datasets package or third-party RAE implementation is required.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import ssl
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SOURCE_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
SOURCE_PAGE = "https://ai.stanford.edu/~amaas/data/sentiment/"
PAPER_URL = "https://arxiv.org/html/2509.25839v1"
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
REVIEW_NAME = re.compile(r"aclImdb/train/(?:pos|neg)/\d+_\d+\.txt\Z")


def checksum(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl_records(path):
    """Preserve Unicode line-separator characters inside JSON string values."""
    # str.splitlines() recognizes U+0085/U+2028/U+2029 as line boundaries too.
    # A JSONL record delimiter is a physical newline, not those string contents.
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def download_archive(destination, source_url=SOURCE_URL, attempts=3):
    """Write atomically; report progress without extracting archive members."""
    destination = Path(destination)
    if destination.exists():
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    if not source_url.startswith("https://"):
        raise ValueError("Archive downloads require HTTPS with certificate verification")
    error = None
    for attempt in range(attempts):
        # Retry a transient configured-proxy failure, then try a process-local
        # direct route. Never alter Windows proxy settings or certificate checks.
        direct = attempt == attempts - 1 and attempts > 1
        handler = urllib.request.ProxyHandler({} if direct else urllib.request.getproxies())
        opener = urllib.request.build_opener(
            handler, urllib.request.HTTPSHandler(context=ssl.create_default_context())
        )
        request = urllib.request.Request(source_url, headers={"User-Agent": "COMP5331-reproduction/1"})
        count = 0
        try:
            with opener.open(request, timeout=45) as response, temporary.open("wb") as stream:
                final_url = response.url
                if not final_url.startswith("https://"):
                    raise ValueError("Refusing a download redirected away from HTTPS")
                expected = int(response.headers.get("Content-Length", "0"))
                while block := response.read(1024 * 1024):
                    stream.write(block)
                    count += len(block)
                    if count // (10 * 1024 * 1024) != (count - len(block)) // (10 * 1024 * 1024):
                        print(f"IMDb download: {count / 1024**2:.1f} MiB", flush=True)
            if not count or (expected and count != expected):
                raise IOError(f"Incomplete archive: received {count} bytes, expected {expected}")
            temporary.replace(destination)
            write_json(destination.parent / "archive_download.json", {
                "requested_url": source_url, "final_url": final_url,
                "bytes": count, "tls_verification": "system trusted certificates; enabled",
                "route": "direct" if direct else "configured proxy",
            })
            return
        except (OSError, ValueError) as exc:
            error = exc
            print(f"Download attempt {attempt + 1}/{attempts} failed: {type(exc).__name__}", flush=True)
            if attempt + 1 < attempts:
                time.sleep(attempt + 1)
    raise IOError("HTTPS archive download failed after verified-TLS retries") from error


def read_training_reviews(archive):
    """Read only regular train/pos and train/neg files, directly in the tar stream.

    No extraction to disk occurs; absolute paths, traversal paths, links, and
    the official test/unsup pools are never accepted as review text.
    """
    reviews = {}
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            if not REVIEW_NAME.fullmatch(member.name):
                continue
            if not member.isfile() or not 0 < member.size <= 1024 * 1024:
                raise ValueError(f"Invalid review member: {member.name}")
            if member.name in reviews:
                raise ValueError(f"Duplicate review ID: {member.name}")
            source = stream.extractfile(member)
            if source is None:
                raise ValueError(f"Missing content: {member.name}")
            raw = source.read()
            if len(raw) != member.size:
                raise ValueError(f"Truncated review: {member.name}")
            reviews[member.name] = {
                "id": member.name, "text": raw.decode("utf-8"),
                "source_text_sha256": hashlib.sha256(raw).hexdigest(),
            }
    return reviews


def sample_and_split(reviews, sample_size, seed):
    if not 2 <= sample_size <= len(reviews) or seed < 0:
        raise ValueError("Invalid sample size or seed")
    ids = sorted(reviews)
    chosen = np.random.default_rng(seed).choice(len(ids), sample_size, replace=False)
    selected = [reviews[ids[index]] for index in chosen]
    # This is the same declared split rule used by reproduce_rae.py, independent
    # of the source sampling generator's consumed state.
    order = np.random.default_rng(seed).permutation(sample_size)
    cutoff = int(sample_size * 0.9)
    split = {"train_rows": order[:cutoff].tolist(), "heldout_rows": order[cutoff:].tolist()}
    return selected, split


def verify_files(folder, files):
    for name, expected in files.items():
        if checksum(folder / name) != expected:
            raise ValueError(f"Prepared input changed: {name}")


def prepare(args):
    if args.dataset_source == "hf":
        return prepare_hf(args)
    folder = args.output
    selection_file = folder / "selection.json"
    request = {"sample_size": args.sample_size, "seed": args.seed,
               "model": args.model, "requested_revision": args.revision,
               "max_length": args.max_length}
    if selection_file.exists():
        saved = read_json(selection_file)
        if saved["request"] != request:
            raise ValueError("Existing preparation uses a different sampling/model specification")
        verify_files(folder, saved["files"])
        if args.archive is not None and checksum(args.archive) != saved["archive_sha256"]:
            raise ValueError("Supplied archive differs from the prepared source")
        return saved
    if folder.exists() and any(folder.iterdir()):
        # A partial archive from an interrupted download is the one safe retry case.
        allowed = {"aclImdb_v1.tar.gz", "aclImdb_v1.tar.gz.part", "archive_download.json",
                   "model_source.json", "model_cache_manifest.json"}
        if any(file.name not in allowed for file in folder.iterdir()):
            raise ValueError("Output must be empty or contain a verifiable earlier preparation")
    folder.mkdir(parents=True, exist_ok=True)
    archive = args.archive or folder / "aclImdb_v1.tar.gz"
    if args.archive is None:
        if args.offline and not archive.exists():
            raise FileNotFoundError("Offline preparation needs --archive or an existing downloaded archive")
        download_archive(archive, args.source_url)
    archive_hash = checksum(archive)
    print("Reading official training review pool", flush=True)
    reviews = read_training_reviews(archive)
    if len(reviews) != 25000:
        raise ValueError(f"Expected 25,000 labeled training reviews, found {len(reviews)}")
    selected, split = sample_and_split(reviews, args.sample_size, args.seed)
    texts_file = folder / "selected_texts.jsonl"
    with texts_file.open("w", encoding="utf-8", newline="\n") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(folder / "source_ids.json", [row["id"] for row in selected])
    write_json(folder / "split_manifest.json", {
        "seed": args.seed, "train_fraction": 0.9,
        "rule": "NumPy default_rng(seed).permutation(sample_size); first floor(0.9*N) train",
        **split,
        "train_ids": [selected[index]["id"] for index in split["train_rows"]],
        "heldout_ids": [selected[index]["id"] for index in split["heldout_rows"]],
        "purpose": "unsupervised geometric holdout; sentiment labels are not used",
    })
    saved = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "request": request, "source_url": args.source_url, "source_page": SOURCE_PAGE,
        "archive_source": "user-supplied local archive" if args.archive else "official URL",
        "archive_sha256": archive_hash, "archive_bytes": archive.stat().st_size,
        "archive_hash_note": "locally calculated provenance hash; no publisher checksum supplied",
        "population": len(reviews), "population_scope": "official train/pos + train/neg only",
        "sample_rule": "uniform without replacement from lexically sorted source paths; no label balancing",
        "sample_count": len(selected),
        "text_preprocessing": "exact UTF-8 review contents; retain HTML breaks; no strip or case changes",
        "claim_scope": "independent re-encoding of a same-family dataset, NOT the authors' original vectors",
        "checkpoint_choice": "RAE v1 says MPNet/768d without exact checkpoint; this project chooses the named ST model",
        "paper_url": PAPER_URL,
        "files": {name: checksum(folder / name) for name in
                  ["selected_texts.jsonl", "source_ids.json", "split_manifest.json"]},
    }
    write_json(selection_file, saved)
    return saved


def prepare_hf(args):
    """Use the official Stanford NLP Hugging Face train parquet as its own source.

    Its stable row IDs cannot recover the original Stanford tar member names.
    No tar metadata is manufactured for this alternative distribution.
    """
    from huggingface_hub import HfApi, hf_hub_download

    folder = args.output
    request = {"sample_size": args.sample_size, "seed": args.seed,
               "model": args.model, "requested_revision": args.revision,
               "max_length": args.max_length, "dataset_source": "hf",
               "dataset_requested_revision": args.dataset_revision}
    selection_file = folder / "selection.json"
    if selection_file.exists():
        saved = read_json(selection_file)
        if saved["request"] != request:
            raise ValueError("Existing HF preparation uses a different sampling/model specification")
        verify_files(folder, saved["files"])
        return saved
    folder.mkdir(parents=True, exist_ok=True)
    source_file = folder / "dataset_source.json"
    repo = "stanfordnlp/imdb"
    filename = "plain_text/train-00000-of-00001.parquet"
    if source_file.exists():
        source = read_json(source_file)
        if source["requested_revision"] != args.dataset_revision:
            raise ValueError("Existing dataset pin differs from the requested revision")
    else:
        if re.fullmatch(r"[0-9a-f]{40}", args.dataset_revision):
            revision = args.dataset_revision
        elif args.offline:
            raise ValueError("Offline HF preparation requires a fixed dataset revision or saved source")
        else:
            revision = HfApi().dataset_info(repo, revision=args.dataset_revision).sha
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("HF dataset did not resolve to an immutable commit SHA")
        source = {"dataset_repo": repo, "requested_revision": args.dataset_revision,
                  "resolved_dataset_revision": revision, "repo_filename": filename,
                  "source_url": f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"}
        write_json(source_file, source)
    parquet = folder / "train.parquet"
    if not parquet.exists():
        cached = hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename,
                                 revision=source["resolved_dataset_revision"],
                                 local_files_only=args.offline)
        partial = parquet.with_suffix(".parquet.part")
        shutil.copyfile(cached, partial)
        partial.replace(parquet)
    import pyarrow.parquet as pq

    texts = pq.read_table(parquet, columns=["text"])["text"].to_pylist()
    if len(texts) != 25000 or any(not isinstance(text, str) for text in texts):
        raise ValueError("Expected 25,000 non-null official IMDb training texts")
    reviews = {}
    for index, text in enumerate(texts):
        row_id = f"stanfordnlp/imdb/train/{index:05d}"
        reviews[row_id] = {"id": row_id, "text": text,
                           "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    selected, split = sample_and_split(reviews, args.sample_size, args.seed)
    with (folder / "selected_texts.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(folder / "source_ids.json", [row["id"] for row in selected])
    write_json(folder / "split_manifest.json", {
        "seed": args.seed, "train_fraction": 0.9, **split,
        "rule": "NumPy default_rng(seed).permutation(sample_size); first floor(0.9*N) train",
        "train_ids": [selected[index]["id"] for index in split["train_rows"]],
        "heldout_ids": [selected[index]["id"] for index in split["heldout_rows"]],
        "source_id_meaning": "zero-based row in immutable official HF train parquet; NOT a tar filename",
        "purpose": "unsupervised geometric holdout; sentiment labels are not used",
    })
    saved = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "request": request,
        **source, "source_format": "official Hugging Face parquet",
        "source_page": "https://huggingface.co/datasets/stanfordnlp/imdb",
        "parquet_sha256": checksum(parquet), "parquet_bytes": parquet.stat().st_size,
        "original_stanford_tar_filenames_available": False,
        "source_id_meaning": "zero-based row in the pinned HF train parquet",
        "population": len(reviews), "population_scope": "official HF IMDb train split only",
        "sample_count": len(selected),
        "sample_rule": "uniform without replacement from sorted fixed-width HF row IDs; no label balancing",
        "text_preprocessing": "parquet text strings unchanged; retain HTML breaks; hash their UTF-8 encoding",
        "claim_scope": "independent re-encoding of a same-family dataset, NOT the authors' original vectors",
        "checkpoint_choice": "RAE v1 says MPNet/768d without exact checkpoint; this project chooses the named ST model",
        "paper_url": PAPER_URL,
        "files": {name: checksum(folder / name) for name in
                  ["train.parquet", "dataset_source.json", "selected_texts.jsonl",
                   "source_ids.json", "split_manifest.json"]},
    }
    write_json(selection_file, saved)
    return saved


def resolve_model_source(args):
    """Resolve once, then reuse an immutable SHA without another API request."""
    from huggingface_hub import HfApi, try_to_load_from_cache

    source_file = args.output / "model_source.json"
    if source_file.exists():
        source = read_json(source_file)
        if source["model"] != args.model:
            raise ValueError("Cached model source belongs to a different model")
        if args.revision not in {source["requested_revision"], source["resolved_model_revision"]}:
            raise ValueError("Cached model source belongs to a different requested revision")
        return source
    if re.fullmatch(r"[0-9a-f]{40}", args.revision):
        revision = args.revision
    else:
        cached = try_to_load_from_cache(args.model, "modules.json", revision=args.revision)
        if isinstance(cached, str) and re.fullmatch(r"[0-9a-f]{40}", Path(cached).parent.name):
            revision = Path(cached).parent.name
        elif args.offline:
            raise FileNotFoundError("Offline model use requires a full --revision SHA or a cached revision")
        else:
            revision = HfApi().model_info(args.model, revision=args.revision).sha
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Model revision did not resolve to an immutable commit SHA")
    source = {"model": args.model, "requested_revision": args.revision,
              "resolved_model_revision": revision,
              "source_url": f"https://huggingface.co/{args.model}/tree/{revision}"}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(source_file, source)
    return source


def cache_model(args):
    """Fetch model files without constructing a model or allocating any GPU memory."""
    from huggingface_hub import snapshot_download

    source = resolve_model_source(args)
    snapshot = Path(snapshot_download(
        repo_id=args.model, revision=source["resolved_model_revision"],
        local_files_only=args.offline,
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "1_Pooling/*", "2_Normalize/*"],
    ))
    if not list(snapshot.glob("*.safetensors")) and not (snapshot / "pytorch_model.bin").exists():
        raise FileNotFoundError("Cached snapshot lacks model weights; expected safetensors for this checkpoint")
    files = {str(path.relative_to(snapshot)).replace("\\", "/"): checksum(path)
             for path in sorted(snapshot.rglob("*")) if path.is_file()}
    manifest = {**source, "snapshot_directory": str(snapshot), "files": files,
                "gpu_used": False, "downloaded_utc": datetime.now(timezone.utc).isoformat()}
    write_json(args.output / "model_cache_manifest.json", manifest)
    print(f"Cached model at immutable revision {source['resolved_model_revision']}", flush=True)
    return manifest


def encode(args, selection):
    # Imported only in the encoding stage, so --prepare-only does not load Torch
    # or consume the GPU while another experiment is running.
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Normalize

    folder = args.output
    final_manifest = folder / "embedding_manifest.json"
    if final_manifest.exists():
        manifest = read_json(final_manifest)
        if manifest["selection_sha256"] != checksum(folder / "selection.json"):
            raise ValueError("Embedding cache and source selection differ")
        verify_files(folder, manifest["files"])
        print("Reusing verified raw embedding cache", flush=True)
        return manifest
    spec_path = folder / "embedding_spec.json"
    if spec_path.exists():
        spec = read_json(spec_path)
        if spec["selection_sha256"] != checksum(folder / "selection.json"):
            raise ValueError("Incomplete embedding run belongs to different source inputs")
    else:
        revision = resolve_model_source(args)["resolved_model_revision"]
        spec = {"model": args.model, "requested_revision": args.revision,
                "resolved_model_revision": revision,
                "selection_sha256": checksum(folder / "selection.json"),
                "max_length": args.max_length,
                "normalization": "remove terminal ST Normalize if present; no encode normalization",
                "precision": "float32 model and stored vectors"}
        write_json(spec_path, spec)
    print(f"Loading frozen {args.model} at {spec['resolved_model_revision']} on {args.device}",
          flush=True)
    model = SentenceTransformer(args.model, revision=spec["resolved_model_revision"],
                                device=args.device, trust_remote_code=False,
                                local_files_only=args.offline)
    original_modules = [type(module).__name__ for module in model.children()]
    modules = list(model.children())
    if any(isinstance(module, Normalize) for module in modules[:-1]):
        raise ValueError("Nonterminal Normalize module requires an explicit preprocessing decision")
    removed_normalizer = bool(modules and isinstance(modules[-1], Normalize))
    if removed_normalizer:
        # Reuse the same frozen modules. Only the terminal unit-length mapping
        # is omitted, to expose the model's raw pooled representation.
        model = SentenceTransformer(modules=modules[:-1], device=args.device)
    model.max_seq_length = args.max_length
    model.float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this Python environment")
        torch.cuda.manual_seed_all(args.seed)
    width = model.get_sentence_embedding_dimension()
    if width is None:
        raise ValueError("Embedding model does not expose its dimension")
    if args.model == DEFAULT_MODEL and width != 768:
        raise ValueError(f"Expected MPNet width 768, found {width}")
    records = read_jsonl_records(folder / "selected_texts.jsonl")
    if len(records) != selection["sample_count"]:
        raise ValueError("Selected review count changed")
    temporary = folder / "raw_embeddings.part.npy"
    raw = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                    shape=(len(records), width))
    started = time.perf_counter()
    with torch.inference_mode():
        for first in range(0, len(records), 256):
            text = [record["text"] for record in records[first:first + 256]]
            result = model.encode(text, batch_size=args.batch_size, convert_to_numpy=True,
                                  normalize_embeddings=False, precision="float32",
                                  show_progress_bar=False)
            if result.shape != (len(text), width) or not np.isfinite(result).all():
                raise ValueError(f"Invalid encoding output at row {first}")
            raw[first:first + len(text)] = result
            raw.flush()
            completed = first + len(text)
            elapsed = time.perf_counter() - started
            eta = elapsed * (len(records) - completed) / completed
            print(f"Encoded {completed}/{len(records)}; elapsed {elapsed:.1f}s; ETA {eta:.1f}s",
                  flush=True)
    norms = np.linalg.norm(raw, axis=1)
    norm_statistics = {"min": float(norms.min()), "mean": float(norms.mean()),
                       "max": float(norms.max()), "std": float(norms.std())}
    if np.any(norms < 1e-12):
        raise ValueError("Model produced zero vectors")
    del raw
    temporary.replace(folder / "raw_embeddings.npy")
    versions = {name: importlib.metadata.version(name) for name in
                ["numpy", "torch", "sentence-transformers", "huggingface-hub"]}
    manifest = {
        **spec, "created_utc": datetime.now(timezone.utc).isoformat(),
        "dimension": width, "rows": len(records), "device": args.device,
        "batch_size": args.batch_size, "model_frozen": True,
        "original_st_modules": original_modules,
        "removed_terminal_normalize": removed_normalizer,
        "effective_modules": [type(module).__name__ for module in model.children()],
        "raw_norm_statistics": norm_statistics,
        "embedding_seconds_excluding_download_and_load": time.perf_counter() - started,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": versions, "script_sha256": checksum(Path(__file__)),
        "gpu_name": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "reproducibility": "fixed inputs, checkpoint and evaluation mode; no bitwise cross-platform guarantee",
        "claim_scope": selection["claim_scope"],
        "files": {name: checksum(folder / name) for name in
                  ["raw_embeddings.npy", "source_ids.json", "split_manifest.json", "embedding_spec.json"]},
    }
    write_json(final_manifest, manifest)
    print(f"Saved {len(records)} x {width} raw float32 embeddings", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--archive", type=Path, help="Existing official aclImdb_v1.tar.gz, if available")
    parser.add_argument("--source-url", default=SOURCE_URL, help="Verified official HTTPS archive URL")
    parser.add_argument("--dataset-source", choices=["stanford", "hf"], default="stanford")
    parser.add_argument("--dataset-revision", default="main", help="HF dataset commit SHA or branch to pin")
    parser.add_argument("--offline", action="store_true", help="Use only verified local data and model cache")
    parser.add_argument("--cache-model-only", action="store_true", help="Cache pinned model without loading Torch/GPU")
    parser.add_argument("--prepare-only", action="store_true", help="Download/select text without loading the model")
    args = parser.parse_args()
    if not 2 <= args.sample_size <= 25000 or args.seed < 0 or min(args.batch_size, args.max_length) < 1:
        parser.error("Invalid sample size, seed, batch size, or token limit")
    if args.cache_model_only:
        cache_model(args)
        return
    selection = prepare(args)
    source_hash = selection.get("archive_sha256") or selection["parquet_sha256"]
    print(f"Prepared {selection['sample_count']} reviews; source SHA256 {source_hash}",
          flush=True)
    if not args.prepare_only:
        encode(args, selection)


if __name__ == "__main__":
    main()
