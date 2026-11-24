"""A small real query path using the exact same frozen model and saved transform."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import faiss
import numpy as np

from .common import fingerprint, read_json, sha256, unit_vectors
from .compressors import Compressor
from .data import load_dataset
from .embedding import validate_cache
from .search import ExactIndex


class _SavedExactIndex(ExactIndex):
    """Use the measured index unchanged with the shared canonical tie handling."""

    def __init__(self, path: Path, threads: int):
        faiss.omp_set_num_threads(threads)
        # Python handles Windows Unicode paths; the native FAISS filename API does not.
        serialized = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
        self.index = faiss.deserialize_index(serialized)
        self.vectors = self.index.reconstruct_n(0, self.index.ntotal)


def query(folder: Path, text: str, k: int) -> None:
    if k < 1:
        raise ValueError("top-k must be positive")
    study = read_json(folder.parent / "study.json")
    cache = Path(study["embedding_cache"])
    metadata = validate_cache(cache)
    if metadata["files"] != study["embedding_files"]:
        raise ValueError("Query cache differs from the saved study")
    if "spec" not in metadata:
        raise ValueError("New-text query requires a real-model run, not the synthetic demo")
    result = read_json(folder / "result.json")
    for name in ("compressor.npz", "compressor.json", "index.faiss"):
        if sha256(folder / name) != result["files"][name]:
            raise ValueError(f"Saved query artifact failed checksum validation: {name}")
    ids = read_json(cache / "doc_ids.json")
    if fingerprint(ids) != study["document_ids_sha256"]:
        raise ValueError("Query document IDs differ from the saved study")
    with tempfile.TemporaryDirectory(prefix="retrieval-query-") as directory:
        target = Path(directory)
        from .common import write_json

        write_json(target / "request.json", {"metadata": metadata, "text": text})
        subprocess.run(
            [sys.executable, "-m", "retrieval_lab.worker", "query", str(target)], check=True
        )
        q = np.load(target / "query.npy", allow_pickle=False)
    compressor = Compressor.load(folder)
    qz = unit_vectors(compressor.transform(q))
    index = _SavedExactIndex(folder / "index.faiss", study["config"]["experiment"]["threads"])
    if index.index.ntotal != len(ids):
        raise ValueError("Saved query index size differs from document IDs")
    scores, rows = index.search(qz, min(k, index.index.ntotal))
    data_config = study["config"]["data"]
    dataset = load_dataset(Path(data_config["directory"]) / data_config["name"], data_config)
    for rank, (score, row) in enumerate(zip(scores[0], rows[0], strict=True), 1):
        document = dataset.corpus[ids[row]]
        print(f"{rank}. {ids[row]} | cosine={score:.5f} | {document.get('title', '')}")
        print(document["text"][:500], "\n")
