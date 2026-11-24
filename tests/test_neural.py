"""Numerical checks run Torch in workers, preserving FAISS process isolation."""

import copy
import os
import subprocess
import sys

import numpy as np
import pytest

from retrieval_lab.common import fingerprint, read_json, unit_vectors, write_json
from retrieval_lab.compressors import Compressor
from retrieval_lab.config import DEFAULTS
from retrieval_lab.data import Dataset
from retrieval_lab.supervision import build_supervision


def training_config(**overrides):
    config = copy.deepcopy(DEFAULTS["neural"])
    config.update(
        threads=1,
        steps=30,
        batch_size=16,
        lr=0.005,
        min_lr=0.0005,
        device="cpu",
        log_every=30,
        negative_strategy="random",
    )
    config.update(overrides)
    return config


def test_rae_zero_regularization_matches_original_ae_numerically():
    x = unit_vectors(np.random.default_rng(19).normal(size=(48, 8)) + 0.3)
    config = training_config(regularization_weight=0.0, center=True)
    ae_config = {key: config[key] for key in DEFAULTS["ae"]}
    ae = Compressor("ae", 3, 17).fit(x, ae_config)
    rae = Compressor("rae", 3, 17).fit(x, neural_config=config)
    for key in ("mean", "encoder", "decoder"):
        np.testing.assert_array_equal(rae.arrays[key], ae.arrays[key])
    for left, right in zip(ae.info["history"], rae.info["history"], strict=True):
        assert left["corpus_probe_reconstruction"] == right["corpus_probe_reconstruction"]
    assert rae.info["supervision"] == "none"


def test_cosine_hinge_hand_calculation_scale_invariance_and_gradient_direction():
    code = """
import torch
from retrieval_lab.neural import ranking_loss
q = torch.tensor([[1., 0.], [1., 0.]], dtype=torch.float64)
p = torch.tensor([[1., 0.], [0., 2.]], dtype=torch.float64)
n = torch.tensor([[0., 1.], [2., 0.]], dtype=torch.float64)
loss, gap = ranking_loss(q, p, n, 0.25)
torch.testing.assert_close(gap, torch.tensor([1., -1.], dtype=torch.float64))
assert loss.item() == 0.625
scaled, scaled_gap = ranking_loss(7 * q, 0.2 * p, 3 * n, 0.25)
torch.testing.assert_close(loss, scaled)
torch.testing.assert_close(gap, scaled_gap)
vectors = [torch.tensor([row], dtype=torch.float64, requires_grad=True)
           for row in ([1., 0.], [0., 1.], [1., 1.])]
before, gap_before = ranking_loss(*vectors, 0.4)
before.backward()
assert all(torch.isfinite(vector.grad).all() for vector in vectors)
updated = [vector.detach() - 0.01 * vector.grad for vector in vectors]
after, gap_after = ranking_loss(*updated, 0.4)
assert after.item() < before.item()
assert gap_after.item() > gap_before.item()
"""
    environment = dict(os.environ, PYTHONUTF8="1")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_supervised_worker_roundtrip_preserves_and_checks_audit(tmp_path):
    documents = unit_vectors(
        np.asarray(
            [
                [1, 0, 3, 0],
                [-1, 0, 3, 0],
                [0, 1, 3, 0],
                [0, -1, 3, 0],
            ],
            dtype=np.float32,
        )
    )
    queries = documents.copy()
    queries[:, 2] *= -1
    doc_ids, qids = [f"d{i}" for i in range(4)], [f"q{i}" for i in range(4)]
    dataset = Dataset(
        "fixture",
        dict.fromkeys(doc_ids, {}),
        {},
        {"train": {qids[i]: {doc_ids[i]: 1} for i in range(4)}},
        {},
    )
    arrays, metadata = build_supervision(
        dataset, doc_ids, qids, queries, documents, {"negative_strategy": "random"}, 17
    )
    model = Compressor("qrae", 2, 17).fit(
        documents,
        neural_config=training_config(),
        training_data=arrays,
        training_metadata=metadata,
    )
    folder = tmp_path / "trained"
    model.save(folder)
    assert read_json(folder / "supervision.json") == metadata
    assert model.info["supervision_sha256"] == fingerprint(metadata)
    loaded = Compressor.load(folder)
    np.testing.assert_array_equal(loaded.transform(queries), model.transform(queries))
    assert "decoder" not in loaded.arrays
    reexport = tmp_path / "reexport"
    loaded.save(reexport)
    assert read_json(reexport / "supervision.json") == metadata
    changed = copy.deepcopy(metadata)
    changed["seed"] += 1
    write_json(reexport / "supervision.json", changed)
    with pytest.raises(ValueError, match="[Ss]upervision|checksum|hash"):
        Compressor.load(reexport)
