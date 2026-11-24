"""Independent geometry and training checks for the RAE reproduction track."""

import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "rae_reproduction", Path(__file__).parents[1] / "scripts" / "reproduce_rae.py"
)
rae = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rae)


def test_exact_geometry_excludes_self_by_identity_even_for_duplicate_vectors():
    points = np.array([[1, 0], [1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    expected = np.array([[1, 2], [0, 2], [0, 1], [2, 0]])
    for metric in ["euclidean", "cosine"]:
        actual = rae.nearest_neighbors(points, points, 2, metric, np.arange(4), block_size=2)
        np.testing.assert_array_equal(actual, expected)
    assert rae.overlaps(expected, expected, [1, 2]) == {
        "nn_overlap@1": 1.0,
        "nn_overlap@2": 1.0,
    }


def test_overlap_uses_set_membership_not_rank_agreement():
    reference = np.array([[0, 1, 2], [1, 2, 3]])
    changed = np.array([[2, 0, 1], [1, 4, 5]])
    assert rae.overlaps(reference, changed, [1, 3]) == {
        "nn_overlap@1": 0.5,
        "nn_overlap@3": pytest.approx(2 / 3),
    }


def test_split_is_disjoint_and_released_budget_records_epoch_rounding():
    vectors = np.zeros((101, 5))
    train, heldout = rae.split_vectors(vectors, 2026)
    assert (len(train), len(heldout)) == (90, 11)
    assert not set(train) & set(heldout)
    assert sorted([*train, *heldout]) == list(range(101))
    np.testing.assert_array_equal(train, rae.split_vectors(vectors, 2026)[0])
    assert rae.training_budget(1000, 128, 3000, "released") == (3000, 375, 8)
    assert rae.training_budget(9000, 128, 3000, "released") == (2982, 42, 71)
    assert rae.training_budget(9000, 128, 3000, "formula")[0] == 3000


def _run_torch_check(function, *arguments):
    """Run the original assertions without loading Torch into the FAISS test process."""
    code = (
        "import runpy, sys; "
        "namespace = runpy.run_path(sys.argv[1]); "
        "assert 'faiss' not in sys.modules; "
        "namespace[sys.argv[2]](*sys.argv[3:])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve()), function, *map(str, arguments)],
        env=dict(os.environ, PYTHONUTF8="1"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_loss_scaling_matches_explicit_coupled_decay_gradient():
    _run_torch_check("_check_loss_scaling")


def _check_loss_scaling():
    import torch

    # Coupled decay adds w*theta, whereas the derivative of c*||theta||^2
    # is 2*c*theta. Changing element mean to sample sum multiplies by D.
    dimension, weight_decay = 7, 0.04
    weights = torch.tensor([[0.1, -0.4], [0.8, 0.5]], requires_grad=True)
    explicit = (dimension * weight_decay / 2) * weights.square().sum()
    explicit.backward()
    torch.testing.assert_close(weights.grad, dimension * weight_decay * weights.detach())


@pytest.mark.parametrize("mode", ["formula", "released"])
def test_independent_training_learns_low_rank_data_and_roundtrips(tmp_path, mode):
    _run_torch_check("_check_training_roundtrip", tmp_path, mode)


def _check_training_roundtrip(tmp_path, mode):
    tmp_path = Path(tmp_path)
    rng = np.random.default_rng(44)
    vectors = (rng.normal(size=(96, 2)) @ rng.normal(size=(2, 8))).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    arrays, info = rae.train_rae(
        vectors,
        2,
        2026,
        mode,
        0.0,
        steps=220,
        batch_size=32,
        lr=0.01,
        min_lr=0.001,
        threads=1,
        log_every=50,
    )
    assert info["final_probe_reconstruction_per_vector"] < (
        0.1 * info["initial_probe_reconstruction_per_vector"]
    )
    assert ("encoder_bias" in arrays) == (mode == "released")
    assert info["centering"] == "none"
    if mode == "formula":
        assert info["actual_steps"] == 220
        assert info["history"][-1]["learning_rate"] == pytest.approx(0.001)
    else:
        assert info["actual_steps"] == 219
    path = tmp_path / "model.npz"
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        restored = dict(saved)
    np.testing.assert_array_equal(rae.project(vectors, arrays), rae.project(vectors, restored))


def test_invalid_cosine_and_too_few_candidates_fail_explicitly():
    with pytest.raises(ValueError, match="nonzero"):
        rae.nearest_neighbors(np.eye(2), np.zeros((1, 2)), 1, "cosine")
    with pytest.raises(ValueError, match="candidates"):
        rae.nearest_neighbors(np.eye(2), np.eye(2), 2, "euclidean", [0, 1])


def test_imdb_preparation_reads_only_official_train_paths_and_preserves_text(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "prepare_imdb", Path(__file__).parents[1] / "scripts" / "prepare_imdb_reproduction.py"
    )
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)
    archive = tmp_path / "fixture.tar.gz"
    entries = {
        **{f"aclImdb/train/pos/{i}_8.txt": f"  Review {i}<br />\n" for i in range(10)},
        "aclImdb/test/pos/99_8.txt": "must not enter sample",
        "aclImdb/train/unsup/99_0.txt": "must not enter sample",
        "../escape.txt": "must never be extracted",
    }
    with tarfile.open(archive, "w:gz") as stream:
        for name, text in entries.items():
            payload = text.encode("utf-8")
            metadata = tarfile.TarInfo(name)
            metadata.size = len(payload)
            stream.addfile(metadata, io.BytesIO(payload))
    reviews = preparation.read_training_reviews(archive)
    assert len(reviews) == 10
    assert reviews["aclImdb/train/pos/0_8.txt"]["text"] == "  Review 0<br />\n"
    selected, split = preparation.sample_and_split(reviews, 8, 2026)
    assert len({row["id"] for row in selected}) == 8
    assert selected == preparation.sample_and_split(reviews, 8, 2026)[0]
    reference_train, reference_heldout = rae.split_vectors(np.zeros((8, 1)), 2026)
    assert split["train_rows"] == reference_train.tolist()
    assert split["heldout_rows"] == reference_heldout.tolist()
    assert not (tmp_path / "escape.txt").exists()


def test_imdb_preparation_refuses_symlink_review_member(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "prepare_imdb", Path(__file__).parents[1] / "scripts" / "prepare_imdb_reproduction.py"
    )
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)
    archive = tmp_path / "link.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        metadata = tarfile.TarInfo("aclImdb/train/neg/0_1.txt")
        metadata.type = tarfile.SYMTYPE
        metadata.linkname = "somewhere-else"
        stream.addfile(metadata)
    with pytest.raises(ValueError, match="Invalid review member"):
        preparation.read_training_reviews(archive)


def test_imdb_fixed_revision_can_be_resolved_offline_without_model_api(tmp_path, monkeypatch):
    import huggingface_hub

    spec = importlib.util.spec_from_file_location(
        "prepare_imdb", Path(__file__).parents[1] / "scripts" / "prepare_imdb_reproduction.py"
    )
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)

    def network_forbidden(*args, **kwargs):
        raise AssertionError("Network must not be contacted for an immutable cached revision")

    monkeypatch.setattr(huggingface_hub, "HfApi", network_forbidden)
    args = SimpleNamespace(
        output=tmp_path, model=preparation.DEFAULT_MODEL, revision="a" * 40, offline=True
    )
    source = preparation.resolve_model_source(args)
    assert source["resolved_model_revision"] == "a" * 40
    assert preparation.resolve_model_source(args) == source


def test_imdb_downloader_requires_verified_https(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "prepare_imdb", Path(__file__).parents[1] / "scripts" / "prepare_imdb_reproduction.py"
    )
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)
    with pytest.raises(ValueError, match="HTTPS"):
        preparation.download_archive(tmp_path / "data.tar.gz", "http://example.com/data.tar.gz")


def test_imdb_jsonl_reader_preserves_unicode_separators_inside_review_text(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "prepare_imdb", Path(__file__).parents[1] / "scripts" / "prepare_imdb_reproduction.py"
    )
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)
    expected = [
        {"id": "review-1", "text": "before\u2028middle\u2029after"},
        {"id": "review-2", "text": "also preserve\u0085this character"},
    ]
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in expected)
    path = tmp_path / "reviews.jsonl"
    path.write_text(payload, encoding="utf-8")
    # This is the observed failure of the previous reader on real IMDb text.
    with pytest.raises(json.JSONDecodeError):
        [json.loads(line) for line in payload.splitlines()]
    assert preparation.read_jsonl_records(path) == expected
