import runpy
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
CHECK = runpy.run_path(str(SCRIPTS / "check_rae_seed_robustness.py"))
HELPER = runpy.run_path(str(SCRIPTS / "reproduce_rae.py"))


def test_saved_split_preserves_order_and_never_resamples(tmp_path, monkeypatch):
    path = tmp_path / "split.json"
    split = {"train_rows": [5, 2, 0, 4], "heldout_rows": [3, 1], "train_probe_positions": [3, 1]}
    CHECK["write"](path, split)

    def forbidden(*args, **kwargs):
        raise AssertionError("Fixed split must never call a random generator")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    train, heldout, probes = CHECK["load_fixed_split"](path, 6, CHECK["sha"](path))
    assert train.tolist() == split["train_rows"]
    assert heldout.tolist() == split["heldout_rows"]
    assert probes.tolist() == split["train_probe_positions"]
    assert train[probes].tolist() == [4, 2]


@pytest.mark.parametrize("change", ["checksum", "overlap", "fractional", "probe"])
def test_invalid_or_changed_split_is_rejected(tmp_path, change):
    path = tmp_path / "split.json"
    split = {"train_rows": [5, 2, 0, 4], "heldout_rows": [3, 1], "train_probe_positions": [3, 1]}
    CHECK["write"](path, split)
    expected = CHECK["sha"](path)
    if change == "checksum":
        split["train_rows"] = list(reversed(split["train_rows"]))
    elif change == "overlap":
        split["heldout_rows"] = [3, 5]
    elif change == "fractional":
        split["train_rows"][0] = 5.0
    else:
        split["train_probe_positions"] = [4, 1]
    CHECK["write"](path, split)
    with pytest.raises(ValueError):
        CHECK["load_fixed_split"](path, 6, expected if change == "checksum" else CHECK["sha"](path))


def test_only_training_seed_changes_and_reused_seed_never_fits():
    calls = []
    base = np.arange(24, dtype=np.float32).reshape(6, 4)
    before = base.copy()

    def record(*args):
        calls.append(args)
        return {}, {}

    plan = CHECK["job_plan"]()
    assert len(plan) == 12 and all(job["reused"] for job in plan[:4])
    for job in plan:
        if job["reused"]:
            with pytest.raises(ValueError, match="declared new fits"):
                CHECK["fit_new_job"]({"train_rae": record}, base, job)
        else:
            CHECK["fit_new_job"]({"train_rae": record}, base, job)
    assert len(calls) == 8
    assert {(args[2], args[3], args[4]) for args in calls} == {
        (seed, mode, coefficient)
        for seed in (17, 42)
        for mode in ("formula", "released")
        for coefficient in (0.0, 3e-7)
    }
    assert all(
        args[0] is base and args[1] == 256 and args[5:] == (3000, 128, 0.001, 0.00001, "cpu", 2)
        for args in calls
    )
    np.testing.assert_array_equal(base, before)


def test_query_metrics_match_original_cosine_preprocessing_and_self_exclusion():
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(16, 7)).astype(np.float32)
    vectors *= np.arange(1, 17, dtype=np.float32)[:, None]
    base, heldout = vectors[:12], vectors[12:]
    anchors = np.array([2, 6, 11])
    arrays = {
        "encoder": rng.normal(size=(4, 7)).astype(np.float32),
        "encoder_bias": rng.normal(size=4).astype(np.float32),
    }
    references = {
        metric: {
            "heldout": HELPER["nearest_neighbors"](base, heldout, 5, metric),
            "train_probe": HELPER["nearest_neighbors"](base, base[anchors], 5, metric, anchors),
        }
        for metric in ("cosine", "euclidean")
    }
    aggregate, individual, neighbors = CHECK["evaluate_with_queries"](
        HELPER, arrays, base, heldout, anchors, references, [1, 5]
    )
    for metric in ("cosine", "euclidean"):
        expected = HELPER["evaluate_projection"](
            arrays, base, heldout, anchors, metric, references[metric], [1, 5]
        )
        assert aggregate[metric] == expected
        assert np.mean(individual[metric]["heldout"]) == expected["heldout_nn_overlap@5"]
        assert all(
            position not in row
            for position, row in zip(anchors, neighbors[metric + "_train_probe"], strict=True)
        )
