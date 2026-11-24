import copy
import json

import numpy as np
import pytest

from retrieval_lab.data import Dataset
from retrieval_lab.supervision import build_supervision, sample_triplets


class TrainOnlyQrels(dict):
    """Fail even on accidental iteration or lookup of held-out label metadata."""

    def __getitem__(self, key):
        assert key == "train", "The training boundary accessed held-out judgments"
        return super().__getitem__(key)

    def __iter__(self):
        raise AssertionError("The training boundary iterated all split names")

    def values(self):
        raise AssertionError("The training boundary accessed all split values")

    def items(self):
        raise AssertionError("The training boundary accessed all split items")


def fixture():
    doc_ids = ["a", "b", "c", "d", "e", "f"]
    documents = np.asarray(
        [
            [1, 0, 0],
            [0.9, 0.4, 0],
            [0.9, -0.4, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    cached_qids = ["held-dev", "train-z", "held-test", "train-a"]
    queries = np.asarray(
        [
            [np.nan, np.nan, np.nan],
            [0, 1, 0],
            [np.nan, np.nan, np.nan],
            [1, 0, 0],
        ],
        dtype=np.float32,
    )
    dataset = Dataset(
        "fixture",
        {did: {} for did in doc_ids},
        {},
        TrainOnlyQrels(
            {
                "train": {"train-z": {"d": 1}, "train-a": {"a": 2, "b": 1, "e": 0}},
                "dev": {"held-dev": {"f": 1}},
                "test": {"held-test": {"c": 1}},
            }
        ),
        {},
    )
    return dataset, doc_ids, cached_qids, queries, documents


def build(strategy="random", **config):
    return build_supervision(
        *fixture(), {"negative_strategy": strategy, "hard_top_k": 6, **config}, 42
    )


def assert_valid_triplets(arrays, triplets):
    qrows, prows, nrows = triplets
    assert qrows.shape == prows.shape == nrows.shape
    for qrow, positive, negative in zip(qrows, prows, nrows, strict=True):
        start, end = arrays["positive_offsets"][qrow : qrow + 2]
        assert positive in arrays["positive_document_indices"][start:end]
        start, end = arrays["negative_block_offsets"][qrow : qrow + 2]
        assert negative not in arrays["negative_block_document_indices"][start:end]


@pytest.mark.parametrize("strategy", ["random", "in_batch", "hard"])
def test_all_strategies_are_train_only_numeric_and_reproducible(strategy):
    arrays, metadata = build(strategy)
    again, repeated_metadata = build(strategy)
    assert metadata == repeated_metadata
    assert metadata["query_ids"] == ["train-a", "train-z"]
    assert metadata["positive_pairs"] == 3
    assert metadata["negatives_per_positive"] == 1
    assert "positive-document count" in metadata["pair_weighting"]
    json.dumps(metadata, allow_nan=False)
    for key, array in arrays.items():
        assert array.dtype.kind in "fiu"
        assert np.isfinite(array).all()
        np.testing.assert_array_equal(array, again[key])
    first = sample_triplets(arrays, np.random.default_rng(42), 256, strategy)
    second = sample_triplets(arrays, np.random.default_rng(42), 256, strategy)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    assert_valid_triplets(arrays, first)
    # Multiple positives are always blocked, even when another positive is the sampled pair.
    assert not set(first[2][first[0] == 0]) & {0, 1}


def test_heldout_labels_and_vectors_cannot_change_training_inputs_or_audit():
    inputs = fixture()
    original, original_metadata = build_supervision(*inputs, {}, 17)
    dataset, ids, qids, queries, documents = inputs
    dataset.qrels = TrainOnlyQrels(
        {
            "train": dict.__getitem__(dataset.qrels, "train"),
            "dev": {"unseen": {"e": 999}},
            "test": {"unseen": {"f": 999}},
        }
    )
    queries[[0, 2]] = 12345
    changed, changed_metadata = build_supervision(dataset, ids, qids, queries, documents, {}, 17)
    assert original_metadata == changed_metadata
    for key in original:
        np.testing.assert_array_equal(original[key], changed[key])


def test_hard_pool_uses_original_full_corpus_ranks_then_filters_all_positives():
    arrays, metadata = build("hard", hard_rank_min=2, hard_rank_max=4)
    candidates = metadata["hard_mining"]["candidates_by_query"]["train-a"]
    # b and c tie at ranks 2/3; b is another known positive and must be removed.
    assert [row["document_id"] for row in candidates] == ["c", "d"]
    assert [row["full_corpus_rank"] for row in candidates] == [3, 4]
    for qrow, _, negative in zip(
        *sample_triplets(arrays, np.random.default_rng(9), 128, "hard"), strict=True
    ):
        start, end = arrays["hard_offsets"][qrow : qrow + 2]
        assert negative in arrays["hard_document_indices"][start:end]


def test_hard_mining_does_not_remove_documents_relevant_only_to_heldout_queries():
    _, metadata = build("hard")
    # c is labeled relevant for held-test, but that label is inaccessible to construction.
    assert "c" in {
        row["document_id"] for row in metadata["hard_mining"]["candidates_by_query"]["train-a"]
    }


def test_in_batch_uses_other_positive_documents_and_falls_back_for_single_pair():
    arrays, _ = build("in_batch")
    qrows, prows, nrows = sample_triplets(arrays, np.random.default_rng(5), 100, "in_batch")
    assert set(qrows) == {0, 1}
    assert set(nrows[qrows == 0]) == {3}
    assert set(nrows[qrows == 1]) <= {0, 1}
    assert set(nrows) <= set(prows)
    # A batch of one has no other positive; a legal random document still must be sampled.
    assert_valid_triplets(arrays, sample_triplets(arrays, np.random.default_rng(9), 1, "in_batch"))


@pytest.mark.parametrize("strategy", ["random", "in_batch", "hard"])
def test_exact_duplicate_text_and_vectors_are_blocked_and_candidates_deduplicated(strategy):
    dataset, ids, qids, queries, documents = fixture()
    dataset.corpus["a"] = {"text": "same content"}
    dataset.corpus["c"] = {"text": "same content"}
    documents[5] = documents[0]  # a/c/f form a transitive exact-duplicate group.
    arrays, metadata = build_supervision(
        dataset,
        ids,
        qids,
        queries,
        documents,
        {"negative_strategy": strategy, "hard_top_k": 6},
        42,
    )
    assert metadata["duplicate_groups"] == [["a", "c", "f"]]
    assert metadata["blocked_document_ids_by_query"]["train-a"] == ["a", "b", "c", "f"]
    triplets = sample_triplets(arrays, np.random.default_rng(42), 256, strategy)
    assert_valid_triplets(arrays, triplets)
    assert not set(triplets[2][triplets[0] == 0]) & {0, 1, 2, 5}
    if strategy == "hard":
        for qrow in range(len(arrays["queries"])):
            start, end = arrays["hard_offsets"][qrow : qrow + 2]
            candidate_groups = arrays["document_duplicate_groups"][
                arrays["hard_document_indices"][start:end]
            ]
            assert len(candidate_groups) == len(set(candidate_groups))


def test_empty_document_content_is_not_a_duplicate_group():
    arrays, metadata = build()
    assert metadata["duplicate_group_count"] == 6
    np.testing.assert_array_equal(arrays["group_representative_indices"], np.arange(6))


def test_pair_uniform_sampling_preserves_declared_multi_positive_query_weighting():
    arrays, _ = build()
    qrows, _, _ = sample_triplets(arrays, np.random.default_rng(42), 12000, "random")
    assert np.mean(qrows == 0) == pytest.approx(2 / 3, abs=0.02)


def test_invalid_or_unavailable_negative_pools_fail_clearly():
    with pytest.raises(ValueError, match="no hard negative"):
        build("hard", hard_rank_min=1, hard_rank_max=1)
    arrays, _ = build()
    with pytest.raises(ValueError, match="No static hard"):
        sample_triplets(arrays, np.random.default_rng(1), 8, "hard")
    with pytest.raises(ValueError, match="positive integer"):
        sample_triplets(arrays, np.random.default_rng(1), 0, "random")
    inputs = list(fixture())
    inputs[4] = np.ones_like(inputs[4])
    with pytest.raises(ValueError, match="no eligible negative"):
        build_supervision(*inputs, {}, 42)


def test_input_alignment_missing_train_pairs_and_duplicate_ids_are_rejected():
    inputs = list(fixture())
    inputs[1] = ["a", "a", "c", "d", "e", "f"]
    with pytest.raises(ValueError, match="unique"):
        build_supervision(*inputs, {}, 42)
    inputs = list(fixture())
    inputs[2] = ["held-dev", "train-z", "held-test", "missing"]
    with pytest.raises(ValueError, match="missing from the embedding"):
        build_supervision(*inputs, {}, 42)
    inputs = list(fixture())
    inputs[0].qrels = {"train": {"train-a": {"e": 0}}}
    with pytest.raises(ValueError, match="No positive train"):
        build_supervision(*inputs, {}, 42)


def test_static_hard_protocol_rejects_invalid_rank_ranges():
    for config in [
        {"hard_rank_min": 0},
        {"hard_rank_min": 4, "hard_rank_max": 3},
        {"hard_top_k": 2, "hard_rank_max": 3},
        {"hard_top_k": 2.5},
    ]:
        with pytest.raises(ValueError):
            build("hard", **config)


def test_document_alignment_supports_permutations_and_ties_still_follow_ids():
    dataset, ids, qids, queries, documents = fixture()
    arrays, metadata = build("hard", hard_rank_min=2, hard_rank_max=4)
    permutation = [5, 4, 3, 2, 1, 0]
    _, reordered = build_supervision(
        dataset,
        [ids[row] for row in permutation],
        qids,
        queries,
        documents[permutation],
        {"negative_strategy": "hard", "hard_top_k": 6, "hard_rank_min": 2, "hard_rank_max": 4},
        42,
    )
    assert (
        metadata["hard_mining"]["candidates_by_query"]
        == reordered["hard_mining"]["candidates_by_query"]
    )
    modified = copy.deepcopy(arrays)
    modified["pair_query_indices"] = np.asarray([], dtype=np.int64)
    with pytest.raises(ValueError, match="empty supervision"):
        sample_triplets(modified, np.random.default_rng(1), 4, "random")
