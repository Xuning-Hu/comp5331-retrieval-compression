"""Saved-query index loading must work in the actual Windows Unicode workspace."""

import faiss
import numpy as np

from retrieval_lab.common import unit_vectors
from retrieval_lab.query import _SavedExactIndex
from retrieval_lab.search import ExactIndex


def test_saved_index_unicode_path_preserves_full_search_and_cutoff_ties(tmp_path):
    directory = tmp_path / "课程项目" / "已保存索引"
    directory.mkdir(parents=True)
    path = directory / "index.faiss"
    vectors = unit_vectors(
        np.array([[1, 0]] * 7 + [[0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    )
    reference = ExactIndex(vectors, threads=1)
    path.write_bytes(faiss.serialize_index(reference.index).tobytes())
    loaded = _SavedExactIndex(path.resolve(), threads=1)
    queries = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    for k in [3, len(vectors)]:
        expected_scores, expected_rows = reference.search(queries, k)
        scores, rows = loaded.search(queries, k)
        np.testing.assert_array_equal(rows, expected_rows)
        np.testing.assert_allclose(scores, expected_scores, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(loaded.search(queries[:1], 3)[1], [[0, 1, 2]])
