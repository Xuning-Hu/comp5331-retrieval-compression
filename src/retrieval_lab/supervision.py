"""Train-only QRAE triplets and auditable, static negative candidate pools.

The construction boundary reads only ``dataset.qrels['train']``. Workers receive
numeric arrays, never a Dataset, query text, or development/test judgments.
Sampling is uniform over positive pairs, so queries with more positive documents
have proportionally greater weight. Every sampled pair has exactly one negative.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

from .common import fingerprint, unit_vectors

if TYPE_CHECKING:
    from .data import Dataset

STRATEGIES = {"random", "in_batch", "hard"}


def _array_digest(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _duplicate_groups(corpus: dict, doc_ids: list[str], vectors: np.ndarray):
    """Group exact text or exact vector duplicates; missing content is not text."""
    parents = np.arange(len(doc_ids), dtype=np.int64)

    def root(row):
        while parents[row] != row:
            parents[row] = parents[parents[row]]
            row = parents[row]
        return int(row)

    def join(left, right):
        left, right = root(left), root(right)
        if left != right:
            parents[max(left, right)] = min(left, right)

    text_rows, vector_rows = {}, {}
    for row, did in enumerate(doc_ids):
        document = corpus[did]
        if isinstance(document.get("text"), str):
            title = document.get("title", "")
            title = title if isinstance(title, str) else ""
            # This is the existing embedding pipeline's exact text construction.
            text = (title + " " + document["text"]).strip()
            if text:
                if text in text_rows:
                    join(row, text_rows[text])
                else:
                    text_rows[text] = row
        key = hashlib.sha256(vectors[row].tobytes()).digest()
        if key in vector_rows:
            previous = vector_rows[key]
            if np.array_equal(vectors[row], vectors[previous]):
                join(row, previous)
        else:
            vector_rows[key] = row

    members = {}
    for row in range(len(doc_ids)):
        members.setdefault(root(row), []).append(row)
    ordered = sorted(members.values(), key=lambda rows: min(doc_ids[row] for row in rows))
    groups = np.empty(len(doc_ids), dtype=np.int64)
    representatives = []
    for group, rows in enumerate(ordered):
        groups[rows] = group
        representatives.append(min(rows, key=lambda row: doc_ids[row]))
    return groups, np.asarray(representatives, dtype=np.int64), ordered


def _canonical_topk(scores: np.ndarray, k: int, id_order: np.ndarray) -> np.ndarray:
    """Exact top-k, resolving all cutoff ties by document ID without sorting N."""
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    tied = tied[np.argsort(id_order[tied], kind="stable")[: k - len(above)]]
    candidates = np.concatenate((above, tied))
    return candidates[np.lexsort((id_order[candidates], -scores[candidates]))]


def build_supervision(
    dataset: Dataset,
    doc_ids: list[str],
    cached_qids: list[str],
    all_queries: np.ndarray,
    documents: np.ndarray,
    config: dict,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Build immutable numeric training inputs and their JSON-serializable audit.

    ``config`` is the ``[supervision]`` section. Supported keys are
    ``negative_strategy`` (random/in_batch/hard), ``hard_top_k`` (default 100),
    ``hard_rank_min`` (default 1), and ``hard_rank_max`` (default hard_top_k).
    Hard ranks are inclusive and 1-based in the original full-corpus ranking,
    before filtering train positives and exact duplicate groups. Mining uses
    normalized frozen vectors and exhaustive float32 inner products, never labels
    from any other split. An empty permitted pool is an error, not an implicit
    relaxation of the declared rank interval.

    Array query indices address ``queries``; document indices address
    ``documents``. ``positive_*`` CSR records the actual train positives;
    ``negative_block_*`` also includes their exact duplicate documents. Hard
    pools are populated only when the configured strategy is ``hard``.
    """
    strategy = config.get("negative_strategy", "random")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown negative strategy: {strategy}")
    top_k = config.get("hard_top_k", 100)
    rank_min = config.get("hard_rank_min", 1)
    rank_max = config.get("hard_rank_max", top_k)
    if any(
        isinstance(v, bool) or not isinstance(v, (int, np.integer))
        for v in [seed, top_k, rank_min, rank_max]
    ):
        raise ValueError("Seed and hard-negative rank settings must be integers")
    if seed < 0 or not 1 <= rank_min <= rank_max <= top_k:
        raise ValueError("Require seed >= 0 and 1 <= hard_rank_min <= hard_rank_max <= hard_top_k")
    if not doc_ids or len(set(doc_ids)) != len(doc_ids):
        raise ValueError("Document IDs must be nonempty and unique")
    if set(doc_ids) != set(dataset.corpus):
        raise ValueError("Document IDs must match the complete corpus")
    if len(set(cached_qids)) != len(cached_qids):
        raise ValueError("Cached query IDs must be unique")
    query_matrix, document_matrix = np.asarray(all_queries), np.asarray(documents)
    if query_matrix.ndim != 2 or query_matrix.shape[0] != len(cached_qids):
        raise ValueError("Cached query IDs and query vectors are not aligned")
    if document_matrix.ndim != 2 or document_matrix.shape[0] != len(doc_ids):
        raise ValueError("Document IDs and document vectors are not aligned")
    if query_matrix.shape[1] != document_matrix.shape[1]:
        raise ValueError("Query and document vector dimensions must agree")
    try:
        train = dataset.qrels["train"]
    except KeyError as error:
        raise ValueError("QRAE supervision requires a train qrels split") from error
    if not train:
        raise ValueError("QRAE supervision requires nonempty train qrels")

    document_positions = {did: row for row, did in enumerate(doc_ids)}
    query_positions = {qid: row for row, qid in enumerate(cached_qids)}
    positives, skipped = {}, []
    for qid in sorted(train):
        labels = train[qid]
        if any(not np.isfinite(grade) or grade < 0 for grade in labels.values()):
            raise ValueError(f"Invalid train relevance grade for query {qid}")
        ids = sorted(did for did, grade in labels.items() if grade > 0)
        if not ids:
            skipped.append(qid)
            continue
        if qid not in query_positions:
            raise ValueError(f"Train query {qid} is missing from the embedding cache")
        if any(did not in document_positions for did in ids):
            raise ValueError(f"Train query {qid} has a positive document outside the corpus")
        positives[qid] = ids
    qids = list(positives)
    if not qids:
        raise ValueError("No positive train query-document pairs are available")
    # Select before validation/normalization: held-out query vectors are not read.
    queries = unit_vectors(query_matrix[[query_positions[qid] for qid in qids]])
    docs = unit_vectors(document_matrix)
    groups, representatives, group_members = _duplicate_groups(dataset.corpus, doc_ids, docs)

    pair_queries, pair_documents = [], []
    offsets, blocked_offsets, blocked_groups_offsets = [0], [0], [0]
    blocked_documents, blocked_groups, blocked_ids = [], [], {}
    for qrow, qid in enumerate(qids):
        rows = sorted(document_positions[did] for did in positives[qid])
        pair_queries.extend([qrow] * len(rows))
        pair_documents.extend(rows)
        offsets.append(len(pair_documents))
        forbidden_groups = sorted(set(groups[rows].tolist()))
        if len(forbidden_groups) == len(representatives):
            raise ValueError(
                f"Train query {qid} has no eligible negative after duplicate filtering"
            )
        forbidden_rows = sorted(row for group in forbidden_groups for row in group_members[group])
        blocked_groups.extend(forbidden_groups)
        blocked_groups_offsets.append(len(blocked_groups))
        blocked_documents.extend(forbidden_rows)
        blocked_offsets.append(len(blocked_documents))
        blocked_ids[qid] = [doc_ids[row] for row in forbidden_rows]

    hard_offsets, hard_documents, hard_ranks, hard_scores = [0], [], [], []
    hard_candidates = {}
    effective_k = min(top_k, len(doc_ids))
    if strategy == "hard":
        id_order = np.argsort(np.argsort(np.asarray(doc_ids), kind="stable"), kind="stable")
        for start in range(0, len(qids), 64):
            # Bounded memory: never allocate the entire query-by-corpus matrix.
            scores = queries[start : start + 64] @ docs.T
            for local, row_scores in enumerate(scores):
                qrow = start + local
                ranked = _canonical_topk(row_scores, effective_k, id_order)
                forbidden = set(
                    blocked_groups[blocked_groups_offsets[qrow] : blocked_groups_offsets[qrow + 1]]
                )
                seen, candidates = set(), []
                for rank in range(rank_min, min(rank_max, effective_k) + 1):
                    drow = int(ranked[rank - 1])
                    group = int(groups[drow])
                    if group in forbidden or group in seen:
                        continue
                    seen.add(group)
                    hard_documents.append(drow)
                    hard_ranks.append(rank)
                    hard_scores.append(float(row_scores[drow]))
                    candidates.append(
                        {
                            "document_id": doc_ids[drow],
                            "full_corpus_rank": rank,
                            "cosine": float(row_scores[drow]),
                        }
                    )
                if not candidates:
                    raise ValueError(
                        f"Train query {qids[qrow]} has no hard negative in full-corpus ranks "
                        f"[{rank_min}, {rank_max}] within top-{effective_k}; "
                        "adjust the declared training protocol explicitly"
                    )
                hard_candidates[qids[qrow]] = candidates
                hard_offsets.append(len(hard_documents))
    else:
        hard_offsets = [0] * (len(qids) + 1)

    arrays = {
        "queries": queries,
        "documents": docs,
        "pair_query_indices": np.asarray(pair_queries, dtype=np.int64),
        "pair_positive_indices": np.asarray(pair_documents, dtype=np.int64),
        "positive_offsets": np.asarray(offsets, dtype=np.int64),
        "positive_document_indices": np.asarray(pair_documents, dtype=np.int64),
        "negative_block_offsets": np.asarray(blocked_offsets, dtype=np.int64),
        "negative_block_document_indices": np.asarray(blocked_documents, dtype=np.int64),
        "negative_block_group_offsets": np.asarray(blocked_groups_offsets, dtype=np.int64),
        "negative_block_group_indices": np.asarray(blocked_groups, dtype=np.int64),
        "document_duplicate_groups": groups,
        "group_representative_indices": representatives,
        "hard_offsets": np.asarray(hard_offsets, dtype=np.int64),
        "hard_document_indices": np.asarray(hard_documents, dtype=np.int64),
        "hard_full_corpus_ranks": np.asarray(hard_ranks, dtype=np.int64),
        "hard_cosine_scores": np.asarray(hard_scores, dtype=np.float32),
    }
    metadata = {
        "schema_version": 1,
        "protocol": "in-domain supervised compression; only train qrels used",
        "negative_strategy": strategy,
        "seed": int(seed),
        "query_ids": qids,
        "query_ids_sha256": fingerprint(qids),
        "document_ids": list(doc_ids),
        "document_ids_sha256": fingerprint(doc_ids),
        "positive_document_ids_by_query": positives,
        "blocked_document_ids_by_query": blocked_ids,
        "skipped_train_queries_without_positive": skipped,
        "train_queries": len(qids),
        "positive_pairs": len(pair_queries),
        "pair_weighting": "uniform positive pairs; queries weighted by positive-document count",
        "negatives_per_positive": 1,
        "random_sampling": "uniform over eligible exact-duplicate groups; canonical member by ID",
        "in_batch_sampling": "uniform distinct groups among sampled batch positive documents",
        "in_batch_fallback": "uniform random eligible group when no in-batch negative exists",
        "duplicate_rule": "exact nonempty title-space-text or exactly equal normalized vectors; "
        "transitive groups; no approximate similarity threshold",
        "duplicate_group_count": len(representatives),
        "duplicate_groups": [
            [doc_ids[row] for row in rows] for rows in group_members if len(rows) > 1
        ],
        "hard_mining": {
            "source": "original frozen normalized vectors; full-corpus exact float32 scores",
            "top_k": int(top_k),
            "effective_top_k": int(effective_k),
            "rank_min": int(rank_min),
            "rank_max": int(rank_max),
            "rank_convention": "1-based inclusive; rank first, filter train positives/duplicates next",
            "tie_break": "document ID ascending",
            "pool": "static once per construction; no dev/test labels or dynamic remining",
            "candidates_by_query": hard_candidates,
            "candidates_sha256": fingerprint(hard_candidates),
        },
        "array_sha256": {key: _array_digest(value) for key, value in arrays.items()},
    }
    metadata["supervision_sha256"] = fingerprint(metadata)
    return arrays, metadata


def sample_triplets(
    arrays: dict[str, np.ndarray],
    rng: np.random.Generator,
    batch_size: int,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample pair-uniform (query, positive, negative) indices with one negative.

    Sampling with replacement permits repeated training examples across the batch;
    duplicate document groups never increase a negative candidate's probability.
    Reproducibility requires a private Generator initialized from the recorded seed.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown negative strategy: {strategy}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
        raise ValueError("batch_size must be a positive integer")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    pair_count = len(arrays["pair_query_indices"])
    if not pair_count:
        raise ValueError("Cannot sample an empty supervision table")
    chosen = rng.integers(pair_count, size=batch_size)
    qrows = arrays["pair_query_indices"][chosen]
    prows = arrays["pair_positive_indices"][chosen]
    nrows = np.empty(batch_size, dtype=np.int64)
    representatives = arrays["group_representative_indices"]
    group_offsets = arrays["negative_block_group_offsets"]
    group_blocks = arrays["negative_block_group_indices"]
    in_batch = (
        np.unique(arrays["document_duplicate_groups"][prows]) if strategy == "in_batch" else None
    )

    def random_document(blocked):
        available = len(representatives) - len(blocked)
        if available < 1:
            raise ValueError("No eligible negative after positive and duplicate filtering")
        # Map a uniform allowed rank to its group ID without rejection or an N-sized mask.
        rank = int(rng.integers(available))
        group = rank + np.searchsorted(blocked - np.arange(len(blocked)), rank, side="right")
        return representatives[group]

    for position, qrow in enumerate(qrows):
        blocked = group_blocks[group_offsets[qrow] : group_offsets[qrow + 1]]
        if strategy == "hard":
            start, end = arrays["hard_offsets"][qrow : qrow + 2]
            if start == end:
                raise ValueError(
                    "No static hard candidates; build supervision with strategy='hard'"
                )
            nrows[position] = arrays["hard_document_indices"][rng.integers(start, end)]
        elif strategy == "in_batch":
            eligible = in_batch[~np.isin(in_batch, blocked)]
            nrows[position] = (
                representatives[eligible[rng.integers(len(eligible))]]
                if len(eligible)
                else random_document(blocked)
            )
        else:
            nrows[position] = random_document(blocked)
    return qrows, prows, nrows
