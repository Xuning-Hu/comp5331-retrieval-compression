# Course experiment protocol, fixed before new dev evaluation

Date: 2026-11-04. This document describes intended experiments; execution and
results are recorded separately. No new test scores were inspected when written.

## Shared task and data

- Frozen `intfloat/e5-base-v2` revision
  `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`, float32 768d, model pooling,
  `query: ` / `passage: ` prefixes, 512-token limit, L2 normalization.
- Official BEIR SciFact, NFCorpus and FiQA archives. Preserve whole document units
  and full corpora. All data/model files and query splits have manifests.
- SciFact official train is split 647/162 train/dev using seed2026; official test
  has 300 queries. NFCorpus/FiQA use official query-disjoint splits.
- All compressors may fit the full target corpus without its query labels. QRAE
  additionally uses only train-query relevance. This is corpus-adaptive and
  in-domain supervised compression, not zero-shot or unseen-corpus generalization.

## Main methods and controlled objective

Original, PCA, Gaussian RP and centered linear AE retain their existing definitions.
Controlled RAE and QRAE use the same centered, no-bias, untied linear layers as AE.
For corpus mean mu, encoder E and decoder D:

    z = E(x - mu)
    L_rec = mean_batch sum_dimensions (D z - (x - mu))^2
    L_reg = ||E||_F^2 + ||D||_F^2
    L_rank = mean_pairs max(0, m - cos(z_q,z_pos) + cos(z_q,z_neg))
    L_RAE = L_rec + lambda * L_reg
    L_QRAE = L_rec + lambda * L_reg + alpha * L_rank

Decoder receives raw latent vectors. Ranking/search use normalized latent vectors.
Adam has weight_decay=0 because the squared-norm penalty is explicit. All neural
methods use lr1e-3, cosine decay to1e-5, batch128, final fixed checkpoint, 3000
updates (pilot1000), no gradient clipping or retrieval early stopping. Reconstruction
minibatches use a separate RNG so ranking sampling does not change them.

The initial seven-method pilot used CPU. Before dev selection, a 250-update
execution/stability probe measured about 10.39s on CPU and 4.14s on CUDA (same
configuration and seed; no retrieval-based selection). All main-study neural
training, including the AE baseline, therefore uses CUDA on the same RTX4060
Laptop GPU. PCA/RP and all exact-search evaluation use CPU. Timing from the pilot
and from concurrent background work is not used as a controlled speed comparison.

This controlled, centered RAE-style comparison is distinct from original-paper
reproduction. The latter includes raw-input/no-bias formula and released-style
bias/element-MSE/coupled-decay tracks; see rae-reproduction-protocol.md.
No global kNN preservation theorem is asserted for these learned projections.

## Supervision and negatives

Sample uniformly from train positive query-document pairs; multiple positives give
a query proportionally more weight. Every pair receives exactly one negative.
All known train positives and their exact duplicate documents/vectors are blocked.
Negative-candidate duplicate groups receive one representative to avoid duplicate
weighting; positive examples remain query-document pairs, including duplicate
positives if present in the labels.

- Random: uniformly select an eligible document duplicate group.
- In-batch: select an eligible unique positive-document group from other sampled
  pairs; use random fallback if none exists. This is one-negative in-batch sampling,
  not an all-negatives InfoNCE loss.
- Hard: mine the frozen full-dimensional retriever for train queries only;
  full-corpus ranks1..100, then filter positives/duplicates, uniformly sample one.
  Never use held-out qrels to clean candidates. Unjudged candidates can be false negatives.

Record training query IDs, positive IDs, candidate IDs/ranks and hashes. Train
workers receive numeric training arrays only, never the Dataset or held-out labels.

## Pilot, selection and experiment budget

1. SciFact128d, seed42, 1000 updates, all six methods plus Rank-only. Initial
   lambda=1e-4, alpha=1, margin=0.1, random negatives. Inspect finite loss/gradients,
   noncollapsed latent norms, learning curves and held-out dev retrieval.
2. Dev selection at128d/seed42/3000 updates: RAE lambda in {1e-5,1e-4,1e-3}.
   Select highest dev nDCG@10 (ties: smaller lambda). Hold it shared between
   RAE/QRAE. QRAE alpha in {0.1,1,10}; margin fixed0.1. Select highest dev nDCG@10
   (ties: smaller alpha). Keep all candidate results. This is modest dev tuning,
   not proof of a globally optimized RAE or QRAE. No changes based on test results.
3. Main dimensions64/128/256/384; random-method seeds17/42/2026. Original once,
   PCA once per dimension. Main QRAE uses random negatives to make the basic
   supervision effect explicit; hard and in-batch remain named ablations.
4. Negative sampling ablation on SciFact64/128 with equal updates, batches and
   one negative per pair. Rank-only at64/128 and a768d supervised projection
   separate reconstruction/regularization effects from generic domain adaptation.
   Additional lambda/alpha grid results supply regularization/loss-weight ablations.
5. Reuse the selected SciFact settings on NFCorpus and FiQA without additional
   dataset-specific search. Record this as transferred hyperparameters. Complete
   at least a second dataset before considering optional extra variants.
6. Freeze config and dev artifact hashes, then load those artifacts for final test;
   no refitting. All predeclared methods/seeds are reported, without best-seed selection.

## Shared evaluation and reporting

Full-corpus FAISS IndexFlatIP, normalized float32, identical threads/tie ordering.
Main metric nDCG@10 with linear relevance gain and full-qrels IDCG; secondary
Recall@10/100 and MRR@10/100. Geometric query-document overlap@10 and self-excluded
document-document overlap@10 use the same reference Original. Compare geometry and
retrieval within dimension. Save all per-query scores, ranking IDs and raw scores.

Report mean and sample SD over training seeds, plus paired query and exploratory
crossed seed/query bootstrap. Queries shared across seeds must remain shared during
resampling; do not treat seed-times-query as independent observations. Small grids
and multiple comparisons mean exploratory conclusions; no unplanned significance claim.

Track vector payload, index file, deployed compressor and training-only decoder
separately. Search p50/p95 excludes text encoding and projection; do not label it
end-to-end latency or infer an equal overall RAM reduction. Timings are compared
only on the same host/protocol. Add query encoding/projection/search breakdown if
time permits. Include improvements and regressions, truncation, domain and relevance
judgment limits, and the exact scope of original-paper reproduction.
