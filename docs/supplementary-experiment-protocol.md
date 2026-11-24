# Bounded supplementary experiment protocol

Recorded: 2026-11-04 (Asia/Shanghai), before the supplementary fits. The execution
runner must record this file's original-byte SHA256 and its launch timestamp before
training. Results and actual completion status belong in separate run records.

This is a post-hoc exploratory extension after the original E5 final-test results
and RAE follow-ups have been inspected. It is not the original study's
preregistration, a newly untouched test, or a replacement for the completed main
study. The user has allowed the PC to work for up to a day; that is permission to
attempt this bounded workload, not a promise that it will finish within a day.
The requested deliverables remain the report and code.

## Relationship to the proposal and feedback

The submitted proposal identifies MPNet, E5 and BGE as candidate frozen encoders,
and promises variation of the reconstruction regularizer and ranking-loss weight.
It does not require every named encoder or an exhaustive parameter search. Raymond's
feedback asks for frozen concrete settings, an independent implementation, a pilot,
multiple compression ratios, documented supervision, shared search settings and
early stability monitoring. These already have evidence in the original study.

| Supplement | Question and value | Requirement boundary |
|---|---|---|
| Second frozen encoder | Do the controlled 128d comparisons change under an MPNet embedding pipeline when the E5-selected coefficients are transferred unchanged? | Strengthens model-scope evidence; an additional encoder was not a missing mandatory experiment. |
| Joint coefficient grid | How sensitive are the reconstruction/regularization/ranking combinations on two existing development sets when both coefficients vary together? | Strengthens the proposal's coefficient ablations beyond the completed sequential single-seed grid; does not retroactively alter the main selection. |

The separate IMDb RAE verification is unchanged. Neither supplement recovers the
authors' original input identities or experiment configuration, and neither turns
the present corpora into a large-scale deployment study.

## Preserved data, definitions and provenance

Use the existing full BEIR document collections, document order, text construction
and query splits, without new sampling, document chunking, duplicate removal or
query reassignment:

| Dataset | Documents | Train queries | Development queries | Test queries |
|---|---:|---:|---:|---:|
| SciFact | 5,183 | 647 | 162 | 300 |
| NFCorpus | 3,633 | 2,590 | 324 | 323 |
| FiQA | 57,638 | 5,500 | 500 | 648 |

SciFact uses the existing seed-2026 internal train/dev split; the other datasets
retain their official splits. Known cross-split identical query wording remains
disclosed; distinct query IDs do not make these a new set of unseen query texts.
Only designated train-query records and labels enter supervision. All methods may
fit the target document vectors: this remains corpus-adaptive compression with
in-domain supervision, not zero-shot transfer to an unseen corpus.

The original frozen protocol is `docs/experiment-protocol.md`, SHA256
`7d983d0a5f54586f7b2aaa7a6d719dd4da98c1069247fa87675356091907fd10`.
The unchanged selection record is `docs/results/course/protocol/selection.json`,
SHA256 `db7121b17cc74418a3d84c106329dfb7ce80219d702ce9b440a529e64a4c3c49`.
This supplement does not edit those files, the original 180 formal final-test
evaluations, their predictions/checkpoints, or the previous RAE results.

Before each study the runner must bind actual source bytes, configuration, this
protocol, model revision, tokenizer/pooling settings, corpus/query/qrels hashes,
ordered document/query IDs, split IDs, cached-vector hashes and supervision
manifests. Archive source before execution; a hash collected only at process exit
does not prove which source was loaded. Use new study/output namespaces. Existing
complete artifacts may be reused only after verifying their identities and must be
marked reused; never relabel historical fits as newly executed.

## A. Second encoder: MPNet, fixed 128d comparison

Use `sentence-transformers/all-mpnet-base-v2` at revision
`e8c3b32edf5434bc2275fc9bab85f82640a19130`. Freeze the text model. Both query and
document prefixes are the empty string. Use its transformer and mean pooling,
maximum length 384 including tokenizer special tokens, 768-dimensional float32
embeddings, and L2-normalized output. Preserve the existing title/text composition
and complete corpus candidates. Bind new MPNet caches separately from E5 caches.
These are normalized BEIR embeddings, distinct from the raw-pooled IMDb inputs
used in the earlier RAE reproduction audit.

For each of the three datasets compare Original, PCA, RP, AE, controlled RAE and
QRAE. All compressed representations are 128d; Original remains 768d. RP/AE/RAE/QRAE
use seeds 17, 42 and 2026. Original and deterministic PCA each have one record.
The planned count is therefore 14 development configurations per dataset, **42
development configurations plus 42 frozen-model test evaluations**. This includes
27 neural fits, nine RP projections and three PCA fits; Original does not train a
compressor. Test evaluation adds no fits.

Transfer the original selected coefficients unchanged: controlled RAE and QRAE
lambda=1e-5, QRAE alpha=10, margin=0.1, random negatives. AE has neither penalty nor
ranking supervision. Do not tune these parameters for MPNet or select a favorable
seed, dataset, checkpoint or method subset. After all 42 development configurations
have completed or been explicitly recorded as failed, freeze their configurations,
results and artifact hashes before any supplementary MPNet test stage. Evaluate
every successful frozen configuration on its designated test records without
refitting. A failed configuration stays a failure in the reported full matrix.

This tests transferred hyperparameters across embedding pipelines. MPNet's empty
prefixes and 384-token limit differ from E5's documented query/passage prefixes and
512-token limit. Any cross-model difference therefore includes those processing
differences; it cannot be attributed solely to the encoder architecture. Geometry
is measured against each encoder's own Original reference, not MPNet against E5's
neighbors. The already inspected test collections make this an exploratory extension
even though the new settings are fixed before its runs.

## B. E5 joint lambda-by-alpha sensitivity, development only

Reuse the existing E5-base-v2 caches at revision
`f52bf8ec8c7124536f0efb74aca902b2995e5bcd`, including their query/passage prefixes,
512-token limit, float32 normalization, document order and split bindings. Use only
SciFact and FiQA, dimension 128, and the existing development queries.

| Factor | Fixed values |
|---|---|
| Explicit Frobenius coefficient lambda | 0, 1e-6, 1e-5, 1e-4 |
| Ranking coefficient alpha | 0, 1, 10, 100 |
| Training seeds | 17, 42, 2026 |
| Other supervision choices | Margin 0.1; one random negative per positive pair |

Run every cell as the **QRAE training class**, including alpha=0. The complete
matrix contains **2 datasets x 4 lambdas x 4 alphas x 3 seeds = 96 development-only
neural fits**. These coefficients multiply the controlled objective, not the
nominal coupled-Adam decay used in the IMDb reproduction.

For a given dataset/seed, keep encoder/decoder initialization, reconstruction
minibatches, positive-pair/negative sampling schedule and fixed training probes
identical across all 16 coefficient cells. Separate RNG streams must keep ranking
sampling from changing reconstruction batches. Draw and diagnose the same triplets
even when alpha=0; do not switch that cell to an unsupervised class that skips the
ranking stream. Its ranking gradient is zero by coefficient, although training
labels remain available for diagnostics. With alpha=0 the mathematical objective
is reconstruction plus lambda regularization; lambda=alpha=0 retains reconstruction,
not Rank-only. This statement does not assume untested bitwise equivalence to a
separately executed AE/RAE implementation.

Keep and report all 16 cells for both datasets, including numerical failures.
Describe sensitivity and interactions using the complete surfaces and planned
zero-component controls. Do not promote a best-looking cell to a new selected
model, claim a global optimum, evaluate this grid on test, or use it to revise
the MPNet or original E5 coefficients. No grid-derived configuration will receive
a supplementary final-test score under this protocol.

## Shared training, evaluation and analysis

Keep the controlled centered, no-bias, untied linear encoder/decoder architecture.
Reconstruction is a batch mean of dimension-summed squared error; regularization
is the sum of encoder/decoder squared Frobenius norms. Ranking is the mean
one-negative normalized-cosine hinge. Pair-uniform train sampling, known-positive
exclusions and duplicate-group rules remain those of the frozen protocol. Unjudged
negatives are not guaranteed irrelevant; never clean them with dev/test qrels.

All neural fits use batch 128, Adam with built-in weight decay zero, 3,000 updates,
learning rate 0.001 with cosine decay to 0.00001, and the final fixed checkpoint.
No retrieval-based early stopping, gradient clipping or checkpoint selection is
introduced. Record component losses, gradients, latent norms, learning rates,
fixed training probes and final spectra; preserve nonfinite/collapse failures.
Use the project GPU for neural fitting and CPU exact search, with actual device,
library versions, encoding batch size and thread counts bound in launch records.
Hardware batching choices must not depend on retrieval scores.

Every comparison uses the same full-corpus normalized-float32 FAISS IndexFlatIP
search, deterministic tie handling, top-100 ranking output and four search CPU
threads. Primary effectiveness is nDCG@10 with linear relevance gain and full-qrels
IDCG. Preserve Recall@10/100, MRR@10/100, query-document overlap@10 and self-excluded
document-document overlap@10 with the existing 256 fixed anchors. Retain per-query
scores, rankings, configuration identities, and all stochastic seeds.

Report means and sample SDs over three seeds; do not pool seed-times-query as
independent observations. Primary MPNet contrasts are QRAE-minus-RAE at 128d for
each dataset; retain AE/PCA/RP/Original context. Any paired seed/query bootstrap
interval uses the existing shared-query/seed resampling, 10,000 draws and seed
2026, remains exploratory, and is not multiplicity-adjusted. The grid reports
development sensitivity, not confirmatory significance or test generalization.
Retain storage and descriptive fit/index/search measurements; concurrent runtime
figures are not a new controlled end-to-end latency benchmark.

## Stopping, failures and reporting boundaries

The scientific scope is fixed at the two matrices above. There are no adaptive
additions of encoders, datasets, dimensions, coefficients, seeds or training
budgets. Interrupted or failed attempts retain their logs, causes and manifests.
Resume an unchanged configuration only with its declared completion semantics;
never disguise a restart as continued optimizer state. Infrastructure repair may
be recorded transparently, but score-driven retries or substitutions are excluded.
If the available day or hardware does not permit completion, publish the actual
coverage and remaining cells rather than silently shrinking the planned matrix.

All old main results and all supplementary successes/failures remain visible.
The final report will distinguish the original 180 frozen evaluations, the second
encoder's 42 development/42 test configurations, and the 96 development-only grid
fits. Do not merge them into a larger claimed original preregistered study. Updated
interpretation may describe conditional transfer or sensitivity, but cannot claim
universal superiority, a fully identified original-author RAE protocol, or
large-corpus performance that was not measured. No result is asserted by this plan.
