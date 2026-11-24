# Source file guide

All commands are run from the repository root after installation.

## Core package

| File | Responsibility |
|---|---|
| `src/retrieval_lab/__init__.py` | Package identity and version. |
| `src/retrieval_lab/cli.py` | Argument parsing for prepare/embed/run/pipeline/demo/report/evaluate/query. |
| `src/retrieval_lab/common.py` | Serialization, numeric normalization, fingerprints and SHA-256 utilities. |
| `src/retrieval_lab/compressors.py` | Original/PCA/Gaussian RP/linear AE transforms, persistence and dispatch. |
| `src/retrieval_lab/config.py` | Default settings, TOML loading, validation and repository-relative paths. |
| `src/retrieval_lab/data.py` | BEIR downloading/parsing, dataset splits and input provenance. |
| `src/retrieval_lab/embedding.py` | Pinned frozen-model encoding and immutable shared embedding caches. |
| `src/retrieval_lab/experiment.py` | Fitting, frozen evaluation, synthetic demo and artifact checksums. |
| `src/retrieval_lab/metrics.py` | Query-macro nDCG, recall and MRR with linear relevance gains. |
| `src/retrieval_lab/neural.py` | Controlled RAE/QRAE/Rank-only objectives and neural optimization. |
| `src/retrieval_lab/query.py` | New-text retrieval using a saved model, compressor and FAISS index. |
| `src/retrieval_lab/reporting.py` | Tables, figures and Markdown reports from measured study records. |
| `src/retrieval_lab/search.py` | Exact cosine FAISS search, deterministic tie handling and CPU timings. |
| `src/retrieval_lab/supervision.py` | Train-only positives and random/in-batch/static-hard negative sampling. |
| `src/retrieval_lab/worker.py` | Separate PyTorch training process to isolate native runtimes from FAISS. |

## Experiment, analysis and support scripts

| File | Responsibility |
|---|---|
| `scripts/analyze_course.py` | Read completed studies into traceable course-report tables and diagnostics. |
| `scripts/analyze_supplementary.py` | Describe fixed exploratory encoder and loss-weight follow-ups, without fitting. |
| `scripts/assemble_course_report.py` | Assemble the editable report fragments after supplementary evidence is complete. |
| `scripts/audit_study.py` | Independently replay saved results and diagnose AE/PCA geometry on dev only. |
| `scripts/audit_tokenization.py` | Measure preprocessing truncation using the exact cached tokenizer revision. |
| `scripts/benchmark_saved.py` | Time real single-query requests against immutable saved retrieval artifacts. |
| `scripts/build_course_assets.py` | Build report figures from complete, frozen test studies without fitting or selection. |
| `scripts/check_publication.py` | Check Git candidate files, including ZIP members, without printing matched data. |
| `scripts/check_rae_seed_robustness.py` | Bounded, post-hoc RAE training-seed check on the existing IMDb geometry split. |
| `scripts/compare_course_ablations.py` | Fixed descriptive ablation contrasts; no new test configuration selection. |
| `scripts/course_experiments.py` | Execute the prespecified course stages; test is an explicit separate command. |
| `scripts/enrich_course_evidence.py` | Attach observed query/document/rank evidence to symmetric numeric cases. |
| `scripts/export_course_snapshot.py` | Export every seed's small course evidence, excluding vectors/indexes/rankings. |
| `scripts/export_snapshot.py` | Export small measured artifacts for Git/report use, excluding models/data/indexes. |
| `scripts/finalize_supplementary_evidence.py` | Seal already verified follow-up exports and the unchanged original evidence. |
| `scripts/package_course_submission.py` | Package the reviewed report, code, compact evidence and execution source snapshot. |
| `scripts/prepare_imdb_reproduction.py` | Prepare an explicitly documented replacement IMDb/MPNet reproduction dataset. |
| `scripts/render_course_report.py` | Render editable Markdown as an academic A4 PDF with ReportLab. |
| `scripts/reproduce_rae.py` | Independent RAE geometry experiment; see docs/rae-reproduction-protocol.md. |
| `scripts/run_course_pipeline.py` | Run the already selected dev matrix, freeze it, then evaluate saved test models. |
| `scripts/run_second_encoder.py` | Bounded post-hoc MPNet transfer check; no new hyperparameter selection. |
| `scripts/run_weight_sensitivity.py` | Bounded E5 coefficient grid: plan once, then resume 96 dev-only QRAE fits. |
| `scripts/sanitize_public_artifacts.py` | Remove local repository paths from shareable snapshots, preserving raw runs. |
| `scripts/summarize_audit.py` | Build scientific diagnostics and balanced failure examples from saved dev runs. |
| `scripts/verify_course_results.py` | Independently recompute saved-ranking metrics without retrieval_lab.metrics. |
| `scripts/verify_metrics.py` | Cross-check a real study's graded nDCG against independent sklearn code. |
| `scripts/verify_supplementary.py` | Verify completed bounded follow-ups and export compact evidence, without fitting. |
| `scripts/replay_public_results.py` | Standard-library hash and saved-score aggregation check for the public numeric exports. |

Report-generation and packaging scripts retain their original input contracts.
They need manuscript fragments and full audit bundles from a complete local study;
they are not required to install the package, run the demo or replay public scores.

## Configurations

| File | Purpose |
|---|---|
| `configs/course-fiqa.toml` | Course CUDA preparation/pilot configuration. Inspect explicit overrides and defaults in `config.py`. |
| `configs/course-nfcorpus.toml` | Course CUDA preparation/pilot configuration. Inspect explicit overrides and defaults in `config.py`. |
| `configs/course-scifact-pilot.toml` | Course CUDA preparation/pilot configuration. Inspect explicit overrides and defaults in `config.py`. |
| `configs/fiqa.toml` | Original baseline dataset/pilot configuration. Inspect explicit overrides and defaults in `config.py`. |
| `configs/nfcorpus.toml` | Original baseline dataset/pilot configuration. Inspect explicit overrides and defaults in `config.py`. |
| `configs/scifact-pilot.toml` | Original baseline dataset/pilot configuration. Inspect explicit overrides and defaults in `config.py`. |
| `configs/scifact.toml` | Original baseline dataset/pilot configuration. Inspect explicit overrides and defaults in `config.py`. |

## Test files

Run `python -m pytest -q`. Tests use synthetic or retained small fixtures; no model download is required.

| File | Coverage |
|---|---|
| `tests/test_audit.py` | audit scalar metrics include missing queries and full qrels; crossed bootstrap keeps training seed uncertainty; crossed bootstrap constant difference is degenerate. |
| `tests/test_benchmark_saved.py` | query schedule is balanced randomized and reproducible; source loading is read only and saved search timing runs offline; source checksum mismatch is rejected; 7 tests total. |
| `tests/test_compressors.py` | pca matches sklearn geometry on out of sample queries; pca shared center and original are different; rp seed persistence and projection distribution; 9 tests total. |
| `tests/test_course_analysis.py` | crossed bootstrap retains seed uncertainty; crossed bootstrap pairs query and matching seed ids; analysis preserves sources and uses actual parameters; 10 tests total. |
| `tests/test_course_assets.py` | complete grid and sample sd and training weight; source dev fit cost is separate from zero frozen test fit; source dev timing rejects changed frozen result; 8 tests total. |
| `tests/test_course_evidence.py` | case text ranks negative metadata and report excerpt; short saved rankings and missing seed are explicit; changed evidence inputs are rejected; 6 tests total. |
| `tests/test_course_freeze.py` | freeze records results and all artifacts; consistent weight and result replacement is rejected before test; existing freeze is never overwritten; 6 tests total. |
| `tests/test_course_snapshot.py` | all seven methods all seeds export without large artifacts; tampered source model is rejected before export; incomplete seed grid is rejected; 9 tests total. |
| `tests/test_course_verification.py` | scalar metrics use linear graded gain and positive recall; dev audit does not open test labels; frozen artifacts and metrics verified across both splits; 8 tests total. |
| `tests/test_data.py` | query disjoint split is reproducible and preserves grades; train test query leakage fails; zip traversal is rejected. |
| `tests/test_frozen_evaluation.py` | frozen same queries reproduce results without fitting; test split uses frozen weights and records source; frozen evaluation rejects mismatched inputs; 6 tests total. |
| `tests/test_metrics.py` | graded ndcg matches independent standard without ties; unreturned queries still count and recall is not hit rate; duplicate retrieval is rejected and same string ids retained; 4 tests total. |
| `tests/test_neural.py` | rae zero regularization matches original ae numerically; cosine hinge hand calculation scale invariance and gradient direction; supervised worker roundtrip preserves and checks audit. |
| `tests/test_pipeline.py` | all four methods share one offline pipeline. |
| `tests/test_provenance.py` | short metric cutoffs still retrieve ten neighbors; environment changes create new studies and reuse preserves metadata. |
| `tests/test_query.py` | saved index unicode path preserves full search and cutoff ties. |
| `tests/test_rae_reproduction.py` | exact geometry excludes self by identity even for duplicate vectors; overlap uses set membership not rank agreement; split is disjoint and released budget records epoch rounding; 11 tests total. |
| `tests/test_rae_seed_robustness.py` | saved split preserves order and never resamples; invalid or changed split is rejected; only training seed changes and reused seed never fits; 4 tests total. |
| `tests/test_search.py` | exact search matches numpy cosine and caps small corpus; cutoff ties expand beyond initial topk and self exclusion. |
| `tests/test_second_encoder.py` | encoder transfer retains protocol; unknown dataset rejected; changed source rejected; 4 tests total. |
| `tests/test_supervision.py` | all strategies are train only numeric and reproducible; heldout labels and vectors cannot change training inputs or audit; hard pool uses original full corpus ranks then filters all positives; 12 tests total. |
| `tests/test_supplementary_analysis.py` | cross encoder analysis preserves all seeds queries and paired differences; exported metadata tampering is rejected; cross encoder different query identity is rejected; 6 tests total. |
| `tests/test_supplementary_delivery.py` | verified delivery then changed evidence; incomplete grid cannot be packaged; no completion record cannot be packaged. |
| `tests/test_supplementary_verification.py` | same verified export can be reused without changes; same study id does not allow changed source manifest; same study id does not allow internally rehashed model; 13 tests total. |
| `tests/test_weight_sensitivity.py` | fixed grid has all 96 cells and keeps zero axes in qrae; protocol departures are rejected before execution; complete zero weight fit keeps supervised diagnostics; 9 tests total. |

## Build and dependency files

| File | Purpose |
|---|---|
| `pyproject.toml` | Package build metadata, dependencies, CLI entry point, pytest and Ruff settings. |
| `uv.lock` | Portable locked dependency resolution. |
| `.python-version` | Python 3.12 used for this release. |
| `requirements-course-windows.txt` | Observed Windows/CUDA experiment dependencies. |
| `requirements-report.txt` | Optional report-generation dependencies. |
| `.github/workflows/ci.yml` | Automated lint, unit tests and numeric export replay. |
| `.gitignore`, `.gitattributes` | Exclude generated/private files and preserve evidence bytes. |
