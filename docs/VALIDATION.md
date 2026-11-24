# Public release validation — 2026-11-25

Checks were run on Windows 11 (build 26200), Python 3.12.4, using the existing
course environment's installed dependencies and the release's own `src` directory
on `PYTHONPATH`. This does not claim a clean-room reinstallation of all third-party
dependencies or a fresh rerun of the large research experiments.

| Check | Result |
|---|---|
| Existing scientific Python files | All 42 copied byte for byte; hashes in `SCIENTIFIC_SOURCE_SHA256.json` |
| Python wheel build | Passed: `python -m pip wheel . --no-deps` |
| Built wheel installation and CLI help | Passed in a separate installation target using existing dependencies |
| Ruff lint | Passed: `python -m ruff check src tests scripts` |
| Full offline test suite | **188 passed** in 195.46 seconds; one existing FAISS/NumPy deprecation warning |
| Offline synthetic demo | Passed; 11 fitted/evaluated configurations, report generated |
| Public numeric evidence replay | Passed: 108 studies, 360 runs, 2,828 mean/SD checks |
| Maximum absolute score aggregation difference | `2.220446049250313e-16` |
| Source-file documentation coverage | Every Python source/test script and TOML configuration is listed |
| Publication candidate check | Passed; generated data/model caches and personal course documents excluded |

Numeric replay preserves the original query identities, grouping, complete seed
coverage and saved numeric-file bytes. It does not independently reconstruct
rankings from original corpora or prove historical training chronology. The demo
uses synthetic inputs and is not counted as research evidence.

The public GitHub workflow separately runs dependency installation, lint, unit
tests, public-score replay and the publication check on a hosted macOS runner.
Its status is available in the repository's Actions tab; local results above do
not imply that a remote CI run has completed.
