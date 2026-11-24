# Public numeric evidence

`manifest.json` lists 108 exported studies: six E5 final-test studies (180 runs),
six MPNet studies (42 dev + 42 test runs), and 96 E5 development-only weight-grid
studies. Each directory contains `per-query.json`, `summary.json`, and `summary.csv`,
copied byte for byte from the original saved export. The manifest records their
SHA-256 hashes and the original snapshot hash; numeric values were not edited.

Run `python scripts/replay_public_results.py` from the repository root. It checks
file identity, query counts, metric means, sample SD across seeds, and complete
run coverage. It also prints the main 128d nDCG@10 table.

This is a lightweight numeric excerpt, not the complete original snapshot.
Absolute host paths, frozen-model files, raw documents, full rankings, and original
audit bundles are not included. The original snapshot hash is a provenance reference;
the original snapshot itself is not provided here. The public release does not
claim to independently reverify training chronology or ranking/qrels correctness.
Use the training/evaluation scripts and public original datasets to reproduce
those steps. `docs/results/scifact-dev` is an earlier four-baseline development
fixture required by a regression test, not the final E5 study.
