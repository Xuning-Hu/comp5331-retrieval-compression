# Working together

Clone this repository and install the development dependencies as described in
README.md. Create a branch for each change and open a pull request. Run the offline
test suite and Ruff before merging. Preserve the documented dataset splits, model
revisions, complete seed sets and frozen test procedure.

Do not commit datasets, embedding/model arrays, indexes, virtual environments,
student identifiers or private course documents. Numeric evidence in `results/`
is a release snapshot; new experiments should receive separately named study
directories and a new manifest, rather than silently replacing measured values.

The public release derives from the upstream baseline repository linked in
README.md. Preserve LICENSE and THIRD_PARTY.md. Describe individual contributions
accurately; the course's private contribution form is completed individually.
