# COMP5331 Group 14 — Task-Aware Embedding Compression

**Research project:** Beyond Nearest-Neighbor Preservation: Task-Aware Embedding
Compression for Dense Retrieval.

This repository contains the group's code for comparing **Original, PCA, Gaussian
random projection, linear AE, controlled RAE, QRAE, and Rank-only** on frozen text
embeddings. We study whether preserving original neighbors also preserves relevant
document rankings. QRAE adds train-query relevance supervision to a shared linear
projection; the projection is not generated separately for each query.

Repository: <https://github.com/Xuning-Hu/comp5331-retrieval-compression>

## For the course marker

| Required code documentation | Location |
|---|---|
| How to compile / install | [Installation](#installation) |
| How to execute | [Quick start](#quick-start) and [Full experiments](#full-experiments) |
| Description of each source file | [Source guide](docs/SOURCE_GUIDE.md) |
| An example run | [Offline example](#offline-example) |
| Operating system tested | [Tested environment](#tested-environment) |
| Additional information | [Research scope](#research-scope), [Results](results/README.md), and [Third-party attribution](THIRD_PARTY.md) |

The presentation PPT and presentation order form are submitted separately through
Canvas. The source-code URL in this README is the code submission location.

## Installation

Use **Python 3.12**. Python source does not require a separate compilation step.
The commands below install the package and prebuilt numerical dependencies.
Run all commands from this repository's root directory.

```text
git clone https://github.com/Xuning-Hu/comp5331-retrieval-compression.git
cd comp5331-retrieval-compression
```

Windows PowerShell (activation is unnecessary):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m retrieval_lab.cli demo
```

macOS / Linux equivalent (portability instructions, not a claim that the full
course experiment was run on Linux):

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m retrieval_lab.cli demo
```

If `uv` is already installed, a locked dependency resolution is also provided:

```text
uv sync --locked --extra dev
uv run retrieval-lab demo
```

The first dependency installation requires internet access. The synthetic demo,
tests, and saved-score replay do not download models or datasets. Dependencies
include NumPy, PyTorch, Sentence Transformers and FAISS; this is not a pure Python
numerical implementation. `requirements-course-windows.txt` records the observed
CUDA experiment environment, while `uv.lock` records the portable dependency
resolution. Hardware-specific CUDA installation is separate from the CPU demo.

## Quick start

In the following commands, `python` means the Python executable in the installed
environment (for example `.\.venv\Scripts\python.exe` on Windows).

```text
python -m retrieval_lab.cli --help
python -m retrieval_lab.cli demo --output runs/demo
python -m pytest -q
python -m ruff check src tests scripts
```

### Offline example

```text
python -m retrieval_lab.cli demo --output runs/demo
```

This generates 160 synthetic document vectors and 24 synthetic queries, runs
Original/PCA/RP/AE at compressed widths 8 and 16 with seeds 17 and 42 where
applicable, and prints the generated report path. Inspect the resulting study's
`report.md`, `summary.csv` and per-run metadata under `runs/demo/`.
This checks that the pipeline works; **synthetic scores are not research results**.
The additional RAE/QRAE objectives, sampling and frozen evaluation are covered by
the test suite and the real-data commands below.

To inspect the recorded research scores without installing machine-learning
dependencies, use Python's standard library only:

```text
python scripts/replay_public_results.py
```

This verifies the numeric export hashes, reaggregates saved query scores, checks
their means and seed standard deviations against the exported summaries, and
prints the main 128-dimensional nDCG@10 table. It is **score aggregation**, not a
fresh retrieval, training, or independent ranking/qrels replay.

## Full experiments

Read the [main protocol](docs/experiment-protocol.md) before starting a new study.
The course runners use CUDA for neural training/embedding and CPU exact search.
The recorded course host had an NVIDIA RTX 4060 Laptop GPU (8 GiB). The small
offline example works on CPU. To use the course runners, first install a PyTorch
build compatible with your CUDA driver and check `torch.cuda.is_available()`.
For CPU-only real-data work, use the general CLI with an explicitly adapted TOML
configuration; the course orchestration also fixes CUDA in `course_experiments.py`.
Record changed hardware/configuration as a separate reproduction.

### 1. Download and embed each complete corpus

```text
python -m retrieval_lab.cli prepare --config configs/course-scifact-pilot.toml
python -m retrieval_lab.cli embed --config configs/course-scifact-pilot.toml
python -m retrieval_lab.cli prepare --config configs/course-nfcorpus.toml
python -m retrieval_lab.cli embed --config configs/course-nfcorpus.toml
python -m retrieval_lab.cli prepare --config configs/course-fiqa.toml
python -m retrieval_lab.cli embed --config configs/course-fiqa.toml
```

These commands download public BEIR data and the pinned E5 checkpoint. All three
caches must exist before the offline course wrapper starts. Large corpora,
embeddings, model weights and FAISS indexes are generated locally and excluded
from Git. The E5 revision is
`f52bf8ec8c7124536f0efb74aca902b2995e5bcd`.

### 2. Select on development data, freeze, then test

```text
python scripts/course_experiments.py tune
python scripts/run_course_pipeline.py
```

`tune` searches the declared SciFact development grid and refuses to overwrite an
existing selection. The wrapper completes the development matrix and freezes
every saved model before the six final-test studies. It records source hashes,
stage logs and execution order under `runs/course/pipeline/`. Training uses the
declared final checkpoint and seeds; no test labels enter training or mining.
These commands are a substantial experiment, not a short smoke test.

For a separately reproduced frozen model, use an actual local manifest:

```text
python scripts/course_experiments.py test --frozen runs/course/frozen-scifact.json
```

Use the guarded `test` stage or CLI `evaluate` to evaluate saved weights. The
general `run --split test` command can fit a new study and is not the frozen
final-evaluation procedure. Do not reuse another host's absolute artifact paths.

### 3. Optional supplementary studies

Follow the [supplementary protocol](docs/supplementary-experiment-protocol.md).
After completing the original E5 pipeline, cache MPNet with internet access:

```text
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2', revision='e8c3b32edf5434bc2275fc9bab85f82640a19130', device='cpu', trust_remote_code=False)"
python scripts/run_second_encoder.py
python scripts/run_weight_sensitivity.py plan
python scripts/run_weight_sensitivity.py run
python scripts/verify_supplementary.py all
```

Create a weight-grid plan once and resume with `run`. MPNet has 42 development
and 42 frozen-test evaluations. The 96 E5 weight-grid fits are development-only;
they do not select a replacement test model. The separate IMDb RAE verification
is described in [its protocol](docs/rae-reproduction-protocol.md).

## Tested environment

| Environment | What was tested |
|---|---|
| Windows 11, Python 3.12.4 | Full course E5/MPNet studies and local release checks; see [validation](docs/VALIDATION.md) |
| Intel i9-13980HX, approximately 32 GiB RAM | Course execution host; CPU exact FAISS search uses four threads |
| NVIDIA RTX 4060 Laptop, 8 GiB VRAM | Course CUDA embeddings and neural training |
| macOS | Earlier upstream baseline work; separate Torch/FAISS processes address a wheel OpenMP conflict |
| Linux | No full-course execution is claimed; the commands above are portability guidance |

## Research scope

- Main frozen E5 evaluation: **180 runs** = 159 main runs plus 21 SciFact ablation
  runs, across SciFact, NFCorpus and FiQA; dimensions 64/128/256/384; stochastic
  seeds 17/42/2026. Original stays at 768 dimensions.
- Supplementary MPNet: **42 dev + 42 frozen test records** at 128 dimensions.
- E5 coefficient sensitivity: **96 development-only fits**. These exploratory
  follow-ups do not replace the original frozen test results.
- Full-corpus exact FAISS `IndexFlatIP`, normalized float32 vectors and common
  relevance metrics are shared across methods. This is corpus-adaptive,
  in-domain compression, not zero-shot transfer to an unseen corpus.

Final-test mean nDCG@10 at 128 dimensions (Original uses 768):

| Method | SciFact | NFCorpus | FiQA |
|---|---:|---:|---:|
| Original | 0.71943 | 0.35444 | 0.39867 |
| PCA | 0.66625 | 0.28591 | 0.30618 |
| Gaussian RP | 0.51519 | 0.22934 | 0.22457 |
| AE | 0.67616 | 0.30803 | 0.33026 |
| Controlled RAE | 0.67738 | 0.30604 | 0.33076 |
| QRAE | 0.76185 | 0.29301 | 0.30130 |

QRAE improves SciFact at these settings but regresses on NFCorpus and FiQA at
128d. This does not support universal superiority. Controlled RAE is an
adaptation for the shared comparison, not an exact reproduction of the original
paper. The authors' complete original inputs and configuration remain unknown.
The corpora are modest in size; exact-search timings are not production ANN
service guarantees. A sixfold vector-size reduction is not a sixfold end-to-end
speedup. See the protocols for methodological details and limitations.

## Repository contents and provenance

- `src/retrieval_lab/`: core implementation, training, exact search and metrics.
- `scripts/`: experiment orchestration, audits, analysis and optional report tools.
- `configs/`: seven TOML experiment configurations.
- `tests/`: offline regression and numerical checks.
- `results/`: unchanged numeric excerpts supporting saved-score replay.
- `docs/`: protocols, per-file guide, validation and the historical test fixture.

[SOURCE_GUIDE.md](docs/SOURCE_GUIDE.md) describes every Python source file and
configuration. Report-production scripts are retained as optional tooling; they
require manuscript fragments and raw evidence that are not part of this public
code export. The latest submitted report/PPT are separate course deliverables.

This public export extends Changxuan Fan's
[upstream repository](https://github.com/changxuanfan/retrieval-compression-lab)
at commit `c19684e4b1d1b9cc75a2d20ccbe82dafc911834b`. The original MIT license is
retained. This is a release snapshot, not a reconstruction of the group's commit
history. The public preparation changes documentation and adds a score-replay
utility; the existing scientific Python sources are copied byte for byte.
No new research experiment is claimed by the publication checks.

AI assistance was used for implementation, analysis, verification and report
editing, as disclosed in the course report. Group members remain responsible
for reviewing the work and describing their own contributions accurately.
Third-party datasets/models retain their original terms; see
[THIRD_PARTY.md](THIRD_PARTY.md) and [LICENSE](LICENSE).
