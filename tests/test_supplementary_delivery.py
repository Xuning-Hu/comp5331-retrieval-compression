"""Submission must reject missing follow-up coverage and altered checked evidence."""
import hashlib
import json
import runpy
from pathlib import Path

import pytest

VERIFY = runpy.run_path(str(Path(__file__).parents[1] / "scripts/package_course_submission.py"))[
    "verify_supplementary_delivery"
]


@pytest.fixture
def delivery(tmp_path):
    (tmp_path / "docs/verification").mkdir(parents=True)
    evidence = tmp_path / "docs/supplementary-experiment-protocol.md"
    evidence.write_text("Fixed two-matrix protocol", encoding="utf-8")
    names = [
        "docs/results/supplementary/protocol/mpnet/plan.json",
        "docs/results/supplementary/protocol/mpnet/execution-source.zip",
        "docs/results/supplementary/protocol/weight-grid/plan.json",
        "docs/results/supplementary/protocol/weight-grid/plan.sha256",
        "docs/results/supplementary/protocol/weight-grid/execution-source.zip",
        "docs/results/supplementary/weight-grid/snapshot-map.json",
        "docs/analysis/supplementary/analysis-manifest.json",
        "docs/analysis/supplementary/summary.json",
    ]
    names += [f"docs/results/supplementary/mpnet/{ds}-{split}/snapshot.json"
              for ds in ("scifact", "nfcorpus", "fiqa") for split in ("dev", "test")]
    names += [f"docs/results/supplementary/weight-grid/cell-{i}/snapshot.json" for i in range(96)]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    for name, splits in (("mpnet", {"dev": 42, "test": 42}), ("weight-grid", {"dev": 96})):
        path = tmp_path / f"docs/verification/supplementary/{name}/verification.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "passed", "run_count": sum(splits.values()),
                                   "studies": [{"split": split, "runs": [{}] * count}
                                               for split, count in splits.items()]}), encoding="utf-8")
    record = {
        "status": "passed", "original_evidence_unchanged": True,
        "counts": {"mpnet_dev": 42, "mpnet_test": 42, "weight_grid_dev": 96,
                   "weight_grid_test": 0, "new_neural_fits": 123},
        "delivery_file_sha256": {p.relative_to(tmp_path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (tmp_path / "docs").rglob("*") if p.is_file()},
    }
    (tmp_path / "docs/verification/supplementary-completion.json").write_text(json.dumps(record))
    return tmp_path, record, evidence


def test_verified_delivery_then_changed_evidence(delivery):
    root, _, evidence = delivery
    VERIFY(root)
    evidence.write_text("Different post-result protocol", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after"):
        VERIFY(root)


def test_incomplete_grid_cannot_be_packaged(delivery):
    root, record, _ = delivery
    record["counts"]["weight_grid_dev"] = 95
    (root / "docs/verification/supplementary-completion.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="incomplete"):
        VERIFY(root)


def test_no_completion_record_cannot_be_packaged(delivery):
    root, _, _ = delivery
    (root / "docs/verification/supplementary-completion.json").unlink()
    with pytest.raises(ValueError, match="lack completion"):
        VERIFY(root)
