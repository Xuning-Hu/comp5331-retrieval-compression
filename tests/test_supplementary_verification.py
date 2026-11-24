"""Offline checks for stale exports and the sealed supplementary chronology.

These fixtures never read real experiment outputs or start a training worker.
"""

import copy
import runpy
from pathlib import Path

import pytest

from retrieval_lab.common import read_json, sha256, write_json

SCRIPTS = Path(__file__).parents[1] / "scripts"
SNAPSHOTS = runpy.run_path(str(Path(__file__).with_name("test_course_snapshot.py")))


@pytest.fixture
def verifier(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return runpy.run_path(str(SCRIPTS / "verify_supplementary.py"))


def make_export(tmp_path):
    source = SNAPSHOTS["all_methods_study"](tmp_path)
    target = SNAPSHOTS["EXPORT"]["export"](source, tmp_path / "export")
    return source, target


def test_same_verified_export_can_be_reused_without_changes(tmp_path, verifier):
    source, target = make_export(tmp_path)
    before = {p: sha256(p) for p in target.rglob("*") if p.is_file()}
    assert verifier["checked_export"](source, target) == target
    assert before == {p: sha256(p) for p in target.rglob("*") if p.is_file()}


def test_same_study_id_does_not_allow_changed_source_manifest(tmp_path, verifier):
    source, target = make_export(tmp_path)
    manifest = read_json(source / "study.json")
    manifest["changed_after_export"] = True
    write_json(source / "study.json", manifest)
    with pytest.raises(ValueError):
        verifier["checked_export"](source, target)


def test_same_study_id_does_not_allow_internally_rehashed_model(tmp_path, verifier):
    source, target = make_export(tmp_path)
    folder = source / "qrae-4d-seed42"
    (folder / "compressor.npz").write_bytes(b"another model with consistent result hash")
    result = read_json(folder / "result.json")
    result["files"]["compressor.npz"] = sha256(folder / "compressor.npz")
    write_json(folder / "result.json", result)
    with pytest.raises(ValueError):
        verifier["checked_export"](source, target)


def test_existing_export_cannot_silently_gain_unbound_verification(tmp_path, verifier):
    source, target = make_export(tmp_path)
    audit = tmp_path / "new-audit.json"
    write_json(audit, {"status": "passed", "studies": []})
    with pytest.raises(ValueError):
        verifier["checked_export"](source, target, verification=audit)


def test_existing_export_checks_ancillary_registered_files(tmp_path, verifier):
    source, target = make_export(tmp_path)
    # The general analysis reader does not consume README.md. Export reuse must
    # still validate it because snapshot.json explicitly registers its bytes.
    (target / "README.md").write_text("changed explanation", encoding="utf-8")
    with pytest.raises(ValueError):
        verifier["checked_export"](source, target)


@pytest.fixture
def grid_chronology(tmp_path, monkeypatch, verifier):
    """Use sealed small receipts; isolate the expensive ranking replay below.

    The production runner's receipt semantics and array checks are covered by
    test_weight_sensitivity. Here the contract is summary-to-receipt identity.
    """
    monkeypatch.chdir(tmp_path)
    folder = Path("runs/supplementary/weight-sensitivity")
    (folder / "completed").mkdir(parents=True)
    jobs, receipts, studies = [], [], []
    for index in range(96):
        job_id = f"fixture-{index:02d}"
        job = {"job_id": job_id}
        study = f"runs/fixture/{job_id}"
        receipt = {
            "plan_id": "fixed-plan",
            "job": job,
            "status": "complete",
            "study_path": study,
            "metrics": {"ndcg@10": 0.5},
            "finished_utc": "2026-11-04T00:01:00+00:00",
        }
        jobs.append(job)
        receipts.append(receipt)
        studies.append(study)
        write_json(folder / "completed" / f"{job_id}.json", receipt)
    plan = {"plan_id": "fixed-plan", "jobs": jobs}
    write_json(folder / "plan.json", plan)
    (folder / "plan.sha256").write_text(sha256(folder / "plan.json"), encoding="ascii")
    (folder / "execution-source.zip").write_bytes(b"fixture archive; not training source")
    progress = {
        "plan_id": plan["plan_id"],
        "plan_sha256": sha256(folder / "plan.json"),
        "status": "complete",
        "expected_fits": 96,
        "test_evaluated": False,
        "jobs": [{**receipt, "reused_this_invocation": False} for receipt in receipts],
        "studies": studies,
        "end_input_verification": {
            "source_protocol_cache_data_hashes_unchanged": True,
            "plan_sha256": sha256(folder / "plan.json"),
        },
    }
    write_json(folder / "progress.json", progress)
    state = {"replays": 0, "exports": 0}
    namespace = verifier["audit_grid"].__globals__
    monkeypatch.setattr(namespace["grid"], "validate_plan", lambda _: plan)
    monkeypatch.setattr(namespace["grid"], "verify_completion", lambda *args: None)

    def replay(paths, output):
        state["replays"] += 1
        assert [path.as_posix() for path in paths] == studies
        write_json(
            output / "verification.json",
            {"status": "passed", "run_count": 96, "studies": [{"split": "dev"}] * 96},
        )

    def export(*args, **kwargs):
        state["exports"] += 1

    monkeypatch.setitem(namespace, "verify", replay)
    monkeypatch.setitem(namespace, "checked_export", export)
    return verifier, folder, copy.deepcopy(progress), state


def test_grid_summary_must_include_every_sealed_job(grid_chronology):
    verifier, folder, progress, state = grid_chronology
    progress["jobs"].pop()
    write_json(folder / "progress.json", progress)
    with pytest.raises(ValueError):
        verifier["audit_grid"]()
    assert state == {"replays": 0, "exports": 0}


def test_grid_summary_scores_cannot_disagree_with_receipts(grid_chronology):
    verifier, folder, progress, state = grid_chronology
    progress["jobs"][0]["metrics"] = {"ndcg@10": 0.99}
    write_json(folder / "progress.json", progress)
    with pytest.raises(ValueError):
        verifier["audit_grid"]()
    assert state == {"replays": 0, "exports": 0}


def test_grid_final_input_verification_is_required(grid_chronology):
    verifier, folder, progress, state = grid_chronology
    progress["end_input_verification"]["source_protocol_cache_data_hashes_unchanged"] = False
    write_json(folder / "progress.json", progress)
    with pytest.raises(ValueError):
        verifier["audit_grid"]()
    assert state == {"replays": 0, "exports": 0}


def test_grid_accepts_resume_flag_without_changing_sealed_contents(grid_chronology):
    verifier, folder, progress, state = grid_chronology
    progress["jobs"][0]["reused_this_invocation"] = True
    write_json(folder / "progress.json", progress)
    record = verifier["audit_grid"]()
    assert record["dev_runs"] == 96 and record["test_runs"] == 0
    assert state == {"replays": 1, "exports": 96}


@pytest.fixture
def mpnet_chronology(tmp_path, monkeypatch, verifier):
    monkeypatch.chdir(tmp_path)
    namespace = verifier["audit_mpnet"].__globals__
    root = Path("runs/supplementary/mpnet")
    invocation = root / "invocations/20261104T000000Z"
    invocation.mkdir(parents=True)
    (root / "execution-source.zip").write_bytes(b"fixture source archive")
    datasets = ("scifact", "nfcorpus", "fiqa")
    config = {"embedding": {"model": "fixture-mpnet"}, "experiment": {"split": "dev"}}
    plan = {
        "configs": {dataset: copy.deepcopy(config) for dataset in datasets},
        "source_archive_sha256": sha256(root / "execution-source.zip"),
    }
    write_json(root / "plan.json", plan)
    jobs, frozen = [], {}
    for stage in ("embed", "dev", "test"):
        for dataset in datasets:
            name = f"{dataset}-{stage}"
            jobs.append(
                {
                    "name": name,
                    "status": "complete",
                    "exit_code": 0,
                    "started_utc": "2026-11-04T00:02:00+00:00"
                    if stage == "test"
                    else "2026-11-04T00:00:00+00:00",
                    "finished_utc": "2026-11-04T00:03:00+00:00"
                    if stage == "test"
                    else "2026-11-04T00:01:00+00:00",
                }
            )
            (invocation / f"{name}.log").write_text("fixture complete\n", encoding="utf-8")
    write_json(
        invocation / "progress.json",
        {
            "status": "complete",
            "jobs": jobs,
            "plan_sha256": sha256(root / "plan.json"),
        },
    )
    for dataset in datasets:
        dev, test = root / dataset / "dev", root / dataset / "test"
        write_json(test / "study.json", {"source_study": str(dev)})
        write_json(root / f"test-{dataset}.json", {"study": str(test), "source": str(dev)})
        write_json(root / f"frozen-{dataset}.json", {"fixture": True})
        frozen[dataset] = (copy.deepcopy(config), dev)
    state = {"replays": 0, "exports": 0}
    monkeypatch.setattr(namespace["second"], "ROOT", root)
    monkeypatch.setattr(namespace["second"], "assert_inputs", lambda _: None)
    monkeypatch.setattr(
        namespace["course"],
        "validate_frozen",
        lambda path: frozen[path.stem.removeprefix("frozen-")],
    )

    def replay(paths, output):
        state["replays"] += 1
        write_json(
            output / "verification.json",
            {
                "status": "passed",
                "run_count": 84,
                "studies": [
                    {"dataset": dataset, "split": split, "runs": [{}] * 14}
                    for dataset in datasets
                    for split in ("dev", "test")
                ],
            },
        )

    def export(*args, **kwargs):
        state["exports"] += 1

    monkeypatch.setitem(namespace, "verify", replay)
    monkeypatch.setitem(namespace, "checked_export", export)
    return verifier, root, frozen, state


def test_mpnet_declared_plan_cannot_be_replaced_by_same_size_other_model(mpnet_chronology):
    verifier, _, frozen, state = mpnet_chronology
    frozen["scifact"][0]["embedding"]["model"] = "fixture-e5"
    with pytest.raises(ValueError):
        verifier["audit_mpnet"]()
    assert state == {"replays": 0, "exports": 0}


@pytest.mark.parametrize("location", ["pointer", "study"])
def test_mpnet_test_must_identify_the_exact_frozen_development(mpnet_chronology, location):
    verifier, root, _, state = mpnet_chronology
    target = root / ("test-scifact.json" if location == "pointer" else "scifact/test/study.json")
    record = read_json(target)
    record["source" if location == "pointer" else "source_study"] = "runs/other-dev"
    write_json(target, record)
    with pytest.raises(ValueError):
        verifier["audit_mpnet"]()
    assert state == {"replays": 0, "exports": 0}


def test_mpnet_complete_chronology_replays_six_studies_then_exports(mpnet_chronology):
    verifier, _, _, state = mpnet_chronology
    record = verifier["audit_mpnet"]()
    assert record["dev_runs"] == record["test_runs"] == 42
    assert record["all_dev_finished_before_test"] is True
    assert state == {"replays": 1, "exports": 6}


def test_package_cannot_accept_counts_with_only_a_protocol_file(tmp_path):
    protocol = tmp_path / "docs/supplementary-experiment-protocol.md"
    protocol.parent.mkdir()
    protocol.write_text("fixed fixture protocol", encoding="utf-8")
    write_json(
        tmp_path / "docs/verification/supplementary-completion.json",
        {
            "status": "passed",
            "original_evidence_unchanged": True,
            "counts": {
                "mpnet_dev": 42,
                "mpnet_test": 42,
                "weight_grid_dev": 96,
                "weight_grid_test": 0,
                "new_neural_fits": 123,
            },
            "delivery_file_sha256": {protocol.relative_to(tmp_path).as_posix(): sha256(protocol)},
        },
    )
    package = runpy.run_path(str(SCRIPTS / "package_course_submission.py"))
    with pytest.raises(ValueError, match="omits required"):
        package["verify_supplementary_delivery"](tmp_path)
