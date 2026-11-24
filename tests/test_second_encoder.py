"""Guard the bounded second-encoder transfer scope and fixed historical weights."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_second_encoder as supplement


@pytest.fixture(autouse=True)
def fixed_selection(tmp_path, monkeypatch):
    path = tmp_path / "selection.json"
    supplement.write_json(path, {"regularization_weight": 1e-5, "ranking_weight": 10.0, "margin": 0.1})
    monkeypatch.setattr(supplement.course, "SELECTION", path)


def test_encoder_transfer_retains_protocol():
    for dataset in supplement.DATASETS:
        config = supplement.config_for(dataset)
        assert config["embedding"]["revision"] == supplement.REVISION
        assert config["embedding"]["query_prefix"] == config["embedding"]["document_prefix"] == ""
        assert config["embedding"]["max_length"] == 384
        assert config["experiment"]["split"] == "dev"
        assert config["experiment"]["dimensions"] == [128]
        assert config["experiment"]["seeds"] == [17, 42, 2026]
        assert config["neural"]["ranking_weight"] == 10.0
        assert config["neural"]["regularization_weight"] == 1e-5
        assert config["ae"]["steps"] == config["neural"]["steps"] == 3000
        assert config["supervision"]["negative_strategy"] == "random"


def test_unknown_dataset_rejected():
    with pytest.raises(ValueError, match="predeclared"):
        supplement.config_for("another")


def test_changed_source_rejected(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("old", encoding="utf-8")
    plan = {"input_sha256": {str(path): supplement.sha256(path)}}
    supplement.assert_inputs(plan)
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        supplement.assert_inputs(plan)


def test_changed_selected_weight_rejected(monkeypatch):
    original = supplement.course.selected_config
    def changed(dataset):
        config = original(dataset)
        config["neural"]["ranking_weight"] = 1.0
        return config
    monkeypatch.setattr(supplement.course, "selected_config", changed)
    with pytest.raises(ValueError, match="weights changed"):
        supplement.config_for("scifact")
