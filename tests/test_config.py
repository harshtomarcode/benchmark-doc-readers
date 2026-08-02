from pathlib import Path

from docbench.config import load_config


def test_load_mock_config():
    cfg = load_config(Path("configs/example_mock.yaml"))
    assert cfg.model.type == "mock"
    assert "text_reading" in cfg.qualitative.tasks
    assert "pure_text" in cfg.stress.workloads
    assert cfg.stress.workloads["image_to_text"].modality == "image"


def test_env_expansion(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_DOC_KEY", "secret-value")
    p = tmp_path / "c.yaml"
    p.write_text(
        "model:\n  type: api\n  id: m\n  api_key: ${TEST_DOC_KEY}\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.model.api_key == "secret-value"