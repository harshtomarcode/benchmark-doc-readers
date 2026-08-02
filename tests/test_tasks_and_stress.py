from pathlib import Path

from docbench.config import load_config
from docbench.models.factory import build_provider
from docbench.models.mock import MockModelProvider
from docbench.runner import run_benchmark
from docbench.tasks.registry import get_task_class
from docbench.stress.latency import run_latency
from docbench.stress.concurrency import run_concurrency


def test_text_task_loads_samples():
    task = get_task_class("text_reading")("datasets/text")
    samples = task.load_samples()
    assert len(samples) >= 3
    assert samples[0].question


def test_chart_task_requires_images():
    task = get_task_class("chart_reading")("datasets/charts")
    assert task.requires_images
    samples = task.load_samples()
    req = task.build_request(samples[0])
    assert req.images


def test_mock_provider_answers():
    p = MockModelProvider(answers={"hello": "world"}, latency_ms=1)
    from docbench.models.base import ModelRequest

    r = p.generate(ModelRequest(prompt="hello"))
    assert r.text == "world"
    assert r.ok


def test_latency_and_concurrency_smoke():
    provider = MockModelProvider(latency_ms=5)
    lat = run_latency(
        provider,
        workload="t",
        dataset="datasets/text",
        modality="text",
        warmup=1,
        iterations=3,
    )
    assert lat.stats["mean_ms"] >= 0
    conc = run_concurrency(
        provider,
        workload="t",
        dataset="datasets/text",
        modality="text",
        concurrency_levels=[1, 2],
        requests_per_level=4,
    )
    assert len(conc.levels) == 2
    provider.close()


def test_end_to_end_mock_run(tmp_path):
    cfg = load_config(Path("configs/example_mock.yaml"))
    cfg.output.dir = str(tmp_path / "results")
    cfg.output.run_name = "pytest_run"
    # Keep stress tiny
    for wl in cfg.stress.workloads.values():
        wl.iterations = 2
        wl.warmup = 0
        wl.concurrency_levels = [1]
        wl.requests_per_level = 2

    # Pin some answers so metrics are non-trivial
    provider_cfg = cfg.model.model_dump(exclude_none=True)
    provider_cfg["options"] = {
        "latency_ms": 1,
        "default_answer": "$12.4 million",
    }
    # Monkey via building answers that match first text sample roughly — just run e2e
    payload = run_benchmark(cfg)
    assert Path(cfg.output.dir, "pytest_run", "results.json").exists()
    assert Path(cfg.output.dir, "pytest_run", "report.md").exists()
    assert len(payload["qualitative"]) == 5
    assert len(payload["latency"]) == 2


def test_build_provider_factory():
    p = build_provider({"type": "mock", "id": "m", "options": {"latency_ms": 1}})
    assert isinstance(p, MockModelProvider)
    p.close()