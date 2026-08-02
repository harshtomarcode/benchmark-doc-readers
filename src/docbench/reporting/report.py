"""Write benchmark results as JSON and Markdown."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Template

MARKDOWN_TMPL = Template(
    """# Document Reading Benchmark Report

- **Run:** {{ run_name }}
- **Model:** {{ model_id }} ({{ model_type }})
- **Generated:** {{ timestamp }}

{% if qualitative %}
## Qualitative Results

| Task | Samples | Errors | exact_match | token_f1 | contains | avg_latency_ms |
|------|---------|--------|-------------|----------|----------|----------------|
{% for t in qualitative -%}
| {{ t.task_name }} | {{ t.n_samples }} | {{ t.n_errors }} | {{ "%.3f"|format(t.metrics.get("exact_match", 0)) }} | {{ "%.3f"|format(t.metrics.get("token_f1", 0)) }} | {{ "%.3f"|format(t.metrics.get("contains", 0)) }} | {{ "%.1f"|format(t.metrics.get("avg_latency_ms", 0)) }} |
{% endfor %}
{% endif %}

{% if latency %}
## Stress — Latency

| Workload | Modality | N | Mean (ms) | P50 | P95 | P99 | Errors |
|----------|----------|---|-----------|-----|-----|-----|--------|
{% for l in latency -%}
| {{ l.workload }} | {{ l.modality }} | {{ l.n }} | {{ "%.1f"|format(l.stats.mean_ms) }} | {{ "%.1f"|format(l.stats.p50_ms) }} | {{ "%.1f"|format(l.stats.p95_ms) }} | {{ "%.1f"|format(l.stats.p99_ms) }} | {{ l.errors }} |
{% endfor %}
{% endif %}

{% if concurrency %}
## Stress — Concurrency

{% for c in concurrency %}
### {{ c.workload }} ({{ c.modality }})

| Concurrency | Requests | Successes | Errors | Throughput (rps) | Mean Latency (ms) | P95 (ms) |
|-------------|----------|-----------|--------|------------------|-------------------|----------|
{% for lv in c.levels -%}
| {{ lv.concurrency }} | {{ lv.n_requests }} | {{ lv.successes }} | {{ lv.errors }} | {{ "%.2f"|format(lv.throughput_rps) }} | {{ "%.1f"|format(lv.mean_latency_ms) }} | {{ "%.1f"|format(lv.p95_latency_ms) }} |
{% endfor %}
{% endfor %}
{% endif %}
"""
)


def write_report(
    payload: dict[str, Any],
    output_dir: str | Path,
    *,
    formats: list[str] | None = None,
    run_name: str | None = None,
) -> Path:
    formats = formats or ["json", "markdown"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = run_name or payload.get("run_name") or f"run_{ts}"
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        **payload,
        "run_name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if "json" in formats:
        (run_dir / "results.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if "markdown" in formats or "md" in formats:
        md = MARKDOWN_TMPL.render(
            run_name=name,
            model_id=payload.get("model", {}).get("id", "?"),
            model_type=payload.get("model", {}).get("type", "?"),
            timestamp=payload["timestamp"],
            qualitative=payload.get("qualitative", []),
            latency=payload.get("latency", []),
            concurrency=payload.get("concurrency", []),
        )
        (run_dir / "report.md").write_text(md, encoding="utf-8")

    return run_dir