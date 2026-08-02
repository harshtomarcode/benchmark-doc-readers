"""Latency benchmarking for text and image-to-text workloads."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from docbench.models.base import ModelProvider, ModelRequest

console = Console()


@dataclass
class LatencyResult:
    workload: str
    modality: str
    n: int
    warmup: int
    latencies_ms: list[float]
    errors: int

    @property
    def stats(self) -> dict[str, float]:
        vals = self.latencies_ms
        if not vals:
            return {
                "mean_ms": 0.0,
                "p50_ms": 0.0,
                "p90_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "stdev_ms": 0.0,
            }
        sorted_v = sorted(vals)

        def pct(p: float) -> float:
            idx = min(len(sorted_v) - 1, max(0, int(round((p / 100) * (len(sorted_v) - 1)))))
            return sorted_v[idx]

        return {
            "mean_ms": statistics.mean(vals),
            "p50_ms": pct(50),
            "p90_ms": pct(90),
            "p95_ms": pct(95),
            "p99_ms": pct(99),
            "min_ms": min(vals),
            "max_ms": max(vals),
            "stdev_ms": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "modality": self.modality,
            "n": self.n,
            "warmup": self.warmup,
            "errors": self.errors,
            "stats": self.stats,
            "latencies_ms": self.latencies_ms,
        }


def _load_requests(
    dataset: str | Path,
    modality: str,
    prompt: str | None,
    n: int,
) -> list[ModelRequest]:
    """Build a pool of ModelRequests from a dataset directory or a single file."""
    path = Path(dataset)
    requests: list[ModelRequest] = []

    default_text_prompt = prompt or "Summarize the following text in one sentence:\n\n{document}"
    default_image_prompt = prompt or "Describe this image briefly."

    if path.is_file():
        text = path.read_text(encoding="utf-8")
        requests.append(ModelRequest(prompt=default_text_prompt.format(document=text)))
    elif (path / "samples.jsonl").exists() or (path / "samples.json").exists():
        manifest = path / "samples.jsonl"
        if manifest.exists():
            rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        else:
            data = json.loads((path / "samples.json").read_text())
            rows = data if isinstance(data, list) else data.get("samples", [])
        for row in rows:
            if modality == "image":
                img = row.get("document") or row.get("image") or row.get("path")
                if img and not Path(img).is_absolute():
                    img = str(path / img)
                q = row.get("question") or default_image_prompt
                requests.append(ModelRequest(prompt=q, images=[img] if img else []))
            else:
                doc = row.get("text") or row.get("document_text") or ""
                doc_path = row.get("document")
                if not doc and doc_path:
                    dp = path / doc_path if not Path(doc_path).is_absolute() else Path(doc_path)
                    if dp.exists() and dp.suffix.lower() in {".txt", ".md", ".csv"}:
                        doc = dp.read_text(encoding="utf-8")
                q = default_text_prompt.format(document=doc)
                requests.append(ModelRequest(prompt=q))
    else:
        # Directory of raw files
        if modality == "image":
            for img in sorted(path.glob("*")):
                if img.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
                    requests.append(
                        ModelRequest(prompt=default_image_prompt, images=[str(img)])
                    )
        else:
            for f in sorted(path.glob("*")):
                if f.suffix.lower() in {".txt", ".md"}:
                    requests.append(
                        ModelRequest(
                            prompt=default_text_prompt.format(
                                document=f.read_text(encoding="utf-8")
                            )
                        )
                    )

    if not requests:
        raise FileNotFoundError(f"No stress samples found under {path}")

    # Cycle to reach n
    out: list[ModelRequest] = []
    i = 0
    while len(out) < n:
        out.append(requests[i % len(requests)])
        i += 1
    return out


def run_latency(
    provider: ModelProvider,
    *,
    workload: str,
    dataset: str | Path,
    modality: str = "text",
    warmup: int = 2,
    iterations: int = 20,
    prompt: str | None = None,
) -> LatencyResult:
    total = warmup + iterations
    pool = _load_requests(dataset, modality, prompt, total)
    latencies: list[float] = []
    errors = 0

    console.print(
        f"[bold]Latency[/bold] {workload} ({modality}): warmup={warmup}, iterations={iterations}"
    )
    for i, req in enumerate(pool):
        resp = provider.generate(req)
        if i < warmup:
            continue
        if resp.ok:
            latencies.append(resp.latency_ms)
        else:
            errors += 1
            # Still record latency for failed calls to reflect end-to-end time
            latencies.append(resp.latency_ms)

    result = LatencyResult(
        workload=workload,
        modality=modality,
        n=iterations,
        warmup=warmup,
        latencies_ms=latencies,
        errors=errors,
    )
    s = result.stats
    console.print(
        f"  mean={s['mean_ms']:.1f}ms  p50={s['p50_ms']:.1f}ms  "
        f"p95={s['p95_ms']:.1f}ms  errors={errors}"
    )
    return result