"""Concurrency / throughput stress testing."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from docbench.models.base import ModelProvider
from docbench.stress.latency import _load_requests

console = Console()


@dataclass
class ConcurrencyLevelResult:
    concurrency: int
    n_requests: int
    successes: int
    errors: int
    wall_time_s: float
    latencies_ms: list[float]

    @property
    def throughput_rps(self) -> float:
        if self.wall_time_s <= 0:
            return 0.0
        return self.successes / self.wall_time_s

    @property
    def mean_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    def to_dict(self) -> dict[str, Any]:
        sorted_v = sorted(self.latencies_ms)

        def pct(p: float) -> float:
            if not sorted_v:
                return 0.0
            idx = min(len(sorted_v) - 1, max(0, int(round((p / 100) * (len(sorted_v) - 1)))))
            return sorted_v[idx]

        return {
            "concurrency": self.concurrency,
            "n_requests": self.n_requests,
            "successes": self.successes,
            "errors": self.errors,
            "wall_time_s": self.wall_time_s,
            "throughput_rps": self.throughput_rps,
            "mean_latency_ms": self.mean_latency_ms,
            "p50_latency_ms": pct(50),
            "p95_latency_ms": pct(95),
        }


@dataclass
class ConcurrencyResult:
    workload: str
    modality: str
    levels: list[ConcurrencyLevelResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "modality": self.modality,
            "levels": [lv.to_dict() for lv in self.levels],
        }


async def _run_level(
    provider: ModelProvider,
    requests: list,
    concurrency: int,
) -> ConcurrencyLevelResult:
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    successes = 0
    errors = 0

    async def one(req: Any) -> None:
        nonlocal successes, errors
        async with sem:
            resp = await provider.agenerate(req)
            latencies.append(resp.latency_ms)
            if resp.ok:
                successes += 1
            else:
                errors += 1

    start = time.perf_counter()
    await asyncio.gather(*(one(r) for r in requests))
    wall = time.perf_counter() - start

    return ConcurrencyLevelResult(
        concurrency=concurrency,
        n_requests=len(requests),
        successes=successes,
        errors=errors,
        wall_time_s=wall,
        latencies_ms=latencies,
    )


def run_concurrency(
    provider: ModelProvider,
    *,
    workload: str,
    dataset: str | Path,
    modality: str = "text",
    concurrency_levels: list[int] | None = None,
    requests_per_level: int = 32,
    prompt: str | None = None,
) -> ConcurrencyResult:
    levels = concurrency_levels or [1, 4, 8, 16]
    pool = _load_requests(dataset, modality, prompt, max(requests_per_level, max(levels)))

    console.print(
        f"[bold]Concurrency[/bold] {workload} ({modality}): levels={levels}, "
        f"requests/level={requests_per_level}"
    )

    results: list[ConcurrencyLevelResult] = []
    for c in levels:
        # Take a slice / cycle of requests for this level
        reqs = [pool[i % len(pool)] for i in range(requests_per_level)]
        level_result = asyncio.run(_run_level(provider, reqs, c))
        results.append(level_result)
        console.print(
            f"  c={c:>3}  rps={level_result.throughput_rps:.2f}  "
            f"mean_lat={level_result.mean_latency_ms:.1f}ms  "
            f"errors={level_result.errors}"
        )

    return ConcurrencyResult(workload=workload, modality=modality, levels=results)