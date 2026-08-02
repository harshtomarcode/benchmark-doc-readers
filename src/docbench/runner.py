"""Orchestrate qualitative + stress suites from a BenchConfig."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel

from docbench.config import BenchConfig
from docbench.models.factory import build_provider
from docbench.reporting.report import write_report
from docbench.stress.concurrency import run_concurrency
from docbench.stress.latency import run_latency
from docbench.tasks.registry import get_task_class

console = Console()


def run_benchmark(cfg: BenchConfig) -> dict[str, Any]:
    console.print(
        Panel.fit(
            f"[bold]docbench[/bold]\nmodel={cfg.model.id} ({cfg.model.type})",
            border_style="cyan",
        )
    )

    provider = build_provider(cfg.model.model_dump(exclude_none=True))
    payload: dict[str, Any] = {
        "model": {"id": cfg.model.id, "type": cfg.model.type},
        "qualitative": [],
        "latency": [],
        "concurrency": [],
        "run_name": cfg.output.run_name,
    }

    try:
        if cfg.qualitative.enabled:
            console.rule("[bold cyan]Qualitative")
            for task_name, task_cfg in cfg.qualitative.tasks.items():
                if not task_cfg.enabled:
                    console.print(f"[dim]skip {task_name} (disabled)[/dim]")
                    continue
                cls = get_task_class(task_name)
                task = cls(
                    task_cfg.dataset,
                    metrics=task_cfg.metrics,
                    max_samples=task_cfg.max_samples,
                    system_prompt=task_cfg.system_prompt,
                    user_prompt_template=task_cfg.user_prompt_template,
                )
                if task.requires_images and not provider.supports_images():
                    console.print(
                        f"[yellow]skip {task_name}: provider does not support images[/yellow]"
                    )
                    continue
                result = task.run(provider)
                payload["qualitative"].append(result.to_dict())
                console.print(
                    f"[green]✓[/green] {task_name}: "
                    + ", ".join(f"{k}={v:.3f}" for k, v in result.metrics.items() if k != "avg_latency_ms")
                )

        if cfg.stress.enabled:
            console.rule("[bold cyan]Stress")
            for name, wl in cfg.stress.workloads.items():
                if not wl.enabled:
                    console.print(f"[dim]skip stress/{name} (disabled)[/dim]")
                    continue
                if wl.modality == "image" and not provider.supports_images():
                    console.print(
                        f"[yellow]skip stress/{name}: provider does not support images[/yellow]"
                    )
                    continue

                lat = run_latency(
                    provider,
                    workload=name,
                    dataset=wl.dataset,
                    modality=wl.modality,
                    warmup=wl.warmup,
                    iterations=wl.iterations,
                    prompt=wl.prompt,
                )
                payload["latency"].append(lat.to_dict())

                conc = run_concurrency(
                    provider,
                    workload=name,
                    dataset=wl.dataset,
                    modality=wl.modality,
                    concurrency_levels=wl.concurrency_levels,
                    requests_per_level=wl.requests_per_level,
                    prompt=wl.prompt,
                )
                payload["concurrency"].append(conc.to_dict())

    finally:
        provider.close()

    out_dir = write_report(
        payload,
        cfg.output.dir,
        formats=cfg.output.formats,
        run_name=cfg.output.run_name,
    )
    console.print(f"\n[bold green]Results written to[/bold green] {out_dir}")
    return payload