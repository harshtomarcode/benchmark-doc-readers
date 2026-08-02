"""CLI entrypoint: `docbench run --config configs/example.yaml`."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from rich.console import Console

from docbench import __version__
from docbench.config import load_config
from docbench.runner import run_benchmark

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docbench",
        description="Benchmark document reading models (qualitative + stress).",
    )
    parser.add_argument("--version", action="version", version=f"docbench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a benchmark from a config file")
    run_p.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to YAML/JSON config (see configs/)",
    )
    run_p.add_argument(
        "--qualitative-only",
        action="store_true",
        help="Skip stress tests",
    )
    run_p.add_argument(
        "--stress-only",
        action="store_true",
        help="Skip qualitative tests",
    )
    run_p.add_argument(
        "--run-name",
        default=None,
        help="Override output run name",
    )

    list_p = sub.add_parser("list-tasks", help="List available qualitative tasks")
    list_p.add_argument("--quiet", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-tasks":
        from docbench.tasks.registry import TASK_REGISTRY

        for name in TASK_REGISTRY:
            console.print(f"• {name}")
        return 0

    if args.command == "run":
        cfg = load_config(args.config)
        if args.qualitative_only:
            cfg.stress.enabled = False
        if args.stress_only:
            cfg.qualitative.enabled = False
        if args.run_name:
            cfg.output.run_name = args.run_name
        run_benchmark(cfg)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())