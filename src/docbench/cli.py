"""CLI entrypoint: `docbench run --config configs/example.yaml`."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from rich.console import Console

from docbench import __version__
from docbench.config import load_config, load_reader_config
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

    readers_p = sub.add_parser("run-readers", help="Compare environment-enabled document parsers")
    readers_p.add_argument("-c", "--config", required=True, help="Reader benchmark YAML config")
    readers_p.add_argument("--env-file", default=".env", help="Environment file (default: .env)")
    readers_p.add_argument("--list", action="store_true", help="Show eligibility without any calls")
    readers_p.add_argument("--limit", type=int, help="Maximum documents per enabled tool")
    readers_p.add_argument(
        "--tool", action="append", help="Run only this reader; repeat for several"
    )
    readers_p.add_argument(
        "--run-name", help="New output directory name; existing runs are preserved"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Explicit blank entries in the selected file must disable ambient shell credentials too.
    if args.command == "run-readers":
        load_dotenv(args.env_file, override=True)
    else:
        load_dotenv()

    if args.command == "run-readers":
        from docbench.reader_benchmark import READERS, reader_plan, run_readers

        cfg = load_reader_config(args.config)
        if args.limit is not None:
            if args.limit < 1:
                parser.error("--limit must be positive")
            cfg.limit = args.limit
        if args.run_name:
            cfg.output.run_name = args.run_name
        if args.tool:
            unknown = set(args.tool) - READERS.keys()
            if unknown:
                parser.error(f"Unknown readers: {', '.join(sorted(unknown))}")
            cfg.tools = {name: cfg.tools.get(name, {}) for name in args.tool}
        if args.list:
            for tool in reader_plan(cfg):
                console.print(
                    f"{tool['reader']}: {tool['reason'] or 'enabled'} "
                    f"(gate: {tool['gate']}; route: {tool['route']})",
                    markup=False,
                )
            return 0
        return 1 if run_readers(cfg)["n_errors"] else 0

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
