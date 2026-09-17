"""Load released ParseBench rules and call its pinned, official evaluators.

Scoring runs in a separate interpreter so optional evaluator dependencies do not
constrain the document readers. No provider inference or corpus download occurs here.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PARSEBENCH_VERSION = "1.0.4"
PARSEBENCH_SOURCE_REVISION = "5f16913e9f824ace9ef4fe89a954b8c2eb29c7eb"
CATEGORIES = ("table", "chart", "text_content", "text_formatting", "layout")


def load_parsebench_documents(manifest: Path, root: Path | None, options: dict) -> list[dict]:
    """Group rules by source document and category, preserving released annotations."""
    manifest = Path(manifest)
    dataset_root = Path(root) if root else (manifest if manifest.is_dir() else manifest.parent)
    selected = options.get("categories", options.get("groups", CATEGORIES))
    selected = {selected} if isinstance(selected, str) else set(selected)
    if selected - set(CATEGORIES):
        raise ValueError(f"Unknown ParseBench categories: {sorted(selected - set(CATEGORIES))}")
    files = (
        [manifest]
        if manifest.is_file()
        else [manifest / f"{category}.jsonl" for category in CATEGORIES if category in selected]
    )
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise ValueError("Missing requested ParseBench category JSONLs: " + ", ".join(missing))
    if not files:
        raise ValueError(f"No released ParseBench category JSONLs selected at {manifest}")
    expected_path = dataset_root / "expected_markdown.json"
    expected_map = json.loads(expected_path.read_text()) if expected_path.exists() else {}
    groups: dict[tuple[str, str], dict] = {}
    for file in files:
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        for line_number, line in enumerate(file.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            category = row.get("category", file.stem)
            if category not in selected:
                continue
            source = row.get("pdf")
            if not isinstance(source, str) or not source:
                raise ValueError(f"{file}:{line_number}: missing pdf path")
            key = (category, source)
            if key not in groups:
                suffix = hashlib.sha256(source.encode()).hexdigest()[:12]
                groups[key] = {
                    "doc_id": f"parsebench-{category}-{Path(source).stem}-{suffix}",
                    "doc_path": str((dataset_root / source).resolve()),
                    "category": category,
                    "categories": [category],
                    "tags": [],
                    "family_id": f"parsebench-{suffix}",
                    "page_count": None,
                    "benchmark_cases": [
                        {
                            "group": category,
                            "source": source,
                            "rules": [],
                            "source_rows": [],
                            "expected_markdown": expected_map.get(source),
                            "rule_metadata": {},
                            "manifest": str(file.resolve()),
                            "manifest_sha256": digest,
                        }
                    ],
                }
            doc = groups[key]
            case = doc["benchmark_cases"][0]
            rule = row.get("rule") or {}
            rule = json.loads(rule) if isinstance(rule, str) else dict(rule)
            rule = {"type": row.get("type"), **rule}
            if row.get("id"):
                rule["id"] = row["id"]
            if row.get("page") is not None:
                rule["page"] = row["page"]
            case["source_rows"].append(row)
            if rule["type"] != "expected_markdown":
                case["rules"].append(rule)
            if row.get("expected_markdown"):
                # Matches the official loader: the document-level annotation is
                # repeated across rows, so it must not be duplicated per rule.
                if case["expected_markdown"] is None:
                    case["expected_markdown"] = row["expected_markdown"]
                elif case["expected_markdown"] != row["expected_markdown"]:
                    raise ValueError(f"{source}: conflicting document-level expected_markdown")
            for tag in row.get("tags") or []:
                if tag not in doc["tags"]:
                    doc["tags"].append(tag)
            for field in (
                "allow_splitting_ambiguous_merged_tables",
                "trm_unsupported",
                "max_top_title_rows",
            ):
                if field in rule:
                    case["rule_metadata"][field] = rule[field]
            if row.get("page_count"):
                doc["page_count"] = int(row["page_count"])
    if not groups:
        raise ValueError("No ParseBench documents remain after category filtering")
    return sorted(groups.values(), key=lambda doc: doc["doc_id"])


def parsebench_configuration_error(options: dict) -> str | None:
    """Check the optional scorer before a caller spends money on inference."""
    python = options.get("python") or os.environ.get("PARSEBENCH_PYTHON", "").strip()
    if python and "/" in python:
        python = os.path.abspath(os.path.expanduser(python))
    try:
        with tempfile.TemporaryDirectory(prefix="docbench-parsebench-check-") as temporary:
            checked = subprocess.run(
                [python or sys.executable, str(Path(__file__).resolve()), "--check"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=temporary,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"ParseBench evaluation interpreter is unavailable: {exc}"
    if checked.returncode:
        return (
            "ParseBench requires Python >=3.12 and parse-bench==1.0.4. "
            "Install '.[parsebench]' or set PARSEBENCH_PYTHON. "
            + (
                (checked.stderr or checked.stdout).strip() or "Interpreter check failed"
            ).splitlines()[-1]
        )
    return None


def evaluate_parsebench(doc: dict, result: dict, artifact: Path, options: dict) -> dict:
    """Return official numeric scores and save complete rules, output and verdicts."""
    artifact = Path(artifact).resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    python = options.get("python") or os.environ.get("PARSEBENCH_PYTHON", "").strip()
    if python and "/" in python:
        python = os.path.abspath(os.path.expanduser(python))
    with tempfile.TemporaryDirectory(prefix="docbench-parsebench-") as temporary:
        request = Path(temporary) / "request.json"
        response = Path(temporary) / "response.json"
        request.write_text(
            json.dumps(
                {
                    "doc": doc,
                    "result": result,
                    "artifact": str(artifact),
                    "options": options,
                },
                ensure_ascii=False,
            )
        )
        try:
            completed = subprocess.run(
                [
                    python or sys.executable,
                    str(Path(__file__).resolve()),
                    str(request),
                    str(response),
                ],
                capture_output=True,
                text=True,
                timeout=float(options.get("timeout_s", 600)),
                cwd=temporary,
                # Scoring is deterministic and offline even if a shell previously
                # opted into ParseBench's optional paid LLM chart judge.
                env={**os.environ, "LLAMACLOUD_BENCH_LLM_NORMALIZATION": "off"},
            )
        except subprocess.TimeoutExpired as exc:
            (artifact / "parsebench_evaluator.log").write_text(str(exc))
            raise RuntimeError("Official ParseBench scoring timed out") from exc
        (artifact / "parsebench_evaluator.log").write_text(completed.stdout + completed.stderr)
        if completed.returncode or not response.exists():
            raise RuntimeError(
                "Official ParseBench scoring failed; inspect parsebench_evaluator.log: "
                + (completed.stderr or completed.stdout)[-1000:]
            )
        return json.loads(response.read_text())


def _native_output(doc: dict, result: dict) -> tuple[Any, dict, str, str | None]:
    """Use upstream normalizers; missing geometry remains explicitly unavailable."""
    from parse_bench.schemas.parse_output import ParseOutput

    name = result.get("reader", "unknown")
    provider = {"unstructured_api": "unstructured", "docling": "docling_parse"}.get(name, name)
    raw = result.get("raw") or {}
    raw = raw if isinstance(raw, dict) else {"elements": raw}
    raw = raw.get("downloaded_result", raw)
    output = ParseOutput(
        example_id=doc["doc_id"], pipeline_name=f"docbench-{name}", markdown=result.get("text", "")
    )
    if result.get("parsebench_output"):
        output = ParseOutput.model_validate(result["parsebench_output"])
        reason = (
            None
            if any(page.items for page in output.layout_pages)
            else ("Explicit ParseBench output has no native element geometry")
        )
        return output, raw, provider, reason
    try:
        if name in {"unstructured", "unstructured_api"}:
            from parse_bench.inference.providers.parse.unstructured import _build_layout_pages

            output.layout_pages = _build_layout_pages(raw)
        elif name == "pymupdf4llm":
            # Native page boxes use source-page coordinates. Read only physical
            # dimensions, never source text, to avoid leaking gold into prediction.
            import pymupdf
            from parse_bench.inference.providers.parse.pymupdf4llm import PyMuPDF4LLMProvider
            from parse_bench.schemas.parse_output import PageIR

            with pymupdf.open(doc["doc_path"]) as source:
                pages = []
                for index, chunk in enumerate(raw.get("pages", [])):
                    number = int(chunk.get("metadata", {}).get("page_number", index + 1))
                    rect = source[number - 1].rect
                    page = {
                        **chunk,
                        "page_index": number - 1,
                        "page_number": number,
                        "width": rect.width,
                        "height": rect.height,
                    }
                    pages.append(page)
                    output.pages.append(PageIR(page_index=number - 1, markdown=chunk["text"]))
                    layout = PyMuPDF4LLMProvider._build_layout_page(
                        page, raw_markdown=chunk["text"]
                    )
                    if layout is not None:
                        output.layout_pages.append(layout)
                raw = {**raw, "pages": pages}
        elif name == "llamaparse":
            from parse_bench.inference.providers.parse.llamaparse_v2_normalization import (
                build_pages_from_sdk_response_payload,
                build_parse_output_from_pages,
            )

            pages = build_pages_from_sdk_response_payload(
                raw_payload=raw, output_tables_as_markdown=False
            )
            normalized = build_parse_output_from_pages(
                pages_payload=pages,
                example_id=doc["doc_id"],
                pipeline_name=f"docbench-{name}",
                job_id=None,
            )
            output.pages, output.layout_pages = normalized.pages, normalized.layout_pages
        elif name == "docling":
            from docling_core.types.doc.document import DoclingDocument
            from parse_bench.inference.providers.parse._docling_common import (
                _build_docling_layout_pages,
            )

            document = DoclingDocument.model_validate(raw.get("document") or result["structured"])
            output.layout_pages = _build_docling_layout_pages(doc=document, raw_pages=[])
        elif name == "reducto":
            from parse_bench.inference.providers.parse.reducto import (
                _build_layout_pages,
                _build_pages,
                _coerce_chunks,
            )

            chunks = _coerce_chunks(raw.get("result", {}).get("result", raw))
            output.pages, output.layout_pages = _build_pages(chunks), _build_layout_pages(chunks)
        elif name in {"datalab", "marker"}:
            from parse_bench.inference.providers.parse.datalab import _build_layout_pages

            provider = "datalab"
            output.layout_pages = _build_layout_pages(raw.get("json") or raw.get("blocks") or {})
    except (ImportError, KeyError, ValueError, TypeError, IndexError) as exc:
        return output, raw, provider, f"Native layout normalization unavailable: {exc}"
    reason = (
        None
        if any(p.items for p in output.layout_pages)
        else (
            "Reader returned no supported native element geometry; "
            "bounding boxes were not fabricated"
        )
    )
    return output, raw, provider, reason


def _evaluate_worker(request: dict) -> dict:
    from parse_bench.evaluation.evaluators.layoutdet import LayoutDetectionEvaluator
    from parse_bench.evaluation.evaluators.parse import ParseEvaluator
    from parse_bench.evaluation.layout_adapters import create_layout_adapter_for_result
    from parse_bench.evaluation.layout_adapters.registry import register_pipeline_resolver
    from parse_bench.schemas.pipeline import PipelineSpec
    from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
    from parse_bench.schemas.product import ProductType
    from parse_bench.test_cases.schema import LayoutDetectionTestCase, ParseTestCase

    doc, result = request["doc"], request["result"]
    artifact, options = Path(request["artifact"]), request["options"]
    output, raw, provider, layout_reason = _native_output(doc, result)
    name = output.pipeline_name
    register_pipeline_resolver(
        lambda pipeline: (
            PipelineSpec(
                pipeline_name=name,
                provider_name=provider,
                product_type=ProductType.PARSE,
            )
            if pipeline == name
            else None
        )
    )
    now = datetime.now(timezone.utc)
    inference = InferenceResult(
        request=InferenceRequest(
            example_id=doc["doc_id"],
            source_file_path=doc["doc_path"],
            product_type=ProductType.PARSE,
        ),
        pipeline_name=name,
        product_type=ProductType.PARSE,
        raw_output=raw,
        output=output,
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )
    verdicts, scores, unsupported = [], {}, []
    for case in doc["benchmark_cases"]:
        rules = case["rules"]
        parse_rules = [rule for rule in rules if rule["type"] != "layout"]
        layout_rules = [rule for rule in rules if rule["type"] == "layout"]
        common = dict(
            test_id=doc["doc_id"],
            group=case["group"],
            file_path=doc["doc_path"],
            tags=doc.get("tags", []),
        )
        if parse_rules or case.get("expected_markdown"):
            test = ParseTestCase(
                **common,
                test_rules=parse_rules,
                expected_markdown=case.get("expected_markdown"),
                **case.get("rule_metadata", {}),
            )
            evaluator = ParseEvaluator(
                enable_teds=bool(options.get("teds", False)), grits_pair_workers=1
            )
            evaluated = evaluator.evaluate(inference, test)
            if not evaluated.success:
                raise ValueError(evaluated.error)
            verdicts.append(evaluated.model_dump(mode="json"))
        if layout_rules:
            if layout_reason:
                unsupported.append(layout_reason)
            else:
                adapter = create_layout_adapter_for_result(inference)
                layout_output = adapter.to_layout_output(inference)
                if not layout_output.predictions:
                    unsupported.append("Official adapter produced no supported native geometry")
                else:
                    layout_inference = inference.model_copy(
                        update={
                            "product_type": ProductType.LAYOUT_DETECTION,
                            "output": layout_output,
                        }
                    )
                    test = LayoutDetectionTestCase(**common, test_rules=layout_rules)
                    evaluated = LayoutDetectionEvaluator().evaluate(layout_inference, test)
                    if not evaluated.success:
                        raise ValueError(evaluated.error)
                    verdicts.append(evaluated.model_dump(mode="json"))
    for verdict in verdicts:
        for metric in verdict["metrics"]:
            if math.isfinite(metric["value"]):
                if metric["metric_name"] in scores:
                    scores["parse_" + metric["metric_name"]] = scores[metric["metric_name"]]
                scores[metric["metric_name"]] = metric["value"]
    detail = {
        "package": "parse-bench",
        "version": importlib.metadata.version("parse-bench"),
        "reviewed_source_revision": PARSEBENCH_SOURCE_REVISION,
        "normalization_provider": provider,
        "normalization_reason": layout_reason,
        "llm_normalization": "off",
        "official_results": verdicts,
        "unsupported": unsupported,
    }
    (artifact / "parsebench_cases.json").write_text(json.dumps(doc["benchmark_cases"], indent=2))
    (artifact / "parsebench_output.json").write_text(output.model_dump_json(indent=2))
    (artifact / "parsebench_verdicts.json").write_text(json.dumps(detail, indent=2))
    reason = "; ".join(unsupported) or (None if scores else "No applicable official metrics")
    return {
        "scores": scores,
        "status": "unsupported" if reason else "scored",
        "reason": reason,
        "details": {
            "version": detail["version"],
            "reviewed_source_revision": PARSEBENCH_SOURCE_REVISION,
            "verdicts_path": str(artifact / "parsebench_verdicts.json"),
            "normalized_output_path": str(artifact / "parsebench_output.json"),
            "cases_path": str(artifact / "parsebench_cases.json"),
            "layout_supported": layout_reason is None,
            "scored_metric_count": len(scores),
        },
    }


if __name__ == "__main__":
    version = importlib.metadata.version("parse-bench")
    if version != PARSEBENCH_VERSION:
        raise RuntimeError(f"Expected parse-bench=={PARSEBENCH_VERSION}; found {version}")
    if sys.argv[1] == "--check":
        from parse_bench.evaluation.evaluators.layoutdet import LayoutDetectionEvaluator
        from parse_bench.evaluation.evaluators.parse import ParseEvaluator

        ParseEvaluator()
        LayoutDetectionEvaluator()
        print(f"parse-bench {version} ready")
    else:
        Path(sys.argv[2]).write_text(
            json.dumps(_evaluate_worker(json.loads(Path(sys.argv[1]).read_text())))
        )
