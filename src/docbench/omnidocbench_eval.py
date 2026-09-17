"""Load OmniDocBench pages and run its official, isolated end-to-end evaluator.

No geometry or ground truth is inferred from Markdown. Scores keep the upstream
scale (0--1); Edit_dist is lower-is-better and TEDS/CDM are higher-is-better.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any

import yaml


def load_omnidocbench_documents(
    manifest: Path, root: Path | None = None, options: dict | None = None
) -> list[dict]:
    """Read the released JSON annotations without downloading pages or models."""
    options = options or {}
    root = (root or manifest.parent).resolve()
    pages = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(pages, list) or not pages:
        raise ValueError("OmniDocBench annotations must be a nonempty JSON list")
    documents, seen = [], set()
    for annotation in pages:
        page = annotation.get("page_info", {})
        image_path = page.get("image_path")
        if not image_path or not isinstance(annotation.get("layout_dets"), list):
            raise ValueError("OmniDocBench pages require page_info.image_path and layout_dets")
        if not isinstance(page.get("page_attribute"), dict) or not isinstance(
            annotation.get("extra", {}).get("relation"), list
        ):
            raise ValueError("OmniDocBench pages require page_attribute and extra.relation")
        image = Path(image_path)
        doc_id = image.stem
        if doc_id in seen:
            raise ValueError(f"Duplicate OmniDocBench page ID: {doc_id}")
        seen.add(doc_id)
        if options.get("image_dir"):
            candidates = [root / options["image_dir"] / image.name]
        else:
            candidates = [root / image, root / "images" / image.name]
        doc_path = next((p for p in candidates if p.is_file()), candidates[0])
        attributes = page.get("page_attribute", {})
        tags = {"benchmark:omnidocbench"}
        for key, value in attributes.items():
            for item in value if isinstance(value, list) else [value]:
                if item not in (None, "None", ""):
                    tags.add(f"{key}:{item}")
        blocks = [block for block in annotation["layout_dets"] if not block.get("ignore")]
        categories = set()
        for block in blocks:
            category = block.get("category_type", "")
            if category == "table":
                categories.add("table")
            elif category == "equation_isolated":
                categories.add("formula")
            elif block.get("text") and category not in {
                "abandon",
                "header",
                "footer",
                "page_number",
            }:
                categories.add("text_content")
            for key, value in block.get("attribute", {}).items():
                for item in value if isinstance(value, list) else [value]:
                    if item not in (None, "None", ""):
                        tags.add(f"{key}:{item}")
        if any(block.get("order") is not None for block in blocks):
            categories.add("reading_order")
        family_map = options.get("family_ids", {})
        family = family_map.get(doc_id)
        family_source = "explicit" if family else "image_stem"
        if not family:
            suffix = f"_{page.get('page_no')}"
            family = doc_id.removesuffix(suffix)
            if family != doc_id:
                family_source = "image_stem_without_annotated_page_number"
        documents.append(
            {
                "doc_id": doc_id,
                "doc_path": str(doc_path.resolve()),
                "benchmark": "omnidocbench",
                "benchmark_annotation": annotation,
                "tags": sorted(tags),
                "categories": sorted(categories),
                "page_count": 1,
                "page_number": page.get("page_no"),
                "family_id": family,
                "family_id_source": family_source,
            }
        )
    return documents


def omnidocbench_configuration_error(options: dict | None = None) -> str | None:
    """Fail before reader uploads when the requested official scorer is unavailable."""
    missing = [
        key
        for key in ("OMNIDOCBENCH_PYTHON", "OMNIDOCBENCH_ROOT")
        if not os.environ.get(key, "").strip()
    ]
    if missing:
        return "Official OmniDocBench evaluation requires: " + ", ".join(missing)
    root = Path(os.environ["OMNIDOCBENCH_ROOT"].strip()).expanduser().resolve()
    python = os.path.expanduser(os.environ["OMNIDOCBENCH_PYTHON"].strip())
    if os.sep in python:
        python = os.path.abspath(python)
    if not shutil.which(python):
        return f"OMNIDOCBENCH_PYTHON is not executable: {python}"
    if not (root / "pdf_validation.py").is_file() or not (root / "configs/end2end.yaml").is_file():
        return f"OMNIDOCBENCH_ROOT is not an official evaluator checkout: {root}"
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from src.core.registry import load_default_registrations; load_default_registrations()"
    )
    try:
        completed = subprocess.run(
            [python, "-c", probe, str(root)],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"OmniDocBench evaluator probe failed: {exc}"
    if completed.returncode:
        return "OmniDocBench dependencies are unavailable: " + completed.stderr[-2000:].strip()
    return None


def evaluate_omnidocbench(doc: dict, result: dict, artifact: Path, options: dict) -> dict:
    """Run upstream scoring on one page and retain its matching and score artifacts.

    Call ``omnidocbench_configuration_error`` once before a benchmark run. CDM is
    optional because its external rendering toolchain is substantial. Disabling
    CDM keeps formula edit distance and explicitly marks CDM unavailable.
    """
    work = (artifact / "omnidocbench").resolve()
    work.mkdir(parents=True, exist_ok=False)
    unavailable: dict[str, str] = {
        "chart": "OmniDocBench end-to-end scoring has no chart-value metric",
        "layout": "End-to-end Markdown evaluation does not score bounding boxes",
    }
    details: dict[str, Any] = {
        "evaluator": "official OmniDocBench end2end_eval",
        "score_scale": "0..1; edit_dist lower is better; TEDS/CDM higher is better",
        "unavailable": unavailable,
        "artifact_dir": str(work),
    }
    missing = [
        key
        for key in ("OMNIDOCBENCH_PYTHON", "OMNIDOCBENCH_ROOT")
        if not os.environ.get(key, "").strip()
    ]
    if missing:
        return {
            "scores": {},
            "status": "unsupported",
            "reason": ", ".join(missing) + " is empty",
            "details": details,
        }
    root = Path(os.environ["OMNIDOCBENCH_ROOT"].strip()).expanduser().resolve()
    python = os.path.expanduser(os.environ["OMNIDOCBENCH_PYTHON"].strip())
    if os.sep in python:
        python = os.path.abspath(python)
    annotation = doc["benchmark_annotation"]
    # Keep the original annotation and filename; the upstream matcher uses basenames.
    (work / "ground_truth.json").write_text(json.dumps([annotation], ensure_ascii=False, indent=2))
    predictions = work / "predictions"
    predictions.mkdir(exist_ok=True)
    image_name = Path(annotation["page_info"]["image_path"]).name
    (predictions / (Path(image_name).stem + ".md")).write_text(
        result.get("text") or "", encoding="utf-8"
    )
    cfg = yaml.safe_load((root / "configs/end2end.yaml").read_text(encoding="utf-8"))
    task = cfg["end2end_eval"]
    task["dataset"].update(
        {
            "ground_truth": {"data_path": str(work / "ground_truth.json")},
            "prediction": {"data_path": str(predictions)},
            "match_method": options.get("match_method", "quick_match"),
            "match_workers": 1,
        }
    )
    if task["dataset"]["match_method"] not in {"quick_match", "simple_match", "no_split"}:
        raise ValueError("OmniDocBench match_method must be quick_match, simple_match, or no_split")
    task["metrics"] = {
        "text_block": {"metric": ["Edit_dist"]},
        "display_formula": {"metric": ["Edit_dist"]},
        "table": {"metric": ["TEDS", "Edit_dist"], "teds_workers": 1},
        "reading_order": {"metric": ["Edit_dist"]},
    }
    if options.get("cdm", False):
        # Missing renderers are evaluator failures, never a zero for the reader.
        missing_tools = [
            name for name in ("pdflatex", "kpsewhich", "gs", "magick") if not shutil.which(name)
        ]
        if missing_tools:
            unavailable["formula_cdm"] = "Missing CDM rendering tools: " + ", ".join(missing_tools)
        else:
            task["metrics"]["display_formula"].update(
                {"metric": ["Edit_dist", "CDM"], "cdm_workers": 1}
            )
    else:
        unavailable["formula_cdm"] = (
            "CDM disabled; set benchmark_options.cdm=true with its rendering dependencies"
        )
    config_path = work / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    command = [python, str(root / "pdf_validation.py"), "--config", str(config_path)]
    details["command"] = command
    details["config"] = cfg
    # A commit and source digest distinguish local changes and exported checkouts.
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    details["source_sha256"] = digest.hexdigest()
    try:
        version = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        details["upstream_commit"] = version.stdout.strip() if version.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        details["upstream_commit"] = None
    (work / "provenance.json").write_text(json.dumps(details, ensure_ascii=False, indent=2))
    env = {**os.environ, "PYTHONPATH": str(root), "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    failure = None
    with (work / "evaluator.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command,
                cwd=work,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                process.wait(timeout=float(options.get("timeout_s", 600)))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                failure = "Official OmniDocBench evaluator timed out; see evaluator.log"
            if process.returncode and failure is None:
                failure = f"Official evaluator exited {process.returncode}; see evaluator.log"
        except OSError as exc:
            failure = f"Official OmniDocBench evaluator could not run: {exc}"
    reports = sorted((work / "result").glob("*_metric_result.json"))
    if not reports or failure:
        return {
            "scores": {},
            "status": "error",
            "reason": failure or "Official score report missing",
            "details": details,
        }
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    details["official_report"] = str(reports[0])
    details["official_artifacts"] = [str(path) for path in sorted((work / "result").glob("*"))]
    details["matching"] = report.get("match_debug", {})
    scores = {}
    # Use official page aggregations, not element-count means or a new combined score.
    metric_paths = {
        "text_edit_dist": ("text_block", "all", "Edit_dist", "ALL_page_avg"),
        "formula_edit_dist": ("display_formula", "all", "Edit_dist", "ALL_page_avg"),
        "formula_cdm": ("display_formula", "page", "CDM", "ALL"),
        "table_teds": ("table", "page", "TEDS", "ALL"),
        "table_teds_structure_only": ("table", "page", "TEDS_structure_only", "ALL"),
        "table_edit_dist": ("table", "all", "Edit_dist", "ALL_page_avg"),
        "reading_order_edit_dist": ("reading_order", "all", "Edit_dist", "ALL_page_avg"),
    }
    for name, path in metric_paths.items():
        if name in unavailable:
            continue
        value: Any = report
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        metric = "TEDS" if path[2] == "TEDS_structure_only" else path[2]
        debug = report.get(path[0], {}).get("metric_debug", {}).get(metric, {})
        errors = {
            key: debug[key]
            for key in ("timeout_cases", "error_cases", "exception_cases")
            if debug.get(key)
        }
        if errors:
            unavailable[name] = (
                "Official evaluator reported metric errors; inspect matched-element artifacts"
            )
            details.setdefault("metric_errors", {})[name] = errors
        elif (
            isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
        ):
            scores["omnidocbench_" + name] = float(value)
        else:
            unavailable[name] = "No applicable scored elements on this page"
    return {
        "scores": scores,
        "status": "scored" if scores else "unsupported",
        "reason": None if scores else "No applicable official scores",
        "details": details,
    }
