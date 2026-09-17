"""Annotation-based checks and descriptive benchmark scorecards (no composite score).

Numeric checks need an explicit field context or a regex with a named ``value``
group. Ambiguous matches are unmeasured, never resolved by searching for the gold
number. CER/WER use exact, case-sensitive Levenshtein distance; Counter measures
are explicitly token proxies, not layout or semantic completeness metrics.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_NUMBER = re.compile(
    r"(?<![\w.])\(?\s*[+\-−]?\s*[$€£¥]?\s*"
    r"(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?"
    r"(?:[eE][+\-]?\d+)?\s*%?\s*\)?(?![\w.])"
)
_CHECK_METRICS = {
    "numeric": "annotated_numeric_accuracy",
    "text": "annotated_text_accuracy",
    "order": "annotated_reading_order_accuracy",
    "location": "annotated_location_accuracy",
}
_RUBRIC_METRICS = {
    "cer": "A text reference is required.",
    "wer": "A text reference is required.",
    "token_omission_rate_proxy": "A text reference is required.",
    "token_extra_rate_proxy": "A text reference is required.",
    "token_duplicate_excess_rate_proxy": "A text reference is required.",
    "annotated_numeric_accuracy": "Explicit numeric field checks are required.",
    "annotated_sign_accuracy": "Explicit numeric field checks are required.",
    "annotated_unit_accuracy": "Numeric checks with explicit units are required.",
    "annotated_field_label_accuracy": "Checks with explicit field contexts are required.",
    "annotated_text_accuracy": "Explicit text checks are required.",
    "annotated_reading_order_accuracy": "Explicit ordered text anchors are required.",
    "annotated_location_accuracy": "Location annotations and normalized layout output are required",
    "text_content": "No applicable official text-content metric is supplied.",
    "table_content": "No applicable official table-content metric is supplied.",
    "table_structure": "No table topology scorer or annotated table ground truth is supplied.",
    "chart_understanding": "No chart-question ground truth and scorer are supplied.",
    "reading_order": "No annotated or official reading-order metric is supplied.",
    "layout_detection": "No annotated or official localization metric is supplied.",
    "formula_fidelity": "No applicable official formula metric is supplied.",
    "downstream_qa": "No downstream questions, gold answers, and model outputs are supplied.",
    "schema_extraction": "No applicable structured-extraction metrics are supplied.",
    "image_retention": "No image-object ground truth and scorer are supplied.",
    "style_fidelity": "No style ground truth and scorer are supplied.",
}
_RUBRIC_SCORE_KEYS = {
    "text_content": {
        "content_faithfulness",
        "normalized_text_score",
        "omnidocbench_text_edit_dist",
    },
    "table_content": {
        "gtrm",
        "grits_trm_composite",
        "table_record_match",
        "grits_con",
        "ref_grits_con",
        "teds",
        "omnidocbench_table_teds",
        "omnidocbench_table_edit_dist",
    },
    "table_structure": {
        "teds",
        "teds_struct",
        "teds_struct_bool",
        "ref_grits_top",
        "grits_top",
        "omnidocbench_table_teds_structure_only",
        "omnidocbench_table_teds",
    },
    "chart_understanding": {
        "chart_data_point_match",
        "rule_chart_data_point_pass_rate",
        "rule_chart_data_array_labels_pass_rate",
        "rule_chart_data_array_data_pass_rate",
    },
    "style_fidelity": {"semantic_formatting", "normalized_text_styling"},
    "reading_order": {
        "annotated_reading_order_accuracy",
        "layout_reading_order_pass_rate",
        "omnidocbench_reading_order_edit_dist",
    },
    "layout_detection": {
        "annotated_location_accuracy",
        "layout_localization_pass_rate",
        "layout_classification_pass_rate",
        "layout_attribution_pass_rate",
    },
    "formula_fidelity": {"omnidocbench_formula_edit_dist", "omnidocbench_formula_cdm"},
    "downstream_qa": {"downstream_qa_exact_match", "downstream_qa_token_f1", "qa_anls_star"},
    "schema_extraction": {"found", "read_right"},
}


def _edit_distance(left: Any, right: Any) -> int:
    """Exact unit-cost Levenshtein distance using Myers' bit-vector algorithm."""
    if left == right:
        return 0
    if len(left) > len(right):
        left, right = right, left
    if not left:
        return len(right)
    masks: dict[Any, int] = {}
    for index, item in enumerate(left):
        masks[item] = masks.get(item, 0) | (1 << index)
    positive, negative = ~0, 0
    distance = len(left)
    top = 1 << (len(left) - 1)
    for item in right:
        equal = masks.get(item, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        plus = negative | ~(horizontal | positive)
        minus = positive & horizontal
        distance += bool(plus & top) - bool(minus & top)
        plus = (plus << 1) | 1
        minus <<= 1
        positive = minus | ~(vertical | plus)
        negative = plus & vertical
    return distance


def _annotated_check(check: dict, text: str, elements: list) -> dict:
    """Evaluate only the field/anchor explicitly selected by an annotation."""
    kind = check.get("kind")
    outcome = {
        "id": check.get("id"),
        "kind": kind,
        "annotation": check,
        "status": "not_measured",
        "scores": {},
    }
    if kind not in _CHECK_METRICS:
        return {**outcome, "reason": f"Unsupported check kind: {kind!r}"}
    sensitive = bool(check.get("case_sensitive", True))
    flags = 0 if sensitive else re.IGNORECASE
    context = check.get("context")
    selected = text
    if context is not None:
        if not isinstance(context, str) or not context:
            return {**outcome, "reason": "context must be a nonempty literal string"}
        lines = [line for line in text.splitlines() if re.search(re.escape(context), line, flags)]
        outcome["scores"]["annotated_field_label_accuracy"] = float(bool(lines))
        if not lines:
            outcome["scores"][_CHECK_METRICS[kind]] = 0.0
            if kind == "numeric":
                outcome["scores"]["annotated_sign_accuracy"] = 0.0
                if check.get("unit"):
                    outcome["scores"]["annotated_unit_accuracy"] = 0.0
            return {**outcome, "status": "fail", "reason": "Annotated field label is missing"}
        match_index = check.get("match_index")
        if len(lines) != 1 and match_index is None:
            return {**outcome, "reason": "Field context matches multiple lines; set match_index"}
        selected = lines[int(match_index or 0)]
    outcome["evidence"] = selected[:4000] if context is not None else None

    if kind == "numeric":
        expected = check.get("expected")
        if type(expected) not in (int, float) or not math.isfinite(expected):
            return {**outcome, "reason": "expected must be a finite JSON number"}
        tolerance = float(check.get("tolerance", 0))
        relative = float(check.get("relative_tolerance", 0))
        if not all(math.isfinite(v) and v >= 0 for v in (tolerance, relative)):
            return {**outcome, "reason": "Numeric tolerances must be finite and nonnegative"}
        if check.get("pattern"):
            pattern = re.compile(check["pattern"], flags)
            if "value" not in pattern.groupindex:
                return {**outcome, "reason": "Numeric pattern requires a named value group"}
            matches = list(pattern.finditer(selected))
            candidates = [(m.group("value"), m.group(0)) for m in matches]
        elif context is not None:
            label = re.search(re.escape(context), selected, flags)
            assert label is not None
            candidates = [(m.group(0), selected) for m in _NUMBER.finditer(selected[label.end() :])]
        else:
            return {**outcome, "reason": "Numeric checks require context or a named-value pattern"}
        number_index = check.get("number_index")
        if len(candidates) > 1 and number_index is None:
            return {
                **outcome,
                "reason": "Multiple candidate values; annotate number_index or pattern",
            }
        if not candidates:
            outcome["scores"].update(
                {"annotated_numeric_accuracy": 0.0, "annotated_sign_accuracy": 0.0}
            )
            if check.get("unit"):
                outcome["scores"]["annotated_unit_accuracy"] = 0.0
            return {**outcome, "status": "fail", "reason": "No numeric value at annotated field"}
        value_text, evidence = candidates[int(number_index or 0)]
        numeric = re.sub(r"[\s,$€£¥%]", "", value_text).replace("−", "-")
        if numeric.startswith("(") and numeric.endswith(")"):
            numeric = "-" + numeric[1:-1]
        observed = float(numeric)
        if not math.isfinite(observed):
            return {**outcome, "reason": "Selected numeric value is not finite"}
        value_ok = abs(observed - expected) <= max(tolerance, relative * abs(expected))
        sign_ok = (observed > 0) - (observed < 0) == (expected > 0) - (expected < 0)
        outcome["scores"].update(
            {
                "annotated_numeric_accuracy": float(value_ok),
                "annotated_sign_accuracy": float(sign_ok),
            }
        )
        unit = check.get("unit")
        if unit:
            aliases = [str(unit), *check.get("unit_aliases", [])]
            unit_ok = any(
                re.search(
                    (r"(?<!\w)" if alias[0].isalnum() else "")
                    + re.escape(alias)
                    + (r"(?!\w)" if alias[-1].isalnum() else ""),
                    evidence,
                    flags,
                )
                for alias in aliases
                if isinstance(alias, str) and alias
            )
            outcome["scores"]["annotated_unit_accuracy"] = float(unit_ok)
        outcome.update(
            {"observed": observed, "selected_value": value_text, "evidence": evidence[:4000]}
        )
    elif kind == "text":
        expected = check.get("expected")
        if not isinstance(expected, str) or not expected:
            return {**outcome, "reason": "Text expected must be a nonempty string"}
        compared, target = (
            (selected, expected) if sensitive else (selected.casefold(), expected.casefold())
        )
        match = check.get("match", "contains")
        if match not in {"contains", "exact"}:
            return {**outcome, "reason": "Text match must be contains or exact"}
        passed = target in compared if match == "contains" else target == compared.strip()
        outcome["scores"]["annotated_text_accuracy"] = float(passed)
    elif kind == "order":
        anchors = check.get("expected")
        if (
            not isinstance(anchors, list)
            or len(anchors) < 2
            or not all(isinstance(anchor, str) and anchor for anchor in anchors)
            or len(set(anchors)) != len(anchors)
        ):
            return {
                **outcome,
                "reason": "Order expected requires at least two distinct text anchors",
            }
        positions = []
        for anchor in anchors:
            matches = list(re.finditer(re.escape(anchor), selected, flags))
            if len(matches) > 1:
                return {**outcome, "reason": f"Order anchor appears more than once: {anchor!r}"}
            positions.append(matches[0].start() if matches else -1)
        passed = -1 not in positions and positions == sorted(positions)
        outcome["scores"]["annotated_reading_order_accuracy"] = float(passed)
        outcome["observed_positions"] = positions
    else:
        anchor = check.get("expected")
        if not isinstance(anchor, str) or not anchor:
            return {**outcome, "reason": "Location expected must be a nonempty text anchor"}
        if not elements:
            return {
                **outcome,
                "reason": "Reader has no normalized elements with page/bbox metadata",
            }
        located = [
            element
            for element in elements
            if isinstance(element, dict)
            and re.search(re.escape(anchor), str(element.get("text", "")), flags)
        ]
        if not located:
            outcome["scores"]["annotated_location_accuracy"] = 0.0
            return {
                **outcome,
                "status": "fail",
                "reason": "Text anchor is absent from layout output",
            }
        if len(located) != 1:
            return {**outcome, "reason": "Location anchor matches multiple elements"}
        element = located[0]
        passed = True
        if check.get("page") is not None:
            if element.get("page") is None:
                return {**outcome, "reason": "Reader element has no page number"}
            passed = element["page"] == check["page"]
        if check.get("bbox") is not None:
            if not check.get("bbox_space") or check["bbox_space"] != element.get("bbox_space"):
                return {**outcome, "reason": "Matching explicit bbox_space metadata is required"}
            gold, pred = check["bbox"], element.get("bbox")
            if not all(isinstance(box, (list, tuple)) and len(box) == 4 for box in (gold, pred)):
                return {**outcome, "reason": "Both bboxes must be [left, top, right, bottom]"}
            gx, gy, gr, gb = map(float, gold)
            px, py, pr, pb = map(float, pred)
            if (
                not all(math.isfinite(v) for v in (*gold, *pred))
                or gx >= gr
                or gy >= gb
                or px >= pr
                or py >= pb
            ):
                return {
                    **outcome,
                    "reason": "Bboxes must have finite coordinates and positive area",
                }
            intersection = max(0, min(gr, pr) - max(gx, px)) * max(0, min(gb, pb) - max(gy, py))
            union = (gr - gx) * (gb - gy) + (pr - px) * (pb - py) - intersection
            iou = intersection / union
            threshold = float(check.get("min_iou", 0.5))
            if not 0 <= threshold <= 1:
                return {**outcome, "reason": "min_iou must be between zero and one"}
            passed = passed and iou >= threshold
            outcome["bbox_iou"] = iou
        if check.get("page") is None and check.get("bbox") is None:
            return {**outcome, "reason": "Location check requires page and/or bbox"}
        outcome["observed"] = element
        outcome["scores"]["annotated_location_accuracy"] = float(passed)
    outcome["status"] = "pass" if all(outcome["scores"].values()) else "fail"
    return outcome


def evaluate_custom(doc: dict, result: dict, artifact: Path) -> dict:
    """Save auditable checks; return numeric scores, details, and rubric statuses.

    ``doc['reference']`` is optional text gold. ``doc['checks']`` is a list of
    annotation dictionaries. Optional ``result['elements']`` contains normalized
    {text, page, bbox, bbox_space}; vendor-specific raw layout is never guessed.
    Attach returned statuses to each sample as ``rubric_statuses`` for aggregation.
    """
    text = result.get("text") or ""
    if not isinstance(text, str):
        raise TypeError("Custom evaluation requires result.text to be a string")
    scores: dict[str, float] = {}
    statuses = {
        name: {"status": "not_measured", "reason": reason}
        for name, reason in _RUBRIC_METRICS.items()
    }
    details: dict[str, Any] = {
        "checks": [],
        "metric_scope": "Only supplied references and annotations",
    }
    reference = doc.get("reference")
    if isinstance(reference, str):
        pred_tokens, gold_tokens = text.split(), reference.split()
        char_edits = _edit_distance(text, reference)
        word_edits = _edit_distance(pred_tokens, gold_tokens)
        if reference:
            scores["cer"] = char_edits / len(reference)
        if gold_tokens:
            scores["wer"] = word_edits / len(gold_tokens)
        pred_counts, gold_counts = Counter(pred_tokens), Counter(gold_tokens)
        missing, extra = gold_counts - pred_counts, pred_counts - gold_counts
        duplicate_excess = {token: count for token, count in extra.items() if token in gold_counts}
        if gold_tokens:
            scores["token_omission_rate_proxy"] = sum(missing.values()) / len(gold_tokens)
            scores["token_duplicate_excess_rate_proxy"] = sum(duplicate_excess.values()) / len(
                gold_tokens
            )
        if pred_tokens:
            scores["token_extra_rate_proxy"] = sum(extra.values()) / len(pred_tokens)
        elif gold_tokens:
            scores["token_extra_rate_proxy"] = 0.0
        details["text_alignment"] = {
            "normalization": "None; case-sensitive code points; words split on whitespace",
            "character_edits": char_edits,
            "reference_characters": len(reference),
            "word_edits": word_edits,
            "reference_words": len(gold_tokens),
            "prediction_words": len(pred_tokens),
            "missing_token_counts": dict(missing),
            "extra_token_counts": dict(extra),
            "duplicate_excess_token_counts": duplicate_excess,
            "proxy_limitation": "Token bags ignore order, layout, and semantic equivalence; "
            "duplicate excess counts repeated gold vocabulary only.",
        }
        for name in (
            "cer",
            "wer",
            "token_omission_rate_proxy",
            "token_extra_rate_proxy",
            "token_duplicate_excess_rate_proxy",
        ):
            if name not in scores:
                statuses[name] = {"status": "not_measured", "reason": "Rate denominator is zero"}
    checks = doc.get("checks") or []
    if not isinstance(checks, list):
        checks = [{"kind": "invalid", "annotation_error": "checks must be a list"}]
    checked_scores: dict[str, list] = defaultdict(list)
    annotated_counts: Counter = Counter()
    unmeasured_reasons: dict[str, set] = defaultdict(set)
    for index, check in enumerate(checks):
        try:
            if not isinstance(check, dict):
                raise TypeError("Check must be an object")
            detail = _annotated_check(
                {"id": f"check_{index + 1}", **check}, text, result.get("elements") or []
            )
        except (ValueError, TypeError, IndexError, KeyError, re.error) as exc:
            detail = {
                "id": check.get("id", f"check_{index + 1}") if isinstance(check, dict) else index,
                "kind": check.get("kind") if isinstance(check, dict) else None,
                "annotation": check,
                "status": "not_measured",
                "scores": {},
                "reason": f"Invalid or ambiguous annotation: {exc}",
            }
        details["checks"].append(detail)
        if isinstance(check, dict) and check.get("kind") in _CHECK_METRICS:
            names = {_CHECK_METRICS[check["kind"]]}
            if check["kind"] == "numeric":
                names.add("annotated_sign_accuracy")
                if check.get("unit"):
                    names.add("annotated_unit_accuracy")
            if check.get("context") is not None:
                names.add("annotated_field_label_accuracy")
            for name in names:
                annotated_counts[name] += 1
                if name not in detail["scores"]:
                    unmeasured_reasons[name].add(detail.get("reason", "Check was not measured"))
        for name, value in detail["scores"].items():
            checked_scores[name].append(value)
    for name, values in checked_scores.items():
        scores[name] = statistics.mean(values)
        statuses[name] = {
            "status": "measured",
            "measured_checks": len(values),
            "scope": "Only unambiguous explicitly annotated checks",
        }
    for name, count in annotated_counts.items():
        statuses[name].update(
            total_checks=count,
            measured_checks=len(checked_scores[name]),
            unmeasured_checks=count - len(checked_scores[name]),
        )
        if unmeasured_reasons[name]:
            statuses[name]["unmeasured_reasons"] = sorted(unmeasured_reasons[name])
            if name not in scores:
                statuses[name]["reason"] = "; ".join(sorted(unmeasured_reasons[name]))
    for name in scores:
        statuses.setdefault(name, {})["status"] = "measured"
        statuses[name].pop("reason", None)
    evaluation = {"scores": scores, "details": details, "statuses": statuses}
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "custom_evaluation.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return evaluation


def _distribution(values: list[float]) -> dict:
    """Descriptive latency distribution; p95 uses linear interpolation."""
    values = sorted(values)
    if not values:
        return {"n": 0, "median": None, "p95": None}
    rank = (len(values) - 1) * 0.95
    lower, upper = math.floor(rank), math.ceil(rank)
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p95": values[lower] + (values[upper] - values[lower]) * (rank - lower),
    }


def _summarize(samples: list[dict], wall_seconds: float | None) -> dict:
    successful = [
        s for s in samples if s.get("inference_status", s.get("status")) in {"parsed", "scored"}
    ]
    failures = [s for s in samples if s.get("inference_status", s.get("status")) == "error"]
    quality_eligible = [s for s in samples if s.get("status") in {"parsed", "scored"}]
    documents = {str(s.get("doc_id")) for s in samples}
    completed_documents = {str(s.get("doc_id")) for s in successful}
    per_doc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for sample in quality_eligible:
        for name, value in (sample.get("scores") or {}).items():
            if type(value) in (int, float) and math.isfinite(value):
                per_doc[name][str(sample.get("doc_id"))].append(float(value))
    metrics = {}
    for name, docs in per_doc.items():
        values = [statistics.mean(repeats) for repeats in docs.values()]
        metrics[name] = {
            "mean": statistics.mean(values),
            "documents": len(docs),
            "scored_runs": sum(map(len, docs.values())),
            "missing_documents": len(documents) - len(docs),
        }
    latency = {}
    for name in ("latency_ms", "parse_ms", "extraction_ms", "total_ms"):
        latency[name] = {}
        for label, rows in (("success", successful), ("failure", failures)):
            values = [
                float(s[name])
                for s in rows
                if type(s.get(name)) in (int, float) and math.isfinite(s[name]) and s[name] >= 0
            ]
            latency[name][label] = _distribution(values)
    successful_pages = [
        float(s["page_count"])
        for s in successful
        if type(s.get("page_count")) in (int, float)
        and math.isfinite(s["page_count"])
        and s["page_count"] > 0
    ]
    attempted_pages = [
        float(s["page_count"])
        for s in samples
        if type(s.get("page_count")) in (int, float)
        and math.isfinite(s["page_count"])
        and s["page_count"] > 0
    ]
    known_costs = [
        float(s["cost_usd"])
        for s in samples
        if type(s.get("cost_usd")) in (int, float)
        and math.isfinite(s["cost_usd"])
        and s["cost_usd"] >= 0
    ]
    total_cost = sum(known_costs) if samples and len(known_costs) == len(samples) else None
    pages_per_second = None
    if wall_seconds is not None and wall_seconds > 0 and len(successful_pages) == len(successful):
        pages_per_second = sum(successful_pages) / wall_seconds
    return {
        "attempted": len(samples),
        "completed": len(successful),
        "scored": sum(s.get("status") == "scored" for s in samples),
        "failures": len(failures),
        "scoring_failures": sum(s.get("status") == "error" for s in successful),
        "unsupported": sum(s.get("status") == "unsupported" for s in samples),
        "documents": len(documents),
        "source_documents": len(
            {str(s.get("sha256") or s.get("doc_path") or s.get("doc_id")) for s in samples}
        ),
        "families": len({str(s.get("family_id") or s.get("doc_id")) for s in samples}),
        "completed_documents": len(completed_documents),
        "completion_rate": len(successful) / len(samples) if samples else None,
        "score_coverage": sum(s.get("status") == "scored" for s in samples) / len(samples)
        if samples
        else None,
        "document_coverage": len(completed_documents) / len(documents) if documents else None,
        "metrics": metrics,
        **latency,
        "throughput": {
            "wall_seconds": wall_seconds,
            "successful_pages": sum(successful_pages),
            "known_page_runs": len(successful_pages),
            "successful_runs": len(successful),
            "pages_per_second": pages_per_second,
            "documents_per_second": len(successful) / wall_seconds if wall_seconds else None,
        },
        "cost": {
            "scope": "Measured attempts, including failed and unsupported records; excludes warmup",
            "total_usd": total_cost,
            "total_known_usd": sum(known_costs) if known_costs else None,
            "known_runs": len(known_costs),
            "attempted_runs": len(samples),
            "known_fraction": len(known_costs) / len(samples) if samples else None,
            "per_1000_attempted_pages": total_cost * 1000 / sum(attempted_pages)
            if total_cost is not None
            and len(attempted_pages) == len(samples)
            and sum(attempted_pages) > 0
            else None,
            "per_usable_document": total_cost / len(completed_documents)
            if total_cost is not None and completed_documents
            else None,
        },
    }


def build_scorecard(samples: list[dict], wall_seconds: float, options: dict) -> dict:
    """Aggregate one reader's runs; first average repeats within each document.

    ``wall_seconds`` is the actual elapsed reader run, not the sum of timings.
    Categories/tags are lists; suite/split are strings. Samples retain every run,
    including errors and unsupported inputs. Metric directions for robustness
    can be supplied as ``options['metric_directions']`` (higher/lower).
    """
    if not math.isfinite(wall_seconds) or wall_seconds < 0:
        raise ValueError("wall_seconds must be finite and nonnegative")
    card = {
        "overall": _summarize(samples, wall_seconds),
        "aggregation": "Metric means weight documents equally after averaging successful repeats. "
        "Failure and unsupported records remain in coverage; scores are conditional. "
        "Throughput uses actual wall time. No composite score is computed.",
    }
    for field, name in (
        ("categories", "category"),
        ("tags", "tag"),
        ("suite", "suite"),
        ("split", "split"),
    ):
        groups: dict[str, list] = defaultdict(list)
        for sample in samples:
            labels = sample.get(field) or ["unspecified"]
            if not isinstance(labels, list):
                labels = [labels]
            for label in set(map(str, labels)):
                groups[label].append(sample)
        card[f"by_{name}"] = {
            label: _summarize(rows, None) for label, rows in sorted(groups.items())
        }

    by_id: dict[str, list] = defaultdict(list)
    for sample in samples:
        by_id[str(sample.get("doc_id"))].append(sample)
    repeat_details = []
    means = {}
    for doc_id, rows in by_id.items():
        summary = _summarize(rows, None)
        means[doc_id] = {name: metric["mean"] for name, metric in summary["metrics"].items()}
        if len(rows) < 2:
            continue
        successful = [
            row
            for row in rows
            if row.get("inference_status", row.get("status")) in {"parsed", "scored"}
        ]
        hashes = [row["output_sha256"] for row in successful if row.get("output_sha256")]
        counts = Counter(hashes)
        comparisons = len(hashes) * (len(hashes) - 1) / 2
        variances = {}
        for name in means[doc_id]:
            values = [
                float(row["scores"][name])
                for row in successful
                if row.get("status") in {"parsed", "scored"}
                and type((row.get("scores") or {}).get(name)) in (int, float)
                and math.isfinite(row["scores"][name])
            ]
            variances[name] = {
                "n": len(values),
                "mean": statistics.mean(values),
                "population_variance": statistics.pvariance(values) if len(values) > 1 else None,
            }
        repeat_details.append(
            {
                "doc_id": doc_id,
                "runs": len(rows),
                "completed": len(successful),
                "hash_runs": len(hashes),
                "distinct_output_hashes": len(counts),
                "all_outputs_identical": len(counts) == 1
                if len(hashes) >= 2 and len(hashes) == len(successful)
                else None,
                "pairwise_identical_fraction": sum(n * (n - 1) / 2 for n in counts.values())
                / comparisons
                if comparisons
                else None,
                "metrics": variances,
            }
        )
    card["repeats"] = {
        "documents_with_repeats": len(repeat_details),
        "documents": repeat_details,
        "scope": "Hashes and metric variance use successful runs only; failures remain counted.",
    }

    directions = {
        name: "higher_is_better" for name in _RUBRIC_METRICS if name.startswith("annotated_")
    }
    directions.update(
        {
            name: "lower_is_better"
            for name in (
                "cer",
                "wer",
                "token_omission_rate_proxy",
                "token_extra_rate_proxy",
                "token_duplicate_excess_rate_proxy",
            )
        }
    )
    directions.update(options.get("metric_directions", {}))
    for name in ("accuracy", "f1", "precision", "recall"):
        directions.setdefault(name, "higher_is_better")
    for name in card["overall"]["metrics"]:
        if name.startswith("omnidocbench_") and name.endswith("_edit_dist"):
            directions.setdefault(name, "lower_is_better")
        elif any(name.removeprefix("parse_") in keys for keys in _RUBRIC_SCORE_KEYS.values()):
            directions.setdefault(name, "higher_is_better")
    directions = {
        name: {"higher": "higher_is_better", "lower": "lower_is_better"}.get(value, value)
        for name, value in directions.items()
    }
    card["metric_directions"] = directions
    pairs = []
    for variant_id, rows in by_id.items():
        baselines = {str(row["variant_of"]) for row in rows if row.get("variant_of") is not None}
        for baseline_id in sorted(baselines):
            pair = {"baseline_id": baseline_id, "variant_id": variant_id, "metrics": {}}
            if baseline_id not in means:
                pair.update(status="not_measured", reason="Baseline document is absent")
            elif baseline_id == variant_id:
                pair.update(status="not_measured", reason="variant_of points to the same document")
            else:
                for name in means[variant_id].keys() & means[baseline_id].keys():
                    baseline, variant = means[baseline_id][name], means[variant_id][name]
                    delta = variant - baseline
                    measured = {
                        "baseline": baseline,
                        "variant": variant,
                        "variant_minus_baseline": delta,
                    }
                    measured["direction"] = directions.get(name, "unspecified")
                    if directions.get(name) in {"higher_is_better", "lower_is_better"}:
                        measured["degradation"] = (
                            delta if directions[name] == "lower_is_better" else -delta
                        )
                    pair["metrics"][name] = measured
                pair["status"] = "measured" if pair["metrics"] else "not_measured"
                if not pair["metrics"]:
                    pair["reason"] = "No shared scored metric; one side may have failed"
            pair["baseline_runs"] = len(by_id.get(baseline_id, []))
            pair["variant_runs"] = len(rows)
            pairs.append(pair)
    card["paired_robustness"] = {
        "pairs": pairs,
        "measured_pairs": sum(p["status"] == "measured" for p in pairs),
        "scope": "Only explicit variant_of pairs and shared score keys; no composite.",
    }
    rubric = {}
    for name, reason in _RUBRIC_METRICS.items():
        relevant = {name, *_RUBRIC_SCORE_KEYS.get(name, set())}
        relevant.update({"parse_" + key for key in _RUBRIC_SCORE_KEYS.get(name, set())})
        observed_keys = set()
        measured = sum(
            s.get("status") in {"parsed", "scored"}
            and any(
                type((s.get("scores") or {}).get(key)) in (int, float)
                and math.isfinite(s["scores"][key])
                for key in relevant
            )
            for s in samples
        )
        for sample in samples:
            if sample.get("status") in {"parsed", "scored"}:
                observed_keys.update(
                    key
                    for key in relevant
                    if type((sample.get("scores") or {}).get(key)) in (int, float)
                    and math.isfinite(sample["scores"][key])
                )
        states = [s.get("rubric_statuses", {}).get(name, {}) for s in samples]
        reasons = sorted({state["reason"] for state in states if state.get("reason")})
        rubric[name] = {
            "status": "measured" if measured else "not_measured",
            "measured_runs": measured,
            "total_runs": len(samples),
            "score_keys": sorted(observed_keys),
        }
        if measured:
            rubric[name]["scope"] = "Only the listed metric definitions and their annotated cases"
        if measured < len(samples) or not measured:
            rubric[name]["reasons"] = reasons or [reason]
    card["rubric"] = rubric
    warmup = options.get("warmup_samples") or []
    if warmup:
        combined_cost = _summarize([*samples, *warmup], None)["cost"]
        combined_cost["scope"] = "Measured attempts and warmup attempts, including failures"
        combined_cost["warmup_runs"] = len(warmup)
        usable = card["overall"]["completed_documents"]
        combined_cost["per_usable_document"] = (
            combined_cost["total_usd"] / usable
            if combined_cost["total_usd"] is not None and usable
            else None
        )
        card["overall"]["cost"] = combined_cost
    return card
