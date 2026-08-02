"""Scoring metrics for qualitative document-reading tasks."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Callable


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s.%$-]", "", s)
    return s


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def contains_answer(prediction: str, reference: str) -> float:
    """1.0 if normalized reference appears in prediction."""
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    if not ref:
        return 0.0
    return 1.0 if ref in pred else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _try_parse_json(s: str) -> Any | None:
    s = s.strip()
    # Strip markdown fences
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Attempt to find first {...} or [...]
        for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
            m = re.search(pattern, s)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    continue
    return None


def json_exact(prediction: str, reference: str) -> float:
    """Structural equality of JSON payloads (order-insensitive for objects)."""
    pred = _try_parse_json(prediction)
    # reference may already be a JSON string or Python object stringified
    ref = _try_parse_json(reference)
    if ref is None:
        try:
            ref = json.loads(reference) if isinstance(reference, str) else reference
        except (json.JSONDecodeError, TypeError):
            return 0.0
    if pred is None:
        return 0.0
    return 1.0 if pred == ref else 0.0


def numeric_tolerance(prediction: str, reference: str, tol: float = 0.01) -> float:
    """Extract first number from each side; score 1 if within relative tolerance."""
    num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    p = num_re.search(prediction.replace(",", ""))
    r = num_re.search(str(reference).replace(",", ""))
    if not p or not r:
        return 0.0
    pv, rv = float(p.group()), float(r.group())
    if rv == 0:
        return 1.0 if abs(pv) <= tol else 0.0
    return 1.0 if abs(pv - rv) / abs(rv) <= tol else 0.0


METRIC_REGISTRY: dict[str, Callable[[str, str], float]] = {
    "exact_match": exact_match,
    "contains": contains_answer,
    "token_f1": token_f1,
    "json_exact": json_exact,
    "numeric_tolerance": numeric_tolerance,
}


def score(prediction: str, reference: str, metrics: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in metrics:
        fn = METRIC_REGISTRY.get(name)
        if fn is None:
            raise ValueError(f"Unknown metric: {name}. Known: {list(METRIC_REGISTRY)}")
        out[name] = float(fn(prediction, reference))
    return out


def aggregate(sample_scores: list[dict[str, float]]) -> dict[str, float]:
    if not sample_scores:
        return {}
    keys = sample_scores[0].keys()
    return {k: sum(s[k] for s in sample_scores) / len(sample_scores) for k in keys}