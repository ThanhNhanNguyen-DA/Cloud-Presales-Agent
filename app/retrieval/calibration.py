"""Calibrate evidence acceptance on reviewed queries, with a separate validation split.

One observation is a QUERY, not a passage: passages from one query are correlated.
An accepted query is safe only when every returned passage was labelled relevant.
The Wilson interval describes observed evidence selection, never answer correctness.
"""

from __future__ import annotations

import math


def wilson_lower_bound(successes: int, total: int) -> float:
    """Lower end of a two-sided 95% Wilson interval."""
    if not total:
        return 0.0
    z = 1.959963984540054
    p = successes / total
    return (p + z * z / (2 * total) - z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    )) / (1 + z * z / total)


def validate_queries(queries: list[dict]) -> None:
    ids, questions = set(), set()
    for query in queries:
        identifier = query.get("id")
        question = " ".join(str(query.get("question") or "").casefold().split())
        if not identifier or identifier in ids or not question or question in questions:
            raise ValueError("Every query needs a unique id and question across both splits.")
        ids.add(identifier)
        questions.add(question)
        if query.get("split") not in {"calibration", "validation"}:
            raise ValueError("Each reviewed query needs split=calibration or validation.")
        if type(query.get("answerable")) is not bool:
            raise ValueError("answerable must be a reviewer-supplied boolean.")
        seen = set()
        for candidate in query.get("candidates", []):
            score = candidate.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Every candidate needs a finite sigmoid score in [0, 1].")
            if type(candidate.get("relevant")) is not bool:
                raise ValueError("Every candidate must be reviewed: relevant=true or false; null is not a label.")
            if not candidate.get("id") or candidate["id"] in seen:
                raise ValueError("Candidate ids must be present and unique within a query.")
            seen.add(candidate["id"])
            if not query["answerable"] and candidate["relevant"]:
                raise ValueError("An unanswerable query cannot have a relevant candidate.")
    for split in ("calibration", "validation"):
        labels = {q["answerable"] for q in queries if q["split"] == split}
        if labels != {True, False}:
            raise ValueError(f"{split} needs both answerable and unanswerable reviewed queries.")


def threshold_metrics(queries: list[dict], threshold: float, top_k: int = 4) -> dict:
    accepted = safe = negative_accepted = answerable_covered = 0
    relevant_returned = relevant_available = 0
    for query in queries:
        ranked = sorted(query.get("candidates", []), key=lambda c: c["score"], reverse=True)
        selected = [c for c in ranked if c["score"] >= threshold][:top_k]
        relevant_available += sum(c["relevant"] for c in ranked)
        relevant_returned += sum(c["relevant"] for c in selected)
        if selected:
            accepted += 1
            safe += all(c["relevant"] for c in selected)
            negative_accepted += not query["answerable"]
            answerable_covered += query["answerable"] and any(c["relevant"] for c in selected)
    positives = sum(q["answerable"] for q in queries)
    negatives = len(queries) - positives
    return {
        "queries": len(queries), "accepted_queries": accepted, "safe_queries": safe,
        "coverage": accepted / len(queries) if queries else 0.0,
        "selective_precision": safe / accepted if accepted else None,
        "precision_lower_bound_95": wilson_lower_bound(safe, accepted),
        "answerable_coverage": answerable_covered / positives if positives else None,
        "unanswerable_acceptance_rate": negative_accepted / negatives if negatives else None,
        "candidate_recall": relevant_returned / relevant_available if relevant_available else None,
    }


def calibrate_threshold(queries: list[dict], *, target_precision: float = 0.95,
                        min_accepted: int = 20, top_k: int = 4) -> dict:
    if (not 0 < target_precision < 1 or type(min_accepted) is not int or min_accepted < 1
            or type(top_k) is not int or top_k < 1):
        raise ValueError("Use 0 < target_precision < 1, min_accepted >= 1, top_k >= 1.")
    validate_queries(queries)
    calibration = [q for q in queries if q["split"] == "calibration"]
    validation = [q for q in queries if q["split"] == "validation"]
    thresholds = sorted({0.0, 1.0, *(c["score"] for q in calibration for c in q.get("candidates", []))})
    curve = [{"threshold": t, **threshold_metrics(calibration, t, top_k)} for t in thresholds]
    eligible = [m for m in curve if m["accepted_queries"] >= min_accepted
                and m["precision_lower_bound_95"] >= target_precision]
    report = {"target_precision": target_precision, "min_accepted": min_accepted,
              "top_k": top_k, "selected_threshold": None, "calibration_curve": curve}
    if not eligible:
        return {**report, "status": "insufficient_calibration_evidence"}
    best = max(eligible, key=lambda m: (m["coverage"], m["candidate_recall"] or 0,
                                      m["precision_lower_bound_95"], m["threshold"]))
    measured = threshold_metrics(validation, best["threshold"], top_k)
    valid = measured["accepted_queries"] >= min_accepted and measured["precision_lower_bound_95"] >= target_precision
    return {**report, "status": "validated" if valid else "validation_failed",
            "candidate_threshold": best["threshold"], "calibration": best,
            "validation": measured, "selected_threshold": best["threshold"] if valid else None}
