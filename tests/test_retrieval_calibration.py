"""Threshold selection must not turn a small/easy sample into a confidence claim."""

import copy

import pytest

from app.retrieval.calibration import calibrate_threshold, threshold_metrics, wilson_lower_bound


def reviewed_queries():
    queries = []
    for split in ("calibration", "validation"):
        for index in range(10):
            queries.append({"id": f"{split}-positive-{index}", "question": f"{split} question {index}",
                            "split": split, "answerable": True,
                            "candidates": [{"id": "gold", "score": 0.9, "relevant": True},
                                           {"id": "noise", "score": 0.4, "relevant": False}]})
            queries.append({"id": f"{split}-negative-{index}", "question": f"{split} unknown {index}",
                            "split": split, "answerable": False,
                            "candidates": [{"id": "noise", "score": 0.2, "relevant": False}]})
    return queries


def test_selects_threshold_then_measures_it_on_separate_queries():
    result = calibrate_threshold(reviewed_queries(), target_precision=0.6, min_accepted=5)
    assert result["status"] == "validated" and result["selected_threshold"] == 0.9
    assert result["validation"]["selective_precision"] == 1.0
    assert result["validation"]["coverage"] == 0.5
    assert result["validation"]["unanswerable_acceptance_rate"] == 0.0


def test_validation_failure_does_not_publish_or_retune_threshold():
    queries = reviewed_queries()
    for query in queries:
        if query["split"] == "validation" and not query["answerable"]:
            query["candidates"][0]["score"] = 0.99
    result = calibrate_threshold(queries, target_precision=0.6, min_accepted=5)
    assert result["candidate_threshold"] == 0.9
    assert result["status"] == "validation_failed" and result["selected_threshold"] is None


def test_tiny_perfect_sample_cannot_claim_95_percent_precision():
    result = calibrate_threshold(reviewed_queries(), target_precision=0.95, min_accepted=1)
    assert result["status"] == "insufficient_calibration_evidence"
    assert result["selected_threshold"] is None
    assert wilson_lower_bound(10, 10) < 0.95


def test_one_wrong_passage_makes_the_query_unsafe_even_with_a_good_top_hit():
    queries = reviewed_queries()[:1]
    measured = threshold_metrics(queries, threshold=0.1)
    assert measured["accepted_queries"] == 1 and measured["safe_queries"] == 0


@pytest.mark.parametrize("mutation", ["unreviewed", "only_positive", "leaked_question", "nan", "contradiction"])
def test_rejects_invalid_or_leaking_labels(mutation):
    queries = copy.deepcopy(reviewed_queries())
    if mutation == "unreviewed":
        queries[0]["candidates"][0]["relevant"] = None
    elif mutation == "only_positive":
        queries = [q for q in queries if q["answerable"]]
    elif mutation == "leaked_question":
        queries[-1]["question"] = queries[0]["question"]
    elif mutation == "nan":
        queries[0]["candidates"][0]["score"] = float("nan")
    else:
        queries[1]["candidates"][0]["relevant"] = True
    with pytest.raises(ValueError):
        calibrate_threshold(queries)


# --------------------------------------------------------------------------- abstention criterion
from app.retrieval.calibration import (  # noqa: E402
    abstention_metrics, calibrate_abstention, negatives_needed, wilson_upper_bound,
)


def abstention_queries(negatives_per_split=80, positives_per_split=40):
    queries = []
    for split in ("calibration", "validation"):
        for i in range(positives_per_split):
            queries.append({"id": f"{split}-p{i}", "question": f"{split} answerable {i}", "split": split,
                            "answerable": True,
                            "candidates": [{"id": "gold", "score": 0.95, "relevant": True},
                                           {"id": "near", "score": 0.6, "relevant": False},   # partly relevant
                                           {"id": "deep", "score": 0.05, "relevant": None}]})  # unreviewed, below top-k
        for i in range(negatives_per_split):
            queries.append({"id": f"{split}-n{i}", "question": f"{split} unanswerable {i}", "split": split,
                            "answerable": False,
                            "candidates": [{"id": "noise", "score": 0.02, "relevant": False}]})
    return queries


def test_required_unanswerable_sample_size_is_reported():
    assert negatives_needed(0.05) == 73 and negatives_needed(0.10) == 35
    assert wilson_upper_bound(0, 0) == 1.0


def test_abstention_selects_a_cutoff_that_rejects_unanswerable_questions():
    result = calibrate_abstention(abstention_queries(), top_k=2)
    assert result["status"] == "validated"
    assert 0.02 < result["selected_threshold"] <= 0.95
    assert result["validation"]["answerable_hit_rate"] == 1.0
    assert result["validation"]["false_accept_rate"] == 0.0


def test_partly_relevant_extra_passages_are_not_failures():
    m = abstention_metrics(abstention_queries(), threshold=0.5, top_k=4)
    assert m["answerable_hit_rate"] == 1.0 and m["top1_precision"] == 1.0


def test_too_few_unanswerable_questions_cannot_be_validated():
    result = calibrate_abstention(abstention_queries(negatives_per_split=20), top_k=2)
    assert result["status"] == "insufficient_calibration_evidence" and result["selected_threshold"] is None
    assert "73" in result["hint"] and "have 20" in result["hint"]


def test_only_top_k_needs_labels():
    queries = abstention_queries()
    queries[0]["candidates"][0]["relevant"] = None   # top-1 unlabelled -> rejected
    with pytest.raises(ValueError):
        calibrate_abstention(queries, top_k=2)
