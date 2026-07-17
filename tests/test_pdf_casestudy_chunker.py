"""
Tests for the customer-story PDF chunker (scripts/pdf_casestudy_chunker.py).

Drives the parsing logic on synthetic reflowed-line lists (no PDF I/O), covering both
observed layouts (metadata block AFTER vs BEFORE "Executive summary") and the text reflow.
"""

import pytest

from scripts.pdf_casestudy_chunker import (
    _reflow_lines, _split_studies, _parse_metadata, _exec_summary,
)


def _study_a():
    # Layout A: title, Executive summary, metadata block, Read the story, body.
    return [
        "Acme Corp",
        "Acme transforms support with Azure OpenAI",
        "Executive summary",
        "Acme cut handle time by 40% using AI.",
        "Discover more details",
        "CUSTOMER", "Acme Corp",
        "PRODUCTS", "Azure", "Azure OpenAI",
        "COUNTRY", "Spain",
        "INDUSTRY", "Retail",
        "Read the story",
        "Acme is a large retailer.",
        "It deployed a support copilot.",
    ]


def _study_b():
    # Layout B: title, metadata block, Executive summary, Read the story, body.
    return [
        "Globex",
        "Globex accelerates banking with Azure",
        "CUSTOMER", "Globex",
        "PARTNER", "Initech",
        "COUNTRY", "Japan",
        "INDUSTRY", "Banking",
        "Executive summary",
        "Globex automated 90% of reconciliations.",
        "Read the story",
        "Globex is a digital bank.",
    ]


def test_reflow_fixes_hyphen_and_wrap():
    raw = "cutting\n-\nedge solution for satel\n  lites and the futur\n e of flight."
    out = " ".join(_reflow_lines(raw))
    assert "cutting-edge" in out
    assert "satellites" in out
    assert "future" in out


def test_layout_a_parses():
    region = _study_a()
    studies = _split_studies(region)
    assert len(studies) == 1
    meta = _parse_metadata(studies[0].lines)
    assert meta["CUSTOMER"] == "Acme Corp"
    assert meta["PRODUCTS"] == "Azure; Azure OpenAI"
    assert meta["INDUSTRY"] == "Retail"
    assert "handle time by 40%" in _exec_summary(studies[0].lines)


def test_layout_b_parses():
    region = _study_b()
    studies = _split_studies(region)
    assert len(studies) == 1
    meta = _parse_metadata(studies[0].lines)
    assert meta["CUSTOMER"] == "Globex"
    assert meta["PARTNER"] == "Initech"
    assert meta["INDUSTRY"] == "Banking"
    assert "90% of reconciliations" in _exec_summary(studies[0].lines)


def test_two_studies_segment_independently():
    lines = _study_a() + _study_b()
    studies = _split_studies(lines)
    assert len(studies) == 2
    customers = {_parse_metadata(s.lines)["CUSTOMER"] for s in studies}
    assert customers == {"Acme Corp", "Globex"}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
