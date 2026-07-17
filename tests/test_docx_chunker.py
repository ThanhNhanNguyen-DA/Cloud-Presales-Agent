"""
Tests for the section-based DOCX chunker (scripts/docx_chunker.py).

Builds a small in-memory .docx mirroring the CMC service-doc template and asserts
the chunker respects heading sections, attaches tables, drops admin sections, and
splits only oversized sections.
"""

import docx
import pytest

from scripts.docx_chunker import chunk_docx_sections


@pytest.fixture
def sample_docx(tmp_path):
    d = docx.Document()
    d.add_heading("Description", level=1)

    d.add_heading("Service overview", level=2)
    d.add_paragraph("This service does things.")
    d.add_paragraph("Fast deploy", style="List Paragraph")
    d.add_paragraph("Low cost", style="List Paragraph")

    d.add_heading("Service specification", level=2)
    table = d.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Engine"
    table.cell(0, 1).text = "MySQL"
    table.cell(1, 0).text = "IOPS"
    table.cell(1, 1).text = "60000"

    # Admin section that must be dropped.
    d.add_heading("Permision", level=2)
    perm = d.add_table(rows=1, cols=2)
    perm.cell(0, 0).text = "Owner"
    perm.cell(0, 1).text = "Read"

    # Oversized section to force splitting.
    d.add_heading("Service guide", level=2)
    for i in range(40):
        d.add_paragraph(f"Step {i}: " + "x" * 60)

    path = tmp_path / "Sample Service.docx"
    d.save(str(path))
    return path


def test_sections_carry_breadcrumb(sample_docx):
    chunks = chunk_docx_sections(sample_docx, doc_title="Sample Service")
    overview = [c for c in chunks if c.section_path == "Description › Service overview"]
    assert overview, "expected a Service overview section"
    assert overview[0].content.startswith("Sample Service › Description › Service overview")
    assert "- Fast deploy" in overview[0].content  # list items kept as bullets


def test_table_is_attached_and_rendered(sample_docx):
    chunks = chunk_docx_sections(sample_docx, doc_title="Sample Service")
    spec = [c for c in chunks if c.section_path.endswith("Service specification")]
    assert spec and spec[0].has_table
    assert "Engine | MySQL" in spec[0].content
    assert "--- | ---" in spec[0].content


def test_admin_section_is_dropped(sample_docx):
    chunks = chunk_docx_sections(sample_docx, doc_title="Sample Service")
    assert not any("Permision" in c.section_path for c in chunks)


def test_oversized_section_is_split(sample_docx):
    chunks = chunk_docx_sections(sample_docx, doc_title="Sample Service", chunk_size=500)
    guide = [c for c in chunks if c.section_path.endswith("Service guide")]
    assert len(guide) > 1, "oversized guide section should split into multiple chunks"
    assert all(len(c.content) <= 500 + len("Sample Service › Description › Service guide") + 8
               for c in guide)
    # every split keeps the same breadcrumb header
    assert all(c.content.startswith("Sample Service › Description › Service guide") for c in guide)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
