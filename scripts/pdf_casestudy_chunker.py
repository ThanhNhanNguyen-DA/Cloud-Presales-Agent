"""
Chunker for the Microsoft/Azure customer-story PDFs (CMC industry collections).

Each industry PDF (01_Automotive.pdf … 09_Media.pdf) is a *collection* of many
independent customer case studies sharing a fixed layout:

    <Industry>
    <Customer>
    <headline>
    Executive summary
    <executive summary paragraphs>
    Discover more details
    CUSTOMER / PRODUCTS / ORGANIZATION SIZE / COUNTRY / BUSINESS NEED / INDUSTRY   <- label/value block
    Read the story
    <body, with section headers and pull quotes>
    (next case study…)

So the meaningful "section" here is one CASE STUDY, not the whole file. This chunker:

  * segments the file into studies on the reliable "Executive summary" anchor;
  * extracts the per-study metadata block (customer, partner, products, country,
    industry…) and the title;
  * reflows the shredded PDF text (joins hyphen-split and mid-word-wrapped lines);
  * emits chunks whose text starts with a "<Customer> — <Industry> (<Country>)"
    breadcrumb, with the full metadata carried on each chunk's row.

The image-only cover PDFs (10_Nonprofit … 16_Other) contain no extractable text and
are detected/skipped by `is_casestudy_pdf`.

Preview without touching the database:

    python -m scripts.pdf_casestudy_chunker "/path/to/05_Financial_Services.pdf" --preview
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from scripts.text_chunking import chunk_text

# Metadata labels that appear on their own line, each followed by one or more value lines.
META_LABELS = [
    "CUSTOMER", "PARTNER", "PRODUCTS", "ORGANIZATION SIZE", "COMPANY SIZE",
    "COUNTRY", "COUNTRY/REGION", "BUSINESS NEED", "INDUSTRY",
]
_META_LABEL_SET = set(META_LABELS)

# Anchors that bound the parts of a single study.
EXEC_SUMMARY = "executive summary"
DISCOVER = "discover more details"
READ_STORY = "read the story"
# Lines that end a metadata value run (anchors + the labels themselves).
_VALUE_STOP = {EXEC_SUMMARY, DISCOVER, READ_STORY, "find out more about"}
# Max line distance to pair a CUSTOMER block with its Executive summary.
_PAIR_WINDOW = 45

# A line ending in one of these is prose (used to stop title look-back).
_TERMINATORS = ('.', '”', '"', '!', '?', ':', '’', ';')


@dataclass
class PdfChunk:
    content: str
    customer: str
    industry: str
    country: str
    partner: str
    products: str
    study_index: int
    chunk_index: int
    chunk_count: int


@dataclass
class _Study:
    title: list[str] = field(default_factory=list)   # leading lines before the first anchor
    lines: list[str] = field(default_factory=list)   # the full study region (header + body)


def _reflow_lines(raw: str) -> list[str]:
    """
    Repair the shredded text pypdf produces, returning clean non-empty lines.

    Handles two break artifacts seen in these PDFs:
      * hyphenated words split across lines as `word \n - \n word`  -> `word-word`
      * mid-word wraps where the continuation is indented + lowercase -> joined, no space
    Other line breaks (paragraphs, headers, labels) are preserved.
    """
    lines = raw.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        stripped = cur.strip()

        # `word \n - \n word`  ->  join previous + "-" + next
        if stripped == "-" and out and i + 1 < len(lines):
            out[-1] = out[-1].rstrip() + "-" + lines[i + 1].strip()
            i += 2
            continue

        # mid-word wrap: indented lowercase continuation of a word on the previous line
        if (out and re.match(r'^\s+[a-z]', cur)
                and re.search(r'[A-Za-z]$', out[-1])
                and not out[-1].rstrip().endswith(_TERMINATORS)):
            out[-1] = out[-1].rstrip() + stripped
            i += 1
            continue

        out.append(cur)
        i += 1

    return [ln.strip() for ln in out if ln.strip()]


def _back_over_title(lines: list[str], idx: int, lower_bound: int) -> int:
    """Return the index where the title block (short, non-prose lines) before `idx` starts."""
    j = idx - 1
    count = 0
    while j > lower_bound and count < 5 and not lines[j].endswith(_TERMINATORS):
        j -= 1
        count += 1
    return j + 1


def _split_studies(lines: list[str]) -> list[_Study]:
    """
    Segment the document into case studies, robust to the two layouts seen in these PDFs
    (the CUSTOMER metadata block appears either just after OR just before "Executive
    summary"). Each Executive summary is paired with its nearest CUSTOMER block; the study
    starts at the title preceding whichever of the two comes first.
    """
    es_idx = [i for i, l in enumerate(lines) if l.lower() == EXEC_SUMMARY]
    cu_idx = [i for i, l in enumerate(lines) if l.upper() == "CUSTOMER"]

    # Anchor each study at min(exec summary, nearest customer block).
    anchors: list[int] = []
    for e in es_idx:
        nearest = min(cu_idx, key=lambda c: abs(c - e), default=None)
        if nearest is not None and abs(nearest - e) <= _PAIR_WINDOW:
            anchors.append(min(e, nearest))
        else:
            anchors.append(e)
    anchors.sort()

    # Study region = [title-start of this anchor, title-start of next anchor).
    starts: list[int] = []
    prev = -1
    for a in anchors:
        starts.append(_back_over_title(lines, a, prev))
        prev = a

    studies: list[_Study] = []
    for k, hs in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        region = lines[hs:end]
        # Title = leading lines before the first structural anchor in the region.
        title_end = len(region)
        for i, l in enumerate(region):
            if l.lower() in (EXEC_SUMMARY, DISCOVER) or l.upper() in _META_LABEL_SET:
                title_end = i
                break
        studies.append(_Study(title=region[:title_end], lines=region))
    return studies


def _parse_metadata(region: list[str]) -> dict[str, str]:
    """Parse the label/value metadata block wherever it sits in the study region."""
    meta: dict[str, str] = {}
    i = 0
    while i < len(region):
        label = region[i].upper()
        if label in _META_LABEL_SET and label not in meta:
            values: list[str] = []
            j = i + 1
            while (j < len(region) and region[j].upper() not in _META_LABEL_SET
                   and region[j].lower() not in _VALUE_STOP and len(values) < 10):
                values.append(region[j])
                j += 1
            meta[label] = "; ".join(values)
            i = j
        else:
            i += 1
    return meta


def _body_prose(region: list[str]) -> str:
    """Return the story body (everything after 'Read the story') as newline-joined lines."""
    lowered = [l.lower() for l in region]
    start = lowered.index(READ_STORY) + 1 if READ_STORY in lowered else 0
    return "\n".join(region[start:])


def _exec_summary(region: list[str]) -> str:
    """Return the executive-summary text (between its anchor and the next structural anchor)."""
    lowered = [l.lower() for l in region]
    if EXEC_SUMMARY not in lowered:
        return ""
    start = lowered.index(EXEC_SUMMARY) + 1
    out: list[str] = []
    for l in region[start:]:
        if l.lower() in _VALUE_STOP or l.upper() in _META_LABEL_SET:
            break
        out.append(l)
    return " ".join(out).strip()


def is_casestudy_pdf(path: Path) -> bool:
    """True if the PDF has at least one 'Executive summary' anchor (real, text-bearing)."""
    try:
        raw = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    except Exception:
        return False
    return bool(re.search(r'(?im)^\s*executive summary\s*$', raw))


def chunk_pdf_casestudies(path: Path, industry_fallback: str,
                          chunk_size: int = 1000, overlap: int = 150) -> list[PdfChunk]:
    """
    Chunk a customer-story collection PDF into per-study, breadcrumb-tagged chunks.

    Args:
        path:             the .pdf file.
        industry_fallback: industry label used when a study has no INDUSTRY field
                           (usually the file's industry, e.g. "Automotive").
        chunk_size/overlap: passed to the recursive splitter for each study's text.
    """
    raw = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    lines = _reflow_lines(raw)
    studies = _split_studies(lines)
    chunks: list[PdfChunk] = []

    for s_idx, study in enumerate(studies):
        meta = _parse_metadata(study.lines)
        customer = meta.get("CUSTOMER") or (study.title[-2] if len(study.title) >= 2 else
                                            (study.title[0] if study.title else "Unknown"))
        industry = meta.get("INDUSTRY") or industry_fallback
        country = meta.get("COUNTRY") or meta.get("COUNTRY/REGION") or ""
        partner = meta.get("PARTNER") or ""
        products = meta.get("PRODUCTS") or ""

        breadcrumb = f"{customer} — {industry}" + (f" ({country})" if country else "")
        headline = " ".join(study.title).strip()
        meta_line = " | ".join(filter(None, [
            f"Customer: {customer}", f"Industry: {industry}",
            f"Country: {country}" if country else "",
            f"Partner: {partner}" if partner else "",
            f"Products: {products}" if products else "",
        ]))

        exec_summary = _exec_summary(study.lines)

        document = "\n".join(filter(None, [
            headline,
            meta_line,
            f"Executive summary: {exec_summary}" if exec_summary else "",
            _body_prose(study.lines),
        ]))
        pieces = chunk_text(document, chunk_size=chunk_size, overlap=overlap)

        for c_idx, piece in enumerate(pieces):
            content = f"{breadcrumb}\n\n{piece}".strip()
            chunks.append(PdfChunk(
                content=content, customer=customer, industry=industry, country=country,
                partner=partner, products=products, study_index=s_idx,
                chunk_index=c_idx, chunk_count=len(pieces),
            ))
    return chunks


def _preview(paths: list[str], chunk_size: int) -> None:
    for raw in paths:
        p = Path(raw)
        if not is_casestudy_pdf(p):
            print(f"\n{p.name}: no 'Executive summary' anchor — image-only/empty, SKIPPED")
            continue
        chunks = chunk_pdf_casestudies(p, industry_fallback=p.stem, chunk_size=chunk_size)
        studies = max((c.study_index for c in chunks), default=-1) + 1
        print(f"\n{'=' * 78}\n{p.name}  →  {studies} studies, {len(chunks)} chunk(s)\n{'=' * 78}")
        for c in chunks[:6]:
            print(f"\n── study {c.study_index + 1}, chunk {c.chunk_index + 1}/{c.chunk_count} "
                  f"(len={len(c.content)}) ──")
            print(c.content[:650] + ("…" if len(c.content) > 650 else ""))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview case-study chunks for an industry PDF.")
    parser.add_argument("path", nargs="+")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    _preview(args.path, args.chunk_size)
