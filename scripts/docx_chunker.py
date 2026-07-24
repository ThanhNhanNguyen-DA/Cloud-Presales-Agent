"""
Section-based chunker for the CMC service-documentation DOCX files.

These documents follow a consistent template built from real Word heading styles
(Heading 1/2/3/4) — e.g.

    Description (H1)
      Service overview (H2)
      Service specification (H2)  -> followed by a spec table
      Service guide (H2)
        Console portal (H3)
          Quản lý RDS (H4)

Instead of splitting blindly on newlines (which throws away that structure), this
chunker walks the document in body order, tracks the heading hierarchy, and emits
ONE chunk per leaf section. Each chunk:

  * carries a breadcrumb header ("CMC Cloud RDS › Description › Service guide › …")
    prepended to its text, so the embedding captures where the content sits and
    retrieval/citations stay precise;
  * keeps tables inline, rendered as Markdown, attached to the section they belong to;
  * is split further only when a single section exceeds `chunk_size` (packing whole
    lines, never cutting mid-line), so we never merge unrelated sections.

Used by scripts/ingest_files.py for `.docx` files. Preview chunks without touching
the database with:

    python -m scripts.docx_chunker "/path/to/CMC Cloud RDS.docx" --preview
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterator

import docx
from docx.document import Document as _Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


# Leaf sections whose title matches these are pure admin/boilerplate (access-control
# matrices, hashtag bookkeeping) and add only noise to retrieval. Edit to taste.
SKIP_SECTION_TITLES = {"permision", "permission", "hashtags"}

# Paragraph styles that represent bullet/numbered list items.
LIST_STYLES = {"List Paragraph"}


@dataclass
class Section:
    path: str           # breadcrumb, e.g. "Description › Service guide › Quản lý RDS"
    level: int          # heading level of the leaf (1..N)
    lines: list[str] = field(default_factory=list)
    has_table: bool = False


@dataclass
class DocxChunk:
    content: str        # breadcrumb header + body, ready to embed
    section_path: str
    section_level: int
    has_table: bool
    chunk_index_in_section: int
    section_chunk_count: int


def iter_block_items(parent: _Document) -> Iterator[object]:
    """Yield Paragraph and Table objects in true document (body) order."""
    body = parent.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def _heading_level(paragraph: Paragraph) -> int | None:
    """Return the heading level for a paragraph styled 'Heading N', else None."""
    style = paragraph.style.name if paragraph.style else ""
    if style and style.startswith("Heading "):
        tail = style.split("Heading ", 1)[1].strip()
        if tail.isdigit():
            return int(tail)
    return None


def _table_to_markdown(table: Table) -> str:
    """Render a Word table as a GitHub-flavoured Markdown table."""
    rows: list[list[str]] = []
    for row in table.rows:
        rows.append([cell.text.strip().replace("\n", " ") for cell in row.cells])
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, *body = rows
    md = [" | ".join(header), " | ".join(["---"] * width)]
    md += [" | ".join(r) for r in body]
    return "\n".join(md)


def _split_into_sections(path: Path) -> list[Section]:
    """Walk the document and group its content into heading-bounded sections."""
    document = docx.Document(str(path))
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []   # (heading level, title)
    current: Section | None = None

    def breadcrumb() -> str:
        return " › ".join(title for _, title in stack)

    for block in iter_block_items(document):
        if isinstance(block, Paragraph):
            level = _heading_level(block)
            if level is not None:
                title = block.text.strip()
                if not title:
                    continue
                # Close the previous leaf section (drop empty parent headings).
                if current and current.lines:
                    sections.append(current)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                current = Section(path=breadcrumb(), level=level)
                continue

            text = block.text.strip()
            if not text:
                continue
            if current is None:                       # content before any heading
                current = Section(path="(intro)", level=0)
            style = block.style.name if block.style else ""
            current.lines.append(f"- {text}" if style in LIST_STYLES else text)

        elif isinstance(block, Table):
            md = _table_to_markdown(block)
            if not md:
                continue
            if current is None:
                current = Section(path="(intro)", level=0)
            current.lines.append(md)
            current.has_table = True

    if current and current.lines:
        sections.append(current)

    # Drop admin/boilerplate leaf sections.
    return [s for s in sections if s.path.split(" › ")[-1].strip().lower() not in SKIP_SECTION_TITLES]


def _pack_lines(lines: list[str], chunk_size: int, overlap_lines: int = 1) -> list[str]:
    """Greedily pack whole lines into <= chunk_size bodies; hard-split giant lines."""
    chunks: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        if len(line) > chunk_size:                    # single line/table too big
            if current:
                chunks.append(current)
                current, current_len = [], 0
            for i in range(0, len(line), chunk_size):
                chunks.append([line[i:i + chunk_size]])
            continue
        if current and current_len + len(line) + 1 > chunk_size:
            chunks.append(current)
            tail = current[-overlap_lines:] if overlap_lines else []
            current = list(tail)
            current_len = sum(len(x) + 1 for x in current)
        current.append(line)
        current_len += len(line) + 1

    if current:
        chunks.append(current)
    return ["\n".join(c) for c in chunks]


def chunk_docx_sections(path: Path, doc_title: str, chunk_size: int = 1000) -> list[DocxChunk]:
    """
    Chunk a CMC service-documentation DOCX into section-based chunks.

    Args:
        path:       path to the .docx file.
        doc_title:  human title prefixed to every breadcrumb (usually the service name).
        chunk_size: target characters per chunk; only oversized sections are split.

    Returns:
        A list of DocxChunk, each with a breadcrumb header already prepended to `content`.
    """
    sections = _split_into_sections(path)
    chunks: list[DocxChunk] = []

    for section in sections:
        breadcrumb = f"{doc_title} › {section.path}" if section.path != "(intro)" else doc_title
        bodies = _pack_lines(section.lines, chunk_size=chunk_size)
        for idx, body in enumerate(bodies):
            content = f"{breadcrumb}\n\n{body}".strip()
            chunks.append(DocxChunk(
                content=content,
                section_path=section.path,
                section_level=section.level,
                has_table=section.has_table,
                chunk_index_in_section=idx,
                section_chunk_count=len(bodies),
            ))
    return chunks


def _preview(paths: list[str], chunk_size: int) -> None:
    for raw in paths:
        p = Path(raw)
        chunks = chunk_docx_sections(p, doc_title=p.stem, chunk_size=chunk_size)
        print(f"\n{'=' * 78}\n{p.name}  →  {len(chunks)} chunk(s)\n{'=' * 78}")
        for i, c in enumerate(chunks):
            flag = " [has_table]" if c.has_table else ""
            print(f"\n── chunk {i}  (len={len(c.content)}, "
                  f"section={c.chunk_index_in_section + 1}/{c.section_chunk_count}){flag} ──")
            print(c.content[:700] + ("…" if len(c.content) > 700 else ""))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview section-based chunks for a DOCX file.")
    parser.add_argument("path", nargs="+", help="One or more .docx files to preview.")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--preview", action="store_true", help="Print chunks (default action).")
    args = parser.parse_args()
    _preview(args.path, args.chunk_size)
