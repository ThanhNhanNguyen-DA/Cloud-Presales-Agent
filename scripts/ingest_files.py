"""
Ingest real documents (PDF / DOCX / TXT / MD) into the `cloud_documents` table for RAG.

For each file it:
  1. Extracts the raw text (PDF via pypdf, DOCX via python-docx, TXT/MD read directly).
  2. Splits the text into overlapping chunks.
  3. Embeds each chunk (Qwen3-Embedding-0.6B, 512 dims, local Hugging Face model).
  4. Inserts each chunk into Supabase `cloud_documents` (content + embedding + metadata).

Usage:
    # Ingest a single file
    python -m scripts.ingest_files docs/object_storage_sla.pdf --service "Object Storage" --category SLA

    # Ingest every supported file in a directory (recursively)
    python -m scripts.ingest_files ./docs --service "Cloud Server" --category Feature

    # Let service/category default to the file name / "General"
    python -m scripts.ingest_files ./docs

Options:
    --service     service_name stored on each chunk (default: the file name without extension)
    --category    category stored on each chunk: SLA | Security | Compliance | Feature | General (default: General)
    --chunk-size  characters per chunk (default: 1000)
    --overlap     overlapping characters between consecutive chunks (default: 150)

Notes:
    - PDF support needs `pypdf`, DOCX support needs `python-docx` (both in requirements.txt).
    - Make sure the database schema exists first (see sql/schema.sql) and `.env` is configured.
"""

import argparse
import sys
from pathlib import Path

from app.db.supabase_client import supabase
from app.tools.embedding_tool import embed_text
from scripts.docx_chunker import chunk_docx_sections
from scripts.pdf_casestudy_chunker import chunk_pdf_casestudies, is_casestudy_pdf
from scripts.text_chunking import chunk_text


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
VALID_CATEGORIES = {"SLA", "Security", "Compliance", "Feature", "General", "CaseStudy"}


def extract_text(path: Path) -> str:
    """Extract plain text from a PDF, DOCX, TXT, or MD file."""
    ext = path.suffix.lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("PDF support requires 'pypdf'. Install it with: pip install pypdf")
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if ext == ".docx":
        try:
            import docx  # python-docx
        except ImportError:
            raise ImportError("DOCX support requires 'python-docx'. Install it with: pip install python-docx")
        document = docx.Document(str(path))
        return "\n".join(p.text for p in document.paragraphs)

    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: {ext}")


def _insert_chunk(service_name: str, category: str, content: str, metadata: dict) -> None:
    """Embed `content` and insert one row into cloud_documents."""
    supabase.table("cloud_documents").insert({
        "service_name": service_name,
        "category": category,
        "content": content,
        "embedding": embed_text(content),
        "metadata": metadata,
    }).execute()


def ingest_file(path: Path, service_name: str, category: str, chunk_size: int, overlap: int) -> int:
    """Ingest a single file; returns the number of chunks inserted."""
    # DOCX service docs are structured (Word heading styles + tables) — chunk by section
    # so each chunk is a coherent unit carrying its heading breadcrumb.
    if path.suffix.lower() == ".docx":
        chunks = chunk_docx_sections(path, doc_title=service_name, chunk_size=chunk_size)
        if not chunks:
            print(f"  ⚠️  No section content in {path.name} — skipped.")
            return 0
        for chunk in chunks:
            _insert_chunk(service_name, category, chunk.content, {
                "source": path.name,
                "type": "file_ingest",
                "chunker": "docx_section",
                "section_path": chunk.section_path,
                "section_level": chunk.section_level,
                "has_table": chunk.has_table,
                "chunk_index": chunk.chunk_index_in_section,
                "total_chunks": chunk.section_chunk_count,
            })
        print(f"  ✓ {path.name}: inserted {len(chunks)} section chunk(s) as [{service_name} / {category}]")
        return len(chunks)

    # Customer-story collection PDFs (Azure/MS industry brochures) hold many case studies.
    # Chunk per study, tagging each chunk with its customer/industry/products metadata.
    if path.suffix.lower() == ".pdf" and is_casestudy_pdf(path):
        industry = path.stem.split("_", 1)[-1].replace("_", " ")  # "05_Financial_Services" -> "Financial Services"
        chunks = chunk_pdf_casestudies(path, industry_fallback=industry,
                                       chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            print(f"  ⚠️  No case studies parsed in {path.name} — skipped.")
            return 0
        for chunk in chunks:
            _insert_chunk(chunk.customer, category, chunk.content, {
                "source": path.name,
                "type": "file_ingest",
                "chunker": "pdf_casestudy",
                "customer": chunk.customer,
                "industry": chunk.industry,
                "country": chunk.country,
                "partner": chunk.partner,
                "products": chunk.products,
                "study_index": chunk.study_index,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.chunk_count,
            })
        studies = max((c.study_index for c in chunks), default=-1) + 1
        print(f"  ✓ {path.name}: inserted {len(chunks)} chunk(s) from {studies} case studies as [{category}]")
        return len(chunks)

    # PDF / TXT / MD: plain extraction + recursive character chunking.
    text = extract_text(path)
    if not text.strip():
        print(f"  ⚠️  No extractable text in {path.name} — skipped.")
        return 0

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    for idx, chunk in enumerate(chunks):
        _insert_chunk(service_name, category, chunk, {
            "source": path.name,
            "type": "file_ingest",
            "chunker": "recursive_char",
            "chunk_index": idx,
            "total_chunks": len(chunks),
        })
    print(f"  ✓ {path.name}: inserted {len(chunks)} chunk(s) as [{service_name} / {category}]")
    return len(chunks)


def collect_files(target: Path) -> list[Path]:
    """Return the list of supported files for a file or directory target."""
    if target.is_file():
        return [target] if target.suffix.lower() in SUPPORTED_EXTENSIONS else []
    if target.is_dir():
        return sorted(p for p in target.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    return []


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF/DOCX/TXT/MD files into cloud_documents for RAG.")
    parser.add_argument("path", nargs="+", help="One or more paths to a file or a directory of files to ingest.")
    parser.add_argument("--service", default=None, help="service_name for the chunks (default: file name).")
    parser.add_argument("--category", default="General", help=f"One of {sorted(VALID_CATEGORIES)} (default: General).")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Characters per chunk (default: 1000).")
    parser.add_argument("--overlap", type=int, default=150, help="Overlap characters between chunks (default: 150).")
    args = parser.parse_args()

    if args.category not in VALID_CATEGORIES:
        print(f"⚠️  '{args.category}' is not a standard category {sorted(VALID_CATEGORIES)}; ingesting anyway.")

    files: list[Path] = []
    seen = set()
    for raw in args.path:
        target = Path(raw)
        if not target.exists():
            print(f"⚠️  Path not found, skipping: {target}")
            continue
        for f in collect_files(target):
            if f not in seen:
                seen.add(f)
                files.append(f)

    if not files:
        print(f"❌ No supported files ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) found in the given path(s).")
        sys.exit(1)

    print(f"Starting ingestion of {len(files)} file(s)...\n")
    total = 0
    for path in files:
        service_name = args.service or path.stem
        try:
            total += ingest_file(path, service_name, args.category, args.chunk_size, args.overlap)
        except Exception as e:
            print(f"  ❌ Failed to ingest {path.name}: {e}")

    print(f"\n✅ Done. Inserted {total} chunk(s) from {len(files)} file(s) into cloud_documents.")


if __name__ == "__main__":
    main()
