"""
Generic recursive character chunking, shared by the ingestion scripts.

Kept dependency-free (no embedding model, no DB) so chunkers and tests can import it
cheaply.
"""
from typing import List


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """
    Split text into overlapping chunks, preferring paragraph (newline) boundaries.

    Paragraphs are accumulated until adding the next one would exceed `chunk_size`;
    the current chunk is then emitted and the next chunk starts with an `overlap`-sized
    tail of the previous chunk so context isn't lost across boundaries.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) + 1 > chunk_size:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap > 0 else ""
            current = (tail + " " + para).strip()
        else:
            current = (current + "\n" + para).strip() if current else para

    if current.strip():
        chunks.append(current.strip())

    # Hard-split any chunk that is still too large (e.g. one giant paragraph).
    final: List[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size * 1.5:
            final.append(chunk)
        else:
            step = chunk_size - overlap
            for i in range(0, len(chunk), step):
                piece = chunk[i:i + chunk_size].strip()
                if piece:
                    final.append(piece)
    return final
