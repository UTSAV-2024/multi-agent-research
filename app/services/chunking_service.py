# ==========================================
# CHUNKING SERVICE
# ==========================================
#
# Splits text into overlapping chunks for
# downstream retrieval and embedding.
#
# ==========================================

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[dict]:

    """
    Split raw text into overlapping chunks.

    Args:
        text: The input text to chunk.
        chunk_size: Max characters per chunk.
        overlap: Overlap between consecutive chunks.

    Returns:
        List of dicts with chunk_id, content, start, end.
    """

    if not text:

        return []

    if overlap >= chunk_size:

        overlap = chunk_size // 2

    chunks = []

    start = 0

    chunk_index = 0

    step = chunk_size - overlap

    if step <= 0:

        step = 1

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk_text_content = text[start:end]

        chunks.append({

            "chunk_id": chunk_index + 1,

            "content": chunk_text_content,

            "start": start,

            "end": end

        })

        chunk_index += 1

        start += step

    return chunks


def chunk_document(
    document: dict,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[dict]:

    """
    Split a document dict (with title, url, content) into
    overlapping retrieval-ready chunks with metadata.

    Args:
        document: Dict with "title", "url", and "content".
        chunk_size: Max characters per chunk.
        overlap: Overlap between consecutive chunks.

    Returns:
        List of dicts with chunk_id, title, url, content,
        start, end.
    """

    content = document.get("content", "")

    if not content:

        return []

    raw_chunks = chunk_text(
        content,
        chunk_size=chunk_size,
        overlap=overlap
    )

    result = []

    for chunk in raw_chunks:

        result.append({

            "chunk_id": chunk["chunk_id"],

            "title": document.get("title", ""),

            "url": document.get("url", ""),

            "content": chunk["content"],

            "start": chunk["start"],

            "end": chunk["end"]

        })

    return result



