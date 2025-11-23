
from typing import List


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split long text into overlapping chunks.

    - chunk_size: number of characters per chunk
    - overlap: characters of overlap between chunks to preserve context
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += max(chunk_size - overlap, 1)

    return chunks
