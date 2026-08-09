"""Technique 1: Fixed-size character chunking (the naive baseline).

Slide a fixed-width window over the raw text with a fixed overlap, with no
awareness of sentences, paragraphs, or document structure. Chunks can cut a
sentence (or a word) in half — that's the point of using it as a baseline.
"""

from chunk_utils import PARSED_DIR, save_chunks

CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters


def fixed_size_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks


def main() -> None:
    text = (PARSED_DIR / "attention.txt").read_text()
    chunks = fixed_size_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    save_chunks(
        technique="fixed_size",
        texts=chunks,
        filename="01_fixed_size.json",
        extra_metadata=[{"chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP} for _ in chunks],
    )


if __name__ == "__main__":
    main()
