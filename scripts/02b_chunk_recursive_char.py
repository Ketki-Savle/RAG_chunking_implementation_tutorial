"""Technique 2: Recursive character chunking.

Uses LangChain's RecursiveCharacterTextSplitter, which tries to split on
paragraph breaks first, then line breaks, then spaces, only falling back to
raw character cuts as a last resort. This keeps chunks aligned to natural
text boundaries far more often than the fixed-size baseline.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunk_utils import PARSED_DIR, save_chunks

CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters


def main() -> None:
    text = (PARSED_DIR / "attention.txt").read_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)

    save_chunks(
        technique="recursive_char",
        texts=chunks,
        filename="02_recursive_char.json",
        extra_metadata=[{"chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP} for _ in chunks],
    )


if __name__ == "__main__":
    main()
