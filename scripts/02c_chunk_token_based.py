"""Technique 3: Token-based chunking.

Splits directly on token counts (via tiktoken) instead of characters, so
chunk sizes map exactly to what matters for an LLM/embedding context window,
regardless of how verbose the underlying text is.
"""

from langchain_text_splitters import TokenTextSplitter

from chunk_utils import PARSED_DIR, save_chunks

CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 32


def main() -> None:
    text = (PARSED_DIR / "attention.txt").read_text()

    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )
    chunks = splitter.split_text(text)

    save_chunks(
        technique="token_based",
        texts=chunks,
        filename="03_token_based.json",
        extra_metadata=[
            {"chunk_size_tokens": CHUNK_SIZE_TOKENS, "overlap_tokens": CHUNK_OVERLAP_TOKENS} for _ in chunks
        ],
    )


if __name__ == "__main__":
    main()
