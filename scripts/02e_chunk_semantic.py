"""Technique 5: Semantic chunking.

Embeds each sentence with a sentence-transformers model, then breaks the
text wherever consecutive sentences' embeddings drift apart (a "topic
shift"), instead of using a fixed size at all. This is the technique most
likely to keep a full idea/section together in one chunk.
"""

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from chunk_utils import PARSED_DIR, save_chunks

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BREAKPOINT_THRESHOLD_TYPE = "percentile"


def main() -> None:
    text = (PARSED_DIR / "attention.txt").read_text()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    splitter = SemanticChunker(embeddings, breakpoint_threshold_type=BREAKPOINT_THRESHOLD_TYPE)
    chunks = splitter.split_text(text)

    save_chunks(
        technique="semantic",
        texts=chunks,
        filename="05_semantic.json",
        extra_metadata=[
            {"embedding_model": EMBEDDING_MODEL, "breakpoint_threshold_type": BREAKPOINT_THRESHOLD_TYPE}
            for _ in chunks
        ],
    )


if __name__ == "__main__":
    main()
