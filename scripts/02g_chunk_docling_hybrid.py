"""Technique 7: Docling hybrid chunking.

Starts from the same structure-aware chunks as HierarchicalChunker, then
splits any chunk that overflows the tokenizer's max-token budget and merges
small adjacent peer chunks together -- so every chunk both respects the
document's structure AND fits a real embedding-model context window. This
is the technique docling recommends for production RAG.
"""

import json

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.document import DoclingDocument

from chunk_utils import PARSED_DIR, save_chunks


def main() -> None:
    doc_dict = json.loads((PARSED_DIR / "attention.json").read_text())
    doc = DoclingDocument.model_validate(doc_dict)

    chunker = HybridChunker()
    chunks = list(chunker.chunk(doc))

    texts = [chunker.contextualize(chunk) for chunk in chunks]
    metadata = [{"headings": chunk.meta.headings} for chunk in chunks]

    save_chunks(
        technique="docling_hybrid",
        texts=texts,
        filename="07_docling_hybrid.json",
        extra_metadata=metadata,
    )


if __name__ == "__main__":
    main()
