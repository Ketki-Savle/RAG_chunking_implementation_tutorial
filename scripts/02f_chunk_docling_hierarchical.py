"""Technique 6: Docling hierarchical chunking.

Runs directly on the parsed DoclingDocument (not raw text), so chunk
boundaries follow the paper's actual structure -- section headings, list
items, table cells -- as detected during PDF parsing. It has no token-budget
awareness, so a long section still becomes one (possibly huge) chunk.
"""

import json

from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.types.doc.document import DoclingDocument

from chunk_utils import PARSED_DIR, save_chunks


def main() -> None:
    doc_dict = json.loads((PARSED_DIR / "attention.json").read_text())
    doc = DoclingDocument.model_validate(doc_dict)

    chunker = HierarchicalChunker()
    chunks = list(chunker.chunk(doc))

    texts = [chunker.contextualize(chunk) for chunk in chunks]
    metadata = [{"headings": chunk.meta.headings} for chunk in chunks]

    save_chunks(
        technique="docling_hierarchical",
        texts=texts,
        filename="06_docling_hierarchical.json",
        extra_metadata=metadata,
    )


if __name__ == "__main__":
    main()
