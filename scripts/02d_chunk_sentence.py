"""Technique 4: Sentence-based chunking.

Uses spaCy to split the text into real sentences, then groups a fixed
number of sentences per chunk. Chunks never cut a sentence in half, unlike
the fixed-size baseline.
"""

import spacy

from chunk_utils import PARSED_DIR, save_chunks

SENTENCES_PER_CHUNK = 5


def main() -> None:
    text = (PARSED_DIR / "attention.txt").read_text()

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks = []
    for i in range(0, len(sentences), SENTENCES_PER_CHUNK):
        group = sentences[i : i + SENTENCES_PER_CHUNK]
        chunks.append(" ".join(group))

    save_chunks(
        technique="sentence_based",
        texts=chunks,
        filename="04_sentence_based.json",
        extra_metadata=[{"sentences_per_chunk": SENTENCES_PER_CHUNK} for _ in chunks],
    )


if __name__ == "__main__":
    main()
