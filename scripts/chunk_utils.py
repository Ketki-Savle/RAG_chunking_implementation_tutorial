"""Shared helpers used by every chunking script: consistent token counting
and a common JSON output schema so outputs/chunks/*.json can be compared
apples-to-apples in 03_compare_chunks.py."""

import json
from pathlib import Path

import tiktoken

REPO_ROOT = Path(__file__).resolve().parent.parent
PARSED_DIR = REPO_ROOT / "outputs" / "parsed"
CHUNKS_DIR = REPO_ROOT / "outputs" / "chunks"

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def save_chunks(technique: str, texts: list[str], filename: str, extra_metadata: list[dict] | None = None) -> None:
    """Write chunks to outputs/chunks/<filename> in the common schema."""
    records = []
    for i, text in enumerate(texts):
        meta = extra_metadata[i] if extra_metadata else {}
        records.append(
            {
                "chunk_id": i,
                "text": text,
                "n_chars": len(text),
                "n_tokens": count_tokens(text),
                "metadata": meta,
            }
        )

    out_path = CHUNKS_DIR / filename
    out_path.write_text(json.dumps({"technique": technique, "chunks": records}, indent=2, ensure_ascii=False))
    print(f"[{technique}] wrote {len(records)} chunks -> {out_path.relative_to(REPO_ROOT)}")
