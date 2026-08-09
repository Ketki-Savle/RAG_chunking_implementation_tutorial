"""Step 4: Compare all 7 chunking techniques side by side.

Reads every outputs/chunks/*.json file produced by the 02* scripts and
writes a single markdown table with chunk-count and char/token stats --
the shareable artifact for the LinkedIn write-up.
"""

import json
import statistics as stats

from chunk_utils import CHUNKS_DIR, REPO_ROOT

FILES = [
    "01_fixed_size.json",
    "02_recursive_char.json",
    "03_token_based.json",
    "04_sentence_based.json",
    "05_semantic.json",
    "06_docling_hierarchical.json",
    "07_docling_hybrid.json",
]


def summarize(chunks: list[dict]) -> dict:
    chars = [c["n_chars"] for c in chunks]
    tokens = [c["n_tokens"] for c in chunks]
    return {
        "n_chunks": len(chunks),
        "avg_chars": round(stats.mean(chars)),
        "min_chars": min(chars),
        "max_chars": max(chars),
        "avg_tokens": round(stats.mean(tokens)),
        "min_tokens": min(tokens),
        "max_tokens": max(tokens),
    }


def main() -> None:
    rows = []
    for filename in FILES:
        path = CHUNKS_DIR / filename
        data = json.loads(path.read_text())
        summary = summarize(data["chunks"])
        rows.append((data["technique"], summary))

    header = "| Technique | # Chunks | Avg chars | Min chars | Max chars | Avg tokens | Min tokens | Max tokens |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [
        "# Chunking Technique Comparison — Attention Is All You Need",
        "",
        "Same paper, parsed once with docling, chunked 7 different ways.",
        "",
        header,
        sep,
    ]
    for technique, s in rows:
        lines.append(
            f"| {technique} | {s['n_chunks']} | {s['avg_chars']} | {s['min_chars']} | {s['max_chars']} "
            f"| {s['avg_tokens']} | {s['min_tokens']} | {s['max_tokens']} |"
        )

    out_path = REPO_ROOT / "outputs" / "comparison_summary.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote comparison -> {out_path.relative_to(REPO_ROOT)}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
