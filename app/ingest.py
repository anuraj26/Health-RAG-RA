"""Document ingestion: load .txt/.md files and split them into chunks.

Each source file gets a stable `document_id` (doc_1, doc_2, ...) derived
from sorted filename order, and each chunk gets a `chunk_id` (chunk_0,
chunk_1, ...) local to its document. We chunk on paragraph/sentence
boundaries where possible so that citations land on coherent text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from app.config import DATA_DIR, get_settings


@dataclass
class Chunk:
    document_id: str
    chunk_id: str
    source_path: str
    title: str
    text: str
    # global integer position in the corpus, used to map FAISS rows -> chunks
    index: int = field(default=-1)


def _read_title(path: Path, text: str) -> str:
    """Use the first markdown H1 as the title, else the filename."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.replace("_", " ").title()


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Greedy paragraph packer with character-based overlap.

    We accumulate whole paragraphs until adding another would exceed
    `chunk_size`, then start a new chunk that carries `overlap` characters
    of trailing context from the previous one. This keeps chunks readable
    (no mid-sentence cuts) while preserving local continuity.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}"
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail}\n\n{para}".strip() if tail else para

    if current:
        chunks.append(current)

    # Hard-split any paragraph that is itself larger than chunk_size.
    final: List[str] = []
    for ch in chunks:
        if len(ch) <= chunk_size * 1.5:
            final.append(ch)
            continue
        for i in range(0, len(ch), chunk_size - overlap):
            final.append(ch[i : i + chunk_size])
    return [c.strip() for c in final if c.strip()]


def load_chunks(data_dir: Path | None = None) -> List[Chunk]:
    settings = get_settings()
    data_dir = data_dir or DATA_DIR
    files = sorted(
        [p for p in data_dir.iterdir() if p.suffix.lower() in {".txt", ".md"}],
        key=lambda p: p.name,
    )
    if not files:
        raise FileNotFoundError(f"No .txt/.md documents found in {data_dir}")

    chunks: List[Chunk] = []
    global_idx = 0
    for doc_num, path in enumerate(files, start=1):
        text = path.read_text(encoding="utf-8")
        title = _read_title(path, text)
        doc_id = f"doc_{doc_num}"
        for local_num, piece in enumerate(
            _split_text(text, settings.chunk_size, settings.chunk_overlap)
        ):
            chunks.append(
                Chunk(
                    document_id=doc_id,
                    chunk_id=f"chunk_{local_num}",
                    source_path=str(path.relative_to(DATA_DIR.parent)),
                    title=title,
                    text=piece,
                    index=global_idx,
                )
            )
            global_idx += 1
    return chunks
