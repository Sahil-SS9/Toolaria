"""Line-aware chunking shared by the structural index and semantic search.

A chunk is a contiguous run of lines with a known line range, so any slice the
model retrieves (outline section, search hit) maps straight back to a
`rescuer_fetch(mode="range", start=..., count=...)` call.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    index: int
    start_line: int  # 0-based, inclusive
    end_line: int    # 0-based, inclusive
    text: str

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(d["index"], d["start_line"], d["end_line"], d["text"])


def chunk_lines(
    text: str,
    target_chars: int = 1200,
    overlap_lines: int = 2,
) -> list[Chunk]:
    """Split *text* into line-aligned chunks of roughly *target_chars* each.

    Chunks overlap by *overlap_lines* so a fact spanning a boundary still
    lands whole in at least one chunk. A single line longer than target_chars
    becomes its own chunk (it is not split mid-line; range/grep handle the
    within-line case).
    """
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    n = len(lines)
    start = 0
    idx = 0
    while start < n:
        size = 0
        end = start
        while end < n and (size == 0 or size + len(lines[end]) + 1 <= target_chars):
            size += len(lines[end]) + 1
            end += 1
        end = max(end, start + 1)  # always make progress
        chunks.append(Chunk(idx, start, end - 1, "\n".join(lines[start:end])))
        idx += 1
        if end >= n:
            break
        # Advance with overlap. max(start+1, ...) guarantees progress;
        # min(end, ...) guarantees no gap, so every line is covered.
        start = max(start + 1, min(end, end - overlap_lines))
    return chunks
