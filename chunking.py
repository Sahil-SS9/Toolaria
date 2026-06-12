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
    """Split *text* into chunks of roughly *target_chars* each.

    Short lines are grouped, overlapping by *overlap_lines* so a fact spanning
    a boundary still lands whole in at least one chunk. A line longer than
    target_chars (common for minified JSON returned on a single line) is split
    into overlapping character windows so search still has distinct chunks to
    rank; each window keeps the line it came from.
    """
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    n = len(lines)
    start = 0
    idx = 0
    char_overlap = max(50, target_chars // 8)
    while start < n:
        if len(lines[start]) > target_chars:
            idx = _window_long_line(chunks, idx, start, lines[start],
                                    target_chars, char_overlap)
            start += 1
            continue
        size = 0
        end = start
        while (end < n and len(lines[end]) <= target_chars
               and (size == 0 or size + len(lines[end]) + 1 <= target_chars)):
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


def _window_long_line(chunks: list, idx: int, line_no: int, line: str,
                      target_chars: int, char_overlap: int) -> int:
    """Append overlapping character windows of a single long line."""
    step = max(1, target_chars - char_overlap)
    pos = 0
    while pos < len(line):
        chunks.append(Chunk(idx, line_no, line_no, line[pos:pos + target_chars]))
        idx += 1
        if pos + target_chars >= len(line):
            break
        pos += step
    return idx
