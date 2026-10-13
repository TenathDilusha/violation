"""Find model-quoted excerpts in the article text (character offsets).

Matching ignores differences in whitespace and zero-width characters (Sinhala
text often contains ZWJ), so a quote that differs only in those still counts.
"""

from __future__ import annotations

import re

ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}
QUOTE_CHARS = "\"'“”‘’"


def normalize(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace and drop zero-width characters; also return the index map."""
    out, index = [], []
    prev_space = False
    for i, ch in enumerate(text):
        if ch in ZERO_WIDTH:
            continue
        if ch.isspace():
            if prev_space:
                continue
            ch, prev_space = " ", True
        else:
            prev_space = False
        out.append(ch)
        index.append(i)
    return "".join(out), index


def locate(content: str, excerpt: str, min_len: int = 6) -> list[tuple[int, int]]:
    """(start, end) offsets in content for the excerpt, [] if it is not in the article.

    An excerpt with "..." is matched piece by piece.
    """
    norm, index = normalize(content)
    found = []
    for piece in (excerpt or "").replace("…", "...").split("..."):
        needle = normalize(piece.strip().strip(QUOTE_CHARS))[0].strip().rstrip(".,;:")
        if len(needle) < min_len:
            continue
        start = norm.find(needle)
        if start >= 0:
            found.append((index[start], index[start + len(needle) - 1] + 1))
    return found


def located(spans: list[dict]) -> list[dict]:
    """Spans found in the article content or heading."""
    return [s for s in spans or [] if s.get("offsets") or s.get("heading_offsets")]


def sentences(content: str) -> list[tuple[int, int]]:
    """(start, end) offsets of the sentences / lines of content with at least 10 characters."""
    out = []
    for m in re.finditer(r"[^\n]+", content):
        line_start = m.start()
        for s in re.finditer(r".+?(?:[.!?](?=\s)|$)", m.group(0)):
            text = s.group(0)
            if len(text.strip()) >= 10:
                lead = len(text) - len(text.lstrip())
                out.append((line_start + s.start() + lead, line_start + s.start() + len(text.rstrip())))
    return out
