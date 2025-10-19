import re
from typing import List
from config import CHUNK_SIZE, CHUNK_OVERLAP

def clean_text(s: str) -> str:
    s = s.replace("\x00", " ").replace("\u200b", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(s: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    s = clean_text(s)
    if not s:
        return []
    chunks, start = [], 0
    while start < len(s):
        end = min(len(s), start + size)
        chunks.append(s[start:end])
        if end == len(s): break
        start = end - overlap
        if start < 0: start = 0
    return chunks
