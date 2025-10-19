import os
import re
from pathlib import Path

SAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]")

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def safe_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    return SAFE_CHARS.sub("_", name)

def ext(path: str) -> str:
    return Path(path).suffix.lower()

def write_text(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
