import numpy as np
import pandas as pd
import faiss
from typing import List, Tuple
from ingest.embedder import encode_texts
from ingest.indexer import load_index, load_meta
from config import TOP_K

def retrieve(query: str, k: int = TOP_K) -> List[Tuple[str, str, float]]:
    index = load_index()
    meta = load_meta()
    if index is None or len(meta) == 0:
        return []
    qv = encode_texts([query])
    scores, idxs = index.search(qv, k)
    idxs = idxs[0]
    scores = scores[0]
    out = []
    for i, sc in zip(idxs, scores):
        if i < 0: continue
        row = meta.iloc[i]
        out.append((row["text"], row["source"], float(sc)))
    return out
