import os
import pandas as pd
import numpy as np
import faiss
from typing import List, Dict
from ingest.embedder import encode_texts
from config import VECTOR_INDEX_PATH, META_PATH, EMBED_DIM
from utils.file_utils import write_text

def build_faiss(embeds: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeds.shape[1])
    index.add(embeds)
    return index

def save_index(index: faiss.IndexFlatIP):
    faiss.write_index(index, VECTOR_INDEX_PATH)

def load_index() -> faiss.IndexFlatIP:
    if not os.path.exists(VECTOR_INDEX_PATH):
        return None
    return faiss.read_index(VECTOR_INDEX_PATH)

def save_meta(rows: List[Dict]):
    df = pd.DataFrame(rows)
    df.to_csv(META_PATH, index=False)

def load_meta() -> pd.DataFrame:
    if not os.path.exists(META_PATH):
        return pd.DataFrame(columns=["source","chunk_id","text"])
    return pd.read_csv(META_PATH)

def index_corpus(chunks: List[str], sources: List[str]):
    """
    chunks: list of chunk_texts
    sources: aligned list of source labels
    """
    assert len(chunks) == len(sources)
    embeds = encode_texts(chunks)
    index = build_faiss(embeds)
    save_index(index)

    rows = []
    for i, (t, s) in enumerate(zip(chunks, sources)):
        rows.append({
            "source": s, "chunk_id": i, "text": t[:200].replace("\n"," ")
        })
    save_meta(rows)

def add_to_index(new_chunks: List[str], new_sources: List[str]):
    # For simplicity: rebuild (small projects). For large scale, switch to IVF or HNSW and merge.
    meta = load_meta()
    prev_chunks = meta["text"].tolist() if len(meta) else []
    prev_sources = meta["source"].tolist() if len(meta) else []
    index_corpus(prev_chunks + new_chunks, prev_sources + new_sources)
