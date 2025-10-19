# app.py
# Multimodal RAG with real OCR/PDF extraction, FAISS retrieval, and Gemini answerer.
# Run: GEMINI_API_KEY=... uvicorn app:app --reload
# UI:  http://localhost:8000/

import os
import io
import re
import faiss
import json
import base64
import asyncio
import google.generativeai as genai

from typing import List, Dict, Tuple, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text as pdf_extract_text
from PIL import Image
import pytesseract
from docx import Document as DocxDocument
from youtube_transcript_api import YouTubeTranscriptApi

# -----------------------------
# Config
# -----------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set. Answers will fall back to a local template.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# embeddings model (fast + decent)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBED = SentenceTransformer(EMBED_MODEL_NAME)

# -----------------------------
# In-memory vector store
# -----------------------------
index: Optional[faiss.IndexFlatIP] = None
vectors_dim: Optional[int] = None
corpus: List[Dict] = []   # [{id, text, source, chunk_id}]
id_to_pos: Dict[int, int] = {}  # faiss row -> corpus idx

def normalize_text(s: str) -> str:
    s = s.replace("\x00", " ").replace("\u0000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def chunk_text(text: str, chunk_size: int = 450, overlap: int = 120) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        piece = " ".join(words[i:i+chunk_size])
        if piece.strip():
            chunks.append(piece.strip())
        i += max(1, chunk_size - overlap)
    return chunks

def upsert_vectors(texts: List[str], source: str) -> None:
    global index, vectors_dim, corpus, id_to_pos
    if not texts:
        return
    embs = EMBED.encode(texts, normalize_embeddings=True)
    embs = embs.astype("float32")

    if index is None:
        vectors_dim = embs.shape[1]
        index = faiss.IndexFlatIP(vectors_dim)

    start_row = index.ntotal
    index.add(embs)

    for j, t in enumerate(texts):
        row = start_row + j
        meta = {"id": len(corpus), "text": t, "source": source, "chunk_id": f"{source}:{j+1}"}
        id_to_pos[row] = len(corpus)
        corpus.append(meta)

def retrieve(query: str, k: int = 3) -> List[Dict]:
    if index is None or index.ntotal == 0:
        return []
    q = EMBED.encode([query], normalize_embeddings=True).astype("float32")
    scores, rows = index.search(q, k)
    out = []
    for s, r in zip(scores[0], rows[0]):
        if r == -1:
            continue
        meta = corpus[id_to_pos[r]]
        out.append({**meta, "score": float(s)})
    return out

def answer_with_gemini(query: str, retrieved_chunks: List[Dict]) -> str:
    context = "\n\n".join(
        [f"[{i+1}] ({c['chunk_id']}) {c['text']}" for i, c in enumerate(retrieved_chunks)]
    )
    prompt = f"""
You are a helpful assistant. Answer the user concisely and ONLY using the facts in the retrieved context.
If you cannot find the answer, say you don't have enough information.

Question: {query}

Retrieved context:
{context}

Return a short answer followed by bullet points if useful. Include inline bracket citations like [1], [2] using the indices above.
"""
    if not GEMINI_API_KEY:
        # simple local template fallback
        facts = " ".join(c["text"] for c in retrieved_chunks)[:800]
        return f"{query}\n\nBased on the retrieved context: {facts[:400]}... [1]"
    model = genai.GenerativeModel(GEMINI_MODEL)
    resp = model.generate_content(prompt)
    return resp.text.strip()

# -----------------------------
# Extractors
# -----------------------------
def extract_pdf(f: bytes) -> str:
    try:
        with io.BytesIO(f) as bio:
            text = pdf_extract_text(bio)
        return normalize_text(text)
    except Exception as e:
        return f"PDF extraction failed: {e}"

def extract_image(f: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(f))
        text = pytesseract.image_to_string(img)
        return normalize_text(text)
    except Exception as e:
        return f"Image OCR failed: {e}"

def extract_txt(f: bytes) -> str:
    try:
        return normalize_text(f.decode("utf-8", errors="ignore"))
    except Exception as e:
        return f"TXT read failed: {e}"

def extract_md(f: bytes) -> str:
    return extract_txt(f)

def extract_docx(f: bytes) -> str:
    try:
        with io.BytesIO(f) as bio:
            doc = DocxDocument(bio)
        text = "\n".join([p.text for p in doc.paragraphs])
        return normalize_text(text)
    except Exception as e:
        return f"DOCX read failed: {e}"

def yt_id_from_url(url: str) -> Optional[str]:
    m = re.search(r"(?:v=|youtu\.be/)([\w\-]{6,})", url)
    return m.group(1) if m else None

def fetch_youtube_transcript(url: str) -> str:
    vid = yt_id_from_url(url)
    if not vid:
        return ""
    try:
        parts = YouTubeTranscriptApi.get_transcript(vid, languages=["en"])
        s = " ".join([p["text"] for p in parts])
        return normalize_text(s)
    except Exception:
        return ""

# -----------------------------
# FastAPI app + CORS
# -----------------------------
app = FastAPI(title="Multimodal RAG (Gemini)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -----------------------------
# UI (single-file HTML)
# -----------------------------
HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Multimodal RAG • Gemini</title>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <style>
    :root { --bg:#0b1320; --card:#121a2a; --muted:#9fb3c8; --text:#e8eef6; --accent:#7dd3fc; }
    *{box-sizing:border-box} body{margin:0;font-family:Inter,system-ui,Segoe UI,Roboto,Arial;background:var(--bg);color:var(--text)}
    .wrap{max-width:1200px;margin:32px auto;padding:0 16px}
    h1{font-size:22px;margin:0 0 16px;display:flex;gap:8px;align-items:center}
    .grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
    .card{background:var(--card);border:1px solid rgba(255,255,255,0.05);border-radius:14px;padding:16px;box-shadow:0 10px 30px rgba(0,0,0,.25)}
    .muted{color:var(--muted)}
    .kpis{display:flex;gap:16px;margin-top:8px}
    .kpi{background:#0e1626;border:1px solid rgba(255,255,255,0.06);padding:10px 12px;border-radius:10px;text-align:center}
    .row{display:flex;gap:8px;align-items:center}
    input[type="file"],input[type="text"]{width:100%;background:#0e1626;border:1px solid rgba(255,255,255,0.08);color:var(--text);padding:10px;border-radius:10px}
    button{background:linear-gradient(90deg,#60a5fa,#22d3ee);color:#00131a;font-weight:700;border:0;border-radius:10px;padding:10px 14px;cursor:pointer}
    .mono{font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;}
    .ctx{white-space:pre-wrap;background:#0c1424;border:1px solid rgba(255,255,255,0.06);padding:12px;border-radius:10px}
    .badge{background:#0e2438;color:#9bd5ff;padding:4px 8px;border-radius:999px;font-size:12px;border:1px solid rgba(125,211,252,.3)}
    .bar{height:8px;background:#0d1b2a;border-radius:999px;overflow:hidden}
    .fill{height:100%;background:linear-gradient(90deg,#22d3ee,#60a5fa)}
    .footer{margin-top:24px;color:#7e93aa;font-size:12px}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>🧠 Multimodal Data Processing System <span class="badge">Gemini + FAISS</span></h1>
    <div class="grid">
      <div class="card">
        <h3>1) Upload Inputs</h3>
        <div class="muted">PDF, DOCX, MD, TXT · Images (png/jpg) · YouTube URL</div>
        <div style="height:8px"></div>
        <input id="files" type="file" multiple/>
        <div style="height:8px"></div>
        <div class="row">
          <input id="yt" type="text" placeholder="YouTube URL (optional)"/>
          <button onclick="ingest()">Ingest</button>
        </div>
        <div class="kpis">
          <div class="kpi"><div id="k_files" class="mono">0</div><div class="muted" style="font-size:12px">Files</div></div>
          <div class="kpi"><div id="k_chunks" class="mono">0</div><div class="muted" style="font-size:12px">Chunks</div></div>
          <div class="kpi"><div id="k_vecs" class="mono">0</div><div class="muted" style="font-size:12px">Vectors</div></div>
        </div>
      </div>

      <div class="card">
        <h3>2) Retrieve</h3>
        <input id="q" type="text" placeholder="e.g., What's the total on the invoice?"/>
        <div style="height:8px"></div>
        <div class="row">
          <input id="k" type="text" value="3" style="max-width:80px" class="mono"/>
          <button onclick="retrieve()">Search</button>
        </div>
        <div style="height:10px"></div>
        <div id="ctx"></div>
      </div>

      <div class="card">
        <h3>3) Ask in Natural Language</h3>
        <div class="row"><button onclick="ask()">Ask (Gemini)</button><span id="lat" class="muted" style="margin-left:8px"></span></div>
        <div style="height:10px"></div>
        <div id="ans" class="ctx"></div>
      </div>
    </div>

    <div class="footer">LLM: Gemini (Free) • DB: FAISS • Embeddings: sentence-transformers • OCR: Tesseract</div>
  </div>

<script>
async function ingest() {
  const fd = new FormData();
  const files = document.getElementById('files').files;
  for (let f of files) fd.append('files', f);
  fd.append('youtube_url', document.getElementById('yt').value || "");
  const t0 = performance.now();
  const r = await fetch('/api/ingest', {method:'POST', body:fd});
  const j = await r.json();
  const t1 = performance.now();
  document.getElementById('k_files').innerText = j.files;
  document.getElementById('k_chunks').innerText = j.chunks;
  document.getElementById('k_vecs').innerText = j.vectors;
  document.getElementById('ctx').innerHTML = '<div class="muted">Ingestion completed in ' + Math.round(t1-t0) + ' ms</div>';
}

async function retrieve() {
  const q = document.getElementById('q').value;
  const k = parseInt(document.getElementById('k').value||'3');
  const r = await fetch('/api/retrieve', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({query:q, k:k})});
  const j = await r.json();
  let html = '';
  j.results.forEach((x, i) => {
    html += `<div style="margin:10px 0"><div class="row" style="justify-content:space-between">
      <div><b>#${i+1}</b> · <span class="mono">${x.chunk_id}</span> · ${x.source}</div>
      <div class="mono">${x.score.toFixed(2)}</div></div>
      <div class="bar"><div class="fill" style="width:${Math.max(2, x.score*100)}%"></div></div>
      <div class="ctx" style="margin-top:6px">${x.text}</div></div>`;
  });
  document.getElementById('ctx').innerHTML = html || '<div class="muted">No results.</div>';
}

async function ask() {
  const q = document.getElementById('q').value;
  const t0 = performance.now();
  const r = await fetch('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({query:q})});
  const j = await r.json();
  const t1 = performance.now();
  document.getElementById('ans').innerText = j.answer || '(no answer)';
  document.getElementById('lat').innerText = '— ' + Math.round(t1-t0) + ' ms';
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

# -----------------------------
# API: ingest
# -----------------------------
@app.post("/api/ingest")
async def api_ingest(
    files: List[UploadFile] = File(default=[]),
    youtube_url: str = Form(default="")
):
    total_files = 0
    total_chunks = 0

    # files
    for f in files:
        data = await f.read()
        name = f.filename or "upload"
        ext = (name.split(".")[-1] or "").lower()

        if ext in ("pdf",):
            text = extract_pdf(data)
        elif ext in ("png", "jpg", "jpeg"):
            text = extract_image(data)
        elif ext in ("txt",):
            text = extract_txt(data)
        elif ext in ("md", "markdown"):
            text = extract_md(data)
        elif ext in ("docx",):
            text = extract_docx(data)
        else:
            text = extract_txt(data)

        chunks = chunk_text(text)
        upsert_vectors(chunks, source=name)
        total_files += 1
        total_chunks += len(chunks)

    # youtube
    if youtube_url.strip():
        yt_text = fetch_youtube_transcript(youtube_url.strip())
        if yt_text:
            chunks = chunk_text(yt_text)
            upsert_vectors(chunks, source="YouTube")
            total_files += 1
            total_chunks += len(chunks)

    return JSONResponse({
        "files": total_files,
        "chunks": total_chunks,
        "vectors": 0 if index is None else int(index.ntotal)
    })

# -----------------------------
# API: retrieve
# -----------------------------
@app.post("/api/retrieve")
async def api_retrieve(payload: Dict):
    q = payload.get("query", "").strip()
    k = int(payload.get("k", 3))
    results = retrieve(q, k=k)
    return JSONResponse({"results": results})

# -----------------------------
# API: ask (retrieve + LLM)
# -----------------------------
@app.post("/api/ask")
async def api_ask(payload: Dict):
    q = payload.get("query", "").strip()
    if not q:
        return JSONResponse({"answer": "Please provide a question."})
    topk = retrieve(q, k=3)
    ans = answer_with_gemini(q, topk)
    return JSONResponse({"answer": ans, "citations": [x["chunk_id"] for x in topk]})
