import os
from dotenv import load_dotenv

load_dotenv()

# LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# Paths
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
VECTOR_DIR = "vectorstore"
VECTOR_INDEX_PATH = os.path.join(VECTOR_DIR, "index.faiss")
META_PATH = os.path.join(VECTOR_DIR, "meta.csv")  # chunk metadata (path, chunk_id, text preview)

# RAG
TOP_K = 6
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# OCR / ASR
TESSERACT_LANG = "eng"
WHISPER_MODEL_SIZE = "small"  # tiny/base/small/medium/large-v3; small is a good compromise
