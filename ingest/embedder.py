from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

# Lazy global to avoid reloading in Streamlit
_model = None

def get_encoder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model

def encode_texts(texts):
    enc = get_encoder()
    return enc.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
