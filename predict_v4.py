from pathlib import Path
import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from model_features import clean_text, handcrafted

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "v4"

_tfidf = None
_scaler = None
_quality = None
_stance = None
_meta = None
_embedder = None

def load_models():
    global _tfidf, _scaler, _quality, _stance, _meta, _embedder
    if _quality is not None:
        return
    _tfidf = joblib.load(MODEL_DIR / "v4_tfidf.pkl")
    _scaler = joblib.load(MODEL_DIR / "v4_feature_scaler.pkl")
    _quality = joblib.load(MODEL_DIR / "v4_quality_model.pkl")
    _stance = joblib.load(MODEL_DIR / "v4_stance_model.pkl")
    meta_path = MODEL_DIR / "v4_metadata.pkl"
    _meta = joblib.load(meta_path) if meta_path.exists() else {}
    _embedder = SentenceTransformer(_meta.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))

def _one(topic, argument):
    load_models()
    combined = clean_text(f"{topic} [ARGUMENT] {argument}")
    tf = _tfidf.transform([combined])
    hc = _scaler.transform(handcrafted([argument]))
    emb = _embedder.encode([combined], normalize_embeddings=True, show_progress_bar=False)
    X = hstack([tf, csr_matrix(hc), csr_matrix(emb)]).tocsr()
    raw = float(_quality.predict(X)[0])
    score = float(np.clip(raw, 0, 1) * 100)
    stance = int(_stance.predict(_tfidf.transform([combined]))[0])
    return {
        "quality": round(score, 1),
        "stance": str(stance),
        "stance_label": "FOR" if stance == 1 else "AGAINST" if stance == -1 else "NEUTRAL",
        "model": "MindProof V4",
        "embedding_model": _meta.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
    }

def analyze_two_arguments(topic, arg1, arg2):
    return _one(topic, arg1), _one(topic, arg2)
