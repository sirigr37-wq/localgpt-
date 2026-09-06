"""
Embeddings Module for LocalGPT (Phase 2 - Step 4: Chunking & Embeddings).
Generates vector representations using sentence-transformers/all-MiniLM-L6-v2.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Module-level cached instance
_CACHED_MODEL = None


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """
    Load SentenceTransformer embedding model. Reuses cached instance if available.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _CACHED_MODEL = SentenceTransformer(model_name)
    return _CACHED_MODEL


def embed_text(text: str, model=None) -> np.ndarray:
    """
    Compute 384-dimensional dense vector embedding for a single text string.
    Returns 1D numpy array of shape (384,).
    """
    if not text or not text.strip():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    if model is None:
        model = load_embedding_model()

    embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return np.array(embedding, dtype=np.float32)


def embed_chunks(
    chunks: List[Dict[str, Any]],
    model=None,
    batch_size: int = 32
) -> List[Dict[str, Any]]:
    """
    Compute dense vector embeddings for a list of document chunks.
    Attaches 'embedding' field (np.ndarray of shape (384,)) to each chunk.
    """
    if not chunks:
        return []

    if model is None:
        model = load_embedding_model()

    texts = [c.get("text", "") for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    for i, emb in enumerate(embeddings):
        chunks[i]["embedding"] = np.array(emb, dtype=np.float32)

    return chunks


def get_embeddings_matrix(chunks: List[Dict[str, Any]]) -> np.ndarray:
    """
    Extract a 2D numpy matrix of shape (N, 384) from a list of embedded chunks.
    Ready for indexing into FAISS in subsequent RAG steps.
    """
    if not chunks:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    embs = []
    for c in chunks:
        if "embedding" in c and c["embedding"] is not None:
            embs.append(c["embedding"])
        else:
            embs.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))

    return np.vstack(embs).astype(np.float32)
