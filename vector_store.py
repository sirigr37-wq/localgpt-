"""
FAISS Vector Store Module for LocalGPT (Phase 2 - Step 5: FAISS Vector Store).
Provides high-performance vector indexing, cosine similarity search, and persistence.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss

from embeddings import EMBEDDING_DIM, get_embeddings_matrix


DEFAULT_VECTOR_STORE_DIR = os.path.join("data", "vector_store")
INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors so that Inner Product (IndexFlatIP) equals Cosine Similarity.
    """
    vecs = np.array(vectors, dtype=np.float32)
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    # Avoid in-place modification of non-contiguous arrays
    vecs_copy = np.ascontiguousarray(vecs, dtype=np.float32)
    faiss.normalize_L2(vecs_copy)
    return vecs_copy


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build and populate a FAISS IndexFlatIP (cosine similarity) from normalized embeddings.
    """
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Cannot build FAISS index with empty embeddings.")

    norm_embs = normalize_vectors(embeddings)
    dim = norm_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(norm_embs)
    return index


def add_embeddings(index: faiss.Index, embeddings: np.ndarray) -> None:
    """
    Normalize and add new vector embeddings to an existing FAISS index.
    """
    if embeddings is None or len(embeddings) == 0:
        return
    norm_embs = normalize_vectors(embeddings)
    index.add(norm_embs)


def search_faiss(
    index: faiss.Index,
    query_embedding: np.ndarray,
    top_k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search a FAISS index for the top-k nearest neighbors of a query vector.
    Returns (distances/scores, indices).
    """
    if index is None or index.ntotal == 0:
        return np.array([[]]), np.array([[]])

    norm_query = normalize_vectors(query_embedding)
    k = min(top_k, index.ntotal)
    scores, indices = index.search(norm_query, k)
    return scores, indices


def save_faiss_index(index: faiss.Index, path: str) -> None:
    """
    Save FAISS index binary to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)


def load_faiss_index(path: str) -> Optional[faiss.Index]:
    """
    Load a FAISS index binary from disk.
    """
    if not os.path.exists(path):
        return None
    return faiss.read_index(path)


def create_vector_store(
    chunks: List[Dict[str, Any]],
    embeddings: Optional[np.ndarray] = None
) -> Optional[Dict[str, Any]]:
    """
    Create a complete vector store dictionary combining FAISS index and chunk metadata.
    """
    if not chunks:
        return None

    if embeddings is None:
        embeddings = get_embeddings_matrix(chunks)

    if len(embeddings) == 0 or embeddings.shape[0] != len(chunks):
        raise ValueError(f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)}).")

    index = build_faiss_index(embeddings)
    dim = int(embeddings.shape[1])

    return {
        "index": index,
        "chunks": chunks,
        "dimension": dim,
        "num_chunks": len(chunks),
        "index_type": "IndexFlatIP (Cosine Similarity)"
    }


def search_vector_store(
    vector_store: Dict[str, Any],
    query_embedding: np.ndarray,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search the vector store and return structured results mapped to chunk metadata.
    """
    if not vector_store or "index" not in vector_store or vector_store["index"] is None:
        return []

    index = vector_store["index"]
    chunks = vector_store.get("chunks", [])

    if index.ntotal == 0 or not chunks:
        return []

    scores, indices = search_faiss(index, query_embedding, top_k=top_k)

    results = []
    if len(scores) > 0 and len(indices) > 0:
        row_scores = scores[0]
        row_indices = indices[0]

        for rank, (score, idx) in enumerate(zip(row_scores, row_indices), start=1):
            if idx >= 0 and idx < len(chunks):
                chunk = chunks[idx]
                results.append({
                    "rank": rank,
                    "score": float(score),
                    "chunk_id": chunk.get("chunk_id", idx),
                    "document_id": chunk.get("document_id", ""),
                    "filename": chunk.get("filename", "unknown"),
                    "filepath": chunk.get("filepath", ""),
                    "file_type": chunk.get("file_type", "TXT"),
                    "page_start": chunk.get("page_start", 1),
                    "page_end": chunk.get("page_end", 1),
                    "text": chunk.get("text", ""),
                    "char_count": chunk.get("char_count", len(chunk.get("text", ""))),
                    "word_count": chunk.get("word_count", len(chunk.get("text", "").split())),
                })

    return results


def save_vector_store(
    vector_store: Dict[str, Any],
    directory: str = DEFAULT_VECTOR_STORE_DIR
) -> bool:
    """
    Persist FAISS index and chunk metadata to a folder.
    """
    if not vector_store or "index" not in vector_store or "chunks" not in vector_store:
        return False

    os.makedirs(directory, exist_ok=True)
    index_path = os.path.join(directory, INDEX_FILENAME)
    chunks_path = os.path.join(directory, CHUNKS_FILENAME)

    # 1. Save FAISS binary
    save_faiss_index(vector_store["index"], index_path)

    # 2. Clean chunks for JSON (remove non-serializable numpy arrays)
    serializable_chunks = []
    for c in vector_store["chunks"]:
        c_clean = dict(c)
        if "embedding" in c_clean:
            del c_clean["embedding"]
        serializable_chunks.append(c_clean)

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(serializable_chunks, f, indent=2)

    return True


def load_vector_store(
    directory: str = DEFAULT_VECTOR_STORE_DIR
) -> Optional[Dict[str, Any]]:
    """
    Load persisted FAISS index and chunk metadata from disk.
    """
    index_path = os.path.join(directory, INDEX_FILENAME)
    chunks_path = os.path.join(directory, CHUNKS_FILENAME)

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return None

    try:
        index = load_faiss_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if index is None or not chunks:
            return None

        return {
            "index": index,
            "chunks": chunks,
            "dimension": index.d,
            "num_chunks": len(chunks),
            "index_type": "IndexFlatIP (Cosine Similarity)"
        }
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None
