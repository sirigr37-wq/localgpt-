"""
FAISS Vector Store Service for Phase 3 RAG Pipeline.
Enforces strict multi-tenant user isolation by partitioning vector indices and metadata per user.
Uses FAISS IndexFlatIP with L2 normalization to achieve exact Cosine Similarity search.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss

from app.core.config import settings
from app.services.rag.embeddings import (
    EMBEDDING_DIM,
    embed_text,
    embed_chunks,
    get_embeddings_matrix,
    load_embedding_model,
)

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"
MAX_EXCERPT_WORDS = 180


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors so that FAISS IndexFlatIP computes exact Cosine Similarity.
    """
    vecs = np.array(vectors, dtype=np.float32)
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    vecs_copy = np.ascontiguousarray(vecs, dtype=np.float32)
    faiss.normalize_L2(vecs_copy)
    return vecs_copy


def get_user_vector_dir(user_id: str) -> str:
    """
    Get or create the isolated vector storage directory for a specific user.
    Enforces path traversal safety against settings.FAISS_INDEX_DIR.
    """
    base_dir = os.path.abspath(settings.FAISS_INDEX_DIR)
    os.makedirs(base_dir, exist_ok=True)
    safe_user = os.path.basename(user_id)
    user_dir = os.path.abspath(os.path.join(base_dir, safe_user))

    if not user_dir.startswith(base_dir):
        raise ValueError(f"Invalid user_id path traversal attempt: {user_id}")

    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def has_user_vectors(user_id: str) -> bool:
    """
    Fast check if the user has an initialized FAISS vector store and chunks.
    Avoids expensive embedding model initialization for queries without indexed documents.
    """
    base_dir = os.path.abspath(settings.FAISS_INDEX_DIR)
    safe_user = os.path.basename(user_id)
    user_dir = os.path.abspath(os.path.join(base_dir, safe_user))
    if not os.path.exists(user_dir):
        return False
    index_path = os.path.join(user_dir, INDEX_FILENAME)
    chunks_path = os.path.join(user_dir, CHUNKS_FILENAME)
    return os.path.exists(index_path) and os.path.exists(chunks_path)


def load_user_vector_store(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a user's FAISS index and chunk metadata from their isolated directory.
    Returns None if the user does not have an existing vector store.
    """
    user_dir = get_user_vector_dir(user_id)
    index_path = os.path.join(user_dir, INDEX_FILENAME)
    chunks_path = os.path.join(user_dir, CHUNKS_FILENAME)

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return None

    try:
        index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if index is None or not chunks:
            return None

        return {
            "index": index,
            "chunks": chunks,
            "dimension": index.d,
            "num_chunks": len(chunks),
        }
    except Exception as e:
        logger.warning(f"Failed to load vector store for user {user_id}: {e}")
        return None


def save_user_vector_store(user_id: str, index: faiss.Index, chunks: List[Dict[str, Any]]) -> bool:
    """
    Persist user's FAISS index binary and clean JSON metadata to their isolated directory.
    """
    user_dir = get_user_vector_dir(user_id)
    index_path = os.path.join(user_dir, INDEX_FILENAME)
    chunks_path = os.path.join(user_dir, CHUNKS_FILENAME)

    try:
        # 1. Write FAISS binary
        faiss.write_index(index, index_path)

        # 2. Serialize chunks without non-JSON serializable numpy arrays
        clean_chunks = []
        for c in chunks:
            c_dict = dict(c)
            if "embedding" in c_dict:
                del c_dict["embedding"]
            clean_chunks.append(c_dict)

        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(clean_chunks, f, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        logger.error(f"Error saving vector store for user {user_id}: {e}")
        return False


def add_document_vectors(
    user_id: str,
    document_id: str,
    chunks: List[Dict[str, Any]],
) -> int:
    """
    Embed and add document chunks to the user's isolated FAISS vector store.
    Returns the total number of chunks currently indexed for this user.
    """
    if not chunks:
        return 0

    # Ensure all chunks have embeddings
    unembedded = [c for c in chunks if "embedding" not in c or c["embedding"] is None]
    if unembedded:
        embed_chunks(chunks)

    new_embeddings = get_embeddings_matrix(chunks)
    if len(new_embeddings) == 0:
        return 0

    norm_new = normalize_vectors(new_embeddings)
    existing_store = load_user_vector_store(user_id)

    if existing_store and existing_store.get("index") is not None:
        index = existing_store["index"]
        all_chunks = existing_store.get("chunks", [])

        # Filter out any pre-existing chunks for the same document_id (avoid duplicate indexing)
        all_chunks = [c for c in all_chunks if c.get("document_id") != document_id]
        
        # If we replaced existing chunks, re-index all to maintain exact index-to-chunk alignment
        all_chunks.extend(chunks)
        # Re-embed all chunks to preserve 1:1 row index alignment
        all_embedded = embed_chunks(all_chunks)
        all_matrix = get_embeddings_matrix(all_embedded)
        norm_all = normalize_vectors(all_matrix)

        new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        new_index.add(norm_all)
        save_user_vector_store(user_id, new_index, all_chunks)
        return len(all_chunks)
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(norm_new)
        save_user_vector_store(user_id, index, chunks)
        return len(chunks)


def remove_document_vectors(user_id: str, document_id: str) -> int:
    """
    Remove all chunks belonging to a document from the user's isolated vector store.
    Rebuilds the user's FAISS index to ensure no orphaned vectors remain.
    Returns the remaining number of indexed chunks for this user.
    """
    existing_store = load_user_vector_store(user_id)
    if not existing_store or not existing_store.get("chunks"):
        return 0

    all_chunks = existing_store["chunks"]
    remaining_chunks = [c for c in all_chunks if c.get("document_id") != document_id]

    user_dir = get_user_vector_dir(user_id)
    index_path = os.path.join(user_dir, INDEX_FILENAME)
    chunks_path = os.path.join(user_dir, CHUNKS_FILENAME)

    if not remaining_chunks:
        # No chunks left for this user -> remove index files
        if os.path.exists(index_path):
            try:
                os.remove(index_path)
            except OSError:
                pass
        if os.path.exists(chunks_path):
            try:
                os.remove(chunks_path)
            except OSError:
                pass
        return 0

    # Rebuild FAISS index from remaining chunks
    all_embedded = embed_chunks(remaining_chunks)
    all_matrix = get_embeddings_matrix(all_embedded)
    norm_all = normalize_vectors(all_matrix)

    new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    new_index.add(norm_all)
    save_user_vector_store(user_id, new_index, remaining_chunks)
    return len(remaining_chunks)


def search_user_vectors(
    user_id: str,
    query_embedding: np.ndarray,
    top_k: int = 3,
    score_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Execute semantic cosine similarity search strictly within the specified user's isolated vector store.
    Never searches or returns vectors belonging to another user.
    """
    store = load_user_vector_store(user_id)
    if not store or "index" not in store or store["index"] is None:
        return []

    index: faiss.Index = store["index"]
    chunks: List[Dict[str, Any]] = store.get("chunks", [])

    if index.ntotal == 0 or not chunks:
        return []

    norm_query = normalize_vectors(query_embedding)
    k = min(top_k, index.ntotal)
    scores, indices = index.search(norm_query, k)

    results = []
    if len(scores) > 0 and len(indices) > 0:
        row_scores = scores[0]
        row_indices = indices[0]

        for rank, (score, idx) in enumerate(zip(row_scores, row_indices), start=1):
            if 0 <= idx < len(chunks):
                chunk = chunks[idx]
                sim_score = float(score)

                if sim_score >= score_threshold:
                    text_content = chunk.get("text", "").strip()
                    words = text_content.split()
                    if len(words) > MAX_EXCERPT_WORDS:
                        excerpt = " ".join(words[:MAX_EXCERPT_WORDS]) + " [...]"
                    else:
                        excerpt = text_content

                    p_start = chunk.get("page_start", 1)
                    p_end = chunk.get("page_end", 1)

                    results.append({
                        "rank": rank,
                        "score": round(sim_score, 4),
                        "chunk_id": chunk.get("chunk_id", idx),
                        "document_id": chunk.get("document_id", ""),
                        "filename": chunk.get("filename", "unknown"),
                        "filepath": chunk.get("filepath", ""),
                        "file_type": chunk.get("file_type", "TXT"),
                        "page_start": p_start,
                        "page_end": p_end,
                        "excerpt": excerpt,
                        "text": text_content,
                    })

    return results
