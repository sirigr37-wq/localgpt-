"""
RAG Retrieval & Context Injection Module for LocalGPT (Phase 2 - Step 6).
Connects the FAISS vector store to the Qwen generation pipeline with relevance filtering,
structured context injection, and source attribution.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from embeddings import embed_text, load_embedding_model
from vector_store import search_vector_store


DEFAULT_TOP_K = 1
DEFAULT_SCORE_THRESHOLD = 0.0  # Default to top-k nearest neighbors (0.0 to 1.0)
MAX_CHUNK_EXCERPT_WORDS = 180  # Compact context excerpt limit (~150-200 words) to ensure fast CPU TTFT


def should_use_rag(
    vector_store: Optional[Dict[str, Any]],
    enable_rag: bool = True
) -> bool:
    """
    Check if RAG retrieval is enabled and the vector store is populated.
    """
    if not enable_rag:
        return False
    if not vector_store or "index" not in vector_store or vector_store["index"] is None:
        return False
    return vector_store["index"].ntotal > 0


def retrieve_relevant_chunks(
    query: str,
    vector_store: Dict[str, Any],
    embedding_model=None,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Embed the user query, search the FAISS vector store, and filter chunks by similarity score threshold.
    """
    if not query or not query.strip():
        return []

    if not should_use_rag(vector_store, enable_rag=True):
        return []

    if embedding_model is None:
        embedding_model = load_embedding_model()

    # 1. Embed query into 384-dim dense vector
    q_emb = embed_text(query.strip(), model=embedding_model)

    # 2. Search FAISS index for top-k candidates
    candidates = search_vector_store(vector_store, q_emb, top_k=top_k)

    # 3. Filter by similarity threshold
    filtered_chunks = [c for c in candidates if c.get("score", 0.0) >= score_threshold]

    # Fallback to top candidates if score_threshold was non-zero and filtered everything
    if not filtered_chunks and candidates:
        filtered_chunks = candidates[:min(top_k, len(candidates))]

    return filtered_chunks


def build_rag_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format a list of retrieved chunks into a clean, structured context string for LLM injection.
    Limits each chunk excerpt to ~180 words to keep prompt prefill fast on CPU.
    """
    if not retrieved_chunks:
        return ""

    context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        filename = chunk.get("filename", "unknown_document")
        p_start = chunk.get("page_start", 1)
        p_end = chunk.get("page_end", 1)
        page_str = f"Page {p_start}" if p_start == p_end else f"Pages {p_start}–{p_end}"
        score = chunk.get("score", 0.0)
        
        raw_content = chunk.get("text", "").strip()
        words = raw_content.split()
        if len(words) > MAX_CHUNK_EXCERPT_WORDS:
            content = " ".join(words[:MAX_CHUNK_EXCERPT_WORDS]) + " [...]"
        else:
            content = raw_content

        block = (
            f"[Source {idx}]\n"
            f"Filename: {filename}\n"
            f"{page_str} (Relevance: {score:.2f})\n"
            f"Content:\n{content}"
        )
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def build_rag_prompt(user_query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Combine the retrieved document context and the user query into a grounded prompt for Qwen.
    """
    if not retrieved_chunks:
        return user_query

    context_str = build_rag_context(retrieved_chunks)

    prompt = (
        f"DOCUMENT CONTEXT:\n"
        f"----------------------------------------\n"
        f"{context_str}\n"
        f"----------------------------------------\n\n"
        f"USER QUESTION:\n"
        f"{user_query}\n\n"
        f"INSTRUCTIONS:\n"
        f"Answer the USER QUESTION accurately and concisely using ONLY the facts provided in the DOCUMENT CONTEXT above.\n"
        f"- Reference source filenames and page numbers where appropriate.\n"
        f"- If the answer cannot be found in or deduced from the provided context, state clearly: "
        f"\"I could not find information about this in the uploaded documents.\" Do not invent facts."
    )
    return prompt


def retrieve_and_build_context(
    query: str,
    vector_store: Optional[Dict[str, Any]],
    embedding_model=None,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Convenience function: runs retrieval and returns (augmented_prompt, retrieved_chunks).
    If no relevant chunks are found or RAG is disabled, returns (original_query, []).
    """
    if not should_use_rag(vector_store, enable_rag=True):
        return query, []

    chunks = retrieve_relevant_chunks(
        query=query,
        vector_store=vector_store,
        embedding_model=embedding_model,
        top_k=top_k,
        score_threshold=score_threshold
    )

    if not chunks:
        return query, []

    rag_prompt = build_rag_prompt(query, chunks)
    return rag_prompt, chunks
