"""
RAG Service Implementation for Phase 3.
Coordinates document parsing, chunking, vector embedding, isolated FAISS indexing,
semantic retrieval, and prompt augmentation with source citations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
import asyncio

from app.services.document_loader import load_document
from app.services.rag.chunking import chunk_document
from app.services.rag.embeddings import embed_text
from app.services.rag.vector_store import (
    add_document_vectors,
    remove_document_vectors,
    search_user_vectors,
)

logger = logging.getLogger(__name__)


class BaseRAGService(ABC):
    """Abstract interface defining RAG document ingestion and retrieval capabilities."""

    @abstractmethod
    async def ingest_document(
        self,
        document_id: str,
        filepath: str,
        user_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Process document into chunks, embed, and index into vector store."""
        pass

    @abstractmethod
    async def retrieve_relevant_chunks(
        self, query: str, user_id: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant document chunks for the given user."""
        pass

    @abstractmethod
    def build_augmented_prompt(self, user_query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved context and user query into grounded prompt."""
        pass

    @abstractmethod
    async def delete_document_vectors(self, document_id: str, user_id: str) -> int:
        """Remove document vectors from user's index."""
        pass


class RAGService(BaseRAGService):
    """
    Production Phase 3 RAG Service.
    Enforces per-user FAISS vector index isolation, supports multi-format documents,
    and grounds hosted LLM completions with source/page citations.
    """

    async def ingest_document(
        self,
        document_id: str,
        filepath: str,
        user_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Parse document, chunk text with page mapping, generate embeddings,
        and index into the user's isolated FAISS index.
        """
        try:
            # 1. Parse document text and page metadata
            extracted = await asyncio.to_thread(load_document, filepath)
            if extracted.get("status") == "error":
                return {
                    "document_id": document_id,
                    "status": "error",
                    "chunk_count": 0,
                    "chunks_count": 0,
                    "error_message": extracted.get("error_message", "Document extraction failed."),
                }

            text = extracted.get("text", "").strip()
            if not text:
                return {
                    "document_id": document_id,
                    "status": "ready",
                    "chunk_count": 0,
                    "chunks_count": 0,
                    "message": "Document contains no extractable text.",
                }

            # 2. Chunk document with user ownership
            chunks = chunk_document(
                document=extracted,
                user_id=user_id,
                document_id=document_id,
            )

            if not chunks:
                return {
                    "document_id": document_id,
                    "status": "ready",
                    "chunk_count": 0,
                    "chunks_count": 0,
                    "message": "No chunks generated from document text.",
                }

            # 3. Add chunks and embeddings to user-isolated FAISS store
            await asyncio.to_thread(
                add_document_vectors,
                user_id=user_id,
                document_id=document_id,
                chunks=chunks,
            )

            chunk_count = len(chunks)
            logger.info(
                f"Successfully ingested document {document_id} ({chunk_count} chunks) for user {user_id}"
            )
            return {
                "document_id": document_id,
                "status": "ready",
                "chunk_count": chunk_count,
                "chunks_count": chunk_count,
                "message": f"Successfully indexed {chunk_count} chunks.",
            }

        except Exception as e:
            logger.error(f"Error ingesting document {document_id} for user {user_id}: {e}", exc_info=True)
            return {
                "document_id": document_id,
                "status": "error",
                "chunk_count": 0,
                "chunks_count": 0,
                "error_message": str(e),
            }

    async def delete_document_vectors(self, document_id: str, user_id: str) -> int:
        """Clean up all vectors associated with document_id in the user's isolated store."""
        try:
            remaining = await asyncio.to_thread(
                remove_document_vectors,
                user_id=user_id,
                document_id=document_id,
            )
            return remaining
        except Exception as e:
            logger.error(f"Error deleting vectors for doc {document_id}, user {user_id}: {e}")
            return 0

    async def retrieve_relevant_chunks(
        self, query: str, user_id: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Embed user query and retrieve top-k semantically relevant chunks
        strictly from the authenticated user's isolated FAISS index.
        """
        if not query or not query.strip() or not user_id:
            return []

        # Fast path: check if user has any documents/chunks indexed before touching embedding model
        from app.services.rag.vector_store import has_user_vectors
        if not has_user_vectors(user_id):
            return []

        try:
            # 1. Embed query
            query_emb = await asyncio.to_thread(embed_text, query.strip())

            # 2. Search user's isolated index
            chunks = await asyncio.to_thread(
                search_user_vectors,
                user_id=user_id,
                query_embedding=query_emb,
                top_k=top_k,
                score_threshold=0.0,
            )
            return chunks
        except Exception as e:
            logger.error(f"Error during vector retrieval for user {user_id}: {e}", exc_info=True)
            return []

    def build_augmented_prompt(self, user_query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Synthesize retrieved document chunks into a grounded, structured prompt
        with explicit instructions to cite filenames and page numbers.
        """
        if not retrieved_chunks:
            return user_query

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            filename = chunk.get("filename", "unknown_document")
            p_start = chunk.get("page_start", 1)
            p_end = chunk.get("page_end", 1)
            page_info = f"Page {p_start}" if p_start == p_end else f"Pages {p_start}-{p_end}"
            score = chunk.get("score", 0.0)

            context_blocks.append(
                f"[Source {i}] {filename} ({page_info}, Relevance: {score:.2f}):\n"
                f"{chunk.get('excerpt', chunk.get('text', '')).strip()}"
            )

        context_str = "\n\n".join(context_blocks)
        return (
            f"DOCUMENT CONTEXT:\n"
            f"----------------------------------------\n"
            f"{context_str}\n"
            f"----------------------------------------\n\n"
            f"USER QUESTION:\n"
            f"{user_query}\n\n"
            f"INSTRUCTIONS:\n"
            f"Answer the USER QUESTION accurately and concisely using ONLY the facts provided in the DOCUMENT CONTEXT above.\n"
            f"- Explicitly reference source filenames and page numbers in your response (e.g. [sample.pdf, Page 1]).\n"
            f"- If the answer cannot be found in or deduced from the provided context, state clearly: "
            f"\"I could not find information about this in the uploaded documents.\" Do not invent facts."
        )


# Module-level singleton
_RAG_SERVICE_INSTANCE = None


def get_rag_service() -> BaseRAGService:
    """Factory returning RAG service singleton."""
    global _RAG_SERVICE_INSTANCE
    if _RAG_SERVICE_INSTANCE is None:
        _RAG_SERVICE_INSTANCE = RAGService()
    return _RAG_SERVICE_INSTANCE
