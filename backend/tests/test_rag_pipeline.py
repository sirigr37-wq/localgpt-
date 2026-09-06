"""
Comprehensive RAG Pipeline Test Suite for Phase 3.
Verifies:
1. Document chunking across multiple formats with page-span preservation
2. 384-dimensional dense vector embedding generation
3. FAISS vector storage with exact cosine similarity search
4. Strict multi-tenant user isolation (User 1 vs User 2 data separation)
5. Document deletion and vector cleanup synchronization
6. Prompt grounding and source citation formatting
7. End-to-end chat streaming with RAG context injection and PostgreSQL persistence
"""

import os
import json
import pytest
import numpy as np
from httpx import AsyncClient, ASGITransport
from sqlalchemy import select

from app.main import app
from app.core.config import settings
from app.models.document import Document
from app.models.message import Message
from app.services.rag.chunking import chunk_document, chunk_text
from app.services.rag.embeddings import embed_text, embed_chunks, EMBEDDING_DIM
from app.services.rag.vector_store import (
    add_document_vectors,
    remove_document_vectors,
    search_user_vectors,
    load_user_vector_store,
)
from app.services.rag.service import get_rag_service
from tests.test_chat_persistence import get_test_users


# ==========================================
# 1. CHUNKING TESTS
# ==========================================

def test_chunking_word_window():
    """Verify word chunking splits at specified word counts with proper overlap."""
    words = [f"word{i}" for i in range(120)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) == 3
    assert len(chunks[0].split()) == 50
    assert len(chunks[1].split()) == 50
    # Overlap check
    overlap_words = set(chunks[0].split()[-10:]).intersection(set(chunks[1].split()[:10]))
    assert len(overlap_words) == 10


def test_chunking_structured_document_with_pages():
    """Verify structured document chunking preserves page-level tracking across boundaries."""
    doc = {
        "filename": "multi_page_doc.pdf",
        "filepath": "/path/to/multi_page_doc.pdf",
        "file_type": "PDF",
        "text": "Page one text content. " * 30 + "\n\n" + "Page two text content. " * 30,
        "pages": [
            {"page_number": 1, "text": "Page one text content. " * 30},
            {"page_number": 2, "text": "Page two text content. " * 30},
        ],
    }

    chunks = chunk_document(doc, user_id="user_test_123", document_id="doc_abc", chunk_size=40, chunk_overlap=5)
    assert len(chunks) >= 2
    for c in chunks:
        assert c["document_id"] == "doc_abc"
        assert c["user_id"] == "user_test_123"
        assert c["filename"] == "multi_page_doc.pdf"
        assert "page_start" in c
        assert "page_end" in c
        assert c["page_start"] <= c["page_end"]
        assert len(c["text"]) > 0


# ==========================================
# 2. EMBEDDINGS TESTS
# ==========================================

def test_embeddings_dimension_and_semantic_similarity():
    """Verify 384-dimensional embeddings and cosine similarity properties."""
    v1 = embed_text("Artificial intelligence deep learning transformer models")
    v2 = embed_text("Neural networks and natural language machine learning")
    v3 = embed_text("Delicious traditional chocolate cake baking recipe")

    assert isinstance(v1, np.ndarray)
    assert v1.shape == (EMBEDDING_DIM,)
    assert v1.dtype == np.float32

    # L2 normalize
    norm_v1 = v1 / np.linalg.norm(v1)
    norm_v2 = v2 / np.linalg.norm(v2)
    norm_v3 = v3 / np.linalg.norm(v3)

    sim_ai = float(np.dot(norm_v1, norm_v2))
    sim_cake = float(np.dot(norm_v1, norm_v3))

    # Semantic similarity check: AI & Neural Nets are much closer than AI & Chocolate Cake
    assert sim_ai > sim_cake
    assert sim_ai > 0.4


# ==========================================
# 3. VECTOR STORE & STRICT USER ISOLATION
# ==========================================

def test_vector_store_strict_user_isolation(tmp_path):
    """
    Verify that User 1 vectors cannot be accessed, searched, or leaked to User 2.
    """
    user1_id = "test_isolation_user_1"
    user2_id = "test_isolation_user_2"

    # User 1 has confidential data
    user1_chunks = [
        {
            "chunk_id": 0,
            "document_id": "doc_u1_1",
            "user_id": user1_id,
            "filename": "quantum_financial_vault.txt",
            "page_start": 1,
            "page_end": 1,
            "text": "The secret vault security passcode for Project Quantum is 8849-XRAY-ALPHA.",
        }
    ]

    # User 2 has public gardening data
    user2_chunks = [
        {
            "chunk_id": 0,
            "document_id": "doc_u2_1",
            "user_id": user2_id,
            "filename": "gardening_tips.txt",
            "page_start": 1,
            "page_end": 1,
            "text": "Tomatoes grow best in full sunlight with well-draining soil and organic compost.",
        }
    ]

    # Index User 1
    add_document_vectors(user1_id, "doc_u1_1", user1_chunks)
    # Index User 2
    add_document_vectors(user2_id, "doc_u2_1", user2_chunks)

    # Query for User 1's confidential info
    q_emb = embed_text("What is the security passcode for Project Quantum?")

    # User 1 searches -> MUST find it
    u1_results = search_user_vectors(user1_id, q_emb, top_k=2)
    assert len(u1_results) > 0
    assert "8849-XRAY-ALPHA" in u1_results[0]["text"]
    assert u1_results[0]["filename"] == "quantum_financial_vault.txt"

    # User 2 searches -> MUST NOT find User 1's secret!
    u2_results = search_user_vectors(user2_id, q_emb, top_k=2)
    # User 2 should either get gardening chunks or low relevance, NEVER User 1's chunks
    for r in u2_results:
        assert "8849-XRAY-ALPHA" not in r["text"]
        assert r["filename"] != "quantum_financial_vault.txt"

    # Clean up test user vectors
    remove_document_vectors(user1_id, "doc_u1_1")
    remove_document_vectors(user2_id, "doc_u2_1")


def test_document_deletion_synchronization():
    """Verify that deleting a document removes its chunks from vector search."""
    user_id = "test_deletion_sync_user"
    doc_id = "doc_to_delete_101"

    chunks = [
        {
            "chunk_id": 0,
            "document_id": doc_id,
            "user_id": user_id,
            "filename": "temporary_manual.txt",
            "page_start": 1,
            "page_end": 1,
            "text": "The protocol requires rotating API cryptographic keys every thirty days.",
        }
    ]

    add_document_vectors(user_id, doc_id, chunks)

    # Verify searchable
    q_emb = embed_text("How often should cryptographic keys be rotated?")
    res_before = search_user_vectors(user_id, q_emb, top_k=1)
    assert len(res_before) == 1
    assert "rotating API cryptographic keys" in res_before[0]["text"]

    # Delete document
    remaining = remove_document_vectors(user_id, doc_id)
    assert remaining == 0

    # Verify no longer searchable
    res_after = search_user_vectors(user_id, q_emb, top_k=1)
    assert len(res_after) == 0


# ==========================================
# 4. RAG PROMPT AUGMENTATION TESTS
# ==========================================

def test_rag_prompt_augmentation():
    """Verify build_augmented_prompt formats citations and instructions correctly."""
    rag_service = get_rag_service()
    chunks = [
        {
            "filename": "system_specs.pdf",
            "page_start": 4,
            "page_end": 4,
            "score": 0.89,
            "excerpt": "The maximum memory allocation per worker node is 64GB DDR5.",
        },
        {
            "filename": "network_config.docx",
            "page_start": 2,
            "page_end": 3,
            "score": 0.81,
            "excerpt": "Port 8443 is designated for secure telemetry ingress.",
        },
    ]

    prompt = rag_service.build_augmented_prompt("What is the memory limit?", chunks)
    assert "DOCUMENT CONTEXT:" in prompt
    assert "[Source 1] system_specs.pdf (Page 4, Relevance: 0.89):" in prompt
    assert "The maximum memory allocation per worker node is 64GB DDR5." in prompt
    assert "[Source 2] network_config.docx (Pages 2-3, Relevance: 0.81):" in prompt
    assert "USER QUESTION:" in prompt
    assert "What is the memory limit?" in prompt
    assert "Answer the USER QUESTION accurately and concisely using ONLY the facts provided in the DOCUMENT CONTEXT above." in prompt


# ==========================================
# 5. END-TO-END RAG PIPELINE API TEST
# ==========================================

@pytest.mark.asyncio
async def test_end_to_end_document_upload_and_rag_chat_stream():
    """
    Test full end-to-end integration:
    1. User 1 uploads document (chunk_count > 0 and status='ready' persisted in DB)
    2. User 1 asks a question about document in a chat stream
    3. Retrieved chunks are cited with filename and page info
    4. Citations are persisted in PostgreSQL Message.sources_json
    5. User 2 asks same question -> cannot retrieve User 1's context
    """
    auth_headers = await get_test_users()
    headers_u1 = auth_headers["user1"]["headers"]
    headers_u2 = auth_headers["user2"]["headers"]
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Step 1: User 1 uploads document
        doc_content = (
            "LocalGPT Architecture Overview:\n"
            "The cluster uses 8 NVIDIA H100 GPUs connected with 3.2 Tbps InfiniBand fabrics.\n"
            "Cold storage uses distributed MinIO object storage with erasure coding 8+4.\n"
            "The default context window is 32768 tokens with rotary positional embeddings."
        ).encode("utf-8")

        upload_res = await ac.post(
            f"{settings.API_V1_STR}/documents/upload",
            headers=headers_u1,
            files={"file": ("cluster_architecture.txt", doc_content, "text/plain")},
        )
        assert upload_res.status_code == 201
        doc_data = upload_res.json()
        assert doc_data["filename"] == "cluster_architecture.txt"
        assert doc_data["chunk_count"] >= 1
        assert doc_data["status"] == "ready"
        doc_id = doc_data["id"]

        # Step 2: User 1 creates conversation
        conv_res = await ac.post(
            f"{settings.API_V1_STR}/conversations",
            headers=headers_u1,
            json={"title": "Cluster QA"},
        )
        assert conv_res.status_code == 201
        conv_id = conv_res.json()["id"]

        # Step 3: User 1 streams a question querying the document
        question = "How many NVIDIA H100 GPUs does the cluster use?"
        stream_res = await ac.post(
            f"{settings.API_V1_STR}/chat/{conv_id}/stream",
            headers=headers_u1,
            json={"role": "user", "content": question},
        )
        assert stream_res.status_code == 200
        sse_text = stream_res.text
        assert "data: " in sse_text
        assert "[DONE]" in sse_text

        # Step 4: Verify conversation message persistence and sources_json in DB
        get_conv = await ac.get(
            f"{settings.API_V1_STR}/conversations/{conv_id}",
            headers=headers_u1,
        )
        assert get_conv.status_code == 200
        conv_detail = get_conv.json()
        messages = conv_detail["messages"]
        assert len(messages) >= 2  # user + assistant

        assistant_msg = messages[-1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg.get("sources_json") is not None
        sources = assistant_msg["sources_json"].get("sources", [])
        assert len(sources) >= 1
        assert sources[0]["filename"] == "cluster_architecture.txt"
        assert "H100" in sources[0]["excerpt"]

        # Step 5: User 2 streams question -> verify strict isolation (no sources found)
        conv_res_u2 = await ac.post(
            f"{settings.API_V1_STR}/conversations",
            headers=headers_u2,
            json={"title": "Intruder QA"},
        )
        assert conv_res_u2.status_code == 201
        conv_id_u2 = conv_res_u2.json()["id"]

        stream_res_u2 = await ac.post(
            f"{settings.API_V1_STR}/chat/{conv_id_u2}/stream",
            headers=headers_u2,
            json={"role": "user", "content": question},
        )
        assert stream_res_u2.status_code == 200

        get_conv_u2 = await ac.get(
            f"{settings.API_V1_STR}/conversations/{conv_id_u2}",
            headers=headers_u2,
        )
        messages_u2 = get_conv_u2.json()["messages"]
        asst_u2 = messages_u2[-1]
        # User 2 has NO documents uploaded -> sources_json MUST be null / empty
        assert asst_u2.get("sources_json") is None or not asst_u2["sources_json"].get("sources")

        # Step 6: Clean up User 1's document
        del_res = await ac.delete(
            f"{settings.API_V1_STR}/documents/{doc_id}",
            headers=headers_u1,
        )
        assert del_res.status_code == 204
