"""
End-to-End Document Question Verification Script.
Executes complete user flow:
1. Registers two distinct authenticated users (User Alpha and User Beta)
2. User Alpha uploads a document with domain facts
3. Persisted document verification (chunk_count > 0, status='ready' in PostgreSQL)
4. User Alpha creates conversation and queries document via SSE streaming endpoint
5. Verification of retrieved context, streamed response, and sources_json citations in PostgreSQL
6. Strict User Isolation Verification: User Beta queries identical question -> gets 0 citations
7. Deletion test: User Alpha deletes document -> vectors cleaned from FAISS store
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.message import Message
from sqlalchemy import select


async def run_e2e_verification():
    print("=" * 65)
    print("STARTING END-TO-END RAG DOCUMENT QUESTION VERIFICATION")
    print("=" * 65)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Register & Login User Alpha
        reg_a = await client.post(
            f"{settings.API_V1_STR}/auth/register",
            json={"email": "alpha_rag_e2e@example.com", "password": "Password123!", "full_name": "Alpha Tester"},
        )
        assert reg_a.status_code in (201, 400), f"Registration failed: {reg_a.text}"
        login_a = await client.post(
            f"{settings.API_V1_STR}/auth/login",
            json={"email": "alpha_rag_e2e@example.com", "password": "Password123!"},
        )
        assert login_a.status_code == 200, f"Login A failed: {login_a.text}"
        token_a = login_a.json()["access_token"]
        headers_a = {"Authorization": f"Bearer {token_a}"}
        print("[PASS] User Alpha authenticated.")

        # 2. Register & Login User Beta
        reg_b = await client.post(
            f"{settings.API_V1_STR}/auth/register",
            json={"email": "beta_rag_e2e@example.com", "password": "Password123!", "full_name": "Beta Tester"},
        )
        assert reg_b.status_code in (201, 400), f"Registration failed: {reg_b.text}"
        login_b = await client.post(
            f"{settings.API_V1_STR}/auth/login",
            json={"email": "beta_rag_e2e@example.com", "password": "Password123!"},
        )
        assert login_b.status_code == 200, f"Login B failed: {login_b.text}"
        token_b = login_b.json()["access_token"]
        headers_b = {"Authorization": f"Bearer {token_b}"}
        print("[PASS] User Beta authenticated.")

        # 3. User Alpha uploads confidential technical document
        doc_payload = (
            "PROJECT CELESTIAL DEEP-SPACE TELEMETRY SPECIFICATION:\n"
            "Satellite Subsystem: Array-9 Radiometric Scanner\n"
            "Communication Carrier Frequency: 28.45 GHz Ka-band\n"
            "Primary Ground Station Coordinates: 34.0522 N, 118.2437 W (Mount Wilson Facility)\n"
            "Emergency Cryptographic Key Rotation Period: Exactly 14 Earth Days\n"
            "Telemetry Transponder Output Power: 150 Watts Traveling Wave Tube Amplifier (TWTA)\n"
        ).encode("utf-8")

        upload_res = await client.post(
            f"{settings.API_V1_STR}/documents/upload",
            headers=headers_a,
            files={"file": ("celestial_telemetry_spec.txt", doc_payload, "text/plain")},
        )
        assert upload_res.status_code == 201, f"Upload failed: {upload_res.text}"
        doc_info = upload_res.json()
        doc_id = doc_info["id"]
        assert doc_info["filename"] == "celestial_telemetry_spec.txt"
        assert doc_info["chunk_count"] >= 1
        assert doc_info["status"] == "ready"
        print(f"[PASS] Document uploaded & indexed: id={doc_id}, chunk_count={doc_info['chunk_count']}, status={doc_info['status']}")

        # Verify in PostgreSQL
        async with AsyncSessionLocal() as session:
            stmt = select(Document).where(Document.id == doc_id)
            res = await session.execute(stmt)
            db_doc = res.scalar_one_or_none()
            assert db_doc is not None
            assert db_doc.chunk_count >= 1
            assert db_doc.status == "ready"
        print("[PASS] PostgreSQL persistence of chunk_count and status verified.")

        # 4. User Alpha creates conversation
        conv_res = await client.post(
            f"{settings.API_V1_STR}/conversations",
            headers=headers_a,
            json={"title": "Celestial Telemetry Q&A"},
        )
        assert conv_res.status_code == 201
        conv_id = conv_res.json()["id"]
        print(f"[PASS] Conversation created: {conv_id}")

        # 5. User Alpha asks question querying the document
        question = "What is the primary communication carrier frequency and emergency key rotation period for Project Celestial?"
        print(f"\n[QUERY] User Alpha: '{question}'")
        stream_res = await client.post(
            f"{settings.API_V1_STR}/chat/{conv_id}/stream",
            headers=headers_a,
            json={"role": "user", "content": question},
        )
        assert stream_res.status_code == 200
        sse_lines = stream_res.text.strip().split("\n")
        assert any("data: " in line for line in sse_lines)
        assert any("[DONE]" in line for line in sse_lines)
        print("[PASS] SSE stream completed with [DONE] signal.")

        # 6. Check PostgreSQL conversation detail & source citations
        conv_detail_res = await client.get(
            f"{settings.API_V1_STR}/conversations/{conv_id}",
            headers=headers_a,
        )
        assert conv_detail_res.status_code == 200
        conv_detail = conv_detail_res.json()
        messages = conv_detail["messages"]
        assert len(messages) >= 2
        user_msg = messages[-2]
        asst_msg = messages[-1]

        assert user_msg["role"] == "user"
        assert asst_msg["role"] == "assistant"
        assert asst_msg.get("sources_json") is not None
        sources = asst_msg["sources_json"].get("sources", [])
        assert len(sources) >= 1
        citation = sources[0]
        print(f"[PASS] Citations verified in PostgreSQL:")
        print(f"       - Filename: {citation.get('filename')}")
        print(f"       - Page: {citation.get('page_start')}")
        print(f"       - Score: {citation.get('score')}")
        print(f"       - Excerpt: {citation.get('excerpt')[:80]}...")

        assert citation["filename"] == "celestial_telemetry_spec.txt"
        assert "28.45 GHz" in citation["excerpt"] or "Celestial" in citation["excerpt"]

        # 7. Strict User Isolation Check: User Beta queries identical question
        print("\n[ISOLATION] User Beta asking identical question...")
        conv_beta = await client.post(
            f"{settings.API_V1_STR}/conversations",
            headers=headers_b,
            json={"title": "Unauthorized Telemetry QA"},
        )
        assert conv_beta.status_code == 201
        beta_conv_id = conv_beta.json()["id"]

        stream_beta = await client.post(
            f"{settings.API_V1_STR}/chat/{beta_conv_id}/stream",
            headers=headers_b,
            json={"role": "user", "content": question},
        )
        assert stream_beta.status_code == 200

        conv_beta_detail = await client.get(
            f"{settings.API_V1_STR}/conversations/{beta_conv_id}",
            headers=headers_b,
        )
        beta_messages = conv_beta_detail.json()["messages"]
        beta_asst = beta_messages[-1]
        assert beta_asst.get("sources_json") is None or not beta_asst["sources_json"].get("sources"), \
            "Security Violation: User Beta retrieved User Alpha's confidential citations!"
        print("[PASS] User isolation confirmed: User Beta retrieved 0 citations from User Alpha's data.")

        # 8. Document deletion & index cleanup check
        del_res = await client.delete(
            f"{settings.API_V1_STR}/documents/{doc_id}",
            headers=headers_a,
        )
        assert del_res.status_code == 204
        print(f"[PASS] Document {doc_id} successfully deleted.")

        # Query again after deletion -> sources must now be empty
        stream_post_del = await client.post(
            f"{settings.API_V1_STR}/chat/{conv_id}/stream",
            headers=headers_a,
            json={"role": "user", "content": question},
        )
        assert stream_post_del.status_code == 200

        conv_after_del = await client.get(
            f"{settings.API_V1_STR}/conversations/{conv_id}",
            headers=headers_a,
        )
        last_msg = conv_after_del.json()["messages"][-1]
        assert last_msg.get("sources_json") is None or not last_msg["sources_json"].get("sources")
        print("[PASS] Post-deletion verification: vectors cleaned and no sources retrieved.")

    print("\n" + "=" * 65)
    print("END-TO-END RAG DOCUMENT QUESTION TEST PASSED 100% SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(run_e2e_verification())
