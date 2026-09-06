import io
import json
import pytest
import pypdf
import docx
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession
import os

from app.main import app
from app.core.config import settings
from tests.test_chat_persistence import get_test_users


def create_minimal_pdf_bytes() -> bytes:
    """Generate a minimal, valid in-memory PDF file."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def create_minimal_docx_bytes() -> bytes:
    """Generate a minimal, valid in-memory DOCX document."""
    doc = docx.Document()
    doc.add_paragraph("LocalGPT Phase 3 Test Document Content.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_multi_format_document_uploads():
    """Verify that PDF, DOCX, TXT, CSV, and JSON can all be uploaded and parsed successfully."""
    auth_headers = await get_test_users()
    headers = auth_headers["user1"]["headers"]
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Upload PDF
        pdf_bytes = create_minimal_pdf_bytes()
        res_pdf = await ac.post(
            f"{settings.API_V1_STR}/documents/upload",
            headers=headers,
            files={"file": ("sample_report.pdf", pdf_bytes, "application/pdf")},
        )
        assert res_pdf.status_code == 201
        pdf_data = res_pdf.json()
        assert pdf_data["file_type"] == "PDF"
        assert pdf_data["filename"] == "sample_report.pdf"
        assert pdf_data["num_pages"] >= 1
        assert pdf_data["status"] == "ready"

        # 2. Upload DOCX
        docx_bytes = create_minimal_docx_bytes()
        res_docx = await ac.post(
            f"{settings.API_V1_STR}/documents/upload",
            headers=headers,
            files={"file": ("project_spec.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert res_docx.status_code == 201
        docx_data = res_docx.json()
        assert docx_data["file_type"] == "DOCX"
        assert docx_data["filename"] == "project_spec.docx"
        assert docx_data["num_pages"] >= 1

        # 3. Upload TXT
        txt_bytes = b"This is a sample plain text file for knowledge extraction.\nIt contains test data."
        res_txt = await ac.post(
            f"{settings.API_V1_STR}/documents/upload",
            headers=headers,
            files={"file": ("notes.txt", txt_bytes, "text/plain")},
        )
        assert res_txt.status_code == 201
        txt_data = res_txt.json()
        assert txt_data["file_type"] == "TXT"
        assert txt_data["filename"] == "notes.txt"

        # 4. Upload CSV
        csv_bytes = b"metric,value,unit\nlatency,12,ms\nthroughput,450,rps\naccuracy,0.98,score"
        res_csv = await ac.post(
            f"{settings.API_V1_STR}/documents/upload",
            headers=headers,
            files={"file": ("metrics.csv", csv_bytes, "text/csv")},
        )
        assert res_csv.status_code == 201
        csv_data = res_csv.json()
        assert csv_data["file_type"] == "CSV"
        assert csv_data["filename"] == "metrics.csv"

        # 5. Upload JSON
        json_bytes = json.dumps({
            "service": "LocalGPT",
            "version": "3.0.0",
            "features": ["SSE streaming", "RAG", "Auth"]
        }).encode("utf-8")
        res_json = await ac.post(
            f"{settings.API_V1_STR}/documents/upload",
            headers=headers,
            files={"file": ("config.json", json_bytes, "application/json")},
        )
        assert res_json.status_code == 201
        json_data = res_json.json()
        assert json_data["file_type"] == "JSON"
        assert json_data["filename"] == "config.json"


@pytest.mark.asyncio
async def test_document_validation_failures():
    """Verify proper validation errors for unsupported extensions, empty files, and corrupted content."""
    auth_headers = await get_test_users()
    headers = auth_headers["user1"]["headers"]
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Unsupported extension (.exe)
        res_bad_ext = await ac.post(
            f"{settings.API_V1_STR}/documents/upload",
            headers=headers,
            files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
        )
        assert res_bad_ext.status_code == 400
        assert "Unsupported file format" in res_bad_ext.json()["detail"]

        # 2. Empty file (0 bytes)
        res_empty = await ac.post(
            f"{settings.API_V1_STR}/documents/upload",
            headers=headers,
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert res_empty.status_code == 400
        assert "empty" in res_empty.json()["detail"].lower()

        # 3. Invalid PDF header (random bytes renamed to .pdf)
        res_fake_pdf = await ac.post(
            f"{settings.API_V1_STR}/documents/upload",
            headers=headers,
            files={"file": ("fake.pdf", b"This is not a pdf file at all", "application/pdf")},
        )
        assert res_fake_pdf.status_code == 400
        assert "Invalid PDF" in res_fake_pdf.json()["detail"]

        # 4. Invalid JSON syntax
        res_bad_json = await ac.post(
            f"{settings.API_V1_STR}/documents/upload",
            headers=headers,
            files={"file": ("broken.json", b"{key: unquoted, value: syntax_error", "application/json")},
        )
        assert res_bad_json.status_code == 400
        assert "JSON syntax" in res_bad_json.json()["detail"]


@pytest.mark.asyncio
async def test_unauthenticated_document_access_blocked():
    """Ensure anonymous users cannot access, list, or upload documents."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # GET /documents without token
        get_res = await ac.get(f"{settings.API_V1_STR}/documents")
        assert get_res.status_code == 401

        # POST /documents/upload without token
        post_res = await ac.post(
            f"{settings.API_V1_STR}/documents/upload",
            files={"file": ("test.txt", b"data", "text/plain")},
        )
        assert post_res.status_code == 401


@pytest.mark.asyncio
async def test_user_document_isolation_and_deletion():
    """Verify that User 1 documents cannot be viewed or deleted by User 2, and User 1 can delete their own."""
    auth_headers = await get_test_users()
    headers1 = auth_headers["user1"]["headers"]
    headers2 = auth_headers["user2"]["headers"]
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # User 1 uploads a document
        txt_content = b"Confidential document belonging exclusively to User 1."
        up_res = await ac.post(
            f"{settings.API_V1_STR}/documents/upload",
            headers=headers1,
            files={"file": ("user1_secret.txt", txt_content, "text/plain")},
        )
        assert up_res.status_code == 201
        doc1 = up_res.json()
        doc1_id = doc1["id"]

        # User 1 lists documents -> doc1 is present
        u1_list = await ac.get(f"{settings.API_V1_STR}/documents", headers=headers1)
        assert u1_list.status_code == 200
        u1_doc_ids = [d["id"] for d in u1_list.json()["documents"]]
        assert doc1_id in u1_doc_ids

        # User 2 lists documents -> doc1 is NOT in User 2's list
        u2_list = await ac.get(f"{settings.API_V1_STR}/documents", headers=headers2)
        assert u2_list.status_code == 200
        u2_doc_ids = [d["id"] for d in u2_list.json()["documents"]]
        assert doc1_id not in u2_doc_ids

        # User 2 tries to GET User 1's document -> 404 Not Found
        u2_get = await ac.get(f"{settings.API_V1_STR}/documents/{doc1_id}", headers=headers2)
        assert u2_get.status_code == 404

        # User 2 tries to DELETE User 1's document -> 404 Not Found
        u2_del = await ac.delete(f"{settings.API_V1_STR}/documents/{doc1_id}", headers=headers2)
        assert u2_del.status_code == 404

        # User 1 can successfully GET their document
        u1_get = await ac.get(f"{settings.API_V1_STR}/documents/{doc1_id}", headers=headers1)
        assert u1_get.status_code == 200
        assert u1_get.json()["id"] == doc1_id

        # User 1 deletes their document -> 204 No Content
        u1_del = await ac.delete(f"{settings.API_V1_STR}/documents/{doc1_id}", headers=headers1)
        assert u1_del.status_code == 204

        # Verify document is gone from User 1's list and returns 404
        u1_get_after = await ac.get(f"{settings.API_V1_STR}/documents/{doc1_id}", headers=headers1)
        assert u1_get_after.status_code == 404
