import os
import re
import io
import csv
import json
import zipfile
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import UploadFile, HTTPException, status

import uuid
from app.models.document import Document
from app.core.config import settings
from app.services.document_loader import load_document
from app.services.rag.service import get_rag_service


class DocumentService:
    """Service handling multi-format document validation, isolated storage, and metadata persistence."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".csv", ".json"}

    @staticmethod
    def get_user_storage_dir(user_id: str) -> str:
        """Create and return an isolated, verified absolute storage path for the given user."""
        base_dir = os.path.abspath(settings.UPLOAD_STORAGE_DIR)
        os.makedirs(base_dir, exist_ok=True)
        user_dir = os.path.abspath(os.path.join(base_dir, user_id))
        
        # Verify user_dir does not escape the base storage directory
        if not user_dir.startswith(base_dir):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid storage path.",
            )
        os.makedirs(user_dir, exist_ok=True)
        return user_dir

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent directory traversal and special character exploits."""
        base = os.path.basename(filename)
        # Strip path characters and keep only safe alphanumerics, dot, dash, and underscore
        cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", base)
        return cleaned or "document"

    @staticmethod
    def validate_file_content(file_bytes: bytes, ext: str) -> None:
        """
        Validate that the file header and internal structure match the declared extension.
        Rejects renamed executables, corrupted archives, or invalid formats.
        """
        if ext == ".pdf":
            if not file_bytes.startswith(b"%PDF"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid PDF file format. The file header does not match standard PDF specifications.",
                )
        elif ext in [".docx", ".doc"]:
            if ext == ".docx":
                if not file_bytes.startswith(b"PK\x03\x04"):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid DOCX file format. File is not a valid OpenXML document package.",
                    )
                try:
                    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                        if "word/document.xml" not in zf.namelist():
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Invalid DOCX file. Missing internal word/document.xml payload.",
                            )
                except zipfile.BadZipFile:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Corrupted DOCX archive file.",
                    )
        elif ext == ".json":
            try:
                decoded = file_bytes.decode("utf-8")
                json.loads(decoded)
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="JSON file is not valid UTF-8 text.",
                )
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid JSON syntax: {str(e)}",
                )
        elif ext == ".csv":
            try:
                decoded = file_bytes.decode("utf-8", errors="replace")
                if "\x00" in decoded[:1024]:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Binary file detected. CSV must contain clean text.",
                    )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid CSV file encoding: {str(e)}",
                )
        elif ext in [".txt", ".md"]:
            try:
                decoded = file_bytes.decode("utf-8", errors="replace")
                if "\x00" in decoded[:1024]:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Binary file detected. Plain text document must not contain null bytes.",
                    )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid text file encoding: {str(e)}",
                )

    @staticmethod
    async def save_uploaded_document(
        db: AsyncSession,
        user_id: str,
        upload_file: UploadFile,
    ) -> Document:
        """
        Validate, isolate, extract metadata, and persist document in user-isolated storage and PostgreSQL.
        """
        raw_filename = upload_file.filename or "uploaded_file"
        safe_name = DocumentService.sanitize_filename(raw_filename)
        ext = os.path.splitext(safe_name)[1].lower()

        # 1. Validate extension
        if ext not in DocumentService.SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{ext}'. Supported formats: PDF, DOCX, TXT, CSV, JSON.",
            )

        # 2. Read bytes and validate file size
        file_bytes = await upload_file.read()
        file_size = len(file_bytes)

        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )

        if file_size > settings.MAX_FILE_SIZE_BYTES:
            max_mb = settings.MAX_FILE_SIZE_BYTES // (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size ({file_size} bytes) exceeds the maximum allowed limit of {max_mb}MB.",
            )

        # 3. Content signature validation
        DocumentService.validate_file_content(file_bytes, ext)

        # 4. Storage isolation and path traversal protection
        user_dir = DocumentService.get_user_storage_dir(user_id)
        target_path = os.path.abspath(os.path.join(user_dir, safe_name))

        if not target_path.startswith(user_dir):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Path traversal attempt detected in filename.",
            )

        # Collision avoidance for same-name uploads
        if os.path.exists(target_path):
            name_part, ext_part = os.path.splitext(safe_name)
            counter = 1
            while os.path.exists(os.path.join(user_dir, f"{name_part}_{counter}{ext_part}")):
                counter += 1
            safe_name = f"{name_part}_{counter}{ext_part}"
            target_path = os.path.abspath(os.path.join(user_dir, safe_name))

        # Write file to isolated disk location
        with open(target_path, "wb") as f:
            f.write(file_bytes)

        # 5. Extract document pages and text statistics via DocumentLoader
        extracted = load_document(target_path)
        if extracted.get("status") == "error":
            # Clean up corrupted or encrypted file from disk
            if os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except OSError:
                    pass
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document parsing error: {extracted.get('error_message')}",
            )

        num_pages = extracted.get("num_pages", 1) or 1
        file_type = ext.upper().lstrip(".")

        # 6. Ingest into RAG pipeline (chunking, embedding, FAISS indexing)
        doc_id = str(uuid.uuid4())
        rag_service = get_rag_service()
        ingest_res = await rag_service.ingest_document(
            document_id=doc_id,
            filepath=target_path,
            user_id=user_id,
            filename=safe_name,
            file_type=file_type,
            num_pages=num_pages,
        )

        chunk_count = ingest_res.get("chunk_count", 0)
        doc_status = ingest_res.get("status", "ready")

        # 7. Save metadata record in PostgreSQL
        doc = Document(
            id=doc_id,
            user_id=user_id,
            filename=safe_name,
            filepath=target_path,
            file_type=file_type,
            file_size_bytes=file_size,
            num_pages=num_pages,
            chunk_count=chunk_count,
            status=doc_status,
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        return doc

    @staticmethod
    async def list_documents(db: AsyncSession, user_id: str) -> List[Document]:
        """Fetch all documents belonging strictly to the authenticated user."""
        stmt = (
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_document(db: AsyncSession, document_id: str, user_id: str) -> Document:
        """Fetch a specific document with user ownership isolation."""
        stmt = select(Document).where(Document.id == document_id, Document.user_id == user_id)
        result = await db.execute(stmt)
        doc = result.scalar_one_or_none()
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )
        return doc

    @staticmethod
    async def delete_document(db: AsyncSession, document_id: str, user_id: str) -> bool:
        """Delete document file from user storage, remove vectors, and remove metadata from PostgreSQL."""
        doc = await DocumentService.get_document(db, document_id, user_id)

        # Remove physical file from isolated disk storage
        if os.path.exists(doc.filepath):
            try:
                os.remove(doc.filepath)
            except OSError:
                pass

        # Remove associated vectors from user's FAISS index
        try:
            await get_rag_service().delete_document_vectors(document_id=doc.id, user_id=user_id)
        except Exception:
            pass

        await db.delete(doc)
        await db.commit()
        return True

    @staticmethod
    async def search_documents(user_id: str, query: str, top_k: int = 3) -> List[dict]:
        """Search user's indexed document chunks using semantic similarity (matches Phase 2 Test Search)."""
        rag_service = get_rag_service()
        chunks = await rag_service.retrieve_relevant_chunks(query=query, user_id=user_id, top_k=top_k)
        results = []
        for idx, ch in enumerate(chunks, start=1):
            text = ch.get("text", "") or ""
            results.append({
                "rank": idx,
                "filename": ch.get("filename", "document"),
                "page_start": ch.get("page_start", 1) or 1,
                "page_end": ch.get("page_end", 1) or 1,
                "score": float(ch.get("score", 0.0) or 0.0),
                "text": text,
                "word_count": len(text.split()),
                "char_count": len(text),
            })
        return results

    @staticmethod
    async def clear_all_documents(db: AsyncSession, user_id: str) -> int:
        """Delete all documents and clear FAISS vector store for the user."""
        docs = await DocumentService.list_documents(db, user_id)
        count = len(docs)
        for doc in docs:
            # Remove physical file
            if os.path.exists(doc.filepath):
                try:
                    os.remove(doc.filepath)
                except OSError:
                    pass
            # Remove associated vectors
            try:
                await get_rag_service().delete_document_vectors(document_id=doc.id, user_id=user_id)
            except Exception:
                pass
            await db.delete(doc)
        await db.commit()
        return count


