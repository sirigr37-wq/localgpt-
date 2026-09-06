from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.user import User
from app.schemas.document import (
    DocumentResponse,
    DocumentListResponse,
    DocumentSearchRequest,
    DocumentSearchResponse,
)
from app.services.auth_service import get_current_user
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("", response_model=DocumentListResponse)
async def list_user_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve all documents uploaded by the current user."""
    docs = await DocumentService.list_documents(db, user_id=current_user.id)
    return DocumentListResponse(total=len(docs), documents=docs)


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a document (PDF, DOCX, TXT, CSV, JSON) to the user's isolated knowledge base."""
    return await DocumentService.save_uploaded_document(db, user_id=current_user.id, upload_file=file)


@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(
    payload: DocumentSearchRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Test document search against the user's indexed FAISS vector store.
    Direct Phase 2 parity: inspect similarity scores, page citations, and chunk text.
    """
    matches = await DocumentService.search_documents(
        user_id=current_user.id,
        query=payload.query.strip(),
        top_k=payload.top_k or 3,
    )
    return DocumentSearchResponse(
        query=payload.query,
        total_matches=len(matches),
        matches=matches,
    )


@router.delete("", status_code=status.HTTP_200_OK)
async def clear_all_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete all documents and clear the FAISS index for the authenticated user."""
    count = await DocumentService.clear_all_documents(db, user_id=current_user.id)
    return {"count": count, "message": f"Successfully deleted {count} document(s) and cleared vector store."}


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve metadata for a specific document owned by the user."""
    return await DocumentService.get_document(db, document_id=document_id, user_id=current_user.id)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete an uploaded document and its stored file."""
    await DocumentService.delete_document(db, document_id=document_id, user_id=current_user.id)
    return None


