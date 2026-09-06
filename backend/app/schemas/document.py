from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    id: str
    user_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    num_pages: int
    chunk_count: int
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentResponse]


class DocumentSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


class DocumentSearchHit(BaseModel):
    rank: int
    filename: str
    page_start: int
    page_end: int
    score: float
    text: str
    word_count: Optional[int] = 0
    char_count: Optional[int] = 0


class DocumentSearchResponse(BaseModel):
    query: str
    total_matches: int
    matches: List[DocumentSearchHit]

