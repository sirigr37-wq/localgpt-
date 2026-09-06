from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict


class SourceCitation(BaseModel):
    filename: str
    page_start: Optional[int] = 1
    page_end: Optional[int] = 1
    score: Optional[float] = 0.0
    excerpt: Optional[str] = None


class MessageBase(BaseModel):
    role: str
    content: str


class MessageCreate(MessageBase):
    sources: Optional[List[SourceCitation]] = None


class MessageFeedback(BaseModel):
    feedback: Optional[str] = None  # 'like', 'dislike', or null


class MessageResponse(MessageBase):
    id: str
    conversation_id: str
    message_order: int
    feedback: Optional[str] = None
    sources_json: Optional[Any] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
