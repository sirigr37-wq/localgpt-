from typing import Optional
from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int


class TokenPayload(BaseModel):
    sub: Optional[str] = None
    exp: Optional[int] = None
