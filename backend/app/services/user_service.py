from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException, status

from app.models.user import User
from app.schemas.user import UserUpdate


class UserService:
    """Service handling user account details and updates."""

    @staticmethod
    async def get_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
        stmt = select(User).where(User.id == user_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def update_user(db: AsyncSession, user: User, update_data: UserUpdate) -> User:
        if update_data.full_name is not None:
            user.full_name = update_data.full_name
        if update_data.avatar_url is not None:
            user.avatar_url = update_data.avatar_url

        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user
