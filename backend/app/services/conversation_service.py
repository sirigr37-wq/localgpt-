from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from fastapi import HTTPException, status

from app.models.conversation import Conversation
from app.models.message import Message
from app.schemas.conversation import ConversationCreate, ConversationUpdate


class ConversationService:
    """Service handling multi-turn conversation management and storage."""

    @staticmethod
    async def list_conversations(
        db: AsyncSession,
        user_id: str,
        search_query: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Conversation]:
        stmt = (
            select(Conversation, func.count(Message.id).label("cnt"))
            .outerjoin(Message, Conversation.id == Message.conversation_id)
            .where(Conversation.user_id == user_id)
        )
        if search_query and search_query.strip():
            term = f"%{search_query.strip()}%"
            msg_subq = select(Message.conversation_id).where(Message.content.ilike(term))
            stmt = stmt.where((Conversation.title.ilike(term)) | (Conversation.id.in_(msg_subq)))

        stmt = (
            stmt.group_by(Conversation.id)
            .order_by(Conversation.updated_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await db.execute(stmt)
        conversations: List[Conversation] = []
        for conv, count in result.all():
            conv.message_count = count
            conversations.append(conv)
        return conversations

    @staticmethod
    async def create_conversation(
        db: AsyncSession,
        user_id: str,
        conv_in: ConversationCreate,
    ) -> Conversation:
        conversation = Conversation(
            user_id=user_id,
            title=conv_in.title or "New Chat",
            system_prompt=conv_in.system_prompt,
        )
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
        return conversation

    @staticmethod
    async def get_conversation(
        db: AsyncSession,
        conversation_id: str,
        user_id: str,
    ) -> Conversation:
        stmt = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        )
        result = await db.execute(stmt)
        conversation = result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found.",
            )
        return conversation

    @staticmethod
    async def update_conversation(
        db: AsyncSession,
        conversation_id: str,
        user_id: str,
        conv_update: ConversationUpdate,
    ) -> Conversation:
        conversation = await ConversationService.get_conversation(db, conversation_id, user_id)
        if conv_update.title is not None:
            conversation.title = conv_update.title
        if conv_update.system_prompt is not None:
            conversation.system_prompt = conv_update.system_prompt

        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
        return conversation

    @staticmethod
    async def delete_conversation(
        db: AsyncSession,
        conversation_id: str,
        user_id: str,
    ) -> bool:
        conversation = await ConversationService.get_conversation(db, conversation_id, user_id)
        await db.delete(conversation)
        await db.commit()
        return True
