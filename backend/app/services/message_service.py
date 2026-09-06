from datetime import datetime, timezone
from typing import List, Optional, Any, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, update
from fastapi import HTTPException, status

from app.models.conversation import Conversation
from app.models.message import Message
from app.schemas.message import MessageCreate, MessageFeedback


class MessageService:
    """Service handling conversation messages, ordering, feedback, and truncation."""

    @staticmethod
    async def add_message(
        db: AsyncSession,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        # 1. Compute next message_order
        stmt = select(func.coalesce(func.max(Message.message_order), 0)).where(
            Message.conversation_id == conversation_id
        )
        result = await db.execute(stmt)
        next_order = (result.scalar() or 0) + 1

        # 2. Create message
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            message_order=next_order,
            sources_json={"sources": sources} if sources else None,
        )
        db.add(message)

        # 3. Touch conversation updated_at and auto-generate title on first message
        conv_stmt = select(Conversation).where(Conversation.id == conversation_id)
        conv_res = await db.execute(conv_stmt)
        conv = conv_res.scalar_one_or_none()
        if conv:
            conv.updated_at = datetime.now(timezone.utc)
            if next_order == 1 and conv.title in ["New Chat", ""]:
                first_line = content.split("\n")[0].strip()
                conv.title = first_line[:48].strip() + ("..." if len(first_line) > 48 else "")
            db.add(conv)

        await db.commit()
        await db.refresh(message)
        return message

    @staticmethod
    async def update_feedback(
        db: AsyncSession,
        message_id: str,
        feedback: Optional[str],
    ) -> Message:
        stmt = select(Message).where(Message.id == message_id)
        result = await db.execute(stmt)
        message = result.scalar_one_or_none()
        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message not found.",
            )

        message.feedback = feedback
        db.add(message)
        await db.commit()
        await db.refresh(message)
        return message

    @staticmethod
    async def edit_and_truncate(
        db: AsyncSession,
        conversation_id: str,
        message_id: str,
        new_content: str,
    ) -> Message:
        stmt = select(Message).where(
            Message.id == message_id,
            Message.conversation_id == conversation_id,
        )
        result = await db.execute(stmt)
        target_msg = result.scalar_one_or_none()
        if not target_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message not found.",
            )

        # Update target message content
        target_msg.content = new_content
        db.add(target_msg)

        # Delete all subsequent messages in this conversation
        del_stmt = delete(Message).where(
            Message.conversation_id == conversation_id,
            Message.message_order > target_msg.message_order,
        )
        await db.execute(del_stmt)
        await db.commit()
        await db.refresh(target_msg)
        return target_msg

    @staticmethod
    async def get_messages(
        db: AsyncSession,
        conversation_id: str,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """Fetch all messages for a conversation ordered chronologically by message_order."""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.message_order.asc())
        )
        if limit:
            stmt = stmt.limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())

