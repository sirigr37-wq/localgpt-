import json
import asyncio
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db, AsyncSessionLocal
from app.models.user import User
from app.models.message import Message
from app.schemas.message import MessageCreate, MessageResponse, MessageFeedback
from app.services.auth_service import get_current_user
from app.services.conversation_service import ConversationService
from app.services.message_service import MessageService
from app.services.llm.provider import (
    get_llm_provider,
    LLMError,
    LLMConfigurationError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMProviderError,
)
from app.services.rag.service import get_rag_service

router = APIRouter(prefix="/chat", tags=["Chat & Streaming"])


@router.post("/{conversation_id}/messages", response_model=MessageResponse)
async def post_message(
    conversation_id: str,
    message_in: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save a user message into the conversation."""
    # Verify ownership
    await ConversationService.get_conversation(db, conversation_id, current_user.id)

    # Save message
    user_msg = await MessageService.add_message(
        db=db,
        conversation_id=conversation_id,
        role=message_in.role,
        content=message_in.content,
        sources=[s.model_dump() for s in message_in.sources] if message_in.sources else None,
    )
    return user_msg


@router.post("/{conversation_id}/stream")
async def stream_chat(
    conversation_id: str,
    message_in: MessageCreate,
    regenerate: bool = False,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Stream AI tokens in real-time via Server-Sent Events (SSE).
    Saves the user prompt, retrieves multi-turn conversation memory, streams response chunks,
    and persists the complete assistant message upon completion.
    """
    # Verify conversation access and ownership
    conv = await ConversationService.get_conversation(db, conversation_id, current_user.id)

    # 1. Handle user message persistence or regeneration
    if regenerate:
        all_msgs = await MessageService.get_messages(db, conversation_id)
        if all_msgs and all_msgs[-1].role == "assistant":
            del_stmt = delete(Message).where(Message.id == all_msgs[-1].id)
            await db.execute(del_stmt)
            await db.commit()
            all_msgs = all_msgs[:-1]

        user_msg = all_msgs[-1] if all_msgs and all_msgs[-1].role == "user" else None
        if not user_msg:
            user_msg = await MessageService.add_message(
                db=db,
                conversation_id=conversation_id,
                role="user",
                content=message_in.content,
            )
    else:
        user_msg = await MessageService.add_message(
            db=db,
            conversation_id=conversation_id,
            role="user",
            content=message_in.content,
        )

    llm_provider = get_llm_provider()
    rag_service = get_rag_service()

    # 2. Retrieve RAG chunks if applicable
    chunks = await rag_service.retrieve_relevant_chunks(
        query=message_in.content,
        user_id=current_user.id,
    )
    augmented_prompt = rag_service.build_augmented_prompt(message_in.content, chunks)

    # 3. Load entire conversation history for multi-turn memory
    history_messages = await MessageService.get_messages(db, conversation_id)
    llm_messages = [
        {"role": "system", "content": conv.system_prompt or "You are a helpful assistant."}
    ]
    for m in history_messages:
        if m.id == user_msg.id and chunks:
            llm_messages.append({"role": m.role, "content": augmented_prompt})
        else:
            llm_messages.append({"role": m.role, "content": m.content})

    # 4. Stream response generator
    async def sse_event_stream():
        full_content = []
        try:
            async for chunk in llm_provider.stream_response(llm_messages):
                full_content.append(chunk)
                data = json.dumps({"token": chunk})
                yield f"data: {data}\n\n"

            # Send citations if available
            if chunks:
                yield f"data: {json.dumps({'sources': chunks})}\n\n"

            # Signal completion
            yield "data: [DONE]\n\n"
        except LLMConfigurationError as e:
            err_msg = f"[LLM Configuration Notice]: {e}"
            full_content.append(err_msg)
            yield f"data: {json.dumps({'token': err_msg})}\n\n"
            yield "data: [DONE]\n\n"
        except LLMAuthenticationError as e:
            err_msg = f"[LLM Authentication Error]: {e}"
            full_content.append(err_msg)
            yield f"data: {json.dumps({'token': err_msg})}\n\n"
            yield "data: [DONE]\n\n"
        except LLMRateLimitError as e:
            err_msg = f"[LLM Rate Limit Exceeded]: {e}"
            full_content.append(err_msg)
            yield f"data: {json.dumps({'token': err_msg})}\n\n"
            yield "data: [DONE]\n\n"
        except LLMTimeoutError as e:
            err_msg = f"[LLM Provider Timeout]: {e}"
            full_content.append(err_msg)
            yield f"data: {json.dumps({'token': err_msg})}\n\n"
            yield "data: [DONE]\n\n"
        except LLMProviderError as e:
            err_msg = f"[LLM Provider Error]: {e}"
            full_content.append(err_msg)
            yield f"data: {json.dumps({'token': err_msg})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err_msg = f"[Unexpected Error]: {e}"
            full_content.append(err_msg)
            yield f"data: {json.dumps({'token': err_msg})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # 5. Persist assistant message in PostgreSQL using isolated session
            assistant_text = "".join(full_content)
            if assistant_text.strip():
                async with AsyncSessionLocal() as persist_db:
                    await MessageService.add_message(
                        db=persist_db,
                        conversation_id=conversation_id,
                        role="assistant",
                        content=assistant_text,
                        sources=chunks if chunks else None,
                    )

    return StreamingResponse(
        sse_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.put("/{conversation_id}/messages/{message_id}", response_model=MessageResponse)
async def edit_message(
    conversation_id: str,
    message_id: str,
    message_in: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Edit a user message and truncate subsequent messages in the conversation."""
    await ConversationService.get_conversation(db, conversation_id, current_user.id)
    return await MessageService.edit_and_truncate(
        db=db,
        conversation_id=conversation_id,
        message_id=message_id,
        new_content=message_in.content,
    )


@router.patch("/messages/{message_id}/feedback", response_model=MessageResponse)
async def update_message_feedback(
    message_id: str,
    feedback_in: MessageFeedback,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update feedback (like, dislike, null) on an assistant message."""
    return await MessageService.update_feedback(db, message_id=message_id, feedback=feedback_in.feedback)
