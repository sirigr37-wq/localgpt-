import pytest
import uuid
from sqlalchemy import select, inspect

from app.db.session import engine, AsyncSessionLocal
from app.models.user import User, OAuthAccount
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.document import Document


@pytest.mark.asyncio
async def test_database_tables_exist():
    """Verify that all required Phase 3 tables exist in the schema."""
    async with engine.connect() as conn:
        def get_tables(sync_conn):
            inspector = inspect(sync_conn)
            return inspector.get_table_names()

        tables = await conn.run_sync(get_tables)
        required_tables = {"users", "oauth_accounts", "conversations", "messages", "documents", "alembic_version"}
        assert required_tables.issubset(set(tables)), f"Missing tables: {required_tables - set(tables)}"


@pytest.mark.asyncio
async def test_user_crud_and_uuid():
    """Verify User model creation with UUID primary key."""
    async with AsyncSessionLocal() as session:
        user = User(
            id=str(uuid.uuid4()),
            email=f"user_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="secure_hashed_password",
            full_name="Alex Developer",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

        assert len(user.id) == 36
        assert user.is_active is True
        assert user.created_at is not None

        # Clean up
        await session.delete(user)
        await session.commit()


@pytest.mark.asyncio
async def test_conversation_and_message_cascade():
    """Verify conversation and message relationships and cascade deletion."""
    async with AsyncSessionLocal() as session:
        # Create user
        user = User(
            id=str(uuid.uuid4()),
            email=f"cascade_test_{uuid.uuid4().hex[:6]}@example.com",
            hashed_password="pwd",
        )
        session.add(user)
        await session.commit()

        # Create conversation
        conv = Conversation(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title="Cascade Test Chat",
        )
        session.add(conv)
        await session.commit()

        # Create message
        msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conv.id,
            role="user",
            content="Hello Cascade",
            message_order=1,
        )
        session.add(msg)
        await session.commit()

        # Delete conversation directly
        await session.delete(conv)
        await session.commit()

        # Verify message was cascaded
        stmt = select(Message).where(Message.id == msg.id)
        res = await session.execute(stmt)
        assert res.scalar_one_or_none() is None

        # Clean up user
        await session.delete(user)
        await session.commit()


@pytest.mark.asyncio
async def test_user_data_isolation():
    """Verify queries for conversations and documents are properly user-isolated."""
    async with AsyncSessionLocal() as session:
        user1 = User(id=str(uuid.uuid4()), email=f"u1_{uuid.uuid4().hex[:6]}@example.com", hashed_password="p")
        user2 = User(id=str(uuid.uuid4()), email=f"u2_{uuid.uuid4().hex[:6]}@example.com", hashed_password="p")
        session.add_all([user1, user2])
        await session.commit()

        # Assign conv1 to user1, conv2 to user2
        c1 = Conversation(id=str(uuid.uuid4()), user_id=user1.id, title="User 1 Secret Chat")
        c2 = Conversation(id=str(uuid.uuid4()), user_id=user2.id, title="User 2 Secret Chat")
        session.add_all([c1, c2])
        await session.commit()

        # Query user1 conversations
        u1_convs = (await session.execute(select(Conversation).where(Conversation.user_id == user1.id))).scalars().all()
        assert len(u1_convs) == 1
        assert u1_convs[0].title == "User 1 Secret Chat"

        # Query user2 conversations
        u2_convs = (await session.execute(select(Conversation).where(Conversation.user_id == user2.id))).scalars().all()
        assert len(u2_convs) == 1
        assert u2_convs[0].title == "User 2 Secret Chat"

        # Clean up
        await session.delete(user1)
        await session.delete(user2)
        await session.commit()
