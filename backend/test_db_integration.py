import asyncio
import uuid
from datetime import datetime, timezone
from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select

from app.core.config import settings
from app.db.session import engine, AsyncSessionLocal
from app.models.user import User, OAuthAccount
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.document import Document


async def verify_database_schema_and_integrity():
    print("=" * 60)
    print("PHASE 3 DATABASE SCHEMA & RELATIONSHIP VERIFICATION")
    print("=" * 60)

    # 1. Verify all required tables exist
    async with engine.connect() as conn:
        def get_tables(sync_conn):
            inspector = inspect(sync_conn)
            return inspector.get_table_names()

        tables = await conn.run_sync(get_tables)
        print(f"Discovered Database Tables: {tables}")
        
        required_tables = {"users", "oauth_accounts", "conversations", "messages", "documents", "alembic_version"}
        missing_tables = required_tables - set(tables)
        assert not missing_tables, f"Missing required tables: {missing_tables}"
        print("[PASS] All 5 required domain tables + alembic_version verified present.")

    # 2. Test CRUD, UUID Primary Keys, Foreign Keys, and Cascade Delete
    async with AsyncSessionLocal() as session:
        # A. Create User with UUID
        test_user_id = str(uuid.uuid4())
        user = User(
            id=test_user_id,
            email=f"pg_test_{uuid.uuid4().hex[:6]}@localgpt.ai",
            hashed_password="test_hashed_password",
            full_name="PostgreSQL Integration Tester",
        )
        session.add(user)
        await session.commit()
        print(f"[PASS] Created User with UUID PK: {user.id} ({user.email})")

        # B. Create OAuth Account linked to User
        oauth = OAuthAccount(
            id=str(uuid.uuid4()),
            user_id=user.id,
            provider="google",
            provider_user_id="google-sub-998877",
            access_token="mock_access_token",
        )
        session.add(oauth)
        await session.commit()
        print(f"[PASS] Created OAuthAccount linked to user: {oauth.provider} (user_id={oauth.user_id})")

        # C. Create Conversation linked to User
        conv = Conversation(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title="Database Integration Test Conversation",
            system_prompt="You are verifying database relationships.",
        )
        session.add(conv)
        await session.commit()
        print(f"[PASS] Created Conversation linked to user: {conv.id} (user_id={conv.user_id})")

        # D. Create Message linked to Conversation
        msg1 = Message(
            id=str(uuid.uuid4()),
            conversation_id=conv.id,
            role="user",
            content="Testing message order and relationships in Phase 3 DB",
            message_order=1,
        )
        msg2 = Message(
            id=str(uuid.uuid4()),
            conversation_id=conv.id,
            role="assistant",
            content="Database relationships and cascade foreign keys are active.",
            message_order=2,
            feedback="like",
            sources_json={"sources": [{"filename": "spec.pdf", "page": 1, "score": 0.95}]},
        )
        session.add_all([msg1, msg2])
        await session.commit()
        print(f"[PASS] Created 2 Messages linked to Conversation: orders {[msg1.message_order, msg2.message_order]}")

        # E. Create Document linked to User
        doc = Document(
            id=str(uuid.uuid4()),
            user_id=user.id,
            filename="architecture_diagram.pdf",
            filepath="./data/documents/architecture_diagram.pdf",
            file_type="PDF",
            file_size_bytes=102400,
            num_pages=5,
            chunk_count=12,
            status="processed",
        )
        session.add(doc)
        await session.commit()
        print(f"[PASS] Created Document linked to user: {doc.filename} (user_id={doc.user_id})")

        # F. Test User-Specific Data Isolation
        user_convs = (await session.execute(select(Conversation).where(Conversation.user_id == user.id))).scalars().all()
        user_docs = (await session.execute(select(Document).where(Document.user_id == user.id))).scalars().all()
        assert len(user_convs) == 1
        assert len(user_docs) == 1
        print("[PASS] User-specific data isolation query verified.")

        # G. Test Cascade Deletion
        # Delete user -> should cascade to oauth_accounts, conversations, messages, documents
        await session.delete(user)
        await session.commit()
        print("[PASS] Executed user deletion to test CASCADE handling.")

        # Verify cascades
        remaining_oauth = (await session.execute(select(OAuthAccount).where(OAuthAccount.user_id == test_user_id))).scalars().all()
        remaining_convs = (await session.execute(select(Conversation).where(Conversation.user_id == test_user_id))).scalars().all()
        remaining_msgs = (await session.execute(select(Message).where(Message.conversation_id == conv.id))).scalars().all()
        remaining_docs = (await session.execute(select(Document).where(Document.user_id == test_user_id))).scalars().all()

        assert len(remaining_oauth) == 0, "OAuthAccount not cascaded"
        assert len(remaining_convs) == 0, "Conversation not cascaded"
        assert len(remaining_msgs) == 0, "Messages not cascaded"
        assert len(remaining_docs) == 0, "Document not cascaded"
        print("[PASS] All cascades verified: OAuth accounts, conversations, messages, and documents cleanly removed.")

    print("=" * 60)
    print("ALL PHASE 3 DATABASE INTEGRITY TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(verify_database_schema_and_integrity())
