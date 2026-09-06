import pytest
import uuid
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.config import settings
from app.services.auth_service import AuthService
from app.db.session import AsyncSessionLocal
from app.models.conversation import Conversation
from app.models.message import Message


from app.schemas.user import UserCreate

async def get_test_users():
    """Create test users and generate JWT authentication headers."""
    async with AsyncSessionLocal() as session:
        # User 1
        user1_email = f"chat_user1_{uuid.uuid4().hex[:8]}@example.com"
        user1 = await AuthService.register_user(
            session, UserCreate(email=user1_email, password="Password123!", full_name="User One")
        )
        token1 = AuthService.create_user_token(user1).access_token

        # User 2 (for isolation tests)
        user2_email = f"chat_user2_{uuid.uuid4().hex[:8]}@example.com"
        user2 = await AuthService.register_user(
            session, UserCreate(email=user2_email, password="Password123!", full_name="User Two")
        )
        token2 = AuthService.create_user_token(user2).access_token

    return {
        "user1": {"headers": {"Authorization": f"Bearer {token1}"}, "id": user1.id, "email": user1_email},
        "user2": {"headers": {"Authorization": f"Bearer {token2}"}, "id": user2.id, "email": user2_email},
    }


@pytest.mark.asyncio
async def test_conversation_crud_and_search():
    auth_headers = await get_test_users()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        headers1 = auth_headers["user1"]["headers"]

        # 1. Create a conversation
        create_res = await ac.post(
            f"{settings.API_V1_STR}/conversations",
            json={"title": "Deep Learning Chat", "system_prompt": "You are a PyTorch expert."},
            headers=headers1,
        )
        assert create_res.status_code == 201
        conv_data = create_res.json()
        conv_id = conv_data["id"]
        assert conv_data["title"] == "Deep Learning Chat"
        assert conv_data["user_id"] == auth_headers["user1"]["id"]

        # 2. Get conversation
        get_res = await ac.get(f"{settings.API_V1_STR}/conversations/{conv_id}", headers=headers1)
        assert get_res.status_code == 200
        assert get_res.json()["title"] == "Deep Learning Chat"
        assert len(get_res.json()["messages"]) == 0

        # 3. Rename conversation
        rename_res = await ac.put(
            f"{settings.API_V1_STR}/conversations/{conv_id}",
            json={"title": "Transformers & Attention"},
            headers=headers1,
        )
        assert rename_res.status_code == 200
        assert rename_res.json()["title"] == "Transformers & Attention"

        # 4. Search conversations
        search_res = await ac.get(
            f"{settings.API_V1_STR}/conversations?q=Transformers",
            headers=headers1,
        )
        assert search_res.status_code == 200
        items = search_res.json()
        assert len(items) >= 1
        assert any(c["id"] == conv_id for c in items)

        # Search negative
        search_neg = await ac.get(
            f"{settings.API_V1_STR}/conversations?q=NonExistentKeywordXYZ",
            headers=headers1,
        )
        assert search_neg.status_code == 200
        assert not any(c["id"] == conv_id for c in search_neg.json())

        # 5. Delete conversation
        del_res = await ac.delete(f"{settings.API_V1_STR}/conversations/{conv_id}", headers=headers1)
        assert del_res.status_code == 204

        # Confirm 404 after delete
        get_deleted = await ac.get(f"{settings.API_V1_STR}/conversations/{conv_id}", headers=headers1)
        assert get_deleted.status_code == 404


@pytest.mark.asyncio
async def test_multi_turn_messages_and_ordering():
    auth_headers = await get_test_users()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        headers1 = auth_headers["user1"]["headers"]

        # Create conversation
        create_res = await ac.post(
            f"{settings.API_V1_STR}/conversations",
            json={"title": "Multi-Turn Chat"},
            headers=headers1,
        )
        conv_id = create_res.json()["id"]

        # Post Turn 1: User message via streaming
        stream1 = await ac.post(
            f"{settings.API_V1_STR}/chat/{conv_id}/stream",
            json={"role": "user", "content": "What is self-attention?"},
            headers=headers1,
        )
        assert stream1.status_code == 200
        content1 = stream1.text
        assert "[DONE]" in content1

        # Post Turn 2: Follow-up question
        stream2 = await ac.post(
            f"{settings.API_V1_STR}/chat/{conv_id}/stream",
            json={"role": "user", "content": "How is multi-head attention related to it?"},
            headers=headers1,
        )
        assert stream2.status_code == 200
        content2 = stream2.text
        assert "[DONE]" in content2

        # Verify full conversation detail and message ordering
        detail_res = await ac.get(f"{settings.API_V1_STR}/conversations/{conv_id}", headers=headers1)
        assert detail_res.status_code == 200
        detail = detail_res.json()
        messages = detail["messages"]
        assert len(messages) == 4  # User1, Assistant1, User2, Assistant2

        # Check strict sequential ordering
        for i, msg in enumerate(messages):
            assert msg["message_order"] == i + 1

        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is self-attention?"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "How is multi-head attention related to it?"
        assert messages[3]["role"] == "assistant"


@pytest.mark.asyncio
async def test_user_conversation_isolation():
    auth_headers = await get_test_users()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        headers1 = auth_headers["user1"]["headers"]
        headers2 = auth_headers["user2"]["headers"]

        # User 1 creates conversation
        conv_res = await ac.post(
            f"{settings.API_V1_STR}/conversations",
            json={"title": "User 1 Secret Discussion"},
            headers=headers1,
        )
        conv_id = conv_res.json()["id"]

        # User 2 tries to GET User 1's conversation -> must fail with 404
        u2_get = await ac.get(f"{settings.API_V1_STR}/conversations/{conv_id}", headers=headers2)
        assert u2_get.status_code == 404

        # User 2 tries to rename User 1's conversation -> must fail with 404
        u2_rename = await ac.put(
            f"{settings.API_V1_STR}/conversations/{conv_id}",
            json={"title": "Hacked Title"},
            headers=headers2,
        )
        assert u2_rename.status_code == 404

        # User 2 tries to post message to User 1's conversation -> must fail with 404
        u2_post = await ac.post(
            f"{settings.API_V1_STR}/chat/{conv_id}/messages",
            json={"role": "user", "content": "Unauthorized message injection"},
            headers=headers2,
        )
        assert u2_post.status_code == 404

        # User 2 tries to stream in User 1's conversation -> must fail with 404
        u2_stream = await ac.post(
            f"{settings.API_V1_STR}/chat/{conv_id}/stream",
            json={"role": "user", "content": "Unauthorized stream"},
            headers=headers2,
        )
        assert u2_stream.status_code == 404

        # User 2 list conversations -> does not include User 1's conversation
        u2_list = await ac.get(f"{settings.API_V1_STR}/conversations", headers=headers2)
        assert u2_list.status_code == 200
        assert not any(c["id"] == conv_id for c in u2_list.json())

        # User 2 delete attempt -> must fail with 404
        u2_del = await ac.delete(f"{settings.API_V1_STR}/conversations/{conv_id}", headers=headers2)
        assert u2_del.status_code == 404


@pytest.mark.asyncio
async def test_message_edit_truncate_and_regenerate_flow():
    auth_headers = await get_test_users()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        headers = auth_headers["user1"]["headers"]

        # 1. Create conversation
        conv_res = await ac.post(
            f"{settings.API_V1_STR}/conversations",
            json={"title": "Edit & Regenerate Test"},
            headers=headers,
        )
        assert conv_res.status_code == 201
        conv_id = conv_res.json()["id"]

        # 2. Stream original turn
        s1 = await ac.post(
            f"{settings.API_V1_STR}/chat/{conv_id}/stream",
            json={"role": "user", "content": "Original user query"},
            headers=headers,
        )
        assert s1.status_code == 200
        assert "[DONE]" in s1.text

        # 3. Check conversation detail (1 user + 1 assistant)
        d1 = await ac.get(f"{settings.API_V1_STR}/conversations/{conv_id}", headers=headers)
        assert d1.status_code == 200
        msgs = d1.json()["messages"]
        assert len(msgs) == 2
        user_msg_id = msgs[0]["id"]
        asst_msg_id = msgs[1]["id"]

        # 4. Test feedback
        fb_res = await ac.patch(
            f"{settings.API_V1_STR}/chat/messages/{asst_msg_id}/feedback",
            json={"feedback": "like"},
            headers=headers,
        )
        assert fb_res.status_code == 200
        assert fb_res.json()["feedback"] == "like"

        # 5. Edit user message (which also truncates assistant message)
        edit_res = await ac.put(
            f"{settings.API_V1_STR}/chat/{conv_id}/messages/{user_msg_id}",
            json={"role": "user", "content": "Updated user query"},
            headers=headers,
        )
        assert edit_res.status_code == 200
        assert edit_res.json()["content"] == "Updated user query"

        # Check detail: assistant message was truncated
        d2 = await ac.get(f"{settings.API_V1_STR}/conversations/{conv_id}", headers=headers)
        msgs2 = d2.json()["messages"]
        assert len(msgs2) == 1
        assert msgs2[0]["content"] == "Updated user query"

        # 6. Regenerate response for the edited query
        s2 = await ac.post(
            f"{settings.API_V1_STR}/chat/{conv_id}/stream?regenerate=true",
            json={"role": "user", "content": "Updated user query"},
            headers=headers,
        )
        assert s2.status_code == 200
        assert "[DONE]" in s2.text

        # Check detail: fresh assistant message persisted
        d3 = await ac.get(f"{settings.API_V1_STR}/conversations/{conv_id}", headers=headers)
        msgs3 = d3.json()["messages"]
        assert len(msgs3) == 2
        assert msgs3[0]["content"] == "Updated user query"
        assert msgs3[1]["role"] == "assistant"

