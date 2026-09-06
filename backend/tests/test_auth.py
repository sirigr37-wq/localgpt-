import pytest
import uuid
from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.models.user import User, OAuthAccount
from app.schemas.user import UserCreate, UserLogin
from app.services.auth_service import AuthService
from app.core.security import verify_password, decode_access_token


@pytest.mark.asyncio
async def test_register_and_authenticate_user():
    """Verify user registration, password hashing, and authentication."""
    async with AsyncSessionLocal() as session:
        email = f"auth_test_{uuid.uuid4().hex[:6]}@example.com"
        password = "SecurePassword123!"

        # Register
        user_in = UserCreate(email=email, password=password, full_name="Auth Tester")
        user = await AuthService.register_user(session, user_in)

        assert user.id is not None
        assert user.email == email
        assert user.hashed_password != password
        assert verify_password(password, user.hashed_password) is True

        # Login with correct password
        creds = UserLogin(email=email, password=password)
        authenticated_user = await AuthService.authenticate_user(session, creds)
        assert authenticated_user.id == user.id

        # Generate token
        token = AuthService.create_user_token(authenticated_user)
        assert token.access_token is not None
        assert token.token_type == "bearer"

        # Decode token
        payload = decode_access_token(token.access_token)
        assert payload is not None
        assert payload["sub"] == user.id
        assert payload["email"] == email

        # Clean up
        await session.delete(user)
        await session.commit()


@pytest.mark.asyncio
async def test_password_reset_flow():
    """Verify requesting reset token, confirming new password, and logging in with new password."""
    async with AsyncSessionLocal() as session:
        email = f"reset_test_{uuid.uuid4().hex[:6]}@example.com"
        old_password = "OldPassword123!"
        new_password = "NewPassword456!"

        # Create user
        user_in = UserCreate(email=email, password=old_password, full_name="Reset Tester")
        user = await AuthService.register_user(session, user_in)

        # Request reset token
        reset_token = await AuthService.request_password_reset(session, email)
        assert reset_token is not None

        payload = decode_access_token(reset_token)
        assert payload is not None
        assert payload.get("purpose") == "password_reset"
        assert payload.get("sub") == user.id

        # Confirm reset with new password
        success = await AuthService.confirm_password_reset(session, reset_token, new_password)
        assert success is True

        # Verify old password fails
        with pytest.raises(Exception):
            await AuthService.authenticate_user(session, UserLogin(email=email, password=old_password))

        # Verify new password succeeds
        authed = await AuthService.authenticate_user(session, UserLogin(email=email, password=new_password))
        assert authed.id == user.id

        # Clean up
        await session.delete(user)
        await session.commit()


@pytest.mark.asyncio
async def test_google_oauth_linking_and_creation():
    """Verify Google OAuth account creation and linking to existing users."""
    async with AsyncSessionLocal() as session:
        google_sub = f"google-sub-{uuid.uuid4().hex[:8]}"
        email = f"google_user_{uuid.uuid4().hex[:6]}@example.com"
        google_info = {
            "sub": google_sub,
            "email": email,
            "name": "Google Test User",
            "picture": "https://lh3.googleusercontent.com/a/mock-pic",
        }
        tokens = {
            "access_token": "mock_google_access_token",
            "refresh_token": "mock_google_refresh_token",
        }

        # 1. First time Google login -> Creates new User and OAuthAccount
        user = await AuthService.link_or_create_google_user(session, google_info, tokens)
        assert user.id is not None
        assert user.email == email
        assert user.is_verified is True
        assert user.avatar_url == "https://lh3.googleusercontent.com/a/mock-pic"

        # Verify OAuthAccount created
        stmt = select(OAuthAccount).where(OAuthAccount.provider_user_id == google_sub)
        oauth = (await session.execute(stmt)).scalar_one_or_none()
        assert oauth is not None
        assert oauth.user_id == user.id
        assert oauth.provider == "google"

        # 2. Subsequent Google login with same sub -> Reuses user and updates tokens
        tokens["access_token"] = "updated_google_access_token"
        same_user = await AuthService.link_or_create_google_user(session, google_info, tokens)
        assert same_user.id == user.id

        # 3. Test linking to existing password-registered user
        linked_email = f"password_user_{uuid.uuid4().hex[:6]}@example.com"
        existing_user = await AuthService.register_user(
            session, UserCreate(email=linked_email, password="password", full_name="Existing User")
        )
        new_google_sub = f"google-sub-{uuid.uuid4().hex[:8]}"
        link_google_info = {
            "sub": new_google_sub,
            "email": linked_email,
            "name": "Existing User",
            "picture": "https://lh3.googleusercontent.com/a/linked-pic",
        }
        linked_user = await AuthService.link_or_create_google_user(session, link_google_info, tokens)
        assert linked_user.id == existing_user.id

        # Clean up
        await session.delete(user)
        await session.delete(existing_user)
        await session.commit()


def test_google_auth_url_generator():
    """Verify Google OAuth authorization URL behavior when unconfigured or configured."""
    url_resp = AuthService.get_google_auth_url()
    assert isinstance(url_resp.configured, bool)
    if url_resp.configured:
        assert "accounts.google.com" in url_resp.auth_url
    else:
        assert url_resp.auth_url is None
        assert "GOOGLE_CLIENT_ID" in url_resp.message
