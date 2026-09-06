from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import urllib.parse
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.security import hash_password, verify_password, create_access_token, decode_access_token
from app.core.config import settings
from app.db.session import get_db
from app.models.user import User, OAuthAccount
from app.schemas.user import (
    UserCreate,
    UserLogin,
    GoogleAuthUrlResponse,
    PasswordResetRequest,
    PasswordResetConfirm,
    PasswordResetResponse,
)
from app.schemas.token import Token

security_scheme = HTTPBearer(auto_error=False)


class AuthService:
    """Authentication service handling user registration, authentication, OAuth, and JWTs."""

    @staticmethod
    async def register_user(db: AsyncSession, user_in: UserCreate) -> User:
        stmt = select(User).where(User.email == user_in.email)
        result = await db.execute(stmt)
        existing_user = result.scalar_one_or_none()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A user with this email already exists.",
            )

        new_user = User(
            email=user_in.email,
            hashed_password=hash_password(user_in.password),
            full_name=user_in.full_name,
            is_active=True,
            is_verified=False,
        )
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        return new_user

    @staticmethod
    async def authenticate_user(db: AsyncSession, credentials: UserLogin) -> User:
        stmt = select(User).where(User.email == credentials.email)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user or not user.hashed_password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
            )

        if not verify_password(credentials.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled.",
            )

        return user

    @staticmethod
    def create_user_token(user: User) -> Token:
        access_token = create_access_token(
            subject=user.id,
            extra_claims={"email": user.email, "full_name": user.full_name},
        )
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        )

    # -------------------------------------------------------------------------
    # Google OAuth 2.0 Flow & Account Linking
    # -------------------------------------------------------------------------
    @staticmethod
    def get_google_auth_url(redirect_uri: Optional[str] = None) -> GoogleAuthUrlResponse:
        """
        Generate Google OAuth 2.0 authorization URL.
        Checks if GOOGLE_CLIENT_ID is configured in environment variables.
        """
        client_id = settings.GOOGLE_CLIENT_ID.strip() if settings.GOOGLE_CLIENT_ID else None
        if not client_id or "your-google-client-id" in client_id:
            return GoogleAuthUrlResponse(
                configured=False,
                auth_url=None,
                message=(
                    "Google OAuth credentials are not configured. "
                    "Please set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI in backend/.env"
                ),
            )

        target_redirect = (redirect_uri or settings.GOOGLE_REDIRECT_URI or "http://localhost:3000/auth/callback/google").strip()
        params = {
            "client_id": client_id,
            "redirect_uri": target_redirect,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
            "prompt": "select_account consent",
        }
        encoded_query = urllib.parse.urlencode(params)
        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{encoded_query}"

        return GoogleAuthUrlResponse(
            configured=True,
            auth_url=auth_url,
            message="Google OAuth authorization URL generated successfully.",
        )

    @staticmethod
    async def link_or_create_google_user(
        db: AsyncSession,
        google_info: Dict[str, Any],
        tokens: Dict[str, Any],
    ) -> User:
        """
        Create a new user or link to an existing user account following Google authentication.
        1. Checks if an OAuthAccount already exists for provider='google' and google_sub.
        2. If not found, checks if a User exists with matching email.
        3. If user exists, links the Google OAuthAccount to that user.
        4. If user does not exist, creates User and links OAuthAccount.
        """
        google_sub = str(google_info.get("sub", ""))
        email = google_info.get("email", "")
        name = google_info.get("name") or google_info.get("given_name") or "Google User"
        picture = google_info.get("picture")

        if not google_sub or not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Google account did not provide required profile identifier or email.",
            )

        # 1. Check existing OAuthAccount
        stmt = select(OAuthAccount).where(
            OAuthAccount.provider == "google",
            OAuthAccount.provider_user_id == google_sub,
        )
        oauth_res = await db.execute(stmt)
        oauth_account = oauth_res.scalar_one_or_none()

        if oauth_account:
            # Existing linked account
            user_stmt = select(User).where(User.id == oauth_account.user_id)
            user_res = await db.execute(user_stmt)
            user = user_res.scalar_one_or_none()
            if not user or not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User account is inactive or disabled.",
                )

            # Update access tokens
            oauth_account.access_token = tokens.get("access_token")
            oauth_account.refresh_token = tokens.get("refresh_token") or oauth_account.refresh_token
            db.add(oauth_account)
            await db.commit()
            await db.refresh(user)
            return user

        # 2. Check if a User with this email already exists
        user_stmt = select(User).where(User.email == email)
        user_res = await db.execute(user_stmt)
        existing_user = user_res.scalar_one_or_none()

        if existing_user:
            user = existing_user
            if picture and not user.avatar_url:
                user.avatar_url = picture
            user.is_verified = True
            db.add(user)
        else:
            # 3. Create brand new User
            user = User(
                email=email,
                hashed_password=None,  # OAuth users have no password initially
                full_name=name,
                avatar_url=picture,
                is_active=True,
                is_verified=True,
            )
            db.add(user)
            await db.flush()  # Flush to obtain user.id

        # 4. Create and link OAuthAccount
        new_oauth = OAuthAccount(
            user_id=user.id,
            provider="google",
            provider_user_id=google_sub,
            access_token=tokens.get("access_token"),
            refresh_token=tokens.get("refresh_token"),
        )
        db.add(new_oauth)
        await db.commit()
        await db.refresh(user)
        return user

    @staticmethod
    async def process_google_callback(
        db: AsyncSession,
        code: str,
        redirect_uri: Optional[str] = None,
    ) -> Token:
        """
        Exchange Google OAuth authorization code for tokens and authenticate user.
        """
        client_id = settings.GOOGLE_CLIENT_ID.strip() if settings.GOOGLE_CLIENT_ID else None
        client_secret = settings.GOOGLE_CLIENT_SECRET.strip() if settings.GOOGLE_CLIENT_SECRET else None
        target_redirect = (redirect_uri or settings.GOOGLE_REDIRECT_URI or "http://localhost:3000/auth/callback/google").strip()
        code = code.strip()

        if not client_id or not client_secret or "your-google" in client_id:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Google OAuth is not configured on the server. "
                    "Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in environment."
                ),
            )

        token_url = "https://oauth2.googleapis.com/token"
        token_payload = {
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": target_redirect,
            "grant_type": "authorization_code",
        }

        async with httpx.AsyncClient() as client:
            try:
                token_resp = await client.post(token_url, data=token_payload, timeout=10.0)
                if token_resp.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Google token exchange failed: {token_resp.text}",
                    )
                tokens = token_resp.json()

                # Fetch user info
                userinfo_url = "https://www.googleapis.com/oauth2/v3/userinfo"
                headers = {"Authorization": f"Bearer {tokens['access_token']}"}
                userinfo_resp = await client.get(userinfo_url, headers=headers, timeout=10.0)
                if userinfo_resp.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Failed to retrieve Google user profile.",
                    )
                google_info = userinfo_resp.json()
            except httpx.RequestError as exc:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Network error communicating with Google OAuth: {str(exc)}",
                )

        user = await AuthService.link_or_create_google_user(db, google_info, tokens)
        return AuthService.create_user_token(user)

    # -------------------------------------------------------------------------
    # Password Reset Flow
    # -------------------------------------------------------------------------
    @staticmethod
    async def request_password_reset(db: AsyncSession, email: str) -> Optional[str]:
        """
        Generate a signed password-reset token for the given user email (valid for 15 mins).
        """
        stmt = select(User).where(User.email == email)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            # Return None silently to prevent email enumeration
            return None

        # Create signed token with purpose claim
        reset_token = create_access_token(
            subject=user.id,
            expires_delta=timedelta(minutes=15),
            extra_claims={"purpose": "password_reset", "email": user.email},
        )
        return reset_token

    @staticmethod
    async def confirm_password_reset(
        db: AsyncSession,
        token: str,
        new_password: str,
    ) -> bool:
        """
        Validate password reset token and update user's password.
        """
        payload = decode_access_token(token)
        if not payload or payload.get("purpose") != "password_reset" or "sub" not in payload:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid, expired, or malformed password reset token.",
            )

        user_id = payload["sub"]
        stmt = select(User).where(User.id == user_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User account not found or disabled.",
            )

        user.hashed_password = hash_password(new_password)
        db.add(user)
        await db.commit()
        return True


async def get_current_user(
    auth_header: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Dependency that extracts and validates the authenticated User from the JWT Bearer token."""
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a Bearer token.",
        )

    payload = decode_access_token(auth_header.credentials)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )

    user_id = payload["sub"]
    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive.",
        )

    return user
