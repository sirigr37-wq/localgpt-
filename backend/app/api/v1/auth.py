from typing import Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.user import User
from app.schemas.user import (
    UserCreate,
    UserLogin,
    UserResponse,
    GoogleAuthUrlResponse,
    GoogleCallbackRequest,
    PasswordResetRequest,
    PasswordResetConfirm,
    PasswordResetResponse,
)
from app.schemas.token import Token
from app.services.auth_service import AuthService, get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication & OAuth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_in: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user account with email and password."""
    user = await AuthService.register_user(db, user_in)
    return user


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, db: AsyncSession = Depends(get_db)):
    """Authenticate with email and password and return a JWT access token."""
    user = await AuthService.authenticate_user(db, credentials)
    return AuthService.create_user_token(user)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get profile of the currently authenticated user."""
    return current_user


@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout(current_user: User = Depends(get_current_user)):
    """
    Logout user session.
    Instructs client to remove local JWT session token.
    """
    return {
        "status": "success",
        "message": f"Successfully logged out user {current_user.email}.",
    }


# -----------------------------------------------------------------------------
# Google OAuth 2.0 Endpoints
# -----------------------------------------------------------------------------
@router.get("/google/url", response_model=GoogleAuthUrlResponse)
async def get_google_auth_url(redirect_uri: Optional[str] = Query(None)):
    """
    Retrieve Google OAuth 2.0 authorization URL.
    Returns status indicating if Google OAuth credentials are configured on server.
    """
    return AuthService.get_google_auth_url(redirect_uri=redirect_uri)


@router.post("/google/callback", response_model=Token)
async def google_callback(
    callback_data: GoogleCallbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Exchange Google OAuth code for tokens, retrieve user info,
    and create or link the user account, returning a session JWT.
    """
    return await AuthService.process_google_callback(
        db=db,
        code=callback_data.code,
        redirect_uri=callback_data.redirect_uri,
    )


# -----------------------------------------------------------------------------
# Password Reset Flow Endpoints
# -----------------------------------------------------------------------------
@router.post("/password-reset/request", response_model=PasswordResetResponse)
async def request_password_reset(
    reset_req: PasswordResetRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Initiate a password reset flow.
    Generates a secure 15-minute signed reset token.
    """
    token = await AuthService.request_password_reset(db, email=reset_req.email)
    return PasswordResetResponse(
        message="If this email is registered, password reset instructions have been dispatched.",
        reset_token=token,  # Provided for development and testing verification
    )


@router.post("/password-reset/confirm", status_code=status.HTTP_200_OK)
async def confirm_password_reset(
    confirm_data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db),
):
    """
    Complete password reset with valid reset token and new password.
    """
    await AuthService.confirm_password_reset(
        db=db,
        token=confirm_data.token,
        new_password=confirm_data.new_password,
    )
    return {
        "status": "success",
        "message": "Password has been successfully updated. You may now log in.",
    }
