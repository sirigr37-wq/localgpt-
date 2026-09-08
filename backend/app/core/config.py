import os
import json
from typing import List, Optional, Any, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "LocalGPT Phase 3 API"
    VERSION: str = "3.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Database connection URL
    # Defaults to PostgreSQL with asyncpg. For local testing without a live PostgreSQL instance,
    # SQLite async can be provided via DATABASE_URL="sqlite+aiosqlite:///./localgpt.db"
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/localgpt_phase3"

    # JWT & Authentication
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-insecure-secret-key-change-in-prod-localgpt")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # CORS origins (supports JSON array or comma-separated URLs or '*' in production env)
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            if v.strip() == "*":
                return ["*"]
            if v.strip().startswith("[") and v.strip().endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        elif isinstance(v, (list, tuple, set)):
            return [str(origin) for origin in v]
        return v

    # Google OAuth (Foundation placeholders)
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GOOGLE_REDIRECT_URI: Optional[str] = "http://localhost:3000/auth/callback/google"

    # Hosted LLM Endpoint (Foundation placeholders)
    LLM_PROVIDER: str = "hosted_endpoint"
    HOSTED_LLM_API_BASE: Optional[str] = None
    HOSTED_LLM_API_KEY: Optional[str] = None
    HOSTED_LLM_MODEL: str = "gpt-4o-mini"

    # Document & Vector Storage
    UPLOAD_STORAGE_DIR: str = os.path.join(".", "data", "documents")
    FAISS_INDEX_DIR: str = os.path.join(".", "data", "vector_store")
    MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB max document upload size

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
