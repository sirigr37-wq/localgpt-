import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1 import api_router
from app.db.session import engine, check_database_health
from app.db.base import Base
import app.models  # Ensure models are registered with Base metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("localgpt.backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown routines."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT}]")
    
    # Initialize database tables
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schemas verified and initialized successfully.")
    except Exception as e:
        logger.warning(
            f"Database auto-creation encountered notice (normal if PostgreSQL server is external or requires migration): {e}"
        )

    yield

    logger.info("Shutting down backend services...")
    await engine.dispose()
    logger.info("Database engine connections disposed.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Phase 3 Multi-User FastAPI Backend with Persistent Chat, RAG, and Streaming",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS)
cors_origins = list(settings.CORS_ORIGINS)
if "*" in cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://.*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=r"^https:\/\/.*\.vercel\.app$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Root-level health check convenience endpoint (Liveness probe)
@app.get("/health", tags=["Health"])
async def root_health():
    """Liveness probe: returns 200 OK if the FastAPI process is running."""
    db_ok = await check_database_health()
    return {
        "status": "healthy" if db_ok else "degraded",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "database": "connected" if db_ok else "disconnected",
    }


# Root-level readiness check convenience endpoint (Readiness probe)
@app.get("/ready", tags=["Health"])
async def root_ready(response: Response):
    """Readiness probe for load balancers/orchestrators: returns 503 if DB is unreachable."""
    db_ok = await check_database_health()
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "unready",
            "version": settings.VERSION,
            "database": "disconnected",
        }
    return {
        "status": "ready",
        "version": settings.VERSION,
        "database": "connected",
    }

# Mount versioned API routes
app.include_router(api_router, prefix=settings.API_V1_STR)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
