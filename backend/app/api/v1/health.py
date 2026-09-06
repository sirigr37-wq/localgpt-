from datetime import datetime, timezone
from fastapi import APIRouter, Response, status
from app.schemas.health import HealthCheckResponse
from app.core.config import settings
from app.db.session import check_database_health

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(response: Response):
    """Health check endpoint verifying backend service and database connectivity."""
    db_ok = await check_database_health()
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthCheckResponse(
        status="healthy" if db_ok else "unhealthy",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        database_connected=db_ok,
        timestamp=datetime.now(timezone.utc),
        services={
            "api": "online",
            "database": "online" if db_ok else "disconnected",
            "llm_provider": settings.LLM_PROVIDER,
            "rag_service": "initialized",
        },
    )


@router.get("/ready")
async def readiness_check(response: Response):
    """Readiness probe for orchestrators: returns 200 OK when DB is ready, 503 if not."""
    db_ok = await check_database_health()
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unready", "detail": "Database connection unavailable"}
    return {"status": "ready", "version": settings.VERSION, "database": "connected"}
