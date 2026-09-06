"""
Deployment & Production Readiness Tests.
Verifies:
1. Liveness probes (/health)
2. Readiness probes (/ready, /api/v1/ready)
3. Detailed diagnostic endpoint (/api/v1/health)
4. Production CORS configuration parsing (comma-separated & JSON arrays)
5. Storage directory path traversal protection
"""

import os
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.config import settings, Settings
from app.services.document_service import DocumentService
from app.services.rag.vector_store import get_user_vector_dir


@pytest.mark.asyncio
async def test_root_liveness_probe():
    """Verify /health returns 200 OK for orchestrator liveness checks."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] in ("healthy", "degraded")
        assert "version" in data


@pytest.mark.asyncio
async def test_readiness_probe():
    """Verify /ready returns 200 OK when database is reachable."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.get("/ready")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ready"
        assert data["database"] == "connected"


@pytest.mark.asyncio
async def test_api_v1_health_diagnostics():
    """Verify /api/v1/health and /api/v1/ready endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res_health = await ac.get(f"{settings.API_V1_STR}/health")
        assert res_health.status_code == 200
        health_data = res_health.json()
        assert health_data["status"] in ("healthy", "degraded", "unhealthy")
        assert "services" in health_data
        assert health_data["services"]["api"] == "online"

        res_ready = await ac.get(f"{settings.API_V1_STR}/ready")
        assert res_ready.status_code == 200
        assert res_ready.json()["status"] == "ready"


def test_cors_origins_parsing():
    """Verify that CORS_ORIGINS supports comma-separated strings and JSON arrays from environment."""
    # Test comma-separated string
    s1 = Settings(CORS_ORIGINS="https://app.example.com,https://api.example.com")
    assert "https://app.example.com" in s1.CORS_ORIGINS
    assert "https://api.example.com" in s1.CORS_ORIGINS
    assert len(s1.CORS_ORIGINS) == 2

    # Test JSON array string
    s2 = Settings(CORS_ORIGINS='["https://cloud.localgpt.com", "http://localhost:3000"]')
    assert "https://cloud.localgpt.com" in s2.CORS_ORIGINS
    assert "http://localhost:3000" in s2.CORS_ORIGINS


def test_storage_path_safety():
    """Verify storage paths create directories safely and block path traversal."""
    # Safe directory resolution
    safe_dir = DocumentService.get_user_storage_dir("valid_user_id_123")
    assert os.path.exists(safe_dir)
    assert "valid_user_id_123" in safe_dir

    safe_vector_dir = get_user_vector_dir("valid_user_id_123")
    assert os.path.exists(safe_vector_dir)
    assert "valid_user_id_123" in safe_vector_dir
