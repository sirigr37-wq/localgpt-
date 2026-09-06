import pytest
import json
import httpx
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.config import settings
from app.services.llm.provider import (
    HostedLLMProvider,
    LLMConfigurationError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMProviderError,
)
from app.services.auth_service import AuthService
from app.schemas.user import UserCreate
from app.db.session import AsyncSessionLocal


@pytest.mark.asyncio
async def test_unconfigured_hosted_provider():
    """Verify provider reports unconfigured and raises LLMConfigurationError without inventing keys."""
    provider = HostedLLMProvider(api_key="")
    assert provider.is_configured is False

    with pytest.raises(LLMConfigurationError) as exc_info:
        await provider.generate_response([{"role": "user", "content": "Hi"}])
    assert "HOSTED_LLM_API_KEY is not configured" in str(exc_info.value)

    with pytest.raises(LLMConfigurationError) as exc_info:
        async for _ in provider.stream_response([{"role": "user", "content": "Hi"}]):
            pass
    assert "HOSTED_LLM_API_KEY is not configured" in str(exc_info.value)


@pytest.mark.asyncio
async def test_openai_compatible_mock_streaming():
    """Verify live OpenAI-compatible SSE chunk parsing from mock HTTP stream."""
    provider = HostedLLMProvider(
        api_base="https://mock.llm.provider/v1",
        api_key="sk-mock-valid-key",
        model_name="mock-model",
    )
    assert provider.is_configured is True

    # Simulated SSE lines returned by an OpenAI-compatible endpoint
    mock_sse_lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" from"}}]}',
        'data: {"choices":[{"delta":{"content":" hosted"}}]}',
        'data: {"choices":[{"delta":{"content":" LLM!"}}]}',
        'data: [DONE]',
    ]

    class MockStreamContext:
        def __init__(self, status_code=200):
            self.status_code = status_code

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def aiter_lines(self):
            for line in mock_sse_lines:
                yield line

    with patch("httpx.AsyncClient.stream", return_value=MockStreamContext(200)):
        tokens = []
        async for chunk in provider.stream_response([{"role": "user", "content": "Hello"}]):
            tokens.append(chunk)

        assert tokens == ["Hello", " from", " hosted", " LLM!"]


@pytest.mark.asyncio
async def test_openai_compatible_mock_generate():
    """Verify non-streaming completion parsing."""
    provider = HostedLLMProvider(
        api_base="https://mock.llm.provider/v1",
        api_key="sk-mock-valid-key",
        model_name="mock-model",
    )

    mock_resp = httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": "Complete mock response."}}]},
        request=httpx.Request("POST", "https://mock.llm.provider/v1/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", AsyncMock(return_value=mock_resp)):
        result = await provider.generate_response([{"role": "user", "content": "Tell me a joke"}])
        assert result == "Complete mock response."


@pytest.mark.asyncio
async def test_hosted_llm_authentication_error():
    """Verify 401 response translates into LLMAuthenticationError."""
    provider = HostedLLMProvider(
        api_base="https://mock.llm.provider/v1",
        api_key="sk-invalid-key",
        model_name="mock-model",
    )

    class MockStream401:
        def __init__(self):
            self.status_code = 401

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("httpx.AsyncClient.stream", return_value=MockStream401()):
        with pytest.raises(LLMAuthenticationError):
            async for _ in provider.stream_response([{"role": "user", "content": "Test"}]):
                pass


@pytest.mark.asyncio
async def test_hosted_llm_rate_limit_error():
    """Verify 429 response translates into LLMRateLimitError."""
    provider = HostedLLMProvider(
        api_base="https://mock.llm.provider/v1",
        api_key="sk-rate-limited-key",
        model_name="mock-model",
    )

    class MockStream429:
        def __init__(self):
            self.status_code = 429

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("httpx.AsyncClient.stream", return_value=MockStream429()):
        with pytest.raises(LLMRateLimitError):
            async for _ in provider.stream_response([{"role": "user", "content": "Test"}]):
                pass


@pytest.mark.asyncio
async def test_hosted_llm_timeout_error():
    """Verify httpx.TimeoutException translates into LLMTimeoutError."""
    provider = HostedLLMProvider(
        api_base="https://mock.llm.provider/v1",
        api_key="sk-timeout-key",
        model_name="mock-model",
    )

    with patch("httpx.AsyncClient.stream", side_effect=httpx.ReadTimeout("Read timed out")):
        with pytest.raises(LLMTimeoutError):
            async for _ in provider.stream_response([{"role": "user", "content": "Test"}]):
                pass


@pytest.mark.asyncio
async def test_chat_endpoint_graceful_configuration_fallback():
    """Verify chat SSE endpoint streams configuration notice when HOSTED_LLM_API_KEY is not set."""
    import uuid
    email = f"llm_fallback_{uuid.uuid4().hex[:8]}@example.com"
    async with AsyncSessionLocal() as session:
        user = await AuthService.register_user(
            session, UserCreate(email=email, password="Password123!", full_name="LLM Tester")
        )
        token = AuthService.create_user_token(user).access_token

    transport = ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {token}"}

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Create conversation
        conv_res = await ac.post(
            f"{settings.API_V1_STR}/conversations",
            json={"title": "Unconfigured LLM Test"},
            headers=headers,
        )
        assert conv_res.status_code == 201
        conv_id = conv_res.json()["id"]

        # Call stream chat endpoint without configured key
        stream_res = await ac.post(
            f"{settings.API_V1_STR}/chat/{conv_id}/stream",
            json={"role": "user", "content": "Hello AI"},
            headers=headers,
        )
        assert stream_res.status_code == 200
        content = stream_res.text
        assert "[LLM Configuration Notice]" in content
        assert "[DONE]" in content
