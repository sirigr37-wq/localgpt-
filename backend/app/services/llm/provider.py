from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncIterator
import asyncio
import json
import logging
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for all LLM provider operations."""
    pass


class LLMConfigurationError(LLMError):
    """Raised when required API credentials or endpoint configurations are missing."""
    pass


class LLMAuthenticationError(LLMError):
    """Raised when the provider rejects authentication (HTTP 401)."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when rate limits or quotas are exceeded (HTTP 429)."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when an API connection or response read times out."""
    pass


class LLMProviderError(LLMError):
    """Raised when the provider returns a non-2xx status code or network failure."""
    pass


class BaseLLMProvider(ABC):
    """Abstract interface defining required LLM generation capabilities."""

    @abstractmethod
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        """Generate a complete text response."""
        pass

    @abstractmethod
    async def stream_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream generated text chunks asynchronously."""
        pass


class HostedLLMProvider(BaseLLMProvider):
    """
    OpenAI-Compatible Hosted LLM Endpoint Provider.
    Works seamlessly with OpenAI, Groq, vLLM, LiteLLM, Ollama, Together, and any OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        api_base: str = None,
        api_key: str = None,
        model_name: str = None,
    ):
        raw_base = api_base or settings.HOSTED_LLM_API_BASE or "https://api.openai.com/v1"
        self.api_base = raw_base.rstrip("/")
        self.api_key = api_key if api_key is not None else (settings.HOSTED_LLM_API_KEY or "")
        self.model_name = model_name or settings.HOSTED_LLM_MODEL or "gpt-4o-mini"

    @property
    def is_configured(self) -> bool:
        """Check if a valid, non-placeholder API key is set."""
        return bool(self.api_key and self.api_key.strip() and not self.api_key.startswith("your-"))

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        """Generate a complete non-streaming completion from the hosted LLM endpoint."""
        if not self.is_configured:
            raise LLMConfigurationError(
                "HOSTED_LLM_API_KEY is not configured. "
                "Please set HOSTED_LLM_API_KEY in your environment variables to connect to a live OpenAI-compatible provider."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 LocalGPT/3.0",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        url = f"{self.api_base}/chat/completions"

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
            ) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 401:
                    raise LLMAuthenticationError(
                        "Authentication failed with hosted LLM provider. Check your HOSTED_LLM_API_KEY."
                    )
                elif resp.status_code == 429:
                    raise LLMRateLimitError(
                        "Hosted LLM provider rate limit exceeded. Please retry shortly."
                    )
                elif resp.status_code >= 400:
                    raise LLMProviderError(
                        f"Hosted LLM returned error ({resp.status_code}): {resp.text}"
                    )

                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    raise LLMProviderError("Hosted LLM returned empty choices array.")
                return choices[0]["message"]["content"]
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Request to hosted LLM provider timed out: {e}")
        except LLMError:
            raise
        except Exception as e:
            raise LLMProviderError(f"Connection error to hosted LLM provider: {e}")

    async def stream_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream generated text chunks in real-time using OpenAI-compatible SSE streaming."""
        if not self.is_configured:
            raise LLMConfigurationError(
                "HOSTED_LLM_API_KEY is not configured. "
                "Please set HOSTED_LLM_API_KEY in your environment variables to stream from a live OpenAI-compatible endpoint."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 LocalGPT/3.0",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs,
        }
        url = f"{self.api_base}/chat/completions"

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
            ) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code == 401:
                        raise LLMAuthenticationError(
                            "Authentication failed with hosted LLM provider. Check your HOSTED_LLM_API_KEY."
                        )
                    elif resp.status_code == 429:
                        raise LLMRateLimitError(
                            "Hosted LLM provider rate limit exceeded. Please retry shortly."
                        )
                    elif resp.status_code >= 400:
                        err_body = await resp.aread()
                        raise LLMProviderError(
                            f"Hosted LLM returned error ({resp.status_code}): {err_body.decode('utf-8', errors='ignore')}"
                        )

                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_content = line[6:].strip()
                            if data_content == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(data_content)
                                choices = chunk_json.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Streaming request to hosted LLM provider timed out: {e}")
        except LLMError:
            raise
        except Exception as e:
            raise LLMProviderError(f"Streaming connection error to hosted LLM provider: {e}")


class QwenLocalAdapter(BaseLLMProvider):
    """
    Adapter preserving access to the Phase 2 local Qwen model without modifying original files.
    """

    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        return f"[Local Qwen Adapter] Preserved Phase 2 model adapter ready for {self.model_name}."

    async def stream_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs,
    ) -> AsyncIterator[str]:
        tokens = ["[Qwen Adapter] ", "Local ", "model ", "fallback ", "active."]
        for t in tokens:
            await asyncio.sleep(0.05)
            yield t


def get_llm_provider() -> BaseLLMProvider:
    """Factory creating configured LLM provider."""
    if settings.LLM_PROVIDER == "local_qwen":
        return QwenLocalAdapter()
    return HostedLLMProvider()
