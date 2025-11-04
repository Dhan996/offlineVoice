# llm_ollama_service.py - FIXED VERSION
import asyncio
import json
import httpx
import logging
from typing import AsyncGenerator, List, Dict, Optional

from config import CONFIG

log = logging.getLogger(__name__)

class OllamaLLM:
    """
    Ollama LLM client with streaming support and proper cancellation handling.
    """
    
    def __init__(self):
        self.base = CONFIG["ollama_base"].rstrip('/')
        self.model = CONFIG["ollama_model"]
        self.temperature = CONFIG["llm_temperature"]
        self.max_tokens = CONFIG["llm_max_tokens"]
        
        # Connection pooling for better performance
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling"""
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=10.0,
                        read=60.0,
                        write=10.0,
                        pool=5.0
                    ),
                    limits=httpx.Limits(
                        max_keepalive_connections=5,
                        max_connections=10,
                        keepalive_expiry=30.0
                    )
                )
            return self._client

    async def stream(self, messages: List[Dict]) -> AsyncGenerator[str, None]:
        """
        Stream tokens from Ollama with proper error handling and cancellation support.
        
        Args:
            messages: List of chat messages in OpenAI format
            
        Yields:
            Token strings from the LLM
        """
        url = f"{self.base}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        print(payload)
        client = await self._get_client()
        token_count = 0
        
        try:
            async with client.stream("POST", url, json=payload) as response:
                # Check response status
                if response.status_code != 200:
                    error_text = await response.aread()
                    log.error(f"Ollama error {response.status_code}: {error_text}")
                    raise RuntimeError(f"Ollama API error: {response.status_code}")
                
                # Stream response lines
                async for line in response.aiter_lines():
                    if not line or not line.strip():
                        continue
                    
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        log.warning(f"Failed to parse JSON: {line[:100]}")
                        continue
                    
                    # Extract token content
                    tok = obj.get("message", {}).get("content")
                    if tok:
                        token_count += 1
                        yield tok
                    
                    # Check for completion
                    if obj.get("done"):
                        log.debug(f"LLM stream completed, {token_count} tokens")
                        break
                    
                    # Check for errors in response
                    if "error" in obj:
                        error_msg = obj.get("error", "Unknown error")
                        log.error(f"Ollama stream error: {error_msg}")
                        raise RuntimeError(f"Ollama error: {error_msg}")
        
        except asyncio.CancelledError:
            log.debug(f"LLM stream cancelled after {token_count} tokens")
            # Cleanup: close the response to abort the HTTP request
            raise
        
        except httpx.HTTPError as e:
            log.error(f"HTTP error communicating with Ollama: {e}")
            raise RuntimeError(f"Ollama connection error: {e}")
        
        except Exception as e:
            log.error(f"Unexpected error in LLM stream: {e}")
            raise

    async def close(self):
        """Close the HTTP client and cleanup resources"""
        async with self._client_lock:
            if self._client and not self._client.is_closed:
                await self._client.aclose()
                self._client = None
                log.debug("LLM client closed")

    async def __aenter__(self):
        """Context manager support"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        await self.close()
