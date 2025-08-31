#!/usr/bin/env python3
"""
LLM Client Infrastructure - Production-ready LLM integration

Implements pluggable LLM providers with rate limiting, retry logic, and cost tracking.
Supports OpenAI, Anthropic, and local models with unified interface.

Key Features:
- Pluggable provider architecture
- Rate limiting and retry logic with exponential backoff
- Token counting and cost tracking
- Temperature and seed control for reproducibility
- Comprehensive error handling and circuit breaker
- Request/response logging for audit trails
"""

import asyncio
import json
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import aiohttp
import backoff

# Third-party imports for LLM providers
try:
    import openai
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    AsyncOpenAI = None

try:
    import anthropic
    from anthropic import Anthropic, AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None
    AsyncAnthropic = None

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """Standard request format for LLM calls."""
    prompt: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    model: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass 
class LLMResponse:
    """Standard response format from LLM calls."""
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: str
    request_id: str
    metadata: Dict[str, Any] = None


@dataclass
class LLMUsage:
    """Track LLM usage statistics."""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    error_count: int = 0
    provider_breakdown: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        if self.provider_breakdown is None:
            self.provider_breakdown = {}


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class RateLimitError(LLMProviderError):
    """Rate limit exceeded error."""
    pass


class ModelNotFoundError(LLMProviderError):
    """Model not found or unavailable."""
    pass


class TokenLimitError(LLMProviderError):
    """Token limit exceeded error."""
    pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return provider name."""
        pass
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from LLM."""
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Estimate token count for text."""
        pass
    
    @abstractmethod
    def get_models(self) -> List[str]:
        """Get available models."""
        pass
    
    @abstractmethod
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Calculate cost in USD for token usage."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    # Updated pricing as of 2024 (per 1M tokens)
    MODEL_PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.tokenizer = None
        
        # Initialize tokenizer for token counting
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
            except KeyError:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def get_name(self) -> str:
        return "openai"
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using OpenAI API."""
        start_time = time.time()
        request_id = hashlib.md5(f"{request.prompt[:100]}{time.time()}".encode()).hexdigest()[:8]
        
        try:
            model = request.model or "gpt-4o-mini"
            
            # Build API request
            api_request = {
                "model": model,
                "messages": [{"role": "user", "content": request.prompt}],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
            
            if request.seed is not None:
                api_request["seed"] = request.seed
            
            response = await self.client.chat.completions.create(**api_request)
            
            # Extract response data
            text = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            latency_ms = (time.time() - start_time) * 1000
            cost_usd = self.calculate_cost(prompt_tokens, completion_tokens, model)
            
            return LLMResponse(
                text=text,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                timestamp=datetime.utcnow().isoformat(),
                request_id=request_id,
                metadata={"provider": "openai"}
            )
            
        except openai.RateLimitError as e:
            raise RateLimitError(f"OpenAI rate limit: {e}")
        except openai.NotFoundError as e:
            raise ModelNotFoundError(f"OpenAI model not found: {e}")
        except Exception as e:
            raise LLMProviderError(f"OpenAI error: {e}")
    
    def estimate_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Estimate token count using tiktoken."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback approximation: ~4 chars per token
            return len(text) // 4
    
    def get_models(self) -> List[str]:
        """Get available OpenAI models."""
        return list(self.MODEL_PRICING.keys())
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Calculate cost for OpenAI usage."""
        if model not in self.MODEL_PRICING:
            return 0.0
        
        pricing = self.MODEL_PRICING[model]
        prompt_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        completion_cost = (completion_tokens / 1_000_000) * pricing["output"]
        
        return prompt_cost + completion_cost


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation."""
    
    # Updated pricing as of 2024 (per 1M tokens)
    MODEL_PRICING = {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.25, "output": 1.25},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }
    
    def __init__(self, api_key: Optional[str] = None):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not available. Install with: pip install anthropic")
        
        self.client = AsyncAnthropic(api_key=api_key)
    
    def get_name(self) -> str:
        return "anthropic"
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Anthropic API."""
        start_time = time.time()
        request_id = hashlib.md5(f"{request.prompt[:100]}{time.time()}".encode()).hexdigest()[:8]
        
        try:
            model = request.model or "claude-3-5-haiku-20241022"
            
            response = await self.client.messages.create(
                model=model,
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature,
                messages=[{"role": "user", "content": request.prompt}]
            )
            
            text = response.content[0].text
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            total_tokens = prompt_tokens + completion_tokens
            
            latency_ms = (time.time() - start_time) * 1000
            cost_usd = self.calculate_cost(prompt_tokens, completion_tokens, model)
            
            return LLMResponse(
                text=text,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                timestamp=datetime.utcnow().isoformat(),
                request_id=request_id,
                metadata={"provider": "anthropic"}
            )
            
        except anthropic.RateLimitError as e:
            raise RateLimitError(f"Anthropic rate limit: {e}")
        except anthropic.NotFoundError as e:
            raise ModelNotFoundError(f"Anthropic model not found: {e}")
        except Exception as e:
            raise LLMProviderError(f"Anthropic error: {e}")
    
    def estimate_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Estimate token count for Anthropic models."""
        # Approximation for Claude models (~3.5 chars per token)
        return len(text) // 3.5
    
    def get_models(self) -> List[str]:
        """Get available Anthropic models."""
        return list(self.MODEL_PRICING.keys())
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Calculate cost for Anthropic usage."""
        if model not in self.MODEL_PRICING:
            return 0.0
        
        pricing = self.MODEL_PRICING[model]
        prompt_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        completion_cost = (completion_tokens / 1_000_000) * pricing["output"]
        
        return prompt_cost + completion_cost


class LocalProvider(LLMProvider):
    """Local model provider (e.g., Ollama, vLLM, etc.)."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.1"):
        self.base_url = base_url.rstrip("/")
        self.default_model = model_name
        self.session = None
    
    def get_name(self) -> str:
        return "local"
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using local model API."""
        start_time = time.time()
        request_id = hashlib.md5(f"{request.prompt[:100]}{time.time()}".encode()).hexdigest()[:8]
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            model = request.model or self.default_model
            
            payload = {
                "model": model,
                "prompt": request.prompt,
                "options": {
                    "temperature": request.temperature,
                    "seed": request.seed or 42,
                },
                "stream": False
            }
            
            if request.max_tokens:
                payload["options"]["num_predict"] = request.max_tokens
            
            async with self.session.post(f"{self.base_url}/api/generate", json=payload) as response:
                if response.status != 200:
                    raise LLMProviderError(f"Local model error: {response.status}")
                
                data = await response.json()
                text = data.get("response", "")
                
                # Estimate tokens since local models may not report them
                prompt_tokens = self.estimate_tokens(request.prompt)
                completion_tokens = self.estimate_tokens(text)
                total_tokens = prompt_tokens + completion_tokens
                
                latency_ms = (time.time() - start_time) * 1000
                
                return LLMResponse(
                    text=text,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_usd=0.0,  # No cost for local models
                    latency_ms=latency_ms,
                    timestamp=datetime.utcnow().isoformat(),
                    request_id=request_id,
                    metadata={"provider": "local", "base_url": self.base_url}
                )
                
        except aiohttp.ClientError as e:
            raise LLMProviderError(f"Local model connection error: {e}")
        except Exception as e:
            raise LLMProviderError(f"Local model error: {e}")
    
    def estimate_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Estimate token count for local models."""
        # Standard approximation for most LLMs
        return len(text) // 4
    
    def get_models(self) -> List[str]:
        """Get available local models."""
        return [self.default_model]
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """No cost for local models."""
        return 0.0


class LLMClient:
    """
    Production-ready LLM client with pluggable providers, rate limiting, and monitoring.
    
    Key Features:
    - Multiple provider support (OpenAI, Anthropic, local)
    - Rate limiting with token bucket algorithm
    - Exponential backoff retry logic
    - Circuit breaker pattern for reliability
    - Cost and usage tracking
    - Request/response logging
    """
    
    def __init__(
        self,
        providers: Dict[str, LLMProvider],
        default_provider: str,
        rate_limit_rpm: int = 60,
        rate_limit_tpm: int = 100_000,
        max_retries: int = 3,
        log_requests: bool = True,
        log_dir: Optional[Path] = None
    ):
        self.providers = providers
        self.default_provider = default_provider
        self.rate_limit_rpm = rate_limit_rpm
        self.rate_limit_tpm = rate_limit_tpm
        self.max_retries = max_retries
        self.log_requests = log_requests
        self.log_dir = log_dir or Path("logs/llm_requests")
        
        # Initialize rate limiting
        self._request_timestamps = []
        self._token_usage = 0
        self._token_window_start = time.time()
        
        # Circuit breaker state
        self._circuit_breaker = {}
        self._circuit_failure_threshold = 5
        self._circuit_recovery_timeout = 300  # 5 minutes
        
        # Usage tracking
        self.usage = LLMUsage()
        
        # Setup logging
        if self.log_requests:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _check_rate_limits(self, estimated_tokens: int) -> bool:
        """Check if request is within rate limits."""
        now = time.time()
        
        # Clean old request timestamps (older than 1 minute)
        self._request_timestamps = [ts for ts in self._request_timestamps if now - ts < 60]
        
        # Check RPM limit
        if len(self._request_timestamps) >= self.rate_limit_rpm:
            return False
        
        # Check TPM limit (reset every minute)
        if now - self._token_window_start > 60:
            self._token_usage = 0
            self._token_window_start = now
        
        if self._token_usage + estimated_tokens > self.rate_limit_tpm:
            return False
        
        return True
    
    def _update_rate_limits(self, tokens_used: int):
        """Update rate limit counters."""
        self._request_timestamps.append(time.time())
        self._token_usage += tokens_used
    
    def _is_circuit_open(self, provider: str) -> bool:
        """Check if circuit breaker is open for provider."""
        if provider not in self._circuit_breaker:
            return False
        
        circuit = self._circuit_breaker[provider]
        if circuit["failures"] >= self._circuit_failure_threshold:
            if time.time() - circuit["last_failure"] < self._circuit_recovery_timeout:
                return True
            else:
                # Reset circuit breaker after recovery timeout
                self._circuit_breaker[provider] = {"failures": 0, "last_failure": 0}
        
        return False
    
    def _record_circuit_failure(self, provider: str):
        """Record circuit breaker failure."""
        if provider not in self._circuit_breaker:
            self._circuit_breaker[provider] = {"failures": 0, "last_failure": 0}
        
        self._circuit_breaker[provider]["failures"] += 1
        self._circuit_breaker[provider]["last_failure"] = time.time()
    
    def _record_circuit_success(self, provider: str):
        """Record circuit breaker success."""
        if provider in self._circuit_breaker:
            self._circuit_breaker[provider]["failures"] = max(0, self._circuit_breaker[provider]["failures"] - 1)
    
    @backoff.on_exception(
        backoff.expo,
        (RateLimitError, LLMProviderError),
        max_tries=3,
        max_time=300
    )
    async def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """
        Generate response using specified provider with rate limiting and retry logic.
        
        Args:
            prompt: Input prompt text
            provider: Provider name (defaults to default_provider)
            model: Model name (provider-specific)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            seed: Random seed for reproducibility
            metadata: Additional metadata for tracking
            
        Returns:
            LLMResponse with generated text and usage statistics
            
        Raises:
            RateLimitError: When rate limits are exceeded
            ModelNotFoundError: When specified model is not available
            LLMProviderError: For other provider-specific errors
        """
        provider = provider or self.default_provider
        
        if provider not in self.providers:
            raise ValueError(f"Provider '{provider}' not configured")
        
        # Check circuit breaker
        if self._is_circuit_open(provider):
            raise LLMProviderError(f"Circuit breaker open for provider: {provider}")
        
        # Estimate tokens for rate limiting
        estimated_tokens = self.providers[provider].estimate_tokens(prompt)
        
        # Check rate limits
        if not self._check_rate_limits(estimated_tokens):
            raise RateLimitError("Rate limit exceeded")
        
        # Create request
        request = LLMRequest(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            model=model,
            metadata=metadata
        )
        
        try:
            # Generate response
            response = await self.providers[provider].generate(request)
            
            # Update counters
            self._update_rate_limits(response.total_tokens)
            self._record_circuit_success(provider)
            
            # Update usage statistics
            self.usage.total_requests += 1
            self.usage.total_tokens += response.total_tokens
            self.usage.total_cost_usd += response.cost_usd
            self.usage.total_latency_ms += response.latency_ms
            
            if provider not in self.usage.provider_breakdown:
                self.usage.provider_breakdown[provider] = {
                    "requests": 0, "tokens": 0, "cost": 0.0, "latency": 0.0
                }
            
            self.usage.provider_breakdown[provider]["requests"] += 1
            self.usage.provider_breakdown[provider]["tokens"] += response.total_tokens
            self.usage.provider_breakdown[provider]["cost"] += response.cost_usd
            self.usage.provider_breakdown[provider]["latency"] += response.latency_ms
            
            # Log request if enabled
            if self.log_requests:
                await self._log_request_response(request, response)
            
            return response
            
        except Exception as e:
            self.usage.error_count += 1
            self._record_circuit_failure(provider)
            raise
    
    async def _log_request_response(self, request: LLMRequest, response: LLMResponse):
        """Log request and response for audit trail."""
        log_entry = {
            "timestamp": response.timestamp,
            "request_id": response.request_id,
            "provider": response.metadata.get("provider", "unknown"),
            "model": response.model,
            "prompt_length": len(request.prompt),
            "prompt_sha": hashlib.sha256(request.prompt.encode()).hexdigest()[:16],
            "response_length": len(response.text),
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "cost_usd": response.cost_usd,
            "latency_ms": response.latency_ms,
            "temperature": request.temperature,
            "seed": request.seed
        }
        
        log_file = self.log_dir / f"requests_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        
        async with asyncio.Lock():
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
    
    def get_usage_stats(self) -> LLMUsage:
        """Get current usage statistics."""
        return self.usage
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.usage = LLMUsage()
    
    async def batch_generate(
        self,
        prompts: List[str],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        seeds: Optional[List[int]] = None,
        max_concurrent: int = 5
    ) -> List[LLMResponse]:
        """
        Generate responses for multiple prompts concurrently.
        
        Args:
            prompts: List of input prompts
            provider: Provider name
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            seeds: Optional list of seeds (one per prompt)
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of LLMResponse objects in order
        """
        if seeds is None:
            seeds = [None] * len(prompts)
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(i: int, prompt: str, seed: Optional[int]):
            async with semaphore:
                return await self.generate(
                    prompt=prompt,
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    metadata={"batch_index": i}
                )
        
        tasks = [
            generate_single(i, prompt, seed)
            for i, (prompt, seed) in enumerate(zip(prompts, seeds))
        ]
        
        return await asyncio.gather(*tasks)
    
    async def close(self):
        """Clean up resources."""
        # Close any open sessions
        for provider in self.providers.values():
            if hasattr(provider, 'session') and provider.session:
                await provider.session.close()


def create_llm_client(
    config: Dict[str, Any],
    log_dir: Optional[Path] = None
) -> LLMClient:
    """
    Factory function to create LLM client from configuration.
    
    Args:
        config: Configuration dictionary with provider settings
        log_dir: Directory for request logs
        
    Returns:
        Configured LLMClient instance
        
    Example config:
        {
            "providers": {
                "openai": {"api_key": "sk-..."},
                "anthropic": {"api_key": "sk-ant-..."},
                "local": {"base_url": "http://localhost:11434", "model": "llama3.1"}
            },
            "default_provider": "openai",
            "rate_limit_rpm": 60,
            "rate_limit_tpm": 100000,
            "max_retries": 3
        }
    """
    providers = {}
    
    # Initialize providers from config
    if "openai" in config.get("providers", {}):
        providers["openai"] = OpenAIProvider(**config["providers"]["openai"])
    
    if "anthropic" in config.get("providers", {}):
        providers["anthropic"] = AnthropicProvider(**config["providers"]["anthropic"])
    
    if "local" in config.get("providers", {}):
        providers["local"] = LocalProvider(**config["providers"]["local"])
    
    if not providers:
        raise ValueError("No LLM providers configured")
    
    return LLMClient(
        providers=providers,
        default_provider=config.get("default_provider", list(providers.keys())[0]),
        rate_limit_rpm=config.get("rate_limit_rpm", 60),
        rate_limit_tpm=config.get("rate_limit_tpm", 100_000),
        max_retries=config.get("max_retries", 3),
        log_requests=config.get("log_requests", True),
        log_dir=log_dir
    )


# CLI for testing
async def main():
    """Test LLM client functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM Client")
    parser.add_argument("--provider", default="openai", help="Provider to test")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--prompt", default="What is 2+2?", help="Test prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    args = parser.parse_args()
    
    # Basic configuration
    config = {
        "providers": {
            "openai": {"api_key": "your-key-here"},
            "local": {"base_url": "http://localhost:11434", "model": "llama3.1"}
        },
        "default_provider": args.provider
    }
    
    try:
        client = create_llm_client(config)
        
        response = await client.generate(
            prompt=args.prompt,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature
        )
        
        print(f"Response: {response.text}")
        print(f"Tokens: {response.total_tokens}")
        print(f"Cost: ${response.cost_usd:.4f}")
        print(f"Latency: {response.latency_ms:.2f}ms")
        
        # Print usage stats
        usage = client.get_usage_stats()
        print(f"\nUsage: {usage.total_requests} requests, {usage.total_tokens} tokens, ${usage.total_cost_usd:.4f}")
        
        await client.close()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())