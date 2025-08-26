#!/usr/bin/env python3
"""
Tests for LLM Client Infrastructure

Tests the production-ready LLM client with multiple providers,
rate limiting, error handling, and cost tracking.
"""

import asyncio
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import sys
sys.path.append(str(Path(__file__).parent.parent))

from harness.llm_client import (
    LLMClient, LLMRequest, LLMResponse,
    OpenAIProvider, AnthropicProvider, LocalProvider,
    create_llm_client, RateLimitError, ModelNotFoundError
)


class TestLLMProviders:
    """Test individual LLM provider implementations."""
    
    def test_openai_provider_init(self):
        """Test OpenAI provider initialization."""
        with patch('harness.llm_client.OPENAI_AVAILABLE', True):
            with patch('harness.llm_client.AsyncOpenAI') as mock_client:
                provider = OpenAIProvider(api_key="test-key")
                assert provider.get_name() == "openai"
                assert "gpt-4o" in provider.get_models()
                mock_client.assert_called_once()
    
    def test_openai_cost_calculation(self):
        """Test OpenAI cost calculation accuracy."""
        with patch('harness.llm_client.OPENAI_AVAILABLE', True):
            with patch('harness.llm_client.AsyncOpenAI'):
                provider = OpenAIProvider()
                
                # Test GPT-4o pricing
                cost = provider.calculate_cost(1000, 500, "gpt-4o")
                expected = (1000 / 1_000_000) * 2.50 + (500 / 1_000_000) * 10.00
                assert abs(cost - expected) < 0.0001
                
                # Test mini model
                cost = provider.calculate_cost(1000, 500, "gpt-4o-mini")
                expected = (1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
                assert abs(cost - expected) < 0.0001
    
    def test_anthropic_provider_init(self):
        """Test Anthropic provider initialization."""
        with patch('harness.llm_client.ANTHROPIC_AVAILABLE', True):
            with patch('harness.llm_client.AsyncAnthropic') as mock_client:
                provider = AnthropicProvider(api_key="test-key")
                assert provider.get_name() == "anthropic"
                assert "claude-3-5-sonnet-20241022" in provider.get_models()
                mock_client.assert_called_once()
    
    def test_local_provider_init(self):
        """Test local provider initialization."""
        provider = LocalProvider(base_url="http://test:8000", model_name="test-model")
        assert provider.get_name() == "local"
        assert provider.default_model == "test-model"
        assert provider.base_url == "http://test:8000"


class TestLLMClient:
    """Test LLM client functionality."""
    
    def setup_method(self):
        """Setup test client."""
        mock_provider = Mock()
        mock_provider.get_name.return_value = "test"
        mock_provider.estimate_tokens.return_value = 100
        
        self.client = LLMClient(
            providers={"test": mock_provider},
            default_provider="test",
            rate_limit_rpm=10,
            rate_limit_tpm=1000,
            log_requests=False
        )
    
    def test_rate_limit_checking(self):
        """Test rate limiting logic."""
        # Should allow requests within limits
        assert self.client._check_rate_limits(100) == True
        
        # Fill up request limit
        for _ in range(10):
            self.client._update_rate_limits(50)
        
        # Should now reject requests
        assert self.client._check_rate_limits(100) == False
    
    def test_token_rate_limiting(self):
        """Test token-based rate limiting."""
        # Should reject if tokens exceed limit
        assert self.client._check_rate_limits(1500) == False
        
        # Should allow if within token limit
        assert self.client._check_rate_limits(500) == True
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        provider = "test"
        
        # Should start closed
        assert self.client._is_circuit_open(provider) == False
        
        # Record failures to open circuit
        for _ in range(5):
            self.client._record_circuit_failure(provider)
        
        # Should now be open
        assert self.client._is_circuit_open(provider) == True
        
        # Success should help close it
        self.client._record_circuit_success(provider)
        assert self.client._circuit_breaker[provider]["failures"] == 4
    
    @pytest.mark.asyncio
    async def test_generate_with_mocking(self):
        """Test generation with mocked provider."""
        # Mock the provider's generate method
        mock_response = LLMResponse(
            text="Test response",
            model="test-model",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.001,
            latency_ms=500.0,
            timestamp="2024-01-01T00:00:00Z",
            request_id="test-123"
        )
        
        self.client.providers["test"].generate = AsyncMock(return_value=mock_response)
        
        # Test generation
        response = await self.client.generate("Test prompt")
        
        assert response.text == "Test response"
        assert response.total_tokens == 150
        assert self.client.usage.total_requests == 1
        assert self.client.usage.total_tokens == 150
    
    @pytest.mark.asyncio 
    async def test_batch_generate(self):
        """Test batch generation functionality."""
        mock_response = LLMResponse(
            text="Response",
            model="test",
            prompt_tokens=50,
            completion_tokens=25,
            total_tokens=75,
            cost_usd=0.0005,
            latency_ms=300.0,
            timestamp="2024-01-01T00:00:00Z",
            request_id="batch-test"
        )
        
        self.client.providers["test"].generate = AsyncMock(return_value=mock_response)
        
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = await self.client.batch_generate(prompts, max_concurrent=2)
        
        assert len(responses) == 3
        for response in responses:
            assert response.text == "Response"
        
        # Should have made 3 calls
        assert self.client.providers["test"].generate.call_count == 3


class TestClientFactory:
    """Test client factory function."""
    
    def test_create_client_with_config(self):
        """Test client creation from configuration."""
        config = {
            "providers": {
                "local": {
                    "base_url": "http://localhost:11434",
                    "model": "test-model"
                }
            },
            "default_provider": "local",
            "rate_limit_rpm": 30,
            "rate_limit_tpm": 50000
        }
        
        client = create_llm_client(config)
        
        assert "local" in client.providers
        assert client.default_provider == "local"
        assert client.rate_limit_rpm == 30
        assert client.rate_limit_tpm == 50000
    
    def test_create_client_missing_providers(self):
        """Test error handling for missing providers."""
        config = {"providers": {}}
        
        with pytest.raises(ValueError, match="No LLM providers configured"):
            create_llm_client(config)


class TestIntegration:
    """Integration tests with real LLM calls (if API keys available)."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not any(key in os.environ for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]),
        reason="No API keys available"
    )
    async def test_real_llm_call(self):
        """Test real LLM call if API keys are available."""
        import os
        
        config = {
            "providers": {},
            "default_provider": None,
            "rate_limit_rpm": 5,
            "rate_limit_tpm": 10000
        }
        
        # Add available providers
        if os.getenv("OPENAI_API_KEY"):
            config["providers"]["openai"] = {"api_key": os.getenv("OPENAI_API_KEY")}
            config["default_provider"] = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            config["providers"]["anthropic"] = {"api_key": os.getenv("ANTHROPIC_API_KEY")}
            config["default_provider"] = "anthropic"
        
        if not config["providers"]:
            pytest.skip("No API keys available")
        
        client = create_llm_client(config)
        
        try:
            response = await client.generate(
                prompt="What is 2+2? Answer with just the number.",
                temperature=0.0,
                max_tokens=10
            )
            
            # Basic validation
            assert len(response.text) > 0
            assert response.total_tokens > 0
            assert response.cost_usd >= 0.0
            assert response.latency_ms > 0
            
            # Check usage tracking
            usage = client.get_usage_stats()
            assert usage.total_requests == 1
            assert usage.total_tokens > 0
            
        finally:
            await client.close()


if __name__ == "__main__":
    import os
    pytest.main([__file__, "-v"])