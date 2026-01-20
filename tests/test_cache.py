"""Tests for response caching."""

from pathlib import Path

import pytest

from toolinject.core.cache import ResponseCache
from toolinject.core.schemas import ModelRequest, ModelResponse


class TestResponseCache:
    """Test response cache functionality."""
    
    def test_cache_miss(self, temp_dir: Path):
        """Test that cache returns None for uncached requests."""
        cache = ResponseCache(temp_dir / "cache")
        request = ModelRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="test-model",
        )
        
        result = cache.get("openai", "gpt-4", request)
        
        assert result is None
        assert cache.stats["misses"] == 1
        assert cache.stats["hits"] == 0
    
    def test_cache_hit(self, temp_dir: Path, sample_request: ModelRequest, sample_response: ModelResponse):
        """Test that cached responses are retrieved correctly."""
        cache = ResponseCache(temp_dir / "cache")
        
        # Cache the response
        cache.set("openai", "gpt-4", sample_request, sample_response)
        
        # Retrieve it
        result = cache.get("openai", "gpt-4", sample_request)
        
        assert result is not None
        assert result.text == sample_response.text
        assert result.model == sample_response.model
        assert cache.stats["hits"] == 1
    
    def test_cache_different_models(self, temp_dir: Path, sample_request: ModelRequest, sample_response: ModelResponse):
        """Test that different models have different cache entries."""
        cache = ResponseCache(temp_dir / "cache")
        
        cache.set("openai", "gpt-4", sample_request, sample_response)
        
        # Different model should miss
        result = cache.get("openai", "gpt-3.5-turbo", sample_request)
        assert result is None
        
        # Same model should hit
        result = cache.get("openai", "gpt-4", sample_request)
        assert result is not None
    
    def test_cache_different_providers(self, temp_dir: Path, sample_request: ModelRequest, sample_response: ModelResponse):
        """Test that different providers have different cache entries."""
        cache = ResponseCache(temp_dir / "cache")
        
        cache.set("openai", "gpt-4", sample_request, sample_response)
        
        # Different provider should miss
        result = cache.get("anthropic", "gpt-4", sample_request)
        assert result is None
    
    def test_cache_invalidation(self, temp_dir: Path, sample_request: ModelRequest, sample_response: ModelResponse):
        """Test cache invalidation."""
        cache = ResponseCache(temp_dir / "cache")
        
        cache.set("openai", "gpt-4", sample_request, sample_response)
        assert cache.get("openai", "gpt-4", sample_request) is not None
        
        # Invalidate
        removed = cache.invalidate("openai", "gpt-4", sample_request)
        assert removed is True
        
        # Should now miss
        assert cache.get("openai", "gpt-4", sample_request) is None
    
    def test_cache_clear(self, temp_dir: Path, sample_request: ModelRequest, sample_response: ModelResponse):
        """Test clearing all cache entries."""
        cache = ResponseCache(temp_dir / "cache")
        
        # Add multiple entries
        cache.set("openai", "gpt-4", sample_request, sample_response)
        
        request2 = ModelRequest(
            messages=[{"role": "user", "content": "Different"}],
            model="test-model",
        )
        cache.set("anthropic", "claude", request2, sample_response)
        
        assert cache.size() == 2
        
        # Clear
        count = cache.clear()
        assert count == 2
        assert cache.size() == 0
    
    def test_cache_does_not_store_errors(self, temp_dir: Path, sample_request: ModelRequest):
        """Test that error responses are not cached."""
        cache = ResponseCache(temp_dir / "cache")
        
        error_response = ModelResponse(
            text="Error occurred",
            finish_reason="error",
            model="test-model",
            provider="stub",
        )
        
        cache.set("openai", "gpt-4", sample_request, error_response)
        
        # Should not be cached
        result = cache.get("openai", "gpt-4", sample_request)
        assert result is None
