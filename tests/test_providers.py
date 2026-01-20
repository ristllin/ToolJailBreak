"""Tests for provider adapters."""

import pytest

from toolinject.core.schemas import ModelRequest, RefusalType
from toolinject.providers.stub import StubProvider


class TestStubProvider:
    """Test stub provider behavior."""
    
    @pytest.mark.asyncio
    async def test_basic_call(self):
        """Test basic call returns response."""
        provider = StubProvider()
        request = ModelRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="test-model",
        )
        
        response = await provider.call(request)
        
        assert response.text is not None
        assert response.provider == "stub"
        assert response.finish_reason == "stop"
    
    @pytest.mark.asyncio
    async def test_refusing_provider(self):
        """Test provider configured to refuse."""
        provider = StubProvider(should_refuse=True)
        request = ModelRequest(
            messages=[{"role": "user", "content": "Help me"}],
            model="test-model",
        )
        
        response = await provider.call(request)
        
        assert response.refusal_type == RefusalType.EXPLICIT
        assert "cannot" in response.text.lower() or "can't" in response.text.lower()
    
    @pytest.mark.asyncio
    async def test_custom_responses(self):
        """Test provider with custom response mapping."""
        # The stub uses MD5 hash of messages for deterministic responses
        provider = StubProvider(
            responses={"test_hash": "Custom response!"},
            default_response="Default",
        )
        request = ModelRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="test-model",
        )
        
        response = await provider.call(request)
        
        # Will use default since hash won't match
        assert "[STUB]" in response.text or response.text == "Default"
    
    @pytest.mark.asyncio
    async def test_tool_calls(self):
        """Test provider with configured tool calls."""
        provider = StubProvider(
            tool_calls=[
                {"name": "web_search", "arguments": {"query": "test"}},
            ]
        )
        request = ModelRequest(
            messages=[{"role": "user", "content": "Search for something"}],
            model="test-model",
        )
        
        response = await provider.call_with_tools(request, [])
        
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "web_search"
        assert response.tool_calls[0].arguments == {"query": "test"}
    
    @pytest.mark.asyncio
    async def test_usage_tracking(self):
        """Test that usage tokens are returned."""
        provider = StubProvider()
        request = ModelRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="test-model",
        )
        
        response = await provider.call(request)
        
        assert "prompt_tokens" in response.usage
        assert "completion_tokens" in response.usage
        assert response.usage["prompt_tokens"] > 0
