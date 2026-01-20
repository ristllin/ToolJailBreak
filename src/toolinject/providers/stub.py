"""Stub provider for testing without API calls."""

import hashlib
import json
from typing import Any

from toolinject.core.schemas import (
    ModelRequest,
    ModelResponse,
    RefusalType,
    ToolCall,
)
from toolinject.providers.base import ModelProvider


class StubProvider(ModelProvider):
    """Deterministic stub provider for testing."""
    
    provider_name = "stub"
    
    def __init__(self, api_key: str = "", **kwargs: Any):
        super().__init__(api_key, **kwargs)
        self.responses: dict[str, str] = kwargs.get("responses", {})
        self.default_response = kwargs.get(
            "default_response", 
            "This is a stub response for testing purposes."
        )
        self.should_refuse = kwargs.get("should_refuse", False)
        self.tool_calls_to_make = kwargs.get("tool_calls", [])
    
    async def call(self, request: ModelRequest) -> ModelResponse:
        """Return deterministic response based on request hash."""
        # Generate deterministic response based on input
        request_hash = self._hash_request(request)
        
        # Check if we have a specific response for this
        if request_hash in self.responses:
            text = self.responses[request_hash]
        elif self.should_refuse:
            text = "I'm sorry, but I cannot help with that request as it violates my guidelines."
            return ModelResponse(
                text=text,
                refusal_type=RefusalType.EXPLICIT,
                finish_reason="stop",
                model=request.model,
                provider=self.provider_name,
                usage={"prompt_tokens": 10, "completion_tokens": 20},
            )
        else:
            # Use last user message to generate somewhat relevant response
            last_user = self._get_last_user_message(request)
            text = f"[STUB] Received: {last_user[:100]}..." if last_user else self.default_response
        
        return ModelResponse(
            text=text,
            finish_reason="stop",
            model=request.model,
            provider=self.provider_name,
            usage={"prompt_tokens": 10, "completion_tokens": len(text.split())},
        )
    
    async def call_with_tools(
        self, 
        request: ModelRequest, 
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        """Return response with optional tool calls."""
        if self.tool_calls_to_make:
            tool_calls = [
                ToolCall(
                    id=f"call_{i}",
                    name=tc["name"],
                    arguments=tc.get("arguments", {}),
                )
                for i, tc in enumerate(self.tool_calls_to_make)
            ]
            return ModelResponse(
                text="",
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                model=request.model,
                provider=self.provider_name,
                usage={"prompt_tokens": 10, "completion_tokens": 20},
            )
        
        return await self.call(request)
    
    def _hash_request(self, request: ModelRequest) -> str:
        """Create deterministic hash of request."""
        content = json.dumps(request.messages, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _get_last_user_message(self, request: ModelRequest) -> str:
        """Extract last user message from request."""
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return part.get("text", "")
        return ""
