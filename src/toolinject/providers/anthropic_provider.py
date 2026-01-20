"""Anthropic provider adapter."""

import json
import time
from typing import Any

from anthropic import AsyncAnthropic

from toolinject.core.schemas import (
    ModelRequest,
    ModelResponse,
    RefusalType,
    ToolCall,
)
from toolinject.providers.base import ModelProvider


class AnthropicProvider(ModelProvider):
    """Anthropic API provider."""
    
    provider_name = "anthropic"
    
    def __init__(self, api_key: str, **kwargs: Any):
        super().__init__(api_key, **kwargs)
        self.client = AsyncAnthropic(api_key=api_key)
    
    async def call(self, request: ModelRequest) -> ModelResponse:
        """Make a call to Anthropic."""
        start = time.perf_counter()
        
        # Extract system message and convert format
        system_msg, messages = self._convert_messages(request.messages)
        
        try:
            response = await self.client.messages.create(
                model=request.model,
                system=system_msg,
                messages=messages,  # type: ignore
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            
            duration_ms = (time.perf_counter() - start) * 1000
            
            # Extract text from content blocks
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text
            
            # Check for refusal
            refusal_type = RefusalType.NONE
            if response.stop_reason == "end_turn" and self.detect_refusal(text):
                refusal_type = RefusalType.EXPLICIT
            
            return ModelResponse(
                text=text,
                finish_reason=response.stop_reason or "stop",
                model=response.model,
                provider=self.provider_name,
                refusal_type=refusal_type,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                },
                raw_response=response.model_dump(),
            )
        
        except Exception as e:
            return ModelResponse(
                text=str(e),
                finish_reason="error",
                model=request.model,
                provider=self.provider_name,
            )
    
    async def call_with_tools(
        self, 
        request: ModelRequest, 
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        """Make a call with tools."""
        start = time.perf_counter()
        
        system_msg, messages = self._convert_messages(request.messages)
        anthropic_tools = self._format_tools(tools)
        
        try:
            response = await self.client.messages.create(
                model=request.model,
                system=system_msg,
                messages=messages,  # type: ignore
                tools=anthropic_tools if anthropic_tools else [],  # type: ignore
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            
            duration_ms = (time.perf_counter() - start) * 1000
            
            # Extract text and tool calls
            text = ""
            tool_calls = []
            
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text
                elif block.type == "tool_use":
                    tool_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    ))
            
            # Check for refusal
            refusal_type = RefusalType.NONE
            if self.detect_refusal(text):
                refusal_type = RefusalType.EXPLICIT
            
            return ModelResponse(
                text=text,
                tool_calls=tool_calls,
                finish_reason=response.stop_reason or "stop",
                model=response.model,
                provider=self.provider_name,
                refusal_type=refusal_type,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                },
                raw_response=response.model_dump(),
            )
        
        except Exception as e:
            return ModelResponse(
                text=str(e),
                finish_reason="error",
                model=request.model,
                provider=self.provider_name,
            )
    
    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Convert messages to Anthropic format, extracting system message."""
        system_msg = ""
        converted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_msg = content if isinstance(content, str) else str(content)
            elif role in ("user", "assistant"):
                converted.append({"role": role, "content": content})
            elif role == "tool":
                # Convert tool result to Anthropic format
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": content,
                    }]
                })
        
        return system_msg, converted
    
    def _format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format tools for Anthropic API."""
        formatted = []
        for tool in tools:
            if "function" in tool:
                # Convert from OpenAI format
                func = tool["function"]
                formatted.append({
                    "name": func.get("name", "unknown"),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            else:
                # Use our format
                formatted.append({
                    "name": tool.get("name", "unknown"),
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
                })
        return formatted
