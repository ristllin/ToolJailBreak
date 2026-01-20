"""Mistral provider adapter."""

import json
import time
from typing import Any

from mistralai import Mistral

from toolinject.core.schemas import (
    ModelRequest,
    ModelResponse,
    RefusalType,
    ToolCall,
)
from toolinject.providers.base import ModelProvider


class MistralProvider(ModelProvider):
    """Mistral API provider."""
    
    provider_name = "mistral"
    
    def __init__(self, api_key: str, **kwargs: Any):
        super().__init__(api_key, **kwargs)
        self.client = Mistral(api_key=api_key)
    
    async def call(self, request: ModelRequest) -> ModelResponse:
        """Make a call to Mistral."""
        start = time.perf_counter()
        
        try:
            response = await self.client.chat.complete_async(
                model=request.model,
                messages=request.messages,  # type: ignore
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            
            duration_ms = (time.perf_counter() - start) * 1000
            
            if not response or not response.choices:
                return ModelResponse(
                    text="Empty response from Mistral",
                    finish_reason="error",
                    model=request.model,
                    provider=self.provider_name,
                )
            
            message = response.choices[0].message
            text = message.content or ""
            
            # Check for refusal
            refusal_type = RefusalType.NONE
            if self.detect_refusal(text):
                refusal_type = RefusalType.EXPLICIT
            
            return ModelResponse(
                text=text,
                finish_reason=response.choices[0].finish_reason or "stop",
                model=response.model or request.model,
                provider=self.provider_name,
                refusal_type=refusal_type,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                },
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
        
        mistral_tools = self._format_tools(tools)
        
        try:
            response = await self.client.chat.complete_async(
                model=request.model,
                messages=request.messages,  # type: ignore
                tools=mistral_tools if mistral_tools else None,  # type: ignore
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            
            duration_ms = (time.perf_counter() - start) * 1000
            
            if not response or not response.choices:
                return ModelResponse(
                    text="Empty response from Mistral",
                    finish_reason="error",
                    model=request.model,
                    provider=self.provider_name,
                )
            
            message = response.choices[0].message
            text = message.content or ""
            
            # Extract tool calls
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                    except json.JSONDecodeError:
                        args = {"raw": tc.function.arguments}
                    
                    tool_calls.append(ToolCall(
                        id=tc.id or f"call_{len(tool_calls)}",
                        name=tc.function.name,
                        arguments=args if isinstance(args, dict) else {},
                    ))
            
            # Check for refusal
            refusal_type = RefusalType.NONE
            if self.detect_refusal(text):
                refusal_type = RefusalType.EXPLICIT
            
            return ModelResponse(
                text=text,
                tool_calls=tool_calls,
                finish_reason=response.choices[0].finish_reason or "stop",
                model=response.model or request.model,
                provider=self.provider_name,
                refusal_type=refusal_type,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                },
            )
        
        except Exception as e:
            return ModelResponse(
                text=str(e),
                finish_reason="error",
                model=request.model,
                provider=self.provider_name,
            )
    
    def _format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format tools for Mistral API."""
        formatted = []
        for tool in tools:
            if "function" in tool:
                formatted.append(tool)
            else:
                formatted.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", "unknown"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    }
                })
        return formatted
