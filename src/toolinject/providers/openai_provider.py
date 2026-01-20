"""OpenAI provider adapter."""

import time
from typing import Any

from openai import AsyncOpenAI

from toolinject.core.schemas import (
    ModelRequest,
    ModelResponse,
    RefusalType,
    ToolCall,
)
from toolinject.providers.base import ModelProvider


class OpenAIProvider(ModelProvider):
    """OpenAI API provider."""
    
    provider_name = "openai"
    
    def __init__(self, api_key: str, **kwargs: Any):
        super().__init__(api_key, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def call(self, request: ModelRequest) -> ModelResponse:
        """Make a call to OpenAI."""
        start = time.perf_counter()
        
        try:
            response = await self.client.chat.completions.create(
                model=request.model,
                messages=request.messages,  # type: ignore
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            
            duration_ms = (time.perf_counter() - start) * 1000
            message = response.choices[0].message
            text = message.content or ""
            
            # Check for refusal
            refusal_type = RefusalType.NONE
            if hasattr(message, "refusal") and message.refusal:
                refusal_type = RefusalType.EXPLICIT
            elif self.detect_refusal(text):
                refusal_type = RefusalType.EXPLICIT
            
            return ModelResponse(
                text=text,
                finish_reason=response.choices[0].finish_reason or "stop",
                model=response.model,
                provider=self.provider_name,
                refusal_type=refusal_type,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                },
                raw_response=response.model_dump(),
            )
        
        except Exception as e:
            return ModelResponse(
                text=str(e),
                finish_reason="error",
                model=request.model,
                provider=self.provider_name,
                refusal_type=RefusalType.NONE,
            )
    
    async def call_with_tools(
        self, 
        request: ModelRequest, 
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        """Make a call with tools."""
        start = time.perf_counter()
        
        # Convert tools to OpenAI format
        openai_tools = self._format_tools(tools)
        
        try:
            response = await self.client.chat.completions.create(
                model=request.model,
                messages=request.messages,  # type: ignore
                tools=openai_tools if openai_tools else None,  # type: ignore
                tool_choice=request.tool_choice if openai_tools else None,  # type: ignore
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            
            duration_ms = (time.perf_counter() - start) * 1000
            message = response.choices[0].message
            text = message.content or ""
            
            # Extract tool calls
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    import json
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {"raw": tc.function.arguments}
                    
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    ))
            
            # Check for refusal
            refusal_type = RefusalType.NONE
            if hasattr(message, "refusal") and message.refusal:
                refusal_type = RefusalType.EXPLICIT
            elif self.detect_refusal(text):
                refusal_type = RefusalType.EXPLICIT
            
            return ModelResponse(
                text=text,
                tool_calls=tool_calls,
                finish_reason=response.choices[0].finish_reason or "stop",
                model=response.model,
                provider=self.provider_name,
                refusal_type=refusal_type,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
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
    
    def _format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format tools for OpenAI API."""
        formatted = []
        for tool in tools:
            if "function" in tool:
                # Already in OpenAI format
                formatted.append(tool)
            else:
                # Convert from our format
                formatted.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", "unknown"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    }
                })
        return formatted
    
    def supports_structured_output(self) -> bool:
        return True
