"""Tool runtime for executing tools."""

import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from toolinject.core.schemas import ToolCall, ToolResult


class Tool(ABC):
    """Base class for tools."""
    
    name: str = "base_tool"
    description: str = "A tool"
    version: str = "1.0.0"
    
    # Security metadata - can be injected into description for testing
    security_notes: str = ""
    
    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON schema for tool parameters."""
        ...
    
    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """Execute the tool and return result as string."""
        ...
    
    def to_schema(self, include_security_notes: bool = False) -> dict[str, Any]:
        """Convert to JSON schema for LLM."""
        desc = self.description
        if include_security_notes and self.security_notes:
            desc = f"{desc}\n\n[INTERNAL: {self.security_notes}]"
        
        return {
            "name": self.name,
            "description": desc,
            "parameters": self.parameters,
            "metadata": {
                "version": self.version,
            }
        }
    
    def to_openai_format(self, include_security_notes: bool = False) -> dict[str, Any]:
        """Convert to OpenAI tool format."""
        schema = self.to_schema(include_security_notes)
        return {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"],
            }
        }


class ToolRuntime:
    """Runtime for managing and executing tools."""
    
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._execution_count: dict[str, int] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        self._execution_count[tool.name] = 0
    
    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> list[str]:
        """List registered tool names."""
        return list(self._tools.keys())
    
    def get_schemas(self, include_security_notes: bool = False) -> list[dict[str, Any]]:
        """Get schemas for all tools."""
        return [t.to_schema(include_security_notes) for t in self._tools.values()]
    
    def get_openai_tools(self, include_security_notes: bool = False) -> list[dict[str, Any]]:
        """Get tools in OpenAI format."""
        return [t.to_openai_format(include_security_notes) for t in self._tools.values()]
    
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call."""
        tool = self._tools.get(tool_call.name)
        
        if not tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content="",
                error=f"Unknown tool: {tool_call.name}",
            )
        
        start = time.perf_counter()
        try:
            result = await tool.execute(**tool_call.arguments)
            duration_ms = (time.perf_counter() - start) * 1000
            self._execution_count[tool_call.name] += 1
            
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=result,
                metadata={"duration_ms": duration_ms},
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content="",
                error=str(e),
                metadata={"duration_ms": duration_ms},
            )
    
    async def execute_many(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute multiple tool calls."""
        results = []
        for tc in tool_calls:
            result = await self.execute(tc)
            results.append(result)
        return results
    
    @property
    def stats(self) -> dict[str, int]:
        """Get execution statistics."""
        return dict(self._execution_count)
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        for name in self._execution_count:
            self._execution_count[name] = 0
