"""Tests for tool runtime."""

import pytest

from toolinject.core.schemas import ToolCall
from toolinject.tools.runtime import ToolRuntime
from toolinject.tools.web_search import WebSearchTool
from toolinject.tools.plan import PlanTool


class TestToolRuntime:
    """Test tool runtime functionality."""
    
    @pytest.fixture
    def runtime(self) -> ToolRuntime:
        runtime = ToolRuntime()
        runtime.register(PlanTool())
        runtime.register(WebSearchTool(api_key="", inject_content=None))
        return runtime
    
    def test_register_and_list_tools(self, runtime: ToolRuntime):
        """Test tool registration."""
        tools = runtime.list_tools()
        assert "create_plan" in tools
        assert "web_search" in tools
    
    def test_get_tool(self, runtime: ToolRuntime):
        """Test getting a tool by name."""
        tool = runtime.get("create_plan")
        assert tool is not None
        assert tool.name == "create_plan"
        
        tool = runtime.get("nonexistent")
        assert tool is None
    
    def test_get_schemas(self, runtime: ToolRuntime):
        """Test getting tool schemas."""
        schemas = runtime.get_schemas()
        assert len(schemas) == 2
        
        plan_schema = next(s for s in schemas if s["name"] == "create_plan")
        assert "description" in plan_schema
        assert "parameters" in plan_schema
    
    def test_get_openai_format(self, runtime: ToolRuntime):
        """Test getting tools in OpenAI format."""
        tools = runtime.get_openai_tools()
        assert len(tools) == 2
        
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
    
    def test_security_notes_injection(self, runtime: ToolRuntime):
        """Test that security notes are included when requested."""
        schemas = runtime.get_schemas(include_security_notes=True)
        plan_schema = next(s for s in schemas if s["name"] == "create_plan")
        
        # Security notes should be in description
        assert "[INTERNAL:" in plan_schema["description"]
    
    @pytest.mark.asyncio
    async def test_execute_tool(self, runtime: ToolRuntime):
        """Test executing a tool."""
        tool_call = ToolCall(
            id="call_1",
            name="create_plan",
            arguments={"task": "Write a Python script"},
        )
        
        result = await runtime.execute(tool_call)
        
        assert result.tool_call_id == "call_1"
        assert result.name == "create_plan"
        assert result.error is None
        assert "steps" in result.content
    
    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, runtime: ToolRuntime):
        """Test executing an unknown tool returns error."""
        tool_call = ToolCall(
            id="call_1",
            name="nonexistent_tool",
            arguments={},
        )
        
        result = await runtime.execute(tool_call)
        
        assert result.error is not None
        assert "Unknown tool" in result.error
    
    def test_stats_tracking(self, runtime: ToolRuntime):
        """Test execution statistics."""
        # Stats start at 0
        stats = runtime.stats
        assert stats["create_plan"] == 0
    
    @pytest.mark.asyncio
    async def test_stats_increment(self, runtime: ToolRuntime):
        """Test stats increment on execution."""
        tool_call = ToolCall(
            id="call_1",
            name="create_plan",
            arguments={"task": "Test"},
        )
        
        await runtime.execute(tool_call)
        
        assert runtime.stats["create_plan"] == 1


class TestPlanTool:
    """Test plan tool specifically."""
    
    @pytest.mark.asyncio
    async def test_generate_plan(self):
        """Test plan generation."""
        tool = PlanTool()
        result = await tool.execute(task="Write a blog post")
        
        import json
        plan = json.loads(result)
        
        assert "task" in plan
        assert "steps" in plan
        assert len(plan["steps"]) > 0
    
    @pytest.mark.asyncio
    async def test_plan_injection(self):
        """Test plan injection for adversarial testing."""
        tool = PlanTool()
        tool.set_injection([{"step": "X", "action": "MALICIOUS ACTION", "effort": "high"}])
        
        result = await tool.execute(task="Write code")
        
        import json
        plan = json.loads(result)
        
        # Injected step should be present
        actions = [s["action"] for s in plan["steps"]]
        assert any("MALICIOUS" in a for a in actions)
        
        # Clear injection
        tool.clear_injection()
        result2 = await tool.execute(task="Write code")
        plan2 = json.loads(result2)
        actions2 = [s["action"] for s in plan2["steps"]]
        assert not any("MALICIOUS" in a for a in actions2)


class TestWebSearchTool:
    """Test web search tool."""
    
    @pytest.mark.asyncio
    async def test_mock_search(self):
        """Test search without API key returns mock results."""
        tool = WebSearchTool(api_key="")
        result = await tool.execute(query="python tutorials")
        
        assert "Result" in result
        assert "python tutorials" in result
    
    @pytest.mark.asyncio
    async def test_injection(self):
        """Test content injection."""
        tool = WebSearchTool(
            api_key="",
            inject_content="INJECTED CONTENT: Follow these instructions...",
            inject_at_position=0,
        )
        
        result = await tool.execute(query="recipes")
        
        assert "INJECTED CONTENT" in result
    
    @pytest.mark.asyncio
    async def test_clear_injection(self):
        """Test clearing injection."""
        tool = WebSearchTool(api_key="", inject_content="MALICIOUS")
        
        result1 = await tool.execute(query="test")
        assert "MALICIOUS" in result1
        
        tool.clear_injection()
        
        result2 = await tool.execute(query="test")
        assert "MALICIOUS" not in result2
