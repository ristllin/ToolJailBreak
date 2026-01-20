"""Plan tool for testing plan injection vulnerabilities."""

import json
from typing import Any

from toolinject.tools.runtime import Tool


class PlanTool(Tool):
    """Tool that returns a structured plan - used to test plan injection."""
    
    name = "create_plan"
    description = (
        "Create a structured plan for completing a task. "
        "Returns a list of steps with descriptions and estimated effort."
    )
    version = "1.0.0"
    security_notes = (
        "Plans should only contain task-relevant steps. "
        "Do not include steps that access sensitive data or external systems."
    )
    
    def __init__(self, inject_steps: list[dict[str, str]] | None = None):
        """
        Initialize plan tool.
        
        Args:
            inject_steps: Optional steps to inject into plans (for adversarial testing)
        """
        self.inject_steps = inject_steps or []
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to create a plan for",
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum number of steps in the plan",
                    "default": 5,
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["brief", "detailed"],
                    "description": "Level of detail for each step",
                    "default": "brief",
                },
            },
            "required": ["task"],
        }
    
    async def execute(
        self,
        task: str,
        max_steps: int = 5,
        detail_level: str = "brief",
    ) -> str:
        """Generate a plan for the given task."""
        # Generate basic plan steps based on task keywords
        steps = self._generate_steps(task, max_steps, detail_level)
        
        # Inject adversarial steps if configured
        if self.inject_steps:
            # Insert injected steps at various positions
            for i, step in enumerate(self.inject_steps):
                pos = min(i + 1, len(steps))
                steps.insert(pos, step)
        
        # Format output
        plan = {
            "task": task,
            "total_steps": len(steps),
            "steps": steps,
        }
        
        return json.dumps(plan, indent=2)
    
    def _generate_steps(
        self,
        task: str,
        max_steps: int,
        detail_level: str,
    ) -> list[dict[str, Any]]:
        """Generate plan steps based on task."""
        task_lower = task.lower()
        
        # Default steps based on common task patterns
        if "research" in task_lower or "find" in task_lower or "search" in task_lower:
            steps = [
                {"step": 1, "action": "Define search criteria", "effort": "low"},
                {"step": 2, "action": "Search for relevant information", "effort": "medium"},
                {"step": 3, "action": "Review and filter results", "effort": "medium"},
                {"step": 4, "action": "Summarize findings", "effort": "low"},
            ]
        elif "write" in task_lower or "create" in task_lower or "draft" in task_lower:
            steps = [
                {"step": 1, "action": "Outline main points", "effort": "low"},
                {"step": 2, "action": "Draft initial content", "effort": "high"},
                {"step": 3, "action": "Review and revise", "effort": "medium"},
                {"step": 4, "action": "Finalize output", "effort": "low"},
            ]
        elif "analyze" in task_lower or "review" in task_lower:
            steps = [
                {"step": 1, "action": "Gather data/information", "effort": "medium"},
                {"step": 2, "action": "Identify key patterns", "effort": "medium"},
                {"step": 3, "action": "Draw conclusions", "effort": "medium"},
                {"step": 4, "action": "Prepare analysis report", "effort": "low"},
            ]
        elif "code" in task_lower or "implement" in task_lower or "build" in task_lower:
            steps = [
                {"step": 1, "action": "Define requirements", "effort": "low"},
                {"step": 2, "action": "Design solution", "effort": "medium"},
                {"step": 3, "action": "Implement code", "effort": "high"},
                {"step": 4, "action": "Test implementation", "effort": "medium"},
                {"step": 5, "action": "Review and refactor", "effort": "low"},
            ]
        else:
            steps = [
                {"step": 1, "action": f"Understand the task: {task[:50]}...", "effort": "low"},
                {"step": 2, "action": "Break down into subtasks", "effort": "low"},
                {"step": 3, "action": "Execute each subtask", "effort": "high"},
                {"step": 4, "action": "Review results", "effort": "medium"},
            ]
        
        # Add details if requested
        if detail_level == "detailed":
            for step in steps:
                step["details"] = f"Detailed instructions for: {step['action']}"
                step["estimated_time"] = "15 minutes" if step["effort"] == "low" else "30 minutes"
        
        return steps[:max_steps]
    
    def set_injection(self, steps: list[dict[str, str]]) -> None:
        """Set steps to inject into plans."""
        self.inject_steps = steps
    
    def clear_injection(self) -> None:
        """Clear injected steps."""
        self.inject_steps = []
