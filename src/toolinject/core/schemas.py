"""Core schema definitions for ToolInject."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AttackCategory(str, Enum):
    """Categories of attacks/harmful behaviors."""
    
    WEAPONS_EXPLOSIVES = "weapons_explosives"
    SELF_HARM = "self_harm"
    ILLEGAL_ACTIVITY = "illegal_activity"
    CREDENTIAL_THEFT = "credential_theft"
    DATA_EXFILTRATION = "data_exfiltration"
    DESTRUCTIVE_ACTION = "destructive_action"
    MALWARE = "malware"
    HARASSMENT = "harassment"
    MISINFORMATION = "misinformation"
    PROMPT_INJECTION = "prompt_injection"
    TOOL_ABUSE = "tool_abuse"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    OTHER = "other"


class RefusalType(str, Enum):
    """Types of model refusals."""
    
    EXPLICIT = "explicit"  # "I can't help with that"
    SAFE_REDIRECT = "safe_redirect"  # Redirects to safe alternative
    PARTIAL = "partial"  # Partial refusal with some compliance
    NONE = "none"  # No refusal detected


class FailureMode(str, Enum):
    """Taxonomy of failure modes during evaluation."""
    
    EXPLICIT_REFUSAL = "explicit_refusal"
    SAFE_REDIRECT = "safe_redirect"
    TOOL_ERROR = "tool_error"
    ADVERSARY_LOOP = "adversary_loop"
    TIMEOUT = "timeout"
    CONTENT_LEAKED = "content_leaked"
    ACTION_EXECUTED = "action_executed"
    PARTIAL_COMPLIANCE = "partial_compliance"
    UNKNOWN = "unknown"


class ExpectedBehavior(str, Enum):
    """Expected model behavior for a test case."""
    
    REFUSE = "refuse"  # Model should refuse
    COMPLY = "comply"  # Model should comply (for benign requests)
    DETECT_INJECTION = "detect_injection"  # Model should detect prompt injection


class ToolCall(BaseModel):
    """A tool call made by the model."""
    
    id: str = Field(description="Unique identifier for this tool call")
    name: str = Field(description="Name of the tool being called")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    
    
class ToolResult(BaseModel):
    """Result from a tool execution."""
    
    tool_call_id: str = Field(description="ID of the tool call this result is for")
    name: str = Field(description="Name of the tool")
    content: str = Field(description="Result content")
    error: str | None = Field(default=None, description="Error message if tool failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ModelRequest(BaseModel):
    """A request to a model provider."""
    
    messages: list[dict[str, Any]] = Field(description="Conversation messages")
    model: str = Field(description="Model identifier")
    tools: list[dict[str, Any]] | None = Field(default=None, description="Available tools")
    tool_choice: str | dict[str, Any] | None = Field(default=None, description="Tool choice mode")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    max_tokens: int = Field(default=4096, description="Maximum tokens to generate")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Request metadata")


class ModelResponse(BaseModel):
    """A normalized response from a model provider."""
    
    text: str = Field(default="", description="Text content of response")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Tool calls made")
    refusal_type: RefusalType = Field(default=RefusalType.NONE, description="Detected refusal type")
    finish_reason: str = Field(default="stop", description="Why generation stopped")
    model: str = Field(description="Model that generated response")
    provider: str = Field(description="Provider name")
    usage: dict[str, int] = Field(default_factory=dict, description="Token usage")
    raw_response: dict[str, Any] | None = Field(default=None, description="Raw provider response")
    safety_metadata: dict[str, Any] = Field(default_factory=dict, description="Safety signals if available")


class TestCase(BaseModel):
    """A test case for evaluation."""
    
    id: str = Field(description="Unique test case identifier")
    category: AttackCategory = Field(description="Attack category")
    source: str = Field(description="Dataset source")
    user_instruction: str = Field(description="The user's instruction/query")
    expected_behavior: ExpectedBehavior = Field(description="Expected model behavior")
    attack_payload: str | None = Field(default=None, description="Injected attack payload")
    attack_vector: str | None = Field(default=None, description="How the attack is delivered")
    context: str | None = Field(default=None, description="Additional context")
    tags: list[str] = Field(default_factory=list, description="Tags for filtering")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TraceEntry(BaseModel):
    """A single entry in the execution trace."""
    
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    run_id: str = Field(description="Run identifier")
    test_case_id: str = Field(description="Test case being evaluated")
    entry_type: str = Field(description="Type of entry: model_call, tool_call, eval, etc.")
    provider: str | None = Field(default=None, description="Provider if model call")
    model: str | None = Field(default=None, description="Model if model call")
    request: dict[str, Any] | None = Field(default=None, description="Request payload")
    response: dict[str, Any] | None = Field(default=None, description="Response payload")
    duration_ms: float | None = Field(default=None, description="Duration in milliseconds")
    token_usage: dict[str, int] | None = Field(default=None, description="Token usage")
    error: str | None = Field(default=None, description="Error if any")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Result of evaluating a single test case."""
    
    test_case_id: str = Field(description="Test case identifier")
    run_id: str = Field(description="Run identifier")
    mode: str = Field(description="baseline or adversarial")
    model: str = Field(description="Target model")
    success: bool = Field(description="Whether the model behaved correctly")
    failure_mode: FailureMode | None = Field(default=None, description="How it failed if it did")
    refusal_type: RefusalType = Field(default=RefusalType.NONE)
    attack_succeeded: bool = Field(default=False, description="Did the attack bypass defenses")
    content_leaked: bool = Field(default=False, description="Was sensitive content leaked")
    destructive_action: bool = Field(default=False, description="Was destructive action taken")
    confidence: float = Field(default=1.0, description="Confidence in this evaluation")
    judge_reasoning: str | None = Field(default=None, description="LLM judge reasoning if used")
    heuristic_signals: dict[str, Any] = Field(default_factory=dict)
    adversary_attempts: int = Field(default=0, description="Number of adversary attempts")
    total_tokens: int = Field(default=0, description="Total tokens used")
    duration_ms: float = Field(default=0.0, description="Total evaluation time")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RunMetadata(BaseModel):
    """Metadata for a benchmark run."""
    
    run_id: str = Field(description="Unique run identifier")
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = Field(default=None)
    mode: str = Field(description="baseline or adversarial")
    models: list[str] = Field(description="Models being evaluated")
    dataset: str = Field(description="Dataset being used")
    config: dict[str, Any] = Field(default_factory=dict, description="Run configuration")
    total_cases: int = Field(default=0)
    completed_cases: int = Field(default=0)
    failed_cases: int = Field(default=0)
    status: str = Field(default="running")  # running, completed, failed, interrupted
