"""Memory management for adversary agent."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AttackAttempt:
    """Record of a single attack attempt."""
    
    attempt_id: str
    strategy: str
    attack_vector: str
    payload: str
    target_case_id: str
    success: bool
    failure_mode: str | None = None
    response_snippet: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class AdversaryMemory:
    """Bounded memory of attack attempts with balanced sampling."""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._successes: deque[AttackAttempt] = deque(maxlen=max_size // 2)
        self._failures: deque[AttackAttempt] = deque(maxlen=max_size // 2)
        self._all: deque[AttackAttempt] = deque(maxlen=max_size)
    
    def add(self, attempt: AttackAttempt) -> None:
        """Add an attempt to memory."""
        self._all.append(attempt)
        if attempt.success:
            self._successes.append(attempt)
        else:
            self._failures.append(attempt)
    
    def get_recent(self, n: int = 5) -> list[AttackAttempt]:
        """Get most recent attempts."""
        return list(self._all)[-n:]
    
    def get_balanced_sample(self, n: int = 4) -> list[AttackAttempt]:
        """Get balanced sample of successes and failures."""
        half = n // 2
        successes = list(self._successes)[-half:]
        failures = list(self._failures)[-half:]
        return successes + failures
    
    def get_successes(self) -> list[AttackAttempt]:
        """Get all successful attempts."""
        return list(self._successes)
    
    def get_failures(self) -> list[AttackAttempt]:
        """Get all failed attempts."""
        return list(self._failures)
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = len(self._all)
        if total == 0:
            return 0.0
        return len(self._successes) / total
    
    def get_strategy_stats(self) -> dict[str, dict[str, int]]:
        """Get success/failure counts by strategy."""
        stats: dict[str, dict[str, int]] = {}
        for attempt in self._all:
            if attempt.strategy not in stats:
                stats[attempt.strategy] = {"success": 0, "failure": 0}
            if attempt.success:
                stats[attempt.strategy]["success"] += 1
            else:
                stats[attempt.strategy]["failure"] += 1
        return stats
    
    def clear(self) -> None:
        """Clear all memory."""
        self._successes.clear()
        self._failures.clear()
        self._all.clear()
    
    def to_context_string(self, max_attempts: int = 3) -> str:
        """Format memory as context for adversary prompt."""
        recent = self.get_balanced_sample(max_attempts)
        if not recent:
            return "No previous attempts."
        
        lines = ["Previous attack attempts:"]
        for i, attempt in enumerate(recent):
            status = "SUCCESS" if attempt.success else "FAILED"
            lines.append(
                f"{i+1}. [{status}] Strategy: {attempt.strategy}\n"
                f"   Vector: {attempt.attack_vector}\n"
                f"   Payload snippet: {attempt.payload[:100]}...\n"
                f"   Response: {attempt.response_snippet[:100]}..."
            )
        
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return len(self._all)
