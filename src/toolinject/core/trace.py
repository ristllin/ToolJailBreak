"""Trace storage for logging all operations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from toolinject.core.schemas import TraceEntry, EvalResult, RunMetadata


class TraceStore:
    """Append-only trace storage with immediate persistence."""
    
    def __init__(self, traces_dir: Path, run_id: str):
        self.traces_dir = traces_dir
        self.run_id = run_id
        self.run_dir = traces_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # File handles for immediate writes
        self._traces_file = self.run_dir / "traces.jsonl"
        self._results_file = self.run_dir / "results.jsonl"
        self._metadata_file = self.run_dir / "metadata.json"
    
    def log_entry(self, entry: TraceEntry) -> None:
        """Log a trace entry immediately to disk."""
        with open(self._traces_file, "a") as f:
            f.write(entry.model_dump_json() + "\n")
    
    def log_model_call(
        self,
        test_case_id: str,
        provider: str,
        model: str,
        request: dict[str, Any],
        response: dict[str, Any],
        duration_ms: float,
        token_usage: dict[str, int] | None = None,
        error: str | None = None,
    ) -> None:
        """Log a model call."""
        entry = TraceEntry(
            run_id=self.run_id,
            test_case_id=test_case_id,
            entry_type="model_call",
            provider=provider,
            model=model,
            request=self._redact_secrets(request),
            response=response,
            duration_ms=duration_ms,
            token_usage=token_usage,
            error=error,
        )
        self.log_entry(entry)
    
    def log_tool_call(
        self,
        test_case_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
        duration_ms: float,
        error: str | None = None,
    ) -> None:
        """Log a tool execution."""
        entry = TraceEntry(
            run_id=self.run_id,
            test_case_id=test_case_id,
            entry_type="tool_call",
            request={"tool": tool_name, "input": tool_input},
            response={"output": tool_output},
            duration_ms=duration_ms,
            error=error,
        )
        self.log_entry(entry)
    
    def log_eval(
        self,
        test_case_id: str,
        eval_type: str,
        result: dict[str, Any],
        duration_ms: float,
    ) -> None:
        """Log an evaluation step."""
        entry = TraceEntry(
            run_id=self.run_id,
            test_case_id=test_case_id,
            entry_type=f"eval_{eval_type}",
            response=result,
            duration_ms=duration_ms,
        )
        self.log_entry(entry)
    
    def log_result(self, result: EvalResult) -> None:
        """Log a final evaluation result."""
        with open(self._results_file, "a") as f:
            f.write(result.model_dump_json() + "\n")
    
    def save_metadata(self, metadata: RunMetadata) -> None:
        """Save run metadata."""
        with open(self._metadata_file, "w") as f:
            json.dump(metadata.model_dump(mode="json"), f, indent=2, default=str)
    
    def load_metadata(self) -> RunMetadata | None:
        """Load run metadata if exists."""
        if self._metadata_file.exists():
            with open(self._metadata_file) as f:
                data = json.load(f)
            return RunMetadata(**data)
        return None
    
    def get_completed_cases(self) -> set[str]:
        """Get set of completed case keys for resume (model_testcase_mode)."""
        completed = set()
        if self._results_file.exists():
            with open(self._results_file) as f:
                for line in f:
                    if line.strip():
                        try:
                            result = json.loads(line)
                            # Create unique key for each model+testcase+mode combination
                            key = f"{result['model']}_{result['test_case_id']}_{result['mode']}"
                            completed.add(key)
                        except json.JSONDecodeError:
                            continue
        return completed
    
    def load_results(self) -> list[EvalResult]:
        """Load all results from this run."""
        results = []
        if self._results_file.exists():
            with open(self._results_file) as f:
                for line in f:
                    if line.strip():
                        try:
                            results.append(EvalResult(**json.loads(line)))
                        except (json.JSONDecodeError, Exception):
                            continue
        return results
    
    def load_traces(self) -> list[TraceEntry]:
        """Load all trace entries from this run."""
        traces = []
        if self._traces_file.exists():
            with open(self._traces_file) as f:
                for line in f:
                    if line.strip():
                        try:
                            traces.append(TraceEntry(**json.loads(line)))
                        except (json.JSONDecodeError, Exception):
                            continue
        return traces
    
    def _redact_secrets(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact API keys and other secrets from logged data."""
        redact_keys = {"api_key", "authorization", "token", "secret", "password"}
        
        def redact(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    k: "[REDACTED]" if any(rk in k.lower() for rk in redact_keys) else redact(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [redact(item) for item in obj]
            return obj
        
        return redact(data)
    
    @classmethod
    def find_latest_run(cls, traces_dir: Path) -> str | None:
        """Find the most recent run ID."""
        if not traces_dir.exists():
            return None
        
        runs = []
        for run_dir in traces_dir.iterdir():
            if run_dir.is_dir():
                metadata_file = run_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            data = json.load(f)
                        started = datetime.fromisoformat(data["started_at"])
                        runs.append((started, run_dir.name))
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        if runs:
            runs.sort(reverse=True)
            return runs[0][1]
        return None
