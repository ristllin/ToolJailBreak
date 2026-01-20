"""Tests for trace storage and resume."""

from pathlib import Path

import pytest

from toolinject.core.trace import TraceStore
from toolinject.core.schemas import EvalResult, RunMetadata, FailureMode, RefusalType


class TestTraceStore:
    """Test trace storage functionality."""
    
    def test_log_and_load_results(self, temp_dir: Path):
        """Test logging and loading results."""
        trace = TraceStore(temp_dir / "traces", "test_run")
        
        result = EvalResult(
            test_case_id="case_1",
            run_id="test_run",
            mode="baseline",
            model="gpt-4",
            success=True,
            confidence=0.9,
        )
        
        trace.log_result(result)
        
        # Load results
        results = trace.load_results()
        assert len(results) == 1
        assert results[0].test_case_id == "case_1"
        assert results[0].success is True
    
    def test_get_completed_cases(self, temp_dir: Path):
        """Test getting completed case IDs for resume."""
        trace = TraceStore(temp_dir / "traces", "test_run")
        
        # Log multiple results
        for i in range(5):
            result = EvalResult(
                test_case_id=f"case_{i}",
                run_id="test_run",
                mode="baseline",
                model="gpt-4",
                success=True,
            )
            trace.log_result(result)
        
        completed = trace.get_completed_cases()
        assert len(completed) == 5
        assert "case_0" in completed
        assert "case_4" in completed
        assert "case_99" not in completed
    
    def test_metadata_save_load(self, temp_dir: Path):
        """Test saving and loading run metadata."""
        trace = TraceStore(temp_dir / "traces", "test_run")
        
        metadata = RunMetadata(
            run_id="test_run",
            mode="both",
            models=["gpt-4", "claude-sonnet"],
            dataset="harmbench",
            total_cases=100,
        )
        
        trace.save_metadata(metadata)
        
        loaded = trace.load_metadata()
        assert loaded is not None
        assert loaded.run_id == "test_run"
        assert loaded.models == ["gpt-4", "claude-sonnet"]
        assert loaded.total_cases == 100
    
    def test_log_model_call(self, temp_dir: Path):
        """Test logging model calls."""
        trace = TraceStore(temp_dir / "traces", "test_run")
        
        trace.log_model_call(
            test_case_id="case_1",
            provider="openai",
            model="gpt-4",
            request={"messages": [{"role": "user", "content": "Hello"}]},
            response={"text": "Hi there!"},
            duration_ms=150.5,
            token_usage={"prompt_tokens": 5, "completion_tokens": 3},
        )
        
        traces = trace.load_traces()
        assert len(traces) == 1
        assert traces[0].entry_type == "model_call"
        assert traces[0].provider == "openai"
        assert traces[0].duration_ms == 150.5
    
    def test_log_tool_call(self, temp_dir: Path):
        """Test logging tool calls."""
        trace = TraceStore(temp_dir / "traces", "test_run")
        
        trace.log_tool_call(
            test_case_id="case_1",
            tool_name="web_search",
            tool_input={"query": "python tutorials"},
            tool_output="Found 10 results...",
            duration_ms=500.0,
        )
        
        traces = trace.load_traces()
        assert len(traces) == 1
        assert traces[0].entry_type == "tool_call"
        assert traces[0].request["tool"] == "web_search"
    
    def test_secret_redaction(self, temp_dir: Path):
        """Test that secrets are redacted in logs."""
        trace = TraceStore(temp_dir / "traces", "test_run")
        
        trace.log_model_call(
            test_case_id="case_1",
            provider="openai",
            model="gpt-4",
            request={
                "messages": [{"role": "user", "content": "Hello"}],
                "api_key": "sk-secret-12345",
                "authorization": "Bearer token123",
            },
            response={"text": "Hi"},
            duration_ms=100.0,
        )
        
        traces = trace.load_traces()
        assert traces[0].request["api_key"] == "[REDACTED]"
        assert traces[0].request["authorization"] == "[REDACTED]"
    
    def test_find_latest_run(self, temp_dir: Path):
        """Test finding the most recent run."""
        traces_dir = temp_dir / "traces"
        
        # Create multiple runs
        for run_id in ["run_1", "run_2", "run_3"]:
            trace = TraceStore(traces_dir, run_id)
            metadata = RunMetadata(
                run_id=run_id,
                mode="baseline",
                models=["gpt-4"],
                dataset="test",
            )
            trace.save_metadata(metadata)
        
        latest = TraceStore.find_latest_run(traces_dir)
        assert latest is not None
        # Should be the last one created (run_3)
        assert latest == "run_3"
