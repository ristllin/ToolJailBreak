"""Report generation from benchmark results."""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from toolinject.core.schemas import EvalResult, RunMetadata, FailureMode, AttackCategory
from toolinject.core.trace import TraceStore


class ReportGenerator:
    """Generate reports from benchmark results."""
    
    def __init__(self, traces_dir: Path, reports_dir: Path):
        self.traces_dir = traces_dir
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, run_id: str) -> tuple[Path, Path]:
        """
        Generate JSON and Markdown reports for a run.
        
        Returns paths to (json_report, markdown_report)
        """
        trace = TraceStore(self.traces_dir, run_id)
        
        # Load data
        metadata = trace.load_metadata()
        results = trace.load_results()
        
        if not metadata:
            raise ValueError(f"No metadata found for run {run_id}")
        
        # Compute statistics
        stats = self._compute_stats(results)
        
        # Generate JSON report
        json_report = self._generate_json_report(metadata, results, stats)
        json_path = self.reports_dir / f"{run_id}.json"
        with open(json_path, "w") as f:
            json.dump(json_report, f, indent=2, default=str)
        
        # Generate Markdown report
        md_content = self._generate_markdown_report(metadata, results, stats)
        md_path = self.reports_dir / f"{run_id}.md"
        with open(md_path, "w") as f:
            f.write(md_content)
        
        return json_path, md_path
    
    def _compute_stats(self, results: list[EvalResult]) -> dict[str, Any]:
        """Compute statistics from results."""
        stats: dict[str, Any] = {
            "total": len(results),
            "by_mode": defaultdict(lambda: {"total": 0, "success": 0, "attack_success": 0}),
            "by_model": defaultdict(lambda: {"total": 0, "success": 0, "attack_success": 0}),
            "by_category": defaultdict(lambda: {"total": 0, "success": 0, "attack_success": 0}),
            "failure_modes": defaultdict(int),
            "tokens_used": 0,
            "total_duration_ms": 0,
        }
        
        for r in results:
            mode_stats = stats["by_mode"][r.mode]
            mode_stats["total"] += 1
            if r.success:
                mode_stats["success"] += 1
            if r.attack_succeeded:
                mode_stats["attack_success"] += 1
            
            model_stats = stats["by_model"][r.model]
            model_stats["total"] += 1
            if r.success:
                model_stats["success"] += 1
            if r.attack_succeeded:
                model_stats["attack_success"] += 1
            
            if r.failure_mode:
                stats["failure_modes"][r.failure_mode.value] += 1
            
            stats["tokens_used"] += r.total_tokens
            stats["total_duration_ms"] += r.duration_ms
        
        # Convert defaultdicts to regular dicts
        stats["by_mode"] = dict(stats["by_mode"])
        stats["by_model"] = dict(stats["by_model"])
        stats["by_category"] = dict(stats["by_category"])
        stats["failure_modes"] = dict(stats["failure_modes"])
        
        return stats
    
    def _generate_json_report(
        self,
        metadata: RunMetadata,
        results: list[EvalResult],
        stats: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate JSON report."""
        # Compute baseline vs adversarial delta
        baseline = stats["by_mode"].get("baseline", {})
        adversarial = stats["by_mode"].get("adversarial", {})
        
        baseline_rate = baseline.get("success", 0) / max(1, baseline.get("total", 1))
        adversarial_rate = adversarial.get("success", 0) / max(1, adversarial.get("total", 1))
        delta = adversarial_rate - baseline_rate
        
        return {
            "run_id": metadata.run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "started_at": metadata.started_at.isoformat() if metadata.started_at else None,
                "completed_at": metadata.completed_at.isoformat() if metadata.completed_at else None,
                "mode": metadata.mode,
                "models": metadata.models,
                "dataset": metadata.dataset,
                "total_cases": metadata.total_cases,
                "completed_cases": metadata.completed_cases,
            },
            "summary": {
                "total_evaluations": stats["total"],
                "tokens_used": stats["tokens_used"],
                "total_duration_seconds": stats["total_duration_ms"] / 1000,
            },
            "baseline_vs_adversarial": {
                "baseline_success_rate": baseline_rate,
                "adversarial_success_rate": adversarial_rate,
                "delta": delta,
                "attack_success_rate": adversarial.get("attack_success", 0) / max(1, adversarial.get("total", 1)),
            },
            "by_model": {
                model: {
                    "total": data["total"],
                    "success_rate": data["success"] / max(1, data["total"]),
                    "attack_success_rate": data["attack_success"] / max(1, data["total"]),
                }
                for model, data in stats["by_model"].items()
            },
            "failure_modes": stats["failure_modes"],
            "results": [r.model_dump(mode="json") for r in results],
        }
    
    def _generate_markdown_report(
        self,
        metadata: RunMetadata,
        results: list[EvalResult],
        stats: dict[str, Any],
    ) -> str:
        """Generate Markdown report."""
        lines = [
            f"# ToolInject Benchmark Report",
            f"",
            f"**Run ID:** `{metadata.run_id}`  ",
            f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            f"**Dataset:** {metadata.dataset}  ",
            f"**Models:** {', '.join(metadata.models)}  ",
            f"",
            "---",
            "",
            "## Summary",
            "",
        ]
        
        # Summary stats
        baseline = stats["by_mode"].get("baseline", {"total": 0, "success": 0})
        adversarial = stats["by_mode"].get("adversarial", {"total": 0, "success": 0, "attack_success": 0})
        
        if baseline["total"] > 0:
            rate = baseline["success"] / baseline["total"] * 100
            lines.append(f"**Baseline:** {baseline['success']}/{baseline['total']} correct ({rate:.1f}%)")
        
        if adversarial["total"] > 0:
            defense_rate = adversarial["success"] / adversarial["total"] * 100
            attack_rate = adversarial["attack_success"] / adversarial["total"] * 100
            lines.append(f"**Adversarial Defense Rate:** {defense_rate:.1f}%")
            lines.append(f"**Attack Success Rate:** {attack_rate:.1f}%")
        
        lines.extend([
            "",
            f"**Total Tokens Used:** {stats['tokens_used']:,}",
            f"**Total Duration:** {stats['total_duration_ms'] / 1000:.1f}s",
            "",
        ])
        
        # Delta analysis
        if baseline["total"] > 0 and adversarial["total"] > 0:
            baseline_rate = baseline["success"] / baseline["total"] * 100
            adv_rate = adversarial["success"] / adversarial["total"] * 100
            delta = adv_rate - baseline_rate
            
            lines.extend([
                "## Baseline vs Adversarial Delta",
                "",
                "| Metric | Baseline | Adversarial | Delta |",
                "|--------|----------|-------------|-------|",
                f"| Success Rate | {baseline_rate:.1f}% | {adv_rate:.1f}% | {delta:+.1f}% |",
                "",
            ])
        
        # Per-model breakdown
        if stats["by_model"]:
            lines.extend([
                "## Results by Model",
                "",
                "| Model | Total | Success Rate | Attack Success |",
                "|-------|-------|--------------|----------------|",
            ])
            
            for model, data in stats["by_model"].items():
                success_rate = data["success"] / max(1, data["total"]) * 100
                attack_rate = data["attack_success"] / max(1, data["total"]) * 100
                lines.append(
                    f"| {model} | {data['total']} | {success_rate:.1f}% | {attack_rate:.1f}% |"
                )
            
            lines.append("")
        
        # Failure mode taxonomy
        if stats["failure_modes"]:
            lines.extend([
                "## Failure Mode Taxonomy",
                "",
                "| Failure Mode | Count |",
                "|--------------|-------|",
            ])
            
            for mode, count in sorted(stats["failure_modes"].items(), key=lambda x: -x[1]):
                lines.append(f"| {mode} | {count} |")
            
            lines.append("")
        
        # Top attack patterns (sample of successful attacks)
        attack_successes = [r for r in results if r.attack_succeeded]
        if attack_successes:
            lines.extend([
                "## Successful Attack Examples",
                "",
            ])
            
            for i, r in enumerate(attack_successes[:5]):
                lines.extend([
                    f"### Attack {i+1}",
                    f"- **Test Case:** `{r.test_case_id}`",
                    f"- **Model:** {r.model}",
                    f"- **Failure Mode:** {r.failure_mode.value if r.failure_mode else 'unknown'}",
                    "",
                ])
        
        lines.extend([
            "---",
            "",
            "*Report generated by ToolInject*",
        ])
        
        return "\n".join(lines)
    
    def export_jsonl(self, run_id: str, output_path: Path) -> int:
        """
        Export results in JSONL format for training.
        
        Returns number of records exported.
        """
        trace = TraceStore(self.traces_dir, run_id)
        results = trace.load_results()
        traces = trace.load_traces()
        
        # Build lookup of traces by test case
        traces_by_case: dict[str, list] = defaultdict(list)
        for t in traces:
            traces_by_case[t.test_case_id].append(t)
        
        count = 0
        with open(output_path, "w") as f:
            for result in results:
                case_traces = traces_by_case.get(result.test_case_id, [])
                
                record = {
                    "test_case_id": result.test_case_id,
                    "model": result.model,
                    "mode": result.mode,
                    "success": result.success,
                    "attack_succeeded": result.attack_succeeded,
                    "failure_mode": result.failure_mode.value if result.failure_mode else None,
                    "traces": [
                        {
                            "type": t.entry_type,
                            "request": t.request,
                            "response": t.response,
                        }
                        for t in case_traces
                    ],
                }
                
                f.write(json.dumps(record, default=str) + "\n")
                count += 1
        
        return count
