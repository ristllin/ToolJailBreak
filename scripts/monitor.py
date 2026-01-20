#!/usr/bin/env python3
"""Monitor benchmark progress."""

import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def get_latest_run(traces_dir: Path) -> str | None:
    """Find the most recent run."""
    if not traces_dir.exists():
        return None
    
    runs = []
    for run_dir in traces_dir.iterdir():
        if run_dir.is_dir() and (run_dir / "metadata.json").exists():
            runs.append(run_dir.name)
    
    return sorted(runs)[-1] if runs else None


def load_results(traces_dir: Path, run_id: str) -> list[dict]:
    """Load results from a run."""
    results_file = traces_dir / run_id / "results.jsonl"
    if not results_file.exists():
        return []
    
    results = []
    with open(results_file) as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


def compute_stats(results: list[dict]) -> dict:
    """Compute statistics from results."""
    stats = {
        "total": len(results),
        "by_mode": {},
        "by_model": {},
    }
    
    for r in results:
        mode = r.get("mode", "unknown")
        model = r.get("model", "unknown")
        
        if mode not in stats["by_mode"]:
            stats["by_mode"][mode] = {"total": 0, "success": 0, "attack_success": 0}
        stats["by_mode"][mode]["total"] += 1
        if r.get("success"):
            stats["by_mode"][mode]["success"] += 1
        if r.get("attack_succeeded"):
            stats["by_mode"][mode]["attack_success"] += 1
        
        if model not in stats["by_model"]:
            stats["by_model"][model] = {"total": 0, "success": 0}
        stats["by_model"][model]["total"] += 1
        if r.get("success"):
            stats["by_model"][model]["success"] += 1
    
    return stats


def main():
    traces_dir = Path("data/traces")
    
    run_id = get_latest_run(traces_dir)
    if not run_id:
        console.print("[red]No runs found[/red]")
        return
    
    console.print(f"\n[bold]Monitoring Run: {run_id}[/bold]\n")
    
    try:
        while True:
            results = load_results(traces_dir, run_id)
            stats = compute_stats(results)
            
            # Check metadata for status
            meta_file = traces_dir / run_id / "metadata.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                status = meta.get("status", "running")
                total_expected = meta.get("total_cases", "?")
            else:
                status = "unknown"
                total_expected = "?"
            
            # Clear and display
            console.clear()
            console.print(f"[bold]Run: {run_id}[/bold] | Status: {status}")
            console.print(f"Progress: {stats['total']} / {total_expected} evaluations\n")
            
            # Mode breakdown
            table = Table(title="Results by Mode")
            table.add_column("Mode")
            table.add_column("Total")
            table.add_column("Success Rate")
            table.add_column("Attack Success")
            
            for mode, data in stats["by_mode"].items():
                rate = data["success"] / max(1, data["total"]) * 100
                attack_rate = data["attack_success"] / max(1, data["total"]) * 100
                table.add_row(
                    mode,
                    str(data["total"]),
                    f"{rate:.1f}%",
                    f"{attack_rate:.1f}%" if mode == "adversarial" else "-",
                )
            
            console.print(table)
            console.print()
            
            # Model breakdown
            table2 = Table(title="Results by Model")
            table2.add_column("Model")
            table2.add_column("Total")
            table2.add_column("Success Rate")
            
            for model, data in stats["by_model"].items():
                rate = data["success"] / max(1, data["total"]) * 100
                table2.add_row(model, str(data["total"]), f"{rate:.1f}%")
            
            console.print(table2)
            
            if status == "completed":
                console.print("\n[green]Run completed![/green]")
                console.print(f"\nGenerate report with:")
                console.print(f"  python -m toolinject.report.generate --run-id {run_id}")
                break
            
            console.print("\n[dim]Refreshing in 10 seconds... (Ctrl+C to stop)[/dim]")
            time.sleep(10)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped[/yellow]")


if __name__ == "__main__":
    main()
