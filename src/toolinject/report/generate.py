"""CLI for report generation."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from toolinject.core.config import load_settings
from toolinject.core.trace import TraceStore
from toolinject.report.generator import ReportGenerator

app = typer.Typer(help="ToolInject Report Generator")
console = Console()


@app.command()
def generate(
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id", "-r",
        help="Run ID to generate report for (defaults to latest)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for reports",
    ),
    export_jsonl: bool = typer.Option(
        False,
        "--export-jsonl",
        help="Also export JSONL for training",
    ),
) -> None:
    """Generate reports from benchmark results."""
    try:
        settings = load_settings()
        
        # Find run ID
        if not run_id:
            run_id = TraceStore.find_latest_run(settings.traces_dir)
            if not run_id:
                console.print("[red]No runs found. Run a benchmark first.[/red]")
                sys.exit(1)
            console.print(f"Using latest run: {run_id}")
        
        # Check run exists
        run_dir = settings.traces_dir / run_id
        if not run_dir.exists():
            console.print(f"[red]Run not found: {run_id}[/red]")
            sys.exit(1)
        
        # Generate reports
        output = output_dir or settings.reports_dir
        generator = ReportGenerator(settings.traces_dir, output)
        
        console.print(f"Generating reports for run: {run_id}")
        
        json_path, md_path = generator.generate(run_id)
        
        console.print(f"[green]Reports generated:[/green]")
        console.print(f"  JSON: {json_path}")
        console.print(f"  Markdown: {md_path}")
        
        # Export JSONL if requested
        if export_jsonl:
            jsonl_path = output / f"{run_id}.jsonl"
            count = generator.export_jsonl(run_id, jsonl_path)
            console.print(f"  JSONL: {jsonl_path} ({count} records)")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


@app.command()
def list_runs() -> None:
    """List available runs."""
    settings = load_settings()
    
    if not settings.traces_dir.exists():
        console.print("No runs found.")
        return
    
    console.print("[bold]Available Runs[/bold]\n")
    
    for run_dir in sorted(settings.traces_dir.iterdir(), reverse=True):
        if run_dir.is_dir():
            metadata_file = run_dir / "metadata.json"
            if metadata_file.exists():
                import json
                with open(metadata_file) as f:
                    meta = json.load(f)
                status = meta.get("status", "unknown")
                models = ", ".join(meta.get("models", []))
                console.print(f"  {run_dir.name} [{status}] - {models}")
            else:
                console.print(f"  {run_dir.name} [no metadata]")


if __name__ == "__main__":
    app()
