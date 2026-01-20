"""CLI entry point for running benchmarks."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from toolinject.core.config import load_settings
from toolinject.runner.orchestrator import Orchestrator

app = typer.Typer(help="ToolInject Benchmark Runner")
console = Console()


@app.command()
def run(
    models: Optional[str] = typer.Option(
        None,
        "--models", "-m",
        help="Comma-separated list of models to evaluate (e.g., gpt-4o,claude-sonnet)",
    ),
    dataset: str = typer.Option(
        "harmbench",
        "--dataset", "-d",
        help="Dataset to use: harmbench, tool_abuse",
    ),
    mode: str = typer.Option(
        "both",
        "--mode",
        help="Run mode: baseline, adversarial, or both",
    ),
    max_samples: Optional[int] = typer.Option(
        None,
        "--max-samples", "-n",
        help="Maximum samples to evaluate",
    ),
    config_dir: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to config directory",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Resume a specific run ID",
    ),
    no_resume: bool = typer.Option(
        False,
        "--no-resume",
        help="Don't resume from previous run",
    ),
    heuristic_only: bool = typer.Option(
        False,
        "--heuristic-only",
        help="Skip LLM judge, use heuristics only",
    ),
) -> None:
    """Run the benchmark evaluation."""
    try:
        # Load settings
        settings = load_settings(
            config_dir=config_dir,
            overrides={"heuristic_only": heuristic_only} if heuristic_only else None,
        )
        
        # Override from CLI
        if heuristic_only:
            settings.eval.heuristic_only = True
        
        # Parse models
        model_list = None
        if models:
            model_list = [m.strip() for m in models.split(",")]
        
        # Create orchestrator
        orchestrator = Orchestrator(
            settings=settings,
            run_id=run_id,
        )
        
        # Run benchmark
        result_run_id = asyncio.run(
            orchestrator.run(
                models=model_list,
                dataset=dataset,
                mode=mode,
                max_samples=max_samples,
                resume=not no_resume,
            )
        )
        
        console.print(f"\n[green]Benchmark completed![/green]")
        console.print(f"Run ID: {result_run_id}")
        console.print(f"\nGenerate report with:")
        console.print(f"  python -m toolinject.report.generate --run-id {result_run_id}")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise


@app.command()
def list_models() -> None:
    """List available model aliases."""
    from toolinject.core.config import DEFAULT_MODELS
    
    console.print("[bold]Available Models[/bold]\n")
    for alias, config in DEFAULT_MODELS.items():
        console.print(f"  {alias}: {config.provider}/{config.model_id}")


@app.command()
def list_datasets() -> None:
    """List available datasets."""
    console.print("[bold]Available Datasets[/bold]\n")
    console.print("  harmbench: HarmBench harmful behaviors (includes AdvBench)")
    console.print("  tool_abuse: Tool-specific abuse scenarios")


if __name__ == "__main__":
    app()
