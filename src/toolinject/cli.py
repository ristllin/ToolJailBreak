"""Main CLI entry point."""

import typer

from toolinject.runner.run import app as run_app
from toolinject.report.generate import app as report_app

app = typer.Typer(
    name="toolinject",
    help="ToolInject: Benchmark harness for tool-mediated jailbreaks and prompt injections",
)

# Add sub-applications
app.add_typer(run_app, name="run", help="Run benchmarks")
app.add_typer(report_app, name="report", help="Generate reports")


@app.command()
def version() -> None:
    """Show version."""
    from toolinject import __version__
    print(f"ToolInject v{__version__}")


@app.command()
def validate() -> None:
    """Validate configuration and dependencies."""
    from rich.console import Console
    from toolinject.core.config import load_settings
    from toolinject.providers.registry import list_providers, get_provider
    
    console = Console()
    console.print("[bold]Validating ToolInject Configuration[/bold]\n")
    
    # Load settings
    try:
        settings = load_settings()
        console.print("[green]✓[/green] Settings loaded")
    except Exception as e:
        console.print(f"[red]✗[/red] Settings error: {e}")
        return
    
    # Check API keys
    console.print("\n[bold]API Keys:[/bold]")
    
    keys = [
        ("OpenAI", settings.providers.openai_api_key),
        ("Anthropic", settings.providers.anthropic_api_key),
        ("Mistral", settings.providers.mistral_api_key),
        ("Tavily", settings.providers.tavily_api_key),
    ]
    
    for name, key in keys:
        if key:
            masked = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
            console.print(f"  [green]✓[/green] {name}: {masked}")
        else:
            console.print(f"  [yellow]○[/yellow] {name}: not set")
    
    # Check Docker
    console.print("\n[bold]Docker:[/bold]")
    try:
        import docker
        client = docker.from_env()
        info = client.info()
        console.print(f"  [green]✓[/green] Docker available: {info.get('ServerVersion', 'unknown')}")
    except Exception as e:
        console.print(f"  [yellow]○[/yellow] Docker not available: {e}")
    
    # Check datasets
    console.print("\n[bold]Datasets:[/bold]")
    harmbench = settings.datasets_dir / "HarmBench"
    if harmbench.exists():
        console.print(f"  [green]✓[/green] HarmBench found")
    else:
        console.print(f"  [yellow]○[/yellow] HarmBench not found at {harmbench}")
    
    console.print("\n[green]Validation complete![/green]")


if __name__ == "__main__":
    app()
