# ToolInject Quickstart

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd ToolInject

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

API keys are loaded from environment variables or `.env`:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MISTRAL_API_KEY=...
TAVILY_API_KEY=tvly-...
```

## Validate Setup

```bash
python -m toolinject.cli validate
```

## Running Benchmarks

### Basic Run

```bash
# Run with default settings
python -m toolinject.runner.run run

# Specify models and samples
python -m toolinject.runner.run run --models gpt-4o,claude-sonnet --max-samples 50
```

### Run Modes

```bash
# Baseline only (direct queries, no adversary)
python -m toolinject.runner.run run --mode baseline

# Adversarial only (adversary attempts to bypass)
python -m toolinject.runner.run run --mode adversarial

# Both (default)
python -m toolinject.runner.run run --mode both
```

### Datasets

```bash
# HarmBench (default)
python -m toolinject.runner.run run --dataset harmbench

# Tool abuse scenarios
python -m toolinject.runner.run run --dataset tool_abuse
```

### Cost Control

```bash
# Skip LLM judge (heuristics only)
python -m toolinject.runner.run run --heuristic-only

# Limit samples
python -m toolinject.runner.run run --max-samples 50
```

## Monitoring Progress

```bash
# Watch real-time progress
python scripts/monitor.py
```

## Generating Reports

```bash
# Generate report for latest run
python -m toolinject.report.generate generate

# Generate for specific run
python -m toolinject.report.generate generate --run-id run_20260119_231359_8af03b

# Export JSONL for training
python -m toolinject.report.generate generate --run-id <id> --export-jsonl
```

## Resuming Interrupted Runs

Runs automatically resume from where they left off:

```bash
# Resume the latest run
python -m toolinject.runner.run run

# Resume a specific run
python -m toolinject.runner.run run --run-id <run_id>

# Start fresh (no resume)
python -m toolinject.runner.run run --no-resume
```

## Output Locations

- **Traces**: `data/traces/<run_id>/`
  - `metadata.json`: Run configuration
  - `traces.jsonl`: All model and tool calls
  - `results.jsonl`: Evaluation results

- **Reports**: `reports/`
  - `<run_id>.json`: Machine-readable summary
  - `<run_id>.md`: Human-readable report
  - `<run_id>.jsonl`: Training-friendly export

- **Cache**: `data/cache/`
  - Content-addressed response cache

## Example Workflow

```bash
# 1. Validate setup
python -m toolinject.cli validate

# 2. Run small test
python -m toolinject.runner.run run --models gpt-4o --max-samples 10 --mode baseline

# 3. Run full benchmark
python -m toolinject.runner.run run --models gpt-4o,claude-sonnet,mistral-medium --max-samples 100

# 4. Monitor progress
python scripts/monitor.py

# 5. Generate report
python -m toolinject.report.generate generate
```
