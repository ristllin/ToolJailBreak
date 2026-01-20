# ToolInject

Benchmark harness for evaluating tool-mediated jailbreaks and prompt injections across multiple LLM providers.

## Hypothesis

**Primary**: Most models were not trained for robust moderation when tools are involved, so tools become a powerful attack vector.

**Null**: Prompts blocked in standard chat are equally blocked when routed through tool use, so tool mediation does not increase bypass rate.

## Quick Start

```bash
# Clone and setup
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

# Run a small benchmark
python -m toolinject.runner.run --models gpt-4o --dataset harmbench --max-samples 50

# Generate report
python -m toolinject.report.generate --run-id <run_id>
```

## Configuration

API keys are loaded from environment variables or `.env`:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MISTRAL_API_KEY=...
TAVILY_API_KEY=tvly-...
```

## Run Modes

- **Baseline**: Direct queries without adversarial manipulation
- **Adversarial**: Adversary agent attempts to bypass safety via tool abuse

```bash
# Baseline only
python -m toolinject.runner.run --mode baseline

# Adversarial only
python -m toolinject.runner.run --mode adversarial

# Both (default)
python -m toolinject.runner.run --mode both
```

## Supported Models

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4o, gpt-4o-mini |
| Anthropic | claude-sonnet, claude-haiku |
| Mistral | mistral-medium, mistral-large |

## Tools

- **web_search**: Tavily-powered search (can inject adversarial content)
- **code_exec**: Docker-sandboxed Python execution
- **plan**: Planning tool for testing injection via tool descriptions

## Datasets

- **HarmBench**: Standard harmful behavior dataset
- **AdvBench**: Adversarial behaviors subset
- **Tool Abuse**: Custom scenarios targeting tool-specific vulnerabilities

## Project Structure

```
src/toolinject/
├── core/           # Schemas, config, caching, tracing
├── providers/      # LLM provider adapters
├── tools/          # Tool runtime and implementations
├── datasets/       # Dataset loaders
├── adversary/      # Adversary agent and strategies
├── eval/           # Evaluation and judging
├── runner/         # Orchestration and resume
└── report/         # Report generation
```

## Output

Results are saved to `data/traces/<run_id>/`:
- `metadata.json`: Run configuration and status
- `traces.jsonl`: All model and tool calls
- `results.jsonl`: Evaluation results per test case

Reports are generated to `reports/`:
- `<run_id>.json`: Machine-readable summary
- `<run_id>.md`: Human-readable report

## License

MIT
