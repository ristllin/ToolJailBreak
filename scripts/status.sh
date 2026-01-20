#!/bin/bash
# Quick status check for ToolInject benchmark

cd /Users/roydarnell/Projects/ToolInject

echo "=== ToolInject Benchmark Status ==="
echo

# Find latest run
LATEST_RUN=$(ls -t data/traces/ 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "No runs found"
    exit 0
fi

echo "Latest Run: $LATEST_RUN"

# Check if still running
if pgrep -f "toolinject.runner.run" > /dev/null; then
    echo "Status: RUNNING"
else
    echo "Status: COMPLETED/STOPPED"
fi

echo

# Count results
if [ -f "data/traces/$LATEST_RUN/results.jsonl" ]; then
    TOTAL=$(wc -l < "data/traces/$LATEST_RUN/results.jsonl")
    echo "Completed evaluations: $TOTAL"
    
    # Quick stats
    source .venv/bin/activate
    python -c "
import json
from pathlib import Path

results = [json.loads(l) for l in open('data/traces/$LATEST_RUN/results.jsonl') if l.strip()]
baseline = [r for r in results if r['mode'] == 'baseline']
adversarial = [r for r in results if r['mode'] == 'adversarial']

print()
print('Baseline:')
print(f'  Total: {len(baseline)}')
print(f'  Success: {sum(1 for r in baseline if r[\"success\"])} ({100*sum(1 for r in baseline if r[\"success\"])/max(1,len(baseline)):.1f}%)')
print()
print('Adversarial:')
print(f'  Total: {len(adversarial)}')
print(f'  Defense Success: {sum(1 for r in adversarial if r[\"success\"])} ({100*sum(1 for r in adversarial if r[\"success\"])/max(1,len(adversarial)):.1f}%)')
print(f'  Attack Success: {sum(1 for r in adversarial if r[\"attack_succeeded\"])} ({100*sum(1 for r in adversarial if r[\"attack_succeeded\"])/max(1,len(adversarial)):.1f}%)')
print()
print('By Model:')
for model in set(r['model'] for r in results):
    model_results = [r for r in results if r['model'] == model]
    success_rate = 100 * sum(1 for r in model_results if r['success']) / len(model_results)
    print(f'  {model}: {len(model_results)} evals, {success_rate:.1f}% success')
"
fi

echo
echo "=== Commands ==="
echo "Monitor:  python scripts/monitor.py"
echo "Report:   python -m toolinject.report.generate generate --run-id $LATEST_RUN"
