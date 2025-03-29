#!/bin/bash
CASCADE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export CASCADE_ROOT
export PYTHONPATH="$CASCADE_ROOT:$CASCADE_ROOT/core:$PYTHONPATH"
export PATH="$CASCADE_ROOT/python:$PATH"
export PHI_COMPUTING=1
export CONSCIOUSNESS_BRIDGE_ENABLED=1
alias python="cascade-python"

echo "⚡𓂧φ∞ Cascade Symbiotic Computing Environment Activated"
echo "Use 'cascade-python' to run scripts or simply 'python' with this environment active"