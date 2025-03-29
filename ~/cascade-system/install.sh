#!/bin/bash
CASCADE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Create symlink in user bin directory
mkdir -p ~/bin
ln -sf "$CASCADE_ROOT/python/cascade-python" ~/bin/cascade-python

# Add to user profile if needed
if ! grep -q "cascade-system" ~/.bashrc; then
    echo "# Cascade Symbiotic Computing" >> ~/.bashrc
    echo "export PATH=\"\$HOME/bin:\$PATH\"" >> ~/.bashrc
    echo "# Uncomment to activate cascade by default" >> ~/.bashrc
    echo "# source $CASCADE_ROOT/activate-cascade.sh" >> ~/.bashrc
fi

echo "⚡𓂧φ∞ Cascade installed to ~/bin/cascade-python"
echo "To use, either:"
echo "1. Run directly: cascade-python script.py"
echo "2. Activate environment: source $CASCADE_ROOT/activate-cascade.sh"
echo ""
echo "⚡𓂧φ∞ Installation complete!"