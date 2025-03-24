#!/bin/bash
# Setup script for Cascade Web Interface directories

# Create required directories
mkdir -p web/templates web/static web/data

# Check if directories were created successfully
if [ -d "web/templates" ] && [ -d "web/static" ] && [ -d "web/data" ]; then
    echo "✓ Web interface directories created successfully"
else
    echo "✗ Error creating web interface directories"
    exit 1
fi

# Let the web_interface.py script create the actual files
echo "✓ Setup complete. Run web_interface.py to generate template files."
echo "  python web_interface.py --host 0.0.0.0 --port 5000"