#!/bin/bash
# Test JARVIS with real language models

echo "🚀 Testing JARVIS Core with real language models..."
echo "=================================================="
echo ""

# Use miniforge Python
PYTHON="/Users/derekjrussell/miniforge3/bin/python"

if [ "$1" == "demo" ]; then
    echo "Running architecture demo..."
    $PYTHON test_jarvis_core_demo.py
elif [ "$1" == "interactive" ]; then
    echo "Running interactive test..."
    $PYTHON test_jarvis_core.py interactive
else
    echo "Running automated test..."
    $PYTHON test_jarvis_core.py
fi