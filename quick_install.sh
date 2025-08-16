#!/bin/bash
# Quick install script for JARVIS dependencies

echo "🚀 JARVIS Quick Install"
echo "======================"

# Install psutil first (needed by start_system.py)
echo "📦 Installing psutil..."
pip install psutil

# Install from backend requirements
echo "📦 Installing all dependencies from requirements.txt..."
pip install -r backend/requirements.txt

# Install optimized dependencies for M1
echo "🔧 Installing M1-optimized dependencies..."
pip install llama-cpp-python

echo ""
echo "✅ Installation complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Run: python jarvis_quick_fix.py  # Set up optimized models"
echo "2. Run: python start_system.py      # Start JARVIS"