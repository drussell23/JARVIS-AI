#!/bin/bash
echo ""
echo "🔒 JARVIS SCREEN UNLOCK TEST"
echo "================================"
echo ""
echo "This will:"
echo "  1. Lock your screen automatically"
echo "  2. Wait 3 seconds"
echo "  3. Type your password to unlock"
echo ""
echo "✅ Make sure you're watching your screen!"
echo ""
read -p "Press Enter to start the test..."
python3 diagnose_unlock.py
