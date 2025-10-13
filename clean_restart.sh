#!/bin/bash
# Smart restart script for JARVIS - now uses integrated ProcessCleanupManager
# This script is simpler because the cleanup logic is built into the system!

cd "$(dirname "$0")"

echo "🚀 Starting JARVIS with smart cleanup..."
echo "   (Process cleanup manager will handle all cleanup automatically)"
echo ""

# Just run start_system.py - it handles everything now!
python start_system.py
