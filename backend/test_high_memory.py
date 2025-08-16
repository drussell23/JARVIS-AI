#!/usr/bin/env python3
"""
Test JARVIS with adjusted memory thresholds for high-memory systems
"""
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Monkey-patch the thresholds before importing
import chatbots.dynamic_chatbot as dc
dc.ModeThresholds.LANGCHAIN_THRESHOLD = 0.85  # Allow LangChain up to 85%
dc.ModeThresholds.UPGRADE_THRESHOLD = 0.90    # Allow Intelligent up to 90%
dc.ModeThresholds.DOWNGRADE_THRESHOLD = 0.95  # Only downgrade at 95%

# Now run the server
import uvicorn
from main import app

if __name__ == "__main__":
    print("Starting JARVIS with adjusted memory thresholds...")
    print("LangChain: < 85%, Intelligent: < 90%, Simple: > 95%")
    uvicorn.run(app, host="0.0.0.0", port=8000)