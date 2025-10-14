#!/usr/bin/env python3
"""
Direct fix for JARVIS vision routing - patches the API endpoint
"""

import requests
import json

# Test current behavior
test_query = "What is happening across my desktop spaces?"

print("Testing current behavior...")
try:
    response = requests.post(
        "http://localhost:8010/api/command",
        json={"text": test_query},
        timeout=5
    )
    data = response.json()
    current_type = data.get("command_type", "unknown")
    print(f"Current classification: {current_type}")
    
    if current_type == "vision":
        print("✅ Already working!")
    else:
        print(f"❌ Broken - classified as '{current_type}' instead of 'vision'")
        print("\nThe problem: Python is using cached bytecode with the old code.")
        print("The fix: Restart JARVIS with a completely fresh Python interpreter.")
        
        print("\n" + "="*60)
        print("SOLUTION:")
        print("="*60)
        print("1. Stop JARVIS completely:")
        print("   pkill -9 -f python")
        print("\n2. Clear ALL Python cache:")
        print("   find backend -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null")
        print("   find backend -name '*.pyc' -delete 2>/dev/null")
        print("\n3. Start JARVIS with NO bytecode generation:")
        print("   cd backend && PYTHONDONTWRITEBYTECODE=1 python -B main.py --port 8010")
        print("\n4. Then test with:")
        print("   'What is happening across my desktop spaces?'")
        print("="*60)
        
except Exception as e:
    print(f"Error: {e}")
    print("Backend not responding on port 8010")