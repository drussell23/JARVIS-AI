#!/usr/bin/env python3
"""
Diagnose why vision command "can you see my screen?" is not responding
"""

import asyncio
import sys
import subprocess
from pathlib import Path

print("=" * 80)
print("JARVIS VISION ISSUE DIAGNOSTIC")
print("=" * 80)

# Check 1: Is backend running?
print("\n[1] Checking if JARVIS backend is running...")
result = subprocess.run(
    ["ps", "aux"], capture_output=True, text=True
)
backend_running = "python" in result.stdout and "main.py" in result.stdout
if backend_running:
    print("✅ Backend appears to be running")
    for line in result.stdout.split('\n'):
        if 'main.py' in line or 'uvicorn' in line:
            print(f"   {line.strip()}")
else:
    print("❌ Backend is NOT running!")
    print("\n💡 Solution: Start JARVIS backend:")
    print("   cd /Users/derekjrussell/Documents/repos/JARVIS-AI-Agent")
    print("   python start_system.py --restart")

# Check 2: Are ports listening?
print("\n[2] Checking if backend ports are listening...")
result = subprocess.run(
    ["lsof", "-i", "-P"], capture_output=True, text=True
)
ports_found = []
for port in ["8000", "8010", "3000"]:
    if f":{port}" in result.stdout and "LISTEN" in result.stdout:
        ports_found.append(port)
        for line in result.stdout.split('\n'):
            if f":{port}" in line and "LISTEN" in line:
                print(f"✅ Port {port}: {line.strip()}")

if not ports_found:
    print("❌ No backend ports (8000, 8010, 3000) are listening!")
    print("\n💡 This confirms backend is not running.")

# Check 3: Check if vision_command_handler.py has our fixes
print("\n[3] Checking if performance fixes are in the code...")
vision_handler_path = Path(__file__).parent / "backend" / "api" / "vision_command_handler.py"
if vision_handler_path.exists():
    content = vision_handler_path.read_text()

    # Check for our fix
    if "FAST PATH: Check for monitoring commands using keywords ONLY" in content:
        print("✅ Performance fix is in the code (monitoring keyword check)")
    else:
        print("❌ Performance fix NOT found in code!")

    if "asyncio.wait_for" in content and "timeout=15.0" in content:
        print("✅ Timeout protection is in the code")
    else:
        print("⚠️  Timeout protection may not be complete")
else:
    print("❌ vision_command_handler.py not found!")

# Check 4: Verify Screen Recording permissions
print("\n[4] Checking Screen Recording permissions...")
print("⚠️  Cannot check programmatically - please verify manually:")
print("   System Settings > Privacy & Security > Screen Recording")
print("   Ensure Python/Terminal has Screen Recording permission")

# Summary
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

if not backend_running:
    print("\n🔴 PRIMARY ISSUE: Backend is not running!")
    print("\n📋 TO FIX:")
    print("   1. Open Terminal")
    print("   2. cd /Users/derekjrussell/Documents/repos/JARVIS-AI-Agent")
    print("   3. Run: python start_system.py --restart")
    print("   4. Wait for 'Backend started on port XXXX' message")
    print("   5. Try 'can you see my screen?' again")
    print("\n⏱️  Expected response time after fix: 4-10 seconds")
else:
    print("\n✅ Backend is running, but vision may still be slow")
    print("\n📋 If still slow after restart:")
    print("   1. Check Screen Recording permissions")
    print("   2. Run: python test_vision_performance.py")
    print("   3. Check logs for errors")

print("=" * 80)
