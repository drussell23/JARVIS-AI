#!/usr/bin/env python3
"""
Fix Screen Recording Permission for JARVIS Vision System
"""

import os
import sys
import subprocess
import platform

print("🔧 JARVIS Vision Permission Fixer")
print("=" * 50)

# Step 1: Show current Python info
print("\n1️⃣ Current Python Information:")
print(f"   Python executable: {sys.executable}")
print(f"   Python version: {sys.version}")
print(f"   Platform: {platform.platform()}")

# Step 2: Test screen capture
print("\n2️⃣ Testing screen capture...")
try:
    import Quartz

    screenshot = Quartz.CGDisplayCreateImage(Quartz.CGMainDisplayID())
    if screenshot:
        print("✅ Screen capture is working!")
        width = Quartz.CGImageGetWidth(screenshot)
        height = Quartz.CGImageGetHeight(screenshot)
        print(f"   Screen resolution: {width}x{height}")
    else:
        print("❌ Screen capture failed - permission issue")
except Exception as e:
    print(f"❌ Error: {e}")

# Step 3: Check parent process
print("\n3️⃣ Checking parent process...")
try:
    parent_pid = os.getppid()
    result = subprocess.run(
        ["ps", "-p", str(parent_pid), "-o", "comm="], capture_output=True, text=True
    )
    if result.returncode == 0:
        parent = result.stdout.strip()
        print(f"   Parent process: {parent}")

        # Get full path of parent
        result2 = subprocess.run(
            ["ps", "-p", str(parent_pid), "-o", "command="],
            capture_output=True,
            text=True,
        )
        if result2.returncode == 0:
            print(f"   Full path: {result2.stdout.strip()}")
except Exception as e:
    print(f"   Could not determine parent process: {e}")

# Step 4: Provide solution
print("\n📋 SOLUTION:")
print("=" * 50)

if not screenshot:
    print("\n🚨 To fix screen recording permission:\n")

    print("Option 1: Grant permission to Python directly")
    print(
        "1. Open System Preferences → Security & Privacy → Privacy → Screen Recording"
    )
    print("2. Click the lock to make changes")
    print("3. Click the '+' button")
    print("4. Press Cmd+Shift+G and paste:", sys.executable)
    print("5. Click 'Go' then 'Open'")
    print("6. Make sure it's checked ✓")
    print("7. IMPORTANT: Quit and restart Terminal completely\n")

    print("Option 2: Use system Python (if miniforge doesn't work)")
    print("Try running JARVIS with system Python:")
    print("   /usr/bin/python3 start_system.py\n")

    print("Option 3: Create an app bundle")
    print("Run this command to create a clickable app:")
    print(
        f"   echo '#!/bin/bash\\ncd {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))} && {sys.executable} start_system.py' > ~/Desktop/JARVIS.command"
    )
    print("   chmod +x ~/Desktop/JARVIS.command")
    print("   Then double-click JARVIS.command on your Desktop\n")

    print("🔄 After any of these steps, restart Terminal and run this test again!")
else:
    print("\n✅ Everything is working! JARVIS can see your screen.")
    print("\n🚀 You can now run JARVIS:")
    print("   cd", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("   python start_system.py")
    print("\n🎯 Vision commands to try:")
    print("   • 'Hey JARVIS, what's on my screen?'")
    print("   • 'Hey JARVIS, check for software updates'")
    print("   • 'Hey JARVIS, what applications are open?'")
