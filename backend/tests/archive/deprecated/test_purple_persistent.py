#!/usr/bin/env python3
"""
Test persistent purple indicator
"""

import asyncio
import subprocess
import time
import os

def test_swift_capture():
    print("\n🟣 Testing Persistent Purple Indicator")
    print("=" * 60)
    
    print("Starting Swift capture directly...")
    
    # Start the Swift capture
    process = subprocess.Popen(
        ["swift", "vision/continuous_swift_capture.swift", "--start"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print("✅ Swift capture started")
    print("🟣 CHECK YOUR MENU BAR - PURPLE INDICATOR SHOULD APPEAR!")
    print("\nMonitoring output for 30 seconds...")
    
    start_time = time.time()
    while time.time() - start_time < 30:
        # Check if process is still running
        if process.poll() is not None:
            print("\n❌ Process ended unexpectedly!")
            stdout, stderr = process.communicate()
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            break
            
        # Read any output
        try:
            line = process.stdout.readline()
            if line:
                print(f"[SWIFT] {line.strip()}")
        except:
            pass
            
        time.sleep(1)
        remaining = 30 - int(time.time() - start_time)
        if remaining % 5 == 0 and remaining > 0:
            print(f"\n⏳ {remaining} seconds remaining...")
    
    print("\n2️⃣ Stopping capture...")
    process.terminate()
    process.wait(timeout=5)
    print("✅ Capture stopped - purple indicator should disappear")
    
    print("\n" + "=" * 60)
    print("Test complete!")

if __name__ == "__main__":
    test_swift_capture()