#!/usr/bin/env python3
"""
Restart JARVIS with Voice Feedback Fix
======================================
"""

import subprocess
import time
import os
import signal
import sys

def restart_jarvis():
    """Restart JARVIS to pick up the voice feedback changes"""
    
    print("🔄 Restarting JARVIS with Voice Feedback Fix")
    print("="*50)
    
    # Find JARVIS process
    print("\n1️⃣ Finding JARVIS process...")
    result = subprocess.run(
        ["pgrep", "-f", "python main.py"],
        capture_output=True, text=True
    )
    
    if result.stdout.strip():
        pid = int(result.stdout.strip())
        print(f"   Found JARVIS running with PID: {pid}")
        
        # Kill the process
        print("\n2️⃣ Stopping JARVIS...")
        try:
            os.kill(pid, signal.SIGTERM)
            print("   Sent termination signal")
            time.sleep(2)
            
            # Check if still running
            try:
                os.kill(pid, 0)
                print("   Process still running, sending SIGKILL...")
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                print("   ✅ JARVIS stopped")
                
        except Exception as e:
            print(f"   ❌ Error stopping JARVIS: {e}")
    else:
        print("   ℹ️  JARVIS not currently running")
    
    # Wait a moment
    time.sleep(2)
    
    # Start JARVIS
    print("\n3️⃣ Starting JARVIS with updated code...")
    
    # Change to backend directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Start JARVIS in background
    process = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True
    )
    
    print(f"   Started JARVIS with PID: {process.pid}")
    
    # Wait for it to initialize
    print("\n4️⃣ Waiting for JARVIS to initialize...")
    for i in range(10):
        print(f"   {10-i}...", end='\r')
        time.sleep(1)
    
    print("\n\n✅ JARVIS restarted with voice feedback fix!")
    print("\n📢 What's fixed:")
    print("   - JARVIS will now speak: 'I see your screen is locked...'")
    print("   - Before unlocking, not after")
    print("   - Clear feedback throughout the process")
    
    print("\n🎤 To test:")
    print("   1. Lock your screen")
    print("   2. Say: 'JARVIS, open Safari and search for dogs'")
    print("   3. Listen for the lock detection announcement")

if __name__ == "__main__":
    restart_jarvis()