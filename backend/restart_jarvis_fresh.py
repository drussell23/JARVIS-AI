#!/usr/bin/env python3
"""Restart JARVIS with fresh code, ensuring old instances are stopped."""

import subprocess
import time
import os
import signal
import sys

print("🔄 Restarting JARVIS with fresh code...")
print("=" * 50)

# Find all JARVIS-related processes
print("\n1. Finding existing JARVIS processes...")
try:
    # Get all Python processes
    result = subprocess.run(
        ["ps", "aux"], 
        capture_output=True, 
        text=True
    )
    
    processes = []
    for line in result.stdout.split('\n'):
        if 'python' in line and ('main.py' in line or 'jarvis' in line.lower()):
            parts = line.split()
            if len(parts) > 1:
                pid = parts[1]
                if 'grep' not in line:
                    processes.append((pid, line.strip()))
    
    if processes:
        print(f"\nFound {len(processes)} JARVIS process(es):")
        for pid, line in processes:
            print(f"  PID {pid}: {line[:100]}...")
            
        # Kill old processes
        print("\n2. Stopping old processes...")
        for pid, _ in processes:
            try:
                os.kill(int(pid), signal.SIGTERM)
                print(f"   ✅ Stopped PID {pid}")
            except ProcessLookupError:
                print(f"   ⚠️  PID {pid} already stopped")
            except Exception as e:
                print(f"   ❌ Could not stop PID {pid}: {e}")
        
        # Wait for processes to stop
        print("\n3. Waiting for processes to stop...")
        time.sleep(2)
    else:
        print("   No existing JARVIS processes found")
        
except Exception as e:
    print(f"Error checking processes: {e}")

# Check if ports are free
print("\n4. Checking ports...")
for port in [8000, 8010, 8765]:
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}"], 
            capture_output=True, 
            text=True
        )
        if result.stdout.strip():
            print(f"   ⚠️  Port {port} is still in use")
        else:
            print(f"   ✅ Port {port} is free")
    except:
        pass

# Change to backend directory
backend_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(backend_dir)
print(f"\n5. Working directory: {os.getcwd()}")

# Start fresh JARVIS instance
print("\n6. Starting fresh JARVIS instance on port 8000...")
print("   This will include all the audio fixes we just made")
print("\n" + "=" * 50)
print("Starting JARVIS...\n")

try:
    # Start on default port 8000 with all our fixes
    subprocess.run([sys.executable, "main.py"], check=False)
except KeyboardInterrupt:
    print("\n\n✅ JARVIS stopped by user")
except Exception as e:
    print(f"\n❌ Error starting JARVIS: {e}")

print("\n✅ Restart complete")