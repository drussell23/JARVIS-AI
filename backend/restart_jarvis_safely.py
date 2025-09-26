#!/usr/bin/env python3
"""
Safely Restart JARVIS
====================

Kills old JARVIS instance and starts a new one with updated code.
"""

import subprocess
import time
import os
import signal
import psutil

def find_jarvis_processes():
    """Find all JARVIS-related processes"""
    jarvis_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cmdline = proc.info.get('cmdline')
            if cmdline and any('main.py' in arg for arg in cmdline):
                # Check if it's in the JARVIS directory
                if any('JARVIS-AI-Agent/backend' in arg for arg in cmdline):
                    jarvis_processes.append({
                        'pid': proc.info['pid'],
                        'cmdline': ' '.join(cmdline),
                        'create_time': proc.info['create_time'],
                        'age_hours': (time.time() - proc.info['create_time']) / 3600
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return jarvis_processes


def kill_old_jarvis():
    """Kill old JARVIS instances"""
    processes = find_jarvis_processes()
    
    if not processes:
        print("No JARVIS processes found.")
        return
    
    print(f"Found {len(processes)} JARVIS process(es):")
    for proc in processes:
        print(f"  PID {proc['pid']}: Running for {proc['age_hours']:.1f} hours")
        print(f"  Command: {proc['cmdline'][:80]}...")
    
    # Kill old processes
    for proc in processes:
        try:
            print(f"\nKilling PID {proc['pid']}...")
            os.kill(proc['pid'], signal.SIGTERM)
            time.sleep(1)
            
            # Check if still running
            if psutil.pid_exists(proc['pid']):
                print(f"Process still running, sending SIGKILL...")
                os.kill(proc['pid'], signal.SIGKILL)
                
            print(f"✅ Killed PID {proc['pid']}")
        except Exception as e:
            print(f"❌ Failed to kill PID {proc['pid']}: {e}")


def start_new_jarvis():
    """Start new JARVIS instance"""
    print("\n🚀 Starting new JARVIS instance...")
    
    jarvis_dir = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(jarvis_dir, "main.py")
    
    if not os.path.exists(main_py):
        print(f"❌ main.py not found at {main_py}")
        return False
    
    # Start JARVIS in background
    try:
        process = subprocess.Popen(
            ["python", main_py],
            cwd=jarvis_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        
        # Wait a bit to see if it starts successfully
        time.sleep(3)
        
        if process.poll() is None:
            print(f"✅ JARVIS started with PID {process.pid}")
            print("\nYou can now test the lock command!")
            print('Say: "Hey JARVIS, lock my screen"')
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ JARVIS failed to start")
            print(f"Error: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to start JARVIS: {e}")
        return False


def main():
    """Main restart process"""
    print("🔄 JARVIS Safe Restart Tool")
    print("="*50)
    
    # Kill old instances
    print("\n1️⃣ Killing old JARVIS instances...")
    kill_old_jarvis()
    
    # Wait a moment
    print("\n⏳ Waiting for processes to terminate...")
    time.sleep(2)
    
    # Verify they're gone
    remaining = find_jarvis_processes()
    if remaining:
        print(f"⚠️  Warning: {len(remaining)} process(es) still running")
    else:
        print("✅ All old processes terminated")
    
    # Start new instance
    print("\n2️⃣ Starting fresh JARVIS instance...")
    if start_new_jarvis():
        print("\n✅ JARVIS restart complete!")
        print("The lock command fix is now loaded.")
    else:
        print("\n❌ Failed to start JARVIS")
        print("You may need to start it manually:")
        print("cd ~/Documents/repos/JARVIS-AI-Agent/backend")
        print("python main.py")


if __name__ == "__main__":
    main()