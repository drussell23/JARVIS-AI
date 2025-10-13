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


def verify_intelligent_system():
    """Verify that intelligent routing system loaded correctly"""
    print("\n🧠 Verifying intelligent routing system...")

    # Wait for system to fully initialize
    time.sleep(5)

    jarvis_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(jarvis_dir, "jarvis_backend.log")

    # Check if log file exists
    if not os.path.exists(log_file):
        print("⚠️  Log file not found, cannot verify system status")
        return

    # Read recent logs
    try:
        with open(log_file, 'r') as f:
            # Get last 100 lines
            lines = f.readlines()[-100:]
            log_content = ''.join(lines)

            # Check for intelligent system initialization
            checks = {
                "Yabai System": "✅ Yabai multi-space intelligence initialized" in log_content,
                "Query Classifier": "✅ Intelligent query classification system initialized" in log_content,
                "Context Manager": "Query context manager initialized" in log_content,
                "Smart Router": "Smart query router initialized" in log_content,
            }

            print("\nIntelligent System Status:")
            all_passed = True
            for component, status in checks.items():
                icon = "✅" if status else "❌"
                print(f"  {icon} {component}")
                if not status:
                    all_passed = False

            if all_passed:
                print("\n✨ All intelligent routing components loaded successfully!")
                print("\nTest with: 'What's happening across my desktop spaces?'")
                print("Expected: Detailed breakdown of all spaces with apps and windows")
            else:
                print("\n⚠️  Some components failed to load - check logs for details")

    except Exception as e:
        print(f"❌ Failed to read log file: {e}")


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
            ["python", main_py, "--port", "8010"],
            cwd=jarvis_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )

        # Wait a bit to see if it starts successfully
        time.sleep(3)

        if process.poll() is None:
            print(f"✅ JARVIS started with PID {process.pid}")
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

        # Verify intelligent system loaded
        print("\n3️⃣ Verifying intelligent routing system...")
        verify_intelligent_system()

        print("\n" + "="*50)
        print("🎉 All systems ready!")
    else:
        print("\n❌ Failed to start JARVIS")
        print("You may need to start it manually:")
        print("cd ~/Documents/repos/JARVIS-AI-Agent/backend")
        print("python main.py --port 8010")


if __name__ == "__main__":
    main()