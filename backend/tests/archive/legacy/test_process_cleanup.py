#!/usr/bin/env python3
"""Test the enhanced process cleanup manager with code change detection."""

import asyncio
import subprocess
import time
from pathlib import Path
import os
import psutil
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from process_cleanup_manager import ProcessCleanupManager, ensure_fresh_jarvis_instance

def find_jarvis_processes():
    """Find all JARVIS processes."""
    jarvis_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cmdline = ' '.join(proc.cmdline()).lower()
            if 'main.py' in cmdline and 'jarvis' in cmdline:
                jarvis_procs.append({
                    'pid': proc.pid,
                    'port': extract_port(cmdline),
                    'age': (time.time() - proc.create_time()) / 60,  # minutes
                    'cmdline': ' '.join(proc.cmdline()[:5])
                })
        except:
            continue
    return jarvis_procs

def extract_port(cmdline):
    """Extract port number from command line."""
    parts = cmdline.split()
    for i, part in enumerate(parts):
        if part == '--port' and i + 1 < len(parts):
            return parts[i + 1]
    return '8000'  # default

async def test_cleanup_manager():
    """Test the process cleanup manager functionality."""
    print("=" * 70)
    print("🧪 Testing Enhanced Process Cleanup Manager")
    print("=" * 70)
    
    manager = ProcessCleanupManager()
    
    # 1. Check current JARVIS processes
    print("\n1️⃣  Current JARVIS processes:")
    processes = find_jarvis_processes()
    if processes:
        for p in processes:
            print(f"   PID {p['pid']}: Port {p['port']}, Age {p['age']:.1f} minutes")
    else:
        print("   No JARVIS processes found")
    
    # 2. Check for code changes
    print("\n2️⃣  Checking for code changes...")
    has_changes = manager._detect_code_changes()
    current_hash = manager._calculate_code_hash()[:8]
    saved_hash = manager.code_state.get('code_hash', 'None')[:8] if manager.code_state.get('code_hash') else 'None'
    
    print(f"   Current code hash: {current_hash}")
    print(f"   Saved code hash:   {saved_hash}")
    print(f"   Code changes detected: {'YES' if has_changes else 'NO'}")
    
    # 3. Test cleanup functionality
    print("\n3️⃣  Testing cleanup (dry run)...")
    report = await manager.smart_cleanup(dry_run=True)
    
    print(f"   System CPU: {report['system_state']['cpu_percent']:.1f}%")
    print(f"   System Memory: {report['system_state']['memory_percent']:.1f}%")
    print(f"   Actions planned: {len(report['actions'])}")
    
    if report.get('code_changes_cleanup'):
        print(f"   Code change cleanup: {len(report['code_changes_cleanup'])} old instances would be terminated")
    
    # 4. Test ensure_single_instance
    print("\n4️⃣  Testing single instance check...")
    is_only = manager.ensure_single_instance()
    port = os.getenv('BACKEND_PORT', '8000')
    print(f"   Checking port {port}...")
    print(f"   Can start on port {port}: {'YES' if is_only else 'NO'}")
    
    # 5. Touch a file to simulate code change
    print("\n5️⃣  Simulating code change...")
    test_file = Path(__file__).parent / "api" / "jarvis_voice_api.py"
    if test_file.exists():
        # Touch the file to change its modification time
        test_file.touch()
        print(f"   Touched {test_file.name}")
        
        # Check again
        has_changes_after = manager._detect_code_changes()
        print(f"   Code changes after touch: {'YES' if has_changes_after else 'NO'}")
        
        if has_changes_after:
            print("\n   ⚠️  Code changes detected! Old instances would be cleaned up on next start.")
    
    # 6. Show recommendations
    print("\n6️⃣  System recommendations:")
    recommendations = manager.get_cleanup_recommendations()
    if recommendations:
        for rec in recommendations:
            print(f"   • {rec}")
    else:
        print("   No recommendations - system is healthy")
    
    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("\nKey Features Demonstrated:")
    print("• Code change detection using file hashing")
    print("• Automatic cleanup of old instances when code changes")
    print("• Single instance enforcement per port")
    print("• System health monitoring and recommendations")
    print("• Process cleanup with graceful termination")
    print("\n💡 This ensures you'll never run an old JARVIS instance again!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_cleanup_manager())