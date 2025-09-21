#!/usr/bin/env python3
"""Test the minimal to full upgrader v2."""

import asyncio
import logging
import sys
import os
from pathlib import Path
import subprocess
import time

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from minimal_to_full_upgrader import MinimalToFullUpgrader

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def start_minimal_backend():
    """Start minimal backend for testing."""
    print("Starting minimal backend...")
    log_file = backend_path / "logs" / f"test_minimal_{int(time.time())}.log"
    
    with open(log_file, "w") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "main_minimal.py", "--port", "8010"],
            cwd=str(backend_path),
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(backend_path)}
        )
    
    # Wait for it to start
    await asyncio.sleep(5)
    return process, log_file

async def main():
    """Test the upgrader functionality."""
    
    # Start minimal backend first
    process, log_file = await start_minimal_backend()
    print(f"Minimal backend PID: {process.pid}")
    print(f"Log file: {log_file}")
    
    # Now test upgrader
    print("\nTesting Minimal to Full Upgrader...")
    
    # Create upgrader with short check interval for testing
    upgrader = MinimalToFullUpgrader(check_interval=10)
    
    # Check if we're in minimal mode
    is_minimal = await upgrader._check_minimal_mode()
    print(f"Is minimal mode: {is_minimal}")
    
    # Check component readiness
    readiness = await upgrader._check_component_readiness()
    print(f"Component readiness:")
    for k, v in readiness.items():
        print(f"  {k}: {v}")
    
    # If in minimal mode, start monitoring
    if is_minimal:
        print("\nSystem is in minimal mode, starting upgrader...")
        await upgrader.start()
        
        # Monitor for a bit
        try:
            await asyncio.sleep(60)
        except KeyboardInterrupt:
            print("\nStopping...")
    else:
        print("\nSystem is already in full mode")
    
    # Cleanup
    await upgrader.stop()
    process.terminate()
    process.wait()
    
    # Show last lines of log
    print(f"\nLast lines of log:")
    subprocess.run(["tail", "-20", str(log_file)])
    
    print("\nTest complete")

if __name__ == "__main__":
    asyncio.run(main())