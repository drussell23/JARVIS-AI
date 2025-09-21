#!/usr/bin/env python3
"""Start minimal backend with upgrader for testing."""

import subprocess
import sys
import os
from pathlib import Path
import time

backend_path = Path(__file__).parent
log_dir = backend_path / "logs"
log_dir.mkdir(exist_ok=True)

# Set environment
env = os.environ.copy()
env["PYTHONPATH"] = str(backend_path)
if "ANTHROPIC_API_KEY" in os.environ:
    env["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]

# Start minimal backend
log_file = log_dir / f"minimal_with_upgrader_{int(time.time())}.log"
print(f"Starting minimal backend with upgrader...")
print(f"Log file: {log_file}")

with open(log_file, "w") as log:
    process = subprocess.Popen(
        [sys.executable, "-u", "main_minimal.py", "--port", "8010"],
        cwd=str(backend_path),
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env
    )
    
    print(f"Started minimal backend (PID: {process.pid})")
    print("Tailing log file (Ctrl+C to stop)...")
    
    # Tail the log file
    try:
        subprocess.run(["tail", "-f", str(log_file)])
    except KeyboardInterrupt:
        print("\nStopping...")
        process.terminate()
        process.wait()