#!/usr/bin/env python3
"""
Quick backend starter - prioritizes getting JARVIS running
Falls back to minimal backend immediately if main fails
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

class QuickBackendStarter:
    def __init__(self):
        self.port = 8010
        self.main_path = backend_path / "main.py"
        self.minimal_path = backend_path / "main_minimal.py"
        
    def set_environment(self):
        """Set minimal environment"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(backend_path)
        
        # Set memory optimization
        env["JARVIS_MEMORY_LEVEL"] = "critical"
        env["JARVIS_MODEL_PRECISION"] = "8bit"
        
        # Set Swift library path
        swift_lib_path = str(backend_path / "swift_bridge" / ".build" / "release")
        env["DYLD_LIBRARY_PATH"] = swift_lib_path
        
        return env
    
    def try_start_backend(self, script_path, name="backend"):
        """Try to start a backend script"""
        logger.info(f"Attempting to start {name}...")
        
        cmd = [sys.executable, str(script_path), "--port", str(self.port)]
        env = self.set_environment()
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=backend_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info(f"{name} process started (PID: {process.pid})")
            
            # Quick check if it's running (3 seconds)
            time.sleep(3)
            
            if process.poll() is None:
                # Still running, check if responsive
                import requests
                try:
                    resp = requests.get(f"http://localhost:{self.port}/health", timeout=2)
                    if resp.status_code == 200:
                        logger.info(f"✅ {name} is running and healthy!")
                        return process
                except:
                    pass
                
                # Give it a bit more time
                time.sleep(2)
                
                try:
                    resp = requests.get(f"http://localhost:{self.port}/health", timeout=2)
                    if resp.status_code == 200:
                        logger.info(f"✅ {name} is running and healthy!")
                        return process
                except:
                    pass
            
            # Not working, kill it
            logger.error(f"{name} failed to start properly")
            process.terminate()
            process.wait(timeout=5)
            return None
            
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")
            return None
    
    def run(self):
        """Main run method"""
        logger.info("🚀 Quick backend starter")
        
        # Try main.py first
        if self.main_path.exists():
            process = self.try_start_backend(self.main_path, "main backend")
            if process:
                return self.monitor_process(process, "main")
        
        # Fallback to minimal
        logger.info("Main backend failed, trying minimal fallback...")
        
        if self.minimal_path.exists():
            process = self.try_start_backend(self.minimal_path, "minimal backend")
            if process:
                logger.warning("⚠️  Running in minimal mode - some features limited")
                return self.monitor_process(process, "minimal")
        
        logger.error("❌ Failed to start any backend")
        return 1
    
    def monitor_process(self, process, name):
        """Monitor the running process"""
        try:
            logger.info(f"{name} backend is running. Press Ctrl+C to stop.")
            while True:
                if process.poll() is not None:
                    logger.error(f"{name} backend terminated!")
                    return 1
                time.sleep(30)
        except KeyboardInterrupt:
            logger.info(f"\nShutting down {name} backend...")
            process.terminate()
            process.wait(timeout=5)
            logger.info("Backend stopped.")
            return 0

def main():
    starter = QuickBackendStarter()
    sys.exit(starter.run())

if __name__ == "__main__":
    main()