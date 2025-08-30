#!/usr/bin/env python3
"""
Robust backend startup script for JARVIS
Handles high CPU situations and ensures successful startup
"""

import os
import sys
import time
import psutil
import subprocess
import signal
from pathlib import Path
from datetime import datetime
import logging

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Setup logging
log_file = backend_path / "logs" / f"startup_robust_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import process cleanup manager
try:
    from backend.process_cleanup_manager import ProcessCleanupManager
    CLEANUP_AVAILABLE = True
except ImportError:
    CLEANUP_AVAILABLE = False
    logger.warning("Process cleanup manager not available")

class RobustBackendStarter:
    """Handles backend startup in challenging conditions"""
    
    def __init__(self):
        self.max_cpu_threshold = float(os.getenv('JARVIS_MAX_CPU_FOR_START', '50'))
        self.wait_timeout = int(os.getenv('JARVIS_CPU_WAIT_TIMEOUT', '60'))
        self.retry_count = int(os.getenv('JARVIS_START_RETRIES', '3'))
        self.cleanup_manager = ProcessCleanupManager() if CLEANUP_AVAILABLE else None
        
    def check_system_resources(self):
        """Check if system resources are suitable for startup"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        logger.info(f"System resources: CPU={cpu_percent:.1f}%, Memory={memory.percent:.1f}%")
        
        issues = []
        if cpu_percent > self.max_cpu_threshold:
            issues.append(f"CPU usage too high: {cpu_percent:.1f}% > {self.max_cpu_threshold}%")
            
        if memory.percent > 90:
            issues.append(f"Memory usage critical: {memory.percent:.1f}%")
            
        return issues
    
    def find_high_cpu_processes(self, threshold=30):
        """Find processes using significant CPU"""
        high_cpu = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            try:
                cpu = proc.cpu_percent(interval=0.1)
                if cpu > threshold:
                    high_cpu.append({
                        'pid': proc.pid,
                        'name': proc.name(),
                        'cpu': cpu
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return sorted(high_cpu, key=lambda x: x['cpu'], reverse=True)
    
    def wait_for_cpu_to_settle(self):
        """Wait for CPU to drop below threshold"""
        logger.info(f"Waiting for CPU to drop below {self.max_cpu_threshold}%...")
        
        start_time = time.time()
        while time.time() - start_time < self.wait_timeout:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent <= self.max_cpu_threshold:
                logger.info(f"CPU settled at {cpu_percent:.1f}%")
                return True
                
            # Show what's using CPU
            high_cpu = self.find_high_cpu_processes()
            if high_cpu:
                process_list = ', '.join([f"{p['name']} ({p['cpu']:.1f}%)" for p in high_cpu[:3]])
                logger.info(f"High CPU processes: {process_list}")
                
            remaining = self.wait_timeout - (time.time() - start_time)
            logger.info(f"CPU at {cpu_percent:.1f}%, waiting... ({remaining:.0f}s remaining)")
            time.sleep(5)
            
        return False
    
    def set_environment_for_low_resources(self):
        """Configure environment for resource-constrained startup"""
        # Set memory optimization level based on available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb < 2:
            os.environ['JARVIS_MEMORY_LEVEL'] = 'critical'
            os.environ['JARVIS_MODEL_PRECISION'] = '8bit'
            logger.warning(f"Critical memory: {available_gb:.1f}GB - using 8-bit models")
        elif available_gb < 4:
            os.environ['JARVIS_MEMORY_LEVEL'] = 'high'
            os.environ['JARVIS_MODEL_PRECISION'] = '16bit'
            logger.info(f"Low memory: {available_gb:.1f}GB - using 16-bit models")
        else:
            os.environ['JARVIS_MEMORY_LEVEL'] = 'normal'
            logger.info(f"Normal memory: {available_gb:.1f}GB")
            
        # Set CPU constraints
        cpu_count = psutil.cpu_count()
        if psutil.cpu_percent(interval=0.1) > 70:
            # Limit threads when CPU is busy
            os.environ['OMP_NUM_THREADS'] = str(max(1, cpu_count // 2))
            os.environ['MKL_NUM_THREADS'] = str(max(1, cpu_count // 2))
            logger.info(f"Limiting threads due to high CPU usage")
            
        # Set Swift library path
        swift_lib_path = backend_path / "swift_bridge/.build/release"
        if swift_lib_path.exists():
            os.environ["DYLD_LIBRARY_PATH"] = str(swift_lib_path)
            logger.info("Swift performance bridges configured")
    
    def cleanup_before_start(self):
        """Clean up problematic processes before starting"""
        if not self.cleanup_manager:
            return
            
        logger.info("Checking for stuck processes...")
        
        try:
            # Get recommendations
            recommendations = self.cleanup_manager.get_cleanup_recommendations()
            if recommendations:
                logger.info("System optimization suggestions:")
                for rec in recommendations:
                    logger.info(f"  • {rec}")
                    
            # Analyze what needs cleanup
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Do a dry run first
            report = loop.run_until_complete(
                self.cleanup_manager.smart_cleanup(dry_run=True)
            )
            
            if report['actions']:
                logger.info(f"Found {len(report['actions'])} processes that need cleanup")
                
                # Actually clean up
                logger.info("Performing cleanup...")
                report = loop.run_until_complete(
                    self.cleanup_manager.smart_cleanup(dry_run=False)
                )
                
                if report['freed_resources']['cpu_percent'] > 0:
                    logger.info(f"Freed ~{report['freed_resources']['cpu_percent']:.1f}% CPU, "
                              f"{report['freed_resources']['memory_mb']}MB memory")
                    
                # Give system a moment to settle
                time.sleep(2)
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def start_backend_process(self):
        """Start the backend process"""
        logger.info("Starting JARVIS backend...")
        
        # Build command
        cmd = [sys.executable, "-m", "uvicorn", "main:app"]
        cmd.extend(["--host", "0.0.0.0"])
        cmd.extend(["--port", "8010"])
        cmd.extend(["--workers", "1"])  # Single worker for stability
        
        # Add reload in development
        if os.getenv('JARVIS_ENV') == 'development':
            cmd.append("--reload")
            
        # Start process
        try:
            process = subprocess.Popen(
                cmd,
                cwd=backend_path,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            logger.info(f"Backend process started with PID: {process.pid}")
            return process
            
        except Exception as e:
            logger.error(f"Failed to start backend: {e}")
            return None
    
    def wait_for_backend_ready(self, process, timeout=30):
        """Wait for backend to be ready"""
        import requests
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if process is still running
            if process.poll() is not None:
                logger.error("Backend process terminated unexpectedly")
                return False
                
            # Try to connect
            try:
                response = requests.get("http://localhost:8010/health", timeout=1)
                if response.status_code == 200:
                    logger.info("✅ Backend is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
                
            time.sleep(1)
            
        logger.error("Backend failed to start within timeout")
        return False
    
    def start_minimal_backend(self):
        """Start minimal backend as fallback"""
        logger.info("Starting minimal backend as fallback...")
        
        minimal_path = backend_path / "main_minimal.py"
        if not minimal_path.exists():
            logger.error("main_minimal.py not found")
            return None
            
        # Build command for minimal backend
        cmd = [sys.executable, str(minimal_path), "--port", "8010"]
        
        # Start process
        try:
            process = subprocess.Popen(
                cmd,
                cwd=backend_path,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            logger.info(f"Minimal backend process started with PID: {process.pid}")
            return process
            
        except Exception as e:
            logger.error(f"Failed to start minimal backend: {e}")
            return None
    
    def run(self):
        """Main startup sequence"""
        logger.info("🚀 Starting JARVIS backend with robust startup sequence")
        
        for attempt in range(self.retry_count):
            logger.info(f"\n--- Startup attempt {attempt + 1}/{self.retry_count} ---")
            
            # Check system resources
            issues = self.check_system_resources()
            if issues:
                logger.warning(f"Resource issues detected: {', '.join(issues)}")
                
                # Try cleanup if available (with timeout)
                if self.cleanup_manager and attempt < 2:  # Only cleanup on first 2 attempts
                    self.cleanup_before_start()
                    
                # Wait for CPU if needed (skip on last attempt)
                if attempt < self.retry_count - 1:
                    if not self.wait_for_cpu_to_settle():
                        logger.warning("CPU did not settle in time, attempting startup anyway...")
                else:
                    logger.info("Last attempt - skipping CPU wait to try minimal backend faster")
                    
            # Configure environment
            self.set_environment_for_low_resources()
            
            # Start backend
            process = self.start_backend_process()
            if not process:
                continue
                
            # Wait for it to be ready
            if self.wait_for_backend_ready(process):
                logger.info("✅ Backend started successfully!")
                
                # Monitor and keep running
                try:
                    logger.info("Backend is running. Press Ctrl+C to stop.")
                    while True:
                        if process.poll() is not None:
                            logger.error("Backend process terminated!")
                            break
                            
                        # Periodic health check
                        time.sleep(30)
                        
                except KeyboardInterrupt:
                    logger.info("\nShutting down backend...")
                    process.terminate()
                    process.wait(timeout=5)
                    logger.info("Backend stopped.")
                    
                return 0
                
            else:
                # Startup failed, terminate process
                logger.error("Backend startup failed")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                
                # Try minimal backend as fallback on last attempt
                if attempt == self.retry_count - 1:
                    logger.info("All attempts failed. Trying minimal backend as fallback...")
                    
                    # Start minimal backend
                    minimal_process = self.start_minimal_backend()
                    if minimal_process:
                        # Wait for minimal backend to be ready
                        if self.wait_for_backend_ready(minimal_process):
                            logger.info("✅ Minimal backend started successfully!")
                            logger.warning("⚠️  Running in minimal mode - some features may be limited")
                            
                            # Monitor and keep running
                            try:
                                logger.info("Minimal backend is running. Press Ctrl+C to stop.")
                                while True:
                                    if minimal_process.poll() is not None:
                                        logger.error("Minimal backend process terminated!")
                                        break
                                        
                                    # Periodic health check
                                    time.sleep(30)
                                    
                            except KeyboardInterrupt:
                                logger.info("\nShutting down minimal backend...")
                                minimal_process.terminate()
                                minimal_process.wait(timeout=5)
                                logger.info("Minimal backend stopped.")
                                
                            return 0
                        else:
                            logger.error("Minimal backend also failed to start")
                            minimal_process.terminate()
                            try:
                                minimal_process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                minimal_process.kill()
                                minimal_process.wait()
                
                # Wait before retry (if not last attempt)
                if attempt < self.retry_count - 1:
                    logger.info("Waiting 10 seconds before retry...")
                    time.sleep(10)
                    
        logger.error("❌ Failed to start backend after all attempts")
        return 1

def main():
    """Main entry point"""
    starter = RobustBackendStarter()
    sys.exit(starter.run())

if __name__ == "__main__":
    main()