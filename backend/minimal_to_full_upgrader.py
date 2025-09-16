"""
Minimal to Full Mode Upgrader for JARVIS.
Monitors system health and automatically upgrades from minimal to full mode when possible.
"""

import asyncio
import sys
import os
import logging
import psutil
import subprocess
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import aiohttp
import signal

logger = logging.getLogger(__name__)

class MinimalToFullUpgrader:
    """
    Monitors JARVIS running in minimal mode and automatically upgrades to full mode
    when all components become available.
    """
    
    def __init__(self, check_interval: int = 30):
        """
        Initialize the upgrader.
        
        Args:
            check_interval: Seconds between upgrade attempts (default: 30)
        """
        self.check_interval = check_interval
        self.backend_dir = Path(__file__).parent
        self.main_port = int(os.getenv('BACKEND_PORT', '8010'))
        self._running = False
        self._upgrade_task: Optional[asyncio.Task] = None
        self._is_minimal_mode = False
        self._upgrade_attempts = 0
        self._max_attempts = 10
        self._main_process: Optional[subprocess.Popen] = None
        
    async def start(self):
        """Start monitoring for upgrade opportunities."""
        if self._running:
            return
            
        self._running = True
        logger.info("Minimal to Full upgrader started")
        
        # Check if we're in minimal mode
        self._is_minimal_mode = await self._check_minimal_mode()
        
        if self._is_minimal_mode:
            logger.info("System running in minimal mode - monitoring for upgrade opportunity")
            self._upgrade_task = asyncio.create_task(self._upgrade_monitor())
        else:
            logger.info("System already running in full mode")
            
    async def stop(self):
        """Stop the upgrader."""
        self._running = False
        
        if self._upgrade_task:
            self._upgrade_task.cancel()
            try:
                await self._upgrade_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Minimal to Full upgrader stopped")
        
    async def _check_minimal_mode(self) -> bool:
        """Check if the system is running in minimal mode."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{self.main_port}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Check for indicators of minimal mode
                        components = data.get('components', {})
                        
                        # In minimal mode, several components will be False/unavailable
                        unavailable_count = sum(1 for v in components.values() if not v)
                        
                        if unavailable_count >= 3:  # Multiple components missing
                            return True
                            
                        # Also check if specific critical components are missing
                        critical_missing = (
                            not components.get('vision') or 
                            not components.get('memory') or
                            not components.get('voice')
                        )
                        
                        return critical_missing
                        
        except Exception as e:
            logger.debug(f"Error checking minimal mode: {e}")
            
        return False
        
    async def _check_component_readiness(self) -> Dict[str, bool]:
        """Check if all components are ready for full mode."""
        readiness = {
            'rust_built': False,
            'memory_available': False,
            'dependencies_met': True,
            'ports_available': True,
            'self_healing_complete': False
        }
        
        # Check Rust build status
        try:
            from vision.rust_self_healer import get_self_healer
            healer = get_self_healer()
            
            # Check if Rust is working
            readiness['rust_built'] = await healer._is_rust_working()
            
            # Check self-healing status
            health_report = healer.get_health_report()
            if health_report.get('running'):
                # If no recent failures and some successes
                recent_fixes = health_report.get('recent_fixes', [])
                if recent_fixes:
                    recent_success = any(f['success'] for f in recent_fixes[-3:])
                    readiness['self_healing_complete'] = recent_success
                else:
                    # No recent fixes needed, consider it complete
                    readiness['self_healing_complete'] = True
                    
        except Exception as e:
            logger.debug(f"Could not check Rust status: {e}")
            
        # Check memory availability (need at least 2GB free)
        memory = psutil.virtual_memory()
        readiness['memory_available'] = memory.available >= 2 * 1024 * 1024 * 1024
        
        # Check if main.py exists and is valid
        main_script = self.backend_dir / "main.py"
        readiness['main_script_exists'] = main_script.exists()
        
        return readiness
        
    async def _upgrade_monitor(self):
        """Monitor and attempt upgrades when ready."""
        while self._running and self._upgrade_attempts < self._max_attempts:
            try:
                await asyncio.sleep(self.check_interval)
                
                if not self._running:
                    break
                    
                logger.info(f"Checking upgrade readiness (attempt {self._upgrade_attempts + 1}/{self._max_attempts})")
                
                # Check if components are ready
                readiness = await self._check_component_readiness()
                logger.info(f"Component readiness: {readiness}")
                
                # Check if we can upgrade
                can_upgrade = (
                    readiness['rust_built'] and
                    readiness['memory_available'] and
                    readiness['dependencies_met'] and
                    readiness['main_script_exists']
                )
                
                if can_upgrade:
                    logger.info("✅ All components ready for upgrade to full mode")
                    success = await self._attempt_upgrade()
                    
                    if success:
                        logger.info("🎉 Successfully upgraded to full mode!")
                        self._is_minimal_mode = False
                        break
                    else:
                        logger.warning("Upgrade attempt failed, will retry")
                        self._upgrade_attempts += 1
                else:
                    missing = [k for k, v in readiness.items() if not v]
                    logger.info(f"Not ready for upgrade. Missing: {missing}")
                    
                    # If it's just waiting for self-healing, increase check frequency
                    if not readiness['self_healing_complete'] and readiness['rust_built']:
                        await asyncio.sleep(10)  # Check again in 10s
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in upgrade monitor: {e}")
                self._upgrade_attempts += 1
                
        if self._upgrade_attempts >= self._max_attempts:
            logger.warning("Max upgrade attempts reached, giving up")
            
    async def _attempt_upgrade(self) -> bool:
        """Attempt to upgrade from minimal to full mode."""
        logger.info("Attempting upgrade from minimal to full mode...")
        
        try:
            # First, check if we can start main.py
            main_script = self.backend_dir / "main.py"
            if not main_script.exists():
                logger.error("main.py not found")
                return False
                
            # Kill the minimal backend
            logger.info("Stopping minimal backend...")
            await self._stop_minimal_backend()
            
            # Wait a moment for port to be released
            await asyncio.sleep(2)
            
            # Start full backend
            logger.info("Starting full backend...")
            success = await self._start_full_backend()
            
            if success:
                # Verify it's running in full mode
                await asyncio.sleep(10)  # Give it time to initialize
                
                is_full = not await self._check_minimal_mode()
                if is_full:
                    logger.info("✅ Full backend started successfully")
                    return True
                else:
                    logger.warning("Backend started but still in minimal mode")
                    return False
            else:
                logger.error("Failed to start full backend")
                # Try to restart minimal backend
                await self._restart_minimal_backend()
                return False
                
        except Exception as e:
            logger.error(f"Upgrade attempt failed: {e}")
            # Try to ensure minimal backend is running
            await self._restart_minimal_backend()
            return False
            
    async def _stop_minimal_backend(self):
        """Stop the minimal backend gracefully."""
        try:
            # Try graceful shutdown via API first
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        f"http://localhost:{self.main_port}/shutdown",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status in [200, 404]:  # 404 if endpoint doesn't exist
                            logger.info("Requested graceful shutdown")
                            await asyncio.sleep(2)
                except:
                    pass
                    
            # Kill process on port
            if sys.platform == "darwin":
                cmd = f"lsof -ti:{self.main_port} | xargs kill -15"  # SIGTERM first
            else:
                cmd = f"fuser -k -TERM {self.main_port}/tcp"
                
            subprocess.run(cmd, shell=True, capture_output=True)
            await asyncio.sleep(2)
            
            # Force kill if still running
            if not await self._check_port_available():
                if sys.platform == "darwin":
                    cmd = f"lsof -ti:{self.main_port} | xargs kill -9"
                else:
                    cmd = f"fuser -k {self.main_port}/tcp"
                subprocess.run(cmd, shell=True, capture_output=True)
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error stopping minimal backend: {e}")
            
    async def _start_full_backend(self) -> bool:
        """Start the full backend."""
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.backend_dir)
            env["OPTIMIZE_STARTUP"] = "true"
            env["BACKEND_PARALLEL_IMPORTS"] = "true"
            env["BACKEND_LAZY_LOAD_MODELS"] = "true"
            
            # Ensure Anthropic key is passed
            if "ANTHROPIC_API_KEY" in os.environ:
                env["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]
                
            # Create log file
            log_dir = self.backend_dir / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"full_upgrade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            # Start main.py
            with open(log_file, "w") as log:
                self._main_process = subprocess.Popen(
                    [sys.executable, "main.py", "--port", str(self.main_port)],
                    cwd=str(self.backend_dir),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env
                )
                
            logger.info(f"Started main.py (PID: {self._main_process.pid})")
            
            # Wait for it to be ready
            ready = await self._wait_for_backend(timeout=60)
            
            if ready:
                # Double-check it's really running
                if self._main_process.poll() is None:
                    return True
                else:
                    logger.error(f"Backend process exited with code: {self._main_process.returncode}")
                    return False
            else:
                logger.error("Backend didn't respond in time")
                if self._main_process:
                    self._main_process.terminate()
                return False
                
        except Exception as e:
            logger.error(f"Failed to start full backend: {e}")
            return False
            
    async def _restart_minimal_backend(self):
        """Restart minimal backend as fallback."""
        logger.info("Restarting minimal backend as fallback...")
        
        try:
            minimal_script = self.backend_dir / "main_minimal.py"
            if not minimal_script.exists():
                logger.error("main_minimal.py not found")
                return
                
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.backend_dir)
            
            if "ANTHROPIC_API_KEY" in os.environ:
                env["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]
                
            log_file = self.backend_dir / "logs" / "minimal_fallback.log"
            
            with open(log_file, "w") as log:
                subprocess.Popen(
                    [sys.executable, "main_minimal.py", "--port", str(self.main_port)],
                    cwd=str(self.backend_dir),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env
                )
                
            logger.info("Minimal backend restart initiated")
            
        except Exception as e:
            logger.error(f"Failed to restart minimal backend: {e}")
            
    async def _check_port_available(self) -> bool:
        """Check if the backend port is available."""
        try:
            reader, writer = await asyncio.open_connection("localhost", self.main_port)
            writer.close()
            await writer.wait_closed()
            return False  # Port is in use
        except:
            return True  # Port is available
            
    async def _wait_for_backend(self, timeout: int = 30) -> bool:
        """Wait for backend to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://localhost:{self.main_port}/health",
                        timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        if resp.status == 200:
                            return True
            except:
                pass
                
            await asyncio.sleep(2)
            
        return False

# Global instance
_upgrader: Optional[MinimalToFullUpgrader] = None

def get_upgrader() -> MinimalToFullUpgrader:
    """Get the global upgrader instance."""
    global _upgrader
    if _upgrader is None:
        _upgrader = MinimalToFullUpgrader()
    return _upgrader

async def start_upgrade_monitoring():
    """Start monitoring for upgrade opportunities."""
    upgrader = get_upgrader()
    await upgrader.start()
    return upgrader

# For running as standalone script
async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    upgrader = await start_upgrade_monitoring()
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await upgrader.stop()

if __name__ == "__main__":
    asyncio.run(main())