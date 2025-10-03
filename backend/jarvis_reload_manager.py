#!/usr/bin/env python3
"""
JARVIS Intelligent Reload Manager
==================================
Monitors code changes and automatically reloads JARVIS with updated code.
No hardcoding - uses dynamic discovery and intelligent restart.
"""

import os
import sys
import time
import hashlib
import pickle
import asyncio
import signal
import psutil
import logging
from pathlib import Path
from typing import Dict, Set, Optional, List, Tuple
from datetime import datetime, timedelta
import subprocess
import json

logger = logging.getLogger(__name__)


class JARVISReloadManager:
    """Intelligent reload manager for JARVIS that ensures latest code is always running"""

    def __init__(self):
        self.repo_root = Path(__file__).parent.parent  # JARVIS-AI-Agent root
        self.backend_dir = self.repo_root / "backend"
        self.cache_dir = self.backend_dir / ".jarvis_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # File tracking
        self.code_hash_file = self.cache_dir / "code_hashes.pkl"
        self.last_reload_file = self.cache_dir / "last_reload.json"
        self.config_file = self.cache_dir / "jarvis_config.json"

        # Monitoring configuration
        self.watch_patterns = [
            "**/*.py",           # Python files
            "**/requirements*.txt",  # Dependencies
            "**/*.yaml",         # Config files
            "**/*.yml",
            "**/*.json",
            "**/*.env"
        ]

        # Exclude patterns
        self.exclude_patterns = [
            "**/.*",             # Hidden files/dirs
            "**/__pycache__/**", # Python cache
            "**/node_modules/**", # Node modules
            "**/venv/**",        # Virtual environments
            "**/env/**",
            "**/*.pyc",          # Compiled Python
            "**/*.log",          # Log files
            "**/logs/**",
            "**/cache/**",
            "**/.git/**",        # Git directory
            "**/build/**",       # Build directories
            "**/dist/**"
        ]

        # Process management
        self.jarvis_process: Optional[asyncio.subprocess.Process] = None
        self.monitor_task: Optional[asyncio.Task] = None
        self.restart_cooldown = 5  # Minimum seconds between restarts
        self.last_restart_time = 0

        # Dynamic configuration
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """Load or create dynamic configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                pass

        # Default configuration
        config = {
            "main_script": "main.py",
            "port": None,  # Will be discovered dynamically
            "auto_reload": True,
            "reload_on_error": True,
            "max_retries": 3,
            "health_check_interval": 30,
            "memory_threshold_percent": 80,
            "cpu_threshold_percent": 70
        }

        self.save_config(config)
        return config

    def save_config(self, config: Dict):
        """Save configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        self.config = config

    def find_available_port(self, start_port: int = 8000, max_port: int = 9000) -> int:
        """Find an available port dynamically"""
        import socket

        for port in range(start_port, max_port):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except:
                continue

        raise RuntimeError(f"No available ports found between {start_port} and {max_port}")

    def calculate_file_hashes(self) -> Dict[str, str]:
        """Calculate hashes for all monitored files"""
        hashes = {}

        for pattern in self.watch_patterns:
            for file_path in self.backend_dir.glob(pattern):
                # Skip excluded patterns
                if any(file_path.match(exc) for exc in self.exclude_patterns):
                    continue

                if file_path.is_file():
                    try:
                        with open(file_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                            rel_path = str(file_path.relative_to(self.backend_dir))
                            hashes[rel_path] = file_hash
                    except:
                        continue

        return hashes

    def detect_code_changes(self) -> Tuple[bool, List[str]]:
        """Detect if any code has changed since last check"""
        current_hashes = self.calculate_file_hashes()

        # Load previous hashes
        previous_hashes = {}
        if self.code_hash_file.exists():
            try:
                with open(self.code_hash_file, 'rb') as f:
                    previous_hashes = pickle.load(f)
            except:
                pass

        # Compare hashes
        changed_files = []

        # Check for modified or new files
        for file_path, current_hash in current_hashes.items():
            if file_path not in previous_hashes or previous_hashes[file_path] != current_hash:
                changed_files.append(file_path)

        # Check for deleted files
        for file_path in previous_hashes:
            if file_path not in current_hashes:
                changed_files.append(f"[DELETED] {file_path}")

        # Save current hashes
        with open(self.code_hash_file, 'wb') as f:
            pickle.dump(current_hashes, f)

        return len(changed_files) > 0, changed_files

    async def find_jarvis_process(self) -> Optional[psutil.Process]:
        """Find running JARVIS process dynamically"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('main.py' in arg for arg in cmdline):
                    # Check if it's in our backend directory
                    if any(str(self.backend_dir) in arg for arg in cmdline):
                        return proc
            except:
                continue
        return None

    async def stop_jarvis(self, force: bool = False):
        """Stop JARVIS process gracefully or forcefully"""
        logger.info("Stopping JARVIS...")

        # Try to stop our managed process first
        if self.jarvis_process and self.jarvis_process.returncode is None:
            try:
                self.jarvis_process.terminate()
                await asyncio.wait_for(self.jarvis_process.wait(), timeout=5)
                logger.info("JARVIS stopped gracefully")
                return
            except asyncio.TimeoutError:
                if force:
                    self.jarvis_process.kill()
                    await self.jarvis_process.wait()
                    logger.info("JARVIS force killed")
                    return

        # Find and stop any running JARVIS process
        jarvis_proc = await self.find_jarvis_process()
        if jarvis_proc:
            try:
                jarvis_proc.terminate()
                jarvis_proc.wait(timeout=5)
                logger.info(f"Stopped JARVIS process (PID: {jarvis_proc.pid})")
            except:
                if force:
                    jarvis_proc.kill()
                    logger.info(f"Force killed JARVIS process (PID: {jarvis_proc.pid})")

    async def start_jarvis(self) -> bool:
        """Start JARVIS with dynamic configuration"""
        try:
            # Find available port if not set
            if not self.config.get('port'):
                self.config['port'] = self.find_available_port()
                self.save_config(self.config)

            # Prepare environment
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.backend_dir)
            env['BACKEND_PORT'] = str(self.config['port'])
            env['JARVIS_AUTO_RELOAD'] = 'true'

            # Enable optimizations
            env['OPTIMIZE_STARTUP'] = 'true'
            env['LAZY_LOAD_MODELS'] = 'true'
            env['PARALLEL_INIT'] = 'true'

            # Start JARVIS
            logger.info(f"Starting JARVIS on port {self.config['port']}...")

            self.jarvis_process = await asyncio.create_subprocess_exec(
                sys.executable,
                self.config['main_script'],
                '--port', str(self.config['port']),
                cwd=str(self.backend_dir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )

            # Wait briefly for startup
            await asyncio.sleep(3)

            # Check if process is still running
            if self.jarvis_process.returncode is not None:
                logger.error("JARVIS failed to start")
                return False

            logger.info(f"JARVIS started successfully (PID: {self.jarvis_process.pid})")

            # Save restart time
            self.last_restart_time = time.time()
            with open(self.last_reload_file, 'w') as f:
                json.dump({
                    'time': datetime.now().isoformat(),
                    'port': self.config['port'],
                    'pid': self.jarvis_process.pid
                }, f)

            return True

        except Exception as e:
            logger.error(f"Failed to start JARVIS: {e}")
            return False

    async def restart_jarvis(self, reason: str = "Manual restart"):
        """Restart JARVIS with cooldown protection"""
        # Check cooldown
        time_since_restart = time.time() - self.last_restart_time
        if time_since_restart < self.restart_cooldown:
            wait_time = self.restart_cooldown - time_since_restart
            logger.info(f"Restart cooldown active, waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)

        logger.info(f"Restarting JARVIS: {reason}")

        # Stop current instance
        await self.stop_jarvis()

        # Clear any cached imports (helps with code changes)
        self.clear_python_cache()

        # Start new instance
        success = await self.start_jarvis()

        if not success:
            logger.error("Failed to restart JARVIS")
            # Try alternative startup methods
            await self.try_alternative_startup()

    def clear_python_cache(self):
        """Clear Python import cache for changed modules"""
        # Clear __pycache__ directories
        for cache_dir in self.backend_dir.glob("**/__pycache__"):
            try:
                import shutil
                shutil.rmtree(cache_dir)
            except:
                pass

        # Clear sys.modules for our modules
        modules_to_clear = []
        for module_name in list(sys.modules.keys()):
            if 'jarvis' in module_name.lower() or 'backend' in module_name:
                modules_to_clear.append(module_name)

        for module_name in modules_to_clear:
            sys.modules.pop(module_name, None)

    async def try_alternative_startup(self):
        """Try alternative startup methods if main fails"""
        alternatives = [
            ('main_minimal.py', 'Minimal mode'),
            ('start_backend.py', 'Legacy startup'),
        ]

        for script, mode in alternatives:
            script_path = self.backend_dir / script
            if script_path.exists():
                logger.info(f"Trying {mode} startup...")
                self.config['main_script'] = script
                if await self.start_jarvis():
                    logger.info(f"Started in {mode}")
                    return

    async def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Starting JARVIS monitor loop...")

        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                # Check for code changes
                if self.config.get('auto_reload', True):
                    has_changes, changed_files = self.detect_code_changes()

                    if has_changes:
                        logger.info(f"Detected {len(changed_files)} file changes")
                        for file in changed_files[:5]:  # Show first 5 changes
                            logger.info(f"  - {file}")
                        if len(changed_files) > 5:
                            logger.info(f"  ... and {len(changed_files) - 5} more")

                        await self.restart_jarvis(f"Code changes detected ({len(changed_files)} files)")

                # Check if JARVIS is still running
                if self.jarvis_process and self.jarvis_process.returncode is not None:
                    logger.warning("JARVIS process died unexpectedly")
                    if self.config.get('reload_on_error', True):
                        await self.restart_jarvis("Process died")

                # Check system resources
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=1)

                if memory.percent > self.config.get('memory_threshold_percent', 80):
                    logger.warning(f"High memory usage: {memory.percent}%")
                    # Could trigger a restart here if needed

                if cpu > self.config.get('cpu_threshold_percent', 70):
                    logger.warning(f"High CPU usage: {cpu}%")

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(10)

    async def run(self):
        """Main entry point"""
        logger.info("JARVIS Reload Manager starting...")

        # Check if JARVIS is already running
        existing = await self.find_jarvis_process()
        if existing:
            logger.info(f"Found existing JARVIS process (PID: {existing.pid})")

            # Check for code changes
            has_changes, changed_files = self.detect_code_changes()

            if has_changes:
                logger.info("Code changes detected, restarting with new code...")
                await self.stop_jarvis(force=True)
                await self.start_jarvis()
            else:
                logger.info("No code changes, attaching to existing process")
                # We could monitor the existing process instead
        else:
            # Start JARVIS
            await self.start_jarvis()

        # Start monitoring
        self.monitor_task = asyncio.create_task(self.monitor_loop())

        # Wait for interrupt
        try:
            await self.monitor_task
        except KeyboardInterrupt:
            logger.info("Shutdown requested...")
            await self.stop_jarvis()


async def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    manager = JARVISReloadManager()
    await manager.run()


if __name__ == "__main__":
    asyncio.run(main())