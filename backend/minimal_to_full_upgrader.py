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
import json
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import aiohttp
import signal
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

class MinimalToFullUpgrader:
    """
    Advanced upgrader that monitors JARVIS running in minimal mode and automatically 
    upgrades to full mode when all components become available.
    
    Features:
    - Dynamic configuration loading
    - Intelligent retry strategies
    - Component dependency tracking
    - Performance metrics
    - Health monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the upgrader with dynamic configuration.
        
        Args:
            config: Optional configuration dict
        """
        self.backend_dir = Path(__file__).parent
        
        # Load dynamic configuration
        self.config = self._load_dynamic_config(config)
        
        # Extract config values
        self.check_interval = self.config['upgrade']['check_interval']
        self.main_port = int(os.getenv('BACKEND_PORT', self.config['network']['backend_port']))
        self._max_attempts = self.config['upgrade']['max_attempts']
        
        # Advanced retry strategy
        self.retry_strategy = self.config['upgrade']['retry_strategy']
        self.backoff_multiplier = self.config['upgrade']['backoff_multiplier']
        
        # Component requirements
        self.required_components = self.config['components']['required']
        self.optional_components = self.config['components']['optional']
        
        # State tracking
        self._running = False
        self._upgrade_task: Optional[asyncio.Task] = None
        self._is_minimal_mode = False
        self._upgrade_attempts = 0
        self._main_process: Optional[subprocess.Popen] = None
        
        # Performance metrics
        self._metrics = {
            'start_time': None,
            'check_count': 0,
            'last_check': None,
            'component_history': [],
            'upgrade_history': []
        }
        
        # Component readiness cache
        self._readiness_cache = {}
        self._cache_ttl = timedelta(seconds=10)
        
    def _load_dynamic_config(self, user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load configuration dynamically from multiple sources."""
        # Default configuration
        default_config = {
            'upgrade': {
                'check_interval': 30,
                'max_attempts': 10,
                'retry_strategy': 'exponential',  # 'linear', 'exponential', 'adaptive'
                'backoff_multiplier': 1.5,
                'initial_delay': 5,
                'max_delay': 300,
                'timeout': 120
            },
            'network': {
                'backend_port': 8010,
                'health_endpoint': '/health',
                'shutdown_endpoint': '/shutdown',
                'connection_timeout': 5,
                'request_timeout': 10
            },
            'components': {
                'required': ['rust', 'memory', 'vision'],
                'optional': ['voice', 'tools'],
                'min_required_ratio': 0.8  # 80% of required components must be ready
            },
            'performance': {
                'memory_threshold_gb': 2.0,
                'cpu_threshold_percent': 80,
                'disk_space_gb': 1.0
            },
            'startup': {
                'parallel_imports': True,
                'lazy_load_models': True,
                'optimize_startup': True
            }
        }
        
        # Load from config file if exists
        config_file = self.backend_dir / "upgrader_config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    file_config = json.load(f)
                    default_config = self._deep_merge(default_config, file_config)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        # Merge with user config
        if user_config:
            default_config = self._deep_merge(default_config, user_config)
            
        # Environment variable overrides
        env_overrides = {
            'upgrade.check_interval': os.getenv('UPGRADER_CHECK_INTERVAL'),
            'upgrade.max_attempts': os.getenv('UPGRADER_MAX_ATTEMPTS'),
            'network.backend_port': os.getenv('BACKEND_PORT')
        }
        
        for key, value in env_overrides.items():
            if value:
                self._set_nested_value(default_config, key, value)
                
        return default_config
        
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
        
    def _set_nested_value(self, config: Dict, key: str, value: Any):
        """Set a nested value in config using dot notation."""
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Convert value types
        if keys[-1].endswith('_interval') or keys[-1].endswith('_attempts'):
            value = int(value)
        elif keys[-1].endswith('_gb') or keys[-1].endswith('_percent'):
            value = float(value)
            
        current[keys[-1]] = value
        
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
                        
                        # Direct check for mode field
                        if data.get('mode') == 'minimal':
                            return True
                        
                        # Check service name
                        if data.get('service') == 'jarvis-minimal':
                            return True
                            
                        # Check for indicators of minimal mode
                        components = data.get('components', {})
                        
                        # If components dict exists and all are False, it's minimal
                        if components and all(not v for v in components.values()):
                            return True
                        
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
            rust_working = await healer._is_rust_working()
            
            # Also check if we can actually import and use Rust components
            actual_rust_working = False
            try:
                from vision.jarvis_rust_core import RustBloomFilter
                # Try to create an instance to ensure it's really working
                test_filter = RustBloomFilter(size_mb=0.1, level="element")
                del test_filter
                actual_rust_working = True
            except Exception as rust_err:
                logger.debug(f"Rust import/usage test failed: {rust_err}")
                
            readiness['rust_built'] = rust_working and actual_rust_working
            
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
            readiness['rust_built'] = False
            
        # Check memory availability (need at least 2GB free)
        memory = psutil.virtual_memory()
        readiness['memory_available'] = memory.available >= 2 * 1024 * 1024 * 1024
        
        # Check if main.py exists and is valid
        main_script = self.backend_dir / "main.py"
        readiness['main_script_exists'] = main_script.exists()
        
        return readiness
        
    async def _upgrade_monitor(self):
        """Advanced monitor with intelligent retry strategies and performance tracking."""
        self._metrics['start_time'] = datetime.now()
        
        while self._running and self._upgrade_attempts < self._max_attempts:
            try:
                # Calculate dynamic check interval
                check_interval = await self._calculate_check_interval()
                await asyncio.sleep(check_interval)
                
                if not self._running:
                    break
                
                self._metrics['check_count'] += 1
                self._metrics['last_check'] = datetime.now()
                    
                logger.info(f"🔍 Checking upgrade readiness (attempt {self._upgrade_attempts + 1}/{self._max_attempts})")
                
                # Parallel component readiness checks
                readiness, performance = await asyncio.gather(
                    self._check_component_readiness(),
                    self._check_system_performance()
                )
                
                # Update metrics
                self._metrics['component_history'].append({
                    'timestamp': datetime.now().isoformat(),
                    'readiness': readiness,
                    'performance': performance
                })
                
                # Determine upgrade eligibility
                can_upgrade, reason = await self._evaluate_upgrade_eligibility(readiness, performance)
                
                if can_upgrade:
                    logger.info("✅ All conditions met for upgrade to full mode")
                    logger.info(f"📊 System performance: CPU {performance['cpu_percent']:.1f}%, "
                              f"Memory {performance['memory_available_gb']:.1f}GB")
                    
                    # Attempt upgrade with advanced recovery
                    success = await self._attempt_upgrade_with_recovery()
                    
                    if success:
                        logger.info("🎉 Successfully upgraded to full mode!")
                        await self._record_upgrade_success()
                        self._is_minimal_mode = False
                        break
                    else:
                        logger.warning("Upgrade attempt failed, applying recovery strategy")
                        self._upgrade_attempts += 1
                        await self._apply_failure_recovery()
                else:
                    logger.info(f"❌ Not ready for upgrade: {reason}")
                    
                    # Intelligent action based on missing components
                    await self._take_corrective_action(readiness, reason)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in upgrade monitor: {e}", exc_info=True)
                self._upgrade_attempts += 1
                await self._handle_monitor_error(e)
                
        if self._upgrade_attempts >= self._max_attempts:
            logger.warning("Max upgrade attempts reached")
            await self._final_recovery_attempt()
            
    async def _calculate_check_interval(self) -> float:
        """Calculate dynamic check interval based on retry strategy."""
        if self.retry_strategy == 'linear':
            return self.check_interval
            
        elif self.retry_strategy == 'exponential':
            base_interval = self.config['upgrade']['initial_delay']
            interval = base_interval * (self.backoff_multiplier ** self._upgrade_attempts)
            return min(interval, self.config['upgrade']['max_delay'])
            
        elif self.retry_strategy == 'adaptive':
            # Adaptive strategy based on component readiness history
            if len(self._metrics['component_history']) < 3:
                return self.check_interval
                
            # Calculate readiness trend
            recent_history = self._metrics['component_history'][-5:]
            ready_counts = [
                sum(1 for v in h['readiness'].values() if v)
                for h in recent_history
            ]
            
            if len(ready_counts) > 1:
                # If improving, check more frequently
                if ready_counts[-1] > ready_counts[0]:
                    return self.check_interval * 0.5
                # If stable, use normal interval
                elif ready_counts[-1] == ready_counts[0]:
                    return self.check_interval
                # If degrading, check less frequently
                else:
                    return self.check_interval * 2
                    
        return self.check_interval
        
    async def _check_system_performance(self) -> Dict[str, Any]:
        """Check system performance metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(self.backend_dir))
        
        return {
            'cpu_percent': cpu_percent,
            'cpu_count': psutil.cpu_count(),
            'memory_available_gb': memory.available / (1024**3),
            'memory_percent': memory.percent,
            'disk_free_gb': disk.free / (1024**3),
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }
        
    async def _evaluate_upgrade_eligibility(self, readiness: Dict[str, bool], 
                                          performance: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate if system is eligible for upgrade."""
        # Check required components
        required_ready = sum(1 for comp in self.required_components 
                           if readiness.get(f'{comp}_built', readiness.get(comp, False)))
        required_ratio = required_ready / len(self.required_components)
        
        if required_ratio < self.config['components']['min_required_ratio']:
            return False, f"Only {required_ratio*100:.0f}% of required components ready"
            
        # Check performance thresholds
        if performance['cpu_percent'] > self.config['performance']['cpu_threshold_percent']:
            return False, f"CPU usage too high: {performance['cpu_percent']:.1f}%"
            
        if performance['memory_available_gb'] < self.config['performance']['memory_threshold_gb']:
            return False, f"Insufficient memory: {performance['memory_available_gb']:.1f}GB"
            
        if performance['disk_free_gb'] < self.config['performance']['disk_space_gb']:
            return False, f"Low disk space: {performance['disk_free_gb']:.1f}GB"
            
        # All checks passed
        return True, "All conditions met"
        
    async def _attempt_upgrade_with_recovery(self) -> bool:
        """Attempt upgrade with advanced recovery mechanisms."""
        upgrade_start = time.time()
        
        try:
            # Pre-upgrade validation
            if not await self._pre_upgrade_validation():
                return False
                
            # Execute upgrade steps with checkpoints
            steps = [
                ("Validate main script", self._validate_main_script),
                ("Stop minimal backend", self._stop_minimal_backend),
                ("Wait for port release", self._wait_for_port_release),
                ("Start full backend", self._start_full_backend),
                ("Verify full mode", self._verify_full_mode)
            ]
            
            for step_name, step_func in steps:
                logger.info(f"📍 {step_name}...")
                success = await step_func()
                
                if not success:
                    logger.error(f"❌ Failed at: {step_name}")
                    # Attempt recovery for this specific step
                    if not await self._recover_from_step_failure(step_name):
                        return False
                        
            # Record successful upgrade
            upgrade_time = time.time() - upgrade_start
            self._metrics['upgrade_history'].append({
                'timestamp': datetime.now().isoformat(),
                'duration': upgrade_time,
                'success': True
            })
            
            logger.info(f"✅ Upgrade completed in {upgrade_time:.1f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Upgrade failed with exception: {e}")
            self._metrics['upgrade_history'].append({
                'timestamp': datetime.now().isoformat(),
                'duration': time.time() - upgrade_start,
                'success': False,
                'error': str(e)
            })
            
            # Ensure minimal backend is running
            await self._ensure_minimal_backend_running()
            return False
            
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
            
    async def _start_full_backend(self, env: Optional[Dict[str, str]] = None) -> bool:
        """Start the full backend with dynamic configuration."""
        try:
            if env is None:
                env = os.environ.copy()
                
            # Add optimizations from config
            env["PYTHONPATH"] = str(self.backend_dir)
            if self.config['startup']['optimize_startup']:
                env["OPTIMIZE_STARTUP"] = "true"
            if self.config['startup']['parallel_imports']:
                env["BACKEND_PARALLEL_IMPORTS"] = "true"
            if self.config['startup']['lazy_load_models']:
                env["BACKEND_LAZY_LOAD_MODELS"] = "true"
            
            # Ensure API keys are passed
            for key in ["ANTHROPIC_API_KEY", "OPENWEATHER_API_KEY", "PICOVOICE_ACCESS_KEY"]:
                if key in os.environ:
                    env[key] = os.environ[key]
                
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

    async def _pre_upgrade_validation(self) -> bool:
        """Validate system state before upgrade."""
        # Check if main.py exists
        if not await self._validate_main_script():
            return False
            
        # Check port availability
        if not await self._check_port_available():
            logger.warning("Port not available, attempting to free it")
            await self._stop_minimal_backend()
            await asyncio.sleep(2)
            
        return True
        
    async def _validate_main_script(self) -> bool:
        """Validate main.py exists and is valid."""
        main_script = self.backend_dir / "main.py"
        return main_script.exists()
        
    async def _wait_for_port_release(self) -> bool:
        """Wait for port to be released."""
        for _ in range(5):
            if await self._check_port_available():
                return True
            await asyncio.sleep(1)
        return False
        
    async def _verify_full_mode(self) -> bool:
        """Verify system is running in full mode."""
        await asyncio.sleep(10)  # Give time to initialize
        return not await self._check_minimal_mode()
        
    async def _recover_from_step_failure(self, step_name: str) -> bool:
        """Attempt to recover from a specific step failure."""
        recovery_actions = {
            "Stop minimal backend": self._force_stop_minimal_backend,
            "Wait for port release": self._force_port_release,
            "Start full backend": self._retry_full_backend_start
        }
        
        if step_name in recovery_actions:
            logger.info(f"Attempting recovery for: {step_name}")
            return await recovery_actions[step_name]()
            
        return False
        
    async def _force_stop_minimal_backend(self) -> bool:
        """Force stop minimal backend."""
        try:
            if sys.platform == "darwin":
                cmd = f"lsof -ti:{self.main_port} | xargs kill -9"
            else:
                cmd = f"fuser -k {self.main_port}/tcp"
            subprocess.run(cmd, shell=True, capture_output=True)
            await asyncio.sleep(2)
            return True
        except Exception as e:
            logger.error(f"Failed to force stop: {e}")
            return False
            
    async def _force_port_release(self) -> bool:
        """Force release the port."""
        await self._force_stop_minimal_backend()
        return await self._wait_for_port_release()
        
    async def _retry_full_backend_start(self) -> bool:
        """Retry starting full backend with different strategies."""
        # Try with increased timeout
        env = os.environ.copy()
        env["BACKEND_STARTUP_TIMEOUT"] = "180"  # 3 minutes
        
        try:
            return await self._start_full_backend(env=env)
        except Exception as e:
            logger.error(f"Retry failed: {e}")
            return False
            
    async def _record_upgrade_success(self):
        """Record successful upgrade metrics."""
        metrics_file = self.backend_dir / "upgrade_metrics.json"
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self._metrics, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")
            
    async def _apply_failure_recovery(self):
        """Apply recovery strategy after upgrade failure."""
        # Ensure minimal backend is running
        await self._ensure_minimal_backend_running()
        
        # Adjust retry strategy if adaptive
        if self.retry_strategy == 'adaptive':
            self.check_interval = min(self.check_interval * 1.5, 300)
            
    async def _take_corrective_action(self, readiness: Dict[str, bool], reason: str):
        """Take corrective action based on what's missing."""
        if not readiness.get('rust_built'):
            logger.info("Triggering Rust self-healer...")
            try:
                from vision.rust_self_healer import get_self_healer
                healer = get_self_healer()
                if not healer._running:
                    await healer.start()
            except Exception as e:
                logger.warning(f"Could not trigger self-healer: {e}")
                
    async def _handle_monitor_error(self, error: Exception):
        """Handle errors in the monitoring loop."""
        logger.error(f"Monitor error: {error}")
        
        # Exponential backoff on errors
        error_delay = min(60 * (2 ** min(self._upgrade_attempts, 5)), 3600)
        logger.info(f"Waiting {error_delay}s before retry...")
        await asyncio.sleep(error_delay)
        
    async def _final_recovery_attempt(self):
        """Final recovery attempt when max attempts reached."""
        logger.warning("Attempting final recovery...")
        
        # Save diagnostic information
        diagnostic_file = self.backend_dir / "upgrade_diagnostics.json"
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self._metrics,
            'config': self.config,
            'attempts': self._upgrade_attempts
        }
        
        try:
            with open(diagnostic_file, 'w') as f:
                json.dump(diagnostics, f, indent=2)
            logger.info(f"Diagnostics saved to {diagnostic_file}")
        except Exception as e:
            logger.error(f"Failed to save diagnostics: {e}")
            
    async def _ensure_minimal_backend_running(self):
        """Ensure minimal backend is running as fallback."""
        # Check if it's already running
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{self.main_port}/health",
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        return  # Already running
        except:
            pass
            
        # Start minimal backend
        await self._restart_minimal_backend()

# Global instance
_upgrader: Optional[MinimalToFullUpgrader] = None

def get_upgrader(config: Optional[Dict[str, Any]] = None) -> MinimalToFullUpgrader:
    """Get the global upgrader instance with optional config."""
    global _upgrader
    if _upgrader is None:
        _upgrader = MinimalToFullUpgrader(config)
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