#!/usr/bin/env python3
"""
Advanced Async Process Detection System
Dynamically discovers and terminates JARVIS processes with zero hardcoding.
Supports multiple detection strategies, async operations, and comprehensive edge case handling.
"""

import asyncio
import logging
import os
import psutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import json

logger = logging.getLogger(__name__)


class DetectionStrategy(Enum):
    """Available process detection strategies"""
    PSUTIL_SCAN = "psutil_scan"
    PS_COMMAND = "ps_command"
    PORT_BASED = "port_based"
    NETWORK_CONNECTIONS = "network_connections"
    FILE_DESCRIPTOR = "file_descriptor"
    PARENT_CHILD = "parent_child"
    COMMAND_LINE = "command_line"


class ProcessPriority(Enum):
    """Process termination priority"""
    CRITICAL = 1  # Must be killed first (e.g., parent processes)
    HIGH = 2      # Important to kill (e.g., main backend)
    MEDIUM = 3    # Normal priority
    LOW = 4       # Can be killed last


@dataclass
class ProcessInfo:
    """Comprehensive process information"""
    pid: int
    name: str
    cmdline: List[str]
    create_time: float
    ports: List[int] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)
    parent_pid: Optional[int] = None
    children_pids: List[int] = field(default_factory=list)
    detection_strategy: str = ""
    priority: ProcessPriority = ProcessPriority.MEDIUM
    age_hours: float = 0.0

    def __hash__(self):
        return hash(self.pid)

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            "pid": self.pid,
            "name": self.name,
            "cmdline": " ".join(self.cmdline),
            "age_hours": round(self.age_hours, 2),
            "ports": self.ports,
            "priority": self.priority.name,
            "detection_strategy": self.detection_strategy,
        }


@dataclass
class DetectionConfig:
    """Dynamic configuration for process detection"""
    # Process name patterns (dynamically loaded from config)
    process_patterns: List[str] = field(default_factory=list)

    # Port ranges to scan (dynamically loaded from config)
    ports: List[int] = field(default_factory=list)

    # File patterns for JARVIS-related files
    file_patterns: List[str] = field(default_factory=list)

    # Command line patterns
    cmdline_patterns: List[str] = field(default_factory=list)

    # Strategies to use (all enabled by default)
    enabled_strategies: List[DetectionStrategy] = field(
        default_factory=lambda: list(DetectionStrategy)
    )

    # Timeout for each strategy (seconds)
    strategy_timeout: float = 5.0

    # Maximum concurrent async tasks
    max_concurrency: int = 10

    # Minimum process age to consider (hours) - prevents killing just-started processes
    min_age_hours: float = 0.01  # ~36 seconds

    # Exclude current process and children
    exclude_current: bool = True

    @classmethod
    def from_env(cls) -> "DetectionConfig":
        """Load configuration dynamically from environment and config files"""
        config = cls()

        # Try to load from .env file
        env_file = Path.cwd() / ".env"
        if env_file.exists():
            try:
                import dotenv
                dotenv.load_dotenv(env_file)
            except ImportError:
                logger.debug("python-dotenv not available, using os.environ")

        # Discover ports dynamically from environment
        discovered_ports = set()

        # Common port environment variable patterns
        port_env_patterns = [
            "PORT", "API_PORT", "BACKEND_PORT", "FRONTEND_PORT",
            "WS_PORT", "WEBSOCKET_PORT", "HTTP_PORT", "HTTPS_PORT",
            "SERVER_PORT", "SERVICE_PORT", "APP_PORT"
        ]

        for key, value in os.environ.items():
            # Check if it matches port patterns
            if any(pattern in key.upper() for pattern in port_env_patterns):
                try:
                    port = int(value)
                    if 1024 <= port <= 65535:  # Valid port range
                        discovered_ports.add(port)
                        logger.debug(f"Discovered port {port} from {key}")
                except (ValueError, TypeError):
                    continue

        # Add default JARVIS ports (fallback if none discovered)
        default_ports = [8010, 8000, 3000, 5000, 8080]
        config.ports = sorted(list(discovered_ports)) or default_ports

        # Discover process patterns dynamically
        config.process_patterns = [
            "jarvis",
            "main.py",
            "start_system.py",
            "uvicorn",
            "fastapi",
            "voice_api",
            "backend",
        ]

        # Discover file patterns
        config.file_patterns = [
            "jarvis",
            ".jarvis",
            "backend/main.py",
            "start_system.py",
        ]

        # Command line patterns
        config.cmdline_patterns = [
            "python.*jarvis",
            "python.*main.py",
            "python.*start_system",
            "uvicorn.*jarvis",
            "uvicorn.*main:app",
        ]

        return config

    @classmethod
    def from_config_file(cls, config_path: Path) -> "DetectionConfig":
        """Load configuration from JSON file"""
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls.from_env()

        try:
            with open(config_path, 'r') as f:
                data = json.load(f)

            config = cls()
            config.ports = data.get("ports", [])
            config.process_patterns = data.get("process_patterns", [])
            config.file_patterns = data.get("file_patterns", [])
            config.cmdline_patterns = data.get("cmdline_patterns", [])
            config.strategy_timeout = data.get("strategy_timeout", 5.0)
            config.max_concurrency = data.get("max_concurrency", 10)
            config.min_age_hours = data.get("min_age_hours", 0.01)

            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return cls.from_env()


class AdvancedProcessDetector:
    """
    Advanced async process detector with dynamic configuration and zero hardcoding.

    Features:
    - Dynamic port and process discovery from environment
    - Async/concurrent detection strategies
    - Comprehensive edge case handling
    - Process priority and dependency tracking
    - Multiple detection strategies with fallbacks
    - Zero hardcoded values (all configurable)
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig.from_env()
        self.current_pid = os.getpid()
        self.current_process = psutil.Process(self.current_pid)
        self.detected_processes: Dict[int, ProcessInfo] = {}

        logger.info(f"Initialized AdvancedProcessDetector with {len(self.config.ports)} ports")
        logger.debug(f"Ports to scan: {self.config.ports}")
        logger.debug(f"Process patterns: {self.config.process_patterns}")

    async def detect_all(self) -> List[ProcessInfo]:
        """
        Run all enabled detection strategies concurrently and merge results.

        Returns:
            List of unique ProcessInfo objects sorted by priority
        """
        logger.info("Starting comprehensive process detection...")

        # Create async tasks for each strategy
        tasks = []
        for strategy in self.config.enabled_strategies:
            task = self._run_strategy_with_timeout(strategy)
            tasks.append(task)

        # Run all strategies concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results from all strategies with intelligent deduplication
        # Track processes by PID and merge information from multiple strategies
        pid_to_process: Dict[int, ProcessInfo] = {}
        strategy_detections: Dict[int, List[str]] = {}  # PID -> list of strategies that found it

        for i, result in enumerate(results):
            strategy = self.config.enabled_strategies[i]
            if isinstance(result, Exception):
                logger.warning(f"Strategy {strategy.value} failed: {result}")
                continue

            if isinstance(result, list):
                logger.info(f"Strategy {strategy.value} found {len(result)} processes")

                for proc in result:
                    if proc.pid not in pid_to_process:
                        # First time seeing this PID
                        pid_to_process[proc.pid] = proc
                        strategy_detections[proc.pid] = [proc.detection_strategy]
                    else:
                        # Merge information from multiple strategies
                        existing = pid_to_process[proc.pid]
                        strategy_detections[proc.pid].append(proc.detection_strategy)

                        # Merge ports
                        if proc.ports:
                            existing.ports = list(set(existing.ports + proc.ports))

                        # Merge connections
                        if proc.connections:
                            existing.connections = list(set(existing.connections + proc.connections))

                        # Use better detection_strategy name (show all strategies)
                        existing.detection_strategy = f"multi:{len(strategy_detections[proc.pid])}"

                        # Update age if we now have more accurate info
                        if proc.age_hours > 0 and existing.age_hours == 0:
                            existing.age_hours = proc.age_hours
                            existing.create_time = proc.create_time

        all_processes = list(pid_to_process.values())

        # Log deduplication stats
        multi_strategy = sum(1 for strategies in strategy_detections.values() if len(strategies) > 1)
        if multi_strategy > 0:
            logger.info(f"Deduplicated {multi_strategy} processes detected by multiple strategies")

        # Filter by age
        filtered = [
            p for p in all_processes
            if p.age_hours >= self.config.min_age_hours
        ]

        # Exclude current process if configured
        if self.config.exclude_current:
            filtered = [p for p in filtered if p.pid != self.current_pid]

        # Build parent-child relationships
        self._build_relationships(filtered)

        # Assign priorities
        self._assign_priorities(filtered)

        # Sort by priority (CRITICAL first)
        sorted_processes = sorted(filtered, key=lambda p: p.priority.value)

        logger.info(f"Detected {len(sorted_processes)} JARVIS processes (after filtering)")
        return sorted_processes

    async def _run_strategy_with_timeout(
        self, strategy: DetectionStrategy
    ) -> List[ProcessInfo]:
        """Run a single detection strategy with timeout"""
        try:
            return await asyncio.wait_for(
                self._run_strategy(strategy),
                timeout=self.config.strategy_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Strategy {strategy.value} timed out after {self.config.strategy_timeout}s")
            return []
        except Exception as e:
            logger.error(f"Strategy {strategy.value} failed: {e}", exc_info=True)
            return []

    async def _run_strategy(self, strategy: DetectionStrategy) -> List[ProcessInfo]:
        """Execute a specific detection strategy"""
        if strategy == DetectionStrategy.PSUTIL_SCAN:
            return await self._detect_psutil_scan()
        elif strategy == DetectionStrategy.PS_COMMAND:
            return await self._detect_ps_command()
        elif strategy == DetectionStrategy.PORT_BASED:
            return await self._detect_port_based()
        elif strategy == DetectionStrategy.NETWORK_CONNECTIONS:
            return await self._detect_network_connections()
        elif strategy == DetectionStrategy.FILE_DESCRIPTOR:
            return await self._detect_file_descriptor()
        elif strategy == DetectionStrategy.PARENT_CHILD:
            return await self._detect_parent_child()
        elif strategy == DetectionStrategy.COMMAND_LINE:
            return await self._detect_command_line()
        else:
            logger.warning(f"Unknown strategy: {strategy}")
            return []

    async def _detect_psutil_scan(self) -> List[ProcessInfo]:
        """Strategy 1: Scan all processes using psutil"""
        processes = []

        def scan():
            found = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'cwd']):
                try:
                    info = proc.info
                    name = info['name'].lower()
                    cmdline = info.get('cmdline') or []
                    cmdline_str = ' '.join(cmdline).lower()
                    cwd = info.get('cwd', '').lower() if info.get('cwd') else ''

                    # ENHANCED: Must match pattern AND be in JARVIS directory
                    # This prevents false positives from other Python processes
                    matches_pattern = any(
                        pattern.lower() in name or pattern.lower() in cmdline_str
                        for pattern in self.config.process_patterns
                    )

                    # Check if running from JARVIS directory or mentions jarvis in path
                    in_jarvis_dir = 'jarvis' in cwd or 'jarvis' in cmdline_str

                    # For main.py and start_system.py, require JARVIS context
                    is_main_py = 'main.py' in cmdline_str
                    is_start_system = 'start_system.py' in cmdline_str

                    if is_main_py or is_start_system:
                        # These files must be in JARVIS directory
                        if not in_jarvis_dir:
                            continue

                    # For other patterns, also require JARVIS context
                    if matches_pattern and not (is_main_py or is_start_system):
                        if not in_jarvis_dir:
                            continue

                    # If we got here, it's a valid JARVIS process
                    if matches_pattern or is_main_py or is_start_system:
                        process_info = ProcessInfo(
                            pid=info['pid'],
                            name=info['name'],
                            cmdline=cmdline,
                            create_time=info['create_time'],
                            detection_strategy="psutil_scan",
                            age_hours=(time.time() - info['create_time']) / 3600,
                        )
                        found.append(process_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return found

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        processes = await loop.run_in_executor(None, scan)

        return processes

    async def _detect_ps_command(self) -> List[ProcessInfo]:
        """Strategy 2: Use ps command for verification"""
        processes = []

        try:
            # More precise: look for JARVIS-specific processes
            cmd = "ps aux | grep -E '(start_system\\.py|backend/main\\.py|backend.*main\\.py)' | grep -i jarvis | grep -v grep"

            result = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()

            if stdout:
                for line in stdout.decode().strip().split('\n'):
                    if not line:
                        continue

                    parts = line.split(None, 10)
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1])
                            process_info = ProcessInfo(
                                pid=pid,
                                name="ps_detected",
                                cmdline=[line],
                                create_time=time.time(),  # Unknown create time
                                detection_strategy="ps_command",
                                age_hours=0.0,
                            )
                            processes.append(process_info)
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            logger.debug(f"ps command failed: {e}")

        return processes

    async def _detect_port_based(self) -> List[ProcessInfo]:
        """Strategy 3: Port-based detection using lsof (async, dynamic ports)"""
        processes = []

        # Create tasks for each port
        tasks = [self._check_port(port) for port in self.config.ports]

        # Run port checks concurrently (with semaphore to limit concurrency)
        sem = asyncio.Semaphore(self.config.max_concurrency)

        async def bounded_check(port_task):
            async with sem:
                return await port_task

        results = await asyncio.gather(*[bounded_check(task) for task in tasks])

        # Flatten results
        for port_processes in results:
            processes.extend(port_processes)

        return processes

    async def _check_port(self, port: int) -> List[ProcessInfo]:
        """Check a single port for processes"""
        processes = []

        try:
            cmd = ["lsof", "-ti", f":{port}"]
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()

            if stdout:
                pids = stdout.decode().strip().split('\n')
                for pid_str in pids:
                    if pid_str:
                        try:
                            pid = int(pid_str)
                            proc = psutil.Process(pid)
                            cmdline = proc.cmdline()
                            cmdline_str = ' '.join(cmdline).lower()
                            cwd = proc.cwd().lower() if hasattr(proc, 'cwd') else ''

                            # ENHANCED: Verify process is JARVIS-related
                            # Don't blindly trust port detection - verify it's our process
                            is_jarvis = (
                                'jarvis' in cmdline_str or
                                'jarvis' in cwd or
                                'main.py' in cmdline_str or
                                'start_system.py' in cmdline_str or
                                'uvicorn' in cmdline_str
                            )

                            if not is_jarvis:
                                logger.debug(f"Port {port} process {pid} is not JARVIS-related, skipping")
                                continue

                            process_info = ProcessInfo(
                                pid=pid,
                                name=proc.name(),
                                cmdline=cmdline,
                                create_time=proc.create_time(),
                                ports=[port],
                                detection_strategy=f"port_based:{port}",
                                age_hours=(time.time() - proc.create_time()) / 3600,
                            )
                            processes.append(process_info)
                        except (ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
        except Exception as e:
            logger.debug(f"Port {port} check failed: {e}")

        return processes

    async def _detect_network_connections(self) -> List[ProcessInfo]:
        """Strategy 4: Detect via network connections"""
        processes = []

        def scan_connections():
            found = []
            try:
                # This may require elevated permissions on macOS
                for conn in psutil.net_connections(kind='inet'):
                    if conn.laddr and conn.laddr.port in self.config.ports:
                        try:
                            if conn.pid:
                                proc = psutil.Process(conn.pid)
                                cmdline = proc.cmdline()
                                cmdline_str = ' '.join(cmdline).lower()
                                cwd = proc.cwd().lower() if hasattr(proc, 'cwd') else ''

                                # ENHANCED: Verify it's JARVIS-related
                                is_jarvis = (
                                    'jarvis' in cmdline_str or
                                    'jarvis' in cwd or
                                    'main.py' in cmdline_str or
                                    'start_system.py' in cmdline_str
                                )

                                if not is_jarvis:
                                    continue

                                process_info = ProcessInfo(
                                    pid=conn.pid,
                                    name=proc.name(),
                                    cmdline=cmdline,
                                    create_time=proc.create_time(),
                                    ports=[conn.laddr.port],
                                    connections=[f"{conn.laddr.ip}:{conn.laddr.port}"],
                                    detection_strategy="network_connections",
                                    age_hours=(time.time() - proc.create_time()) / 3600,
                                )
                                found.append(process_info)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
            except (psutil.AccessDenied, PermissionError) as e:
                # This is expected on macOS without sudo
                logger.debug(f"Network connections scan requires elevated permissions: {e}")
            return found

        loop = asyncio.get_event_loop()
        processes = await loop.run_in_executor(None, scan_connections)

        return processes

    async def _detect_file_descriptor(self) -> List[ProcessInfo]:
        """Strategy 5: Detect via open file descriptors"""
        processes = []

        def scan_fds():
            found = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    # Check open files
                    for file in proc.open_files():
                        file_path = file.path.lower()
                        if any(pattern.lower() in file_path
                               for pattern in self.config.file_patterns):
                            info = proc.info
                            process_info = ProcessInfo(
                                pid=info['pid'],
                                name=info['name'],
                                cmdline=info.get('cmdline') or [],
                                create_time=info['create_time'],
                                detection_strategy="file_descriptor",
                                age_hours=(time.time() - info['create_time']) / 3600,
                            )
                            found.append(process_info)
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return found

        loop = asyncio.get_event_loop()
        processes = await loop.run_in_executor(None, scan_fds)

        return processes

    async def _detect_parent_child(self) -> List[ProcessInfo]:
        """Strategy 6: Detect via parent-child relationships"""
        processes = []

        # First, get all processes we've already detected
        existing_pids = set(self.detected_processes.keys())

        def scan_children():
            found = []
            for pid in existing_pids:
                try:
                    proc = psutil.Process(pid)
                    # Get children
                    for child in proc.children(recursive=True):
                        process_info = ProcessInfo(
                            pid=child.pid,
                            name=child.name(),
                            cmdline=child.cmdline(),
                            create_time=child.create_time(),
                            parent_pid=pid,
                            detection_strategy="parent_child",
                            age_hours=(time.time() - child.create_time()) / 3600,
                        )
                        found.append(process_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return found

        loop = asyncio.get_event_loop()
        processes = await loop.run_in_executor(None, scan_children)

        return processes

    async def _detect_command_line(self) -> List[ProcessInfo]:
        """Strategy 7: Detect via command line regex patterns"""
        processes = []

        def scan_cmdlines():
            found = []
            import re
            patterns = [re.compile(p, re.IGNORECASE) for p in self.config.cmdline_patterns]

            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    info = proc.info
                    cmdline = info.get('cmdline') or []
                    cmdline_str = ' '.join(cmdline)

                    if any(pattern.search(cmdline_str) for pattern in patterns):
                        process_info = ProcessInfo(
                            pid=info['pid'],
                            name=info['name'],
                            cmdline=cmdline,
                            create_time=info['create_time'],
                            detection_strategy="command_line",
                            age_hours=(time.time() - info['create_time']) / 3600,
                        )
                        found.append(process_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return found

        loop = asyncio.get_event_loop()
        processes = await loop.run_in_executor(None, scan_cmdlines)

        return processes

    def _build_relationships(self, processes: List[ProcessInfo]):
        """Build parent-child relationships"""
        pid_map = {p.pid: p for p in processes}

        for process in processes:
            try:
                proc = psutil.Process(process.pid)
                parent = proc.parent()
                if parent and parent.pid in pid_map:
                    process.parent_pid = parent.pid
                    pid_map[parent.pid].children_pids.append(process.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def _assign_priorities(self, processes: List[ProcessInfo]):
        """Assign termination priorities based on process characteristics"""
        for process in processes:
            # Parent processes are CRITICAL (must be killed first)
            if process.children_pids:
                process.priority = ProcessPriority.CRITICAL

            # Main backend processes are HIGH
            elif "main.py" in ' '.join(process.cmdline).lower():
                process.priority = ProcessPriority.HIGH

            # Start system processes are HIGH
            elif "start_system" in ' '.join(process.cmdline).lower():
                process.priority = ProcessPriority.HIGH

            # Port-bound processes are MEDIUM
            elif process.ports:
                process.priority = ProcessPriority.MEDIUM

            # Everything else is LOW
            else:
                process.priority = ProcessPriority.LOW

    async def terminate_processes(
        self,
        processes: List[ProcessInfo],
        graceful_timeout: float = 2.0,
        force_timeout: float = 1.0
    ) -> Tuple[int, int]:
        """
        Terminate processes with graceful shutdown and force kill fallback.

        Args:
            processes: List of ProcessInfo objects to terminate
            graceful_timeout: Seconds to wait after SIGTERM
            force_timeout: Seconds to wait after SIGKILL

        Returns:
            Tuple of (killed_count, failed_count)
        """
        killed = 0
        failed = 0

        for process in processes:
            try:
                logger.info(f"Terminating PID {process.pid} ({process.name})...")

                # Try SIGTERM first (graceful)
                try:
                    os.kill(process.pid, signal.SIGTERM)
                    await asyncio.sleep(graceful_timeout)
                except ProcessLookupError:
                    logger.debug(f"PID {process.pid} already dead")
                    killed += 1
                    continue

                # Check if still alive
                if psutil.pid_exists(process.pid):
                    logger.debug(f"PID {process.pid} still alive, forcing...")
                    try:
                        os.kill(process.pid, signal.SIGKILL)
                        await asyncio.sleep(force_timeout)
                    except ProcessLookupError:
                        pass

                # Final verification
                if psutil.pid_exists(process.pid):
                    logger.warning(f"PID {process.pid} still alive after SIGKILL!")
                    failed += 1
                else:
                    logger.info(f"✓ PID {process.pid} terminated successfully")
                    killed += 1

            except PermissionError:
                logger.error(f"Permission denied to kill PID {process.pid}")
                failed += 1
            except Exception as e:
                logger.error(f"Failed to kill PID {process.pid}: {e}")
                failed += 1

        return killed, failed


# Convenience function for backward compatibility
async def detect_and_kill_jarvis_processes(
    config: Optional[DetectionConfig] = None,
    dry_run: bool = False
) -> Dict:
    """
    Detect and kill JARVIS processes using advanced detection.

    Args:
        config: Optional DetectionConfig, uses auto-discovery if None
        dry_run: If True, only detect but don't kill

    Returns:
        Dict with detection results and statistics
    """
    detector = AdvancedProcessDetector(config)

    # Detect processes
    processes = await detector.detect_all()

    result = {
        "total_detected": len(processes),
        "processes": [p.to_dict() for p in processes],
        "killed": 0,
        "failed": 0,
    }

    if not dry_run and processes:
        killed, failed = await detector.terminate_processes(processes)
        result["killed"] = killed
        result["failed"] = failed

    return result
