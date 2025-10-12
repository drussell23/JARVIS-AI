"""
Robust Async Subprocess Manager for JARVIS Vision System

This module provides a production-ready async subprocess management system with:
- Resource pooling and limits
- Automatic cleanup and lifecycle management
- Comprehensive error handling
- Semaphore leak prevention
- Process tracking and monitoring
"""

import asyncio
import logging
import os
import signal
import sys
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import deque
import atexit
import psutil

logger = logging.getLogger(__name__)


class ProcessState(Enum):
    """Process lifecycle states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"
    TIMEOUT = "timeout"


@dataclass
class ProcessInfo:
    """Information about a managed subprocess"""
    process_id: str
    command: List[str]
    state: ProcessState
    process: Optional[asyncio.subprocess.Process] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    stdout: Optional[bytes] = None
    stderr: Optional[bytes] = None
    return_code: Optional[int] = None
    error: Optional[str] = None
    pid: Optional[int] = None


class AsyncSubprocessManager:
    """
    Advanced async subprocess manager with resource management and cleanup.

    Features:
    - Subprocess pooling with configurable limits
    - Automatic cleanup on shutdown
    - Process lifecycle tracking
    - Memory and resource monitoring
    - Comprehensive error handling
    - Semaphore leak prevention
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern to ensure single manager instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        max_concurrent: int = 5,
        max_queue_size: int = 50,
        process_timeout: float = 30.0,
        cleanup_interval: float = 10.0,
        enable_monitoring: bool = True
    ):
        """
        Initialize the subprocess manager.

        Args:
            max_concurrent: Maximum concurrent subprocesses
            max_queue_size: Maximum queued subprocess requests
            process_timeout: Default timeout for processes (seconds)
            cleanup_interval: Interval for cleanup tasks (seconds)
            enable_monitoring: Enable resource monitoring
        """
        if self._initialized:
            return

        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.process_timeout = process_timeout
        self.cleanup_interval = cleanup_interval
        self.enable_monitoring = enable_monitoring

        # Process tracking
        self._processes: Dict[str, ProcessInfo] = {}
        self._active_processes: Set[str] = set()
        self._process_queue: deque = deque(maxlen=max_queue_size)

        # Resource management
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._shutdown_event = asyncio.Event()
        self._cleanup_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = {
            "total_started": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_timeout": 0,
            "total_terminated": 0,
            "peak_concurrent": 0
        }

        # Register cleanup handlers
        atexit.register(self._sync_cleanup)
        self._setup_signal_handlers()

        # Start background tasks
        self._start_background_tasks()

        self._initialized = True
        logger.info(f"AsyncSubprocessManager initialized: max_concurrent={max_concurrent}")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())

        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, signal_handler)

    def _start_background_tasks(self):
        """Start background cleanup and monitoring tasks"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """Background task for periodic cleanup"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_dead_processes()

                if self.enable_monitoring:
                    self._log_resource_usage()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_dead_processes(self):
        """Clean up terminated processes and free resources"""
        to_remove = []

        for process_id, info in self._processes.items():
            if info.state in [ProcessState.COMPLETED, ProcessState.FAILED,
                             ProcessState.TERMINATED, ProcessState.TIMEOUT]:
                # Process is done, check if it's been long enough to clean up
                if info.end_time and (time.time() - info.end_time) > 60:
                    to_remove.append(process_id)
            elif info.process and info.process.returncode is not None:
                # Process terminated but not yet marked
                info.state = ProcessState.COMPLETED
                info.end_time = time.time()
                info.return_code = info.process.returncode
                self._active_processes.discard(process_id)

        # Remove old processes
        for process_id in to_remove:
            del self._processes[process_id]
            logger.debug(f"Cleaned up old process: {process_id}")

    def _log_resource_usage(self):
        """Log current resource usage"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            num_fds = process.num_fds() if hasattr(process, 'num_fds') else 0

            logger.debug(
                f"Subprocess Manager Stats: "
                f"active={len(self._active_processes)}/{self.max_concurrent}, "
                f"tracked={len(self._processes)}, "
                f"memory={memory_mb:.1f}MB, "
                f"fds={num_fds}"
            )
        except Exception as e:
            logger.debug(f"Could not get resource usage: {e}")

    @asynccontextmanager
    async def subprocess(
        self,
        command: List[str],
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        capture_output: bool = True,
        **kwargs
    ):
        """
        Context manager for running a subprocess with automatic cleanup.

        Args:
            command: Command to execute
            timeout: Process timeout (uses default if None)
            cwd: Working directory
            env: Environment variables
            capture_output: Whether to capture stdout/stderr
            **kwargs: Additional arguments for create_subprocess_exec

        Yields:
            ProcessInfo object with process details
        """
        process_id = f"proc_{time.time()}_{id(command)}"
        timeout = timeout or self.process_timeout
        info = ProcessInfo(process_id=process_id, command=command, state=ProcessState.PENDING)

        async with self._semaphore:
            try:
                # Track process
                self._processes[process_id] = info
                self._active_processes.add(process_id)
                self._stats["total_started"] += 1

                # Update peak concurrent
                current_concurrent = len(self._active_processes)
                if current_concurrent > self._stats["peak_concurrent"]:
                    self._stats["peak_concurrent"] = current_concurrent

                # Prepare subprocess arguments
                kwargs.update({
                    'stdout': asyncio.subprocess.PIPE if capture_output else None,
                    'stderr': asyncio.subprocess.PIPE if capture_output else None,
                    'cwd': cwd,
                    'env': env or os.environ.copy(),
                    # Ensure process gets its own process group for clean termination
                    'start_new_session': True
                })

                # Create subprocess
                info.process = await asyncio.create_subprocess_exec(
                    *command,
                    **kwargs
                )
                info.pid = info.process.pid
                info.state = ProcessState.RUNNING

                logger.debug(f"Started subprocess {process_id}: {' '.join(command)} (PID: {info.pid})")

                # Yield control to caller
                yield info

                # Wait for completion with timeout
                try:
                    stdout, stderr = await asyncio.wait_for(
                        info.process.communicate(),
                        timeout=timeout
                    )
                    info.stdout = stdout
                    info.stderr = stderr
                    info.return_code = info.process.returncode
                    info.state = ProcessState.COMPLETED
                    info.end_time = time.time()
                    self._stats["total_completed"] += 1

                    logger.debug(f"Subprocess {process_id} completed with code {info.return_code}")

                except asyncio.TimeoutError:
                    logger.warning(f"Subprocess {process_id} timed out after {timeout}s")
                    info.state = ProcessState.TIMEOUT
                    info.error = f"Process timed out after {timeout} seconds"
                    self._stats["total_timeout"] += 1
                    await self._terminate_process(info)

            except Exception as e:
                logger.error(f"Error running subprocess {process_id}: {e}")
                info.state = ProcessState.FAILED
                info.error = str(e)
                info.end_time = time.time()
                self._stats["total_failed"] += 1

                if info.process:
                    await self._terminate_process(info)

                raise

            finally:
                # Clean up
                self._active_processes.discard(process_id)

                # Ensure process is terminated
                if info.process and info.process.returncode is None:
                    await self._terminate_process(info)

    async def _terminate_process(self, info: ProcessInfo):
        """Terminate a process gracefully, then forcefully if needed"""
        if not info.process:
            return

        try:
            # Try graceful termination first
            info.process.terminate()

            try:
                await asyncio.wait_for(info.process.wait(), timeout=5.0)
                logger.debug(f"Process {info.process_id} terminated gracefully")
            except asyncio.TimeoutError:
                # Force kill if graceful termination fails
                try:
                    info.process.kill()
                    await info.process.wait()
                    logger.warning(f"Process {info.process_id} force killed")
                except ProcessLookupError:
                    pass  # Process already dead

            info.state = ProcessState.TERMINATED
            info.end_time = time.time()
            self._stats["total_terminated"] += 1

        except Exception as e:
            logger.error(f"Error terminating process {info.process_id}: {e}")

    async def run_command(
        self,
        command: List[str],
        timeout: Optional[float] = None,
        check: bool = False,
        **kwargs
    ) -> Tuple[int, Optional[bytes], Optional[bytes]]:
        """
        Run a command and return results.

        Args:
            command: Command to execute
            timeout: Process timeout
            check: Raise exception if return code is non-zero
            **kwargs: Additional arguments for subprocess

        Returns:
            Tuple of (return_code, stdout, stderr)

        Raises:
            subprocess.CalledProcessError: If check=True and process fails
        """
        async with self.subprocess(command, timeout=timeout, **kwargs) as info:
            pass  # Process runs in context manager

        if check and info.return_code != 0:
            from subprocess import CalledProcessError
            raise CalledProcessError(
                info.return_code or -1,
                command,
                output=info.stdout,
                stderr=info.stderr
            )

        return info.return_code or 0, info.stdout, info.stderr

    async def shutdown(self, timeout: float = 10.0):
        """
        Shutdown the subprocess manager and clean up all resources.

        Args:
            timeout: Maximum time to wait for processes to terminate
        """
        logger.info("Shutting down AsyncSubprocessManager...")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel background tasks
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Terminate all active processes
        tasks = []
        for process_id in list(self._active_processes):
            if process_id in self._processes:
                info = self._processes[process_id]
                tasks.append(self._terminate_process(info))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Log final statistics
        logger.info(
            f"Subprocess Manager Statistics: "
            f"started={self._stats['total_started']}, "
            f"completed={self._stats['total_completed']}, "
            f"failed={self._stats['total_failed']}, "
            f"timeout={self._stats['total_timeout']}, "
            f"terminated={self._stats['total_terminated']}, "
            f"peak_concurrent={self._stats['peak_concurrent']}"
        )

        # Clear all tracking
        self._processes.clear()
        self._active_processes.clear()

        logger.info("AsyncSubprocessManager shutdown complete")

    def _sync_cleanup(self):
        """Synchronous cleanup for atexit handler"""
        try:
            # Try to get the event loop
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                # Run async shutdown
                loop.run_until_complete(self.shutdown())
        except Exception as e:
            logger.error(f"Error in sync cleanup: {e}")
            # Force kill any remaining processes
            for info in self._processes.values():
                if info.process and info.pid:
                    try:
                        os.kill(info.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            **self._stats,
            "active_processes": len(self._active_processes),
            "tracked_processes": len(self._processes),
            "queue_size": len(self._process_queue)
        }

    def get_active_processes(self) -> List[ProcessInfo]:
        """Get list of currently active processes"""
        return [
            info for process_id, info in self._processes.items()
            if process_id in self._active_processes
        ]


# Global singleton instance
_subprocess_manager: Optional[AsyncSubprocessManager] = None


def get_subprocess_manager() -> AsyncSubprocessManager:
    """Get or create the global subprocess manager instance"""
    global _subprocess_manager
    if _subprocess_manager is None:
        _subprocess_manager = AsyncSubprocessManager()
    return _subprocess_manager


# Convenience function for simple command execution
async def run_command(
    command: List[str],
    timeout: Optional[float] = None,
    **kwargs
) -> Tuple[int, Optional[bytes], Optional[bytes]]:
    """
    Convenience function to run a command using the global manager.

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    manager = get_subprocess_manager()
    return await manager.run_command(command, timeout=timeout, **kwargs)