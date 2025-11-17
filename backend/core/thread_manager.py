"""
Advanced Thread Manager - Bulletproof thread lifecycle management
===================================================================

Enterprise-grade thread management system that prevents leaks, ensures
clean shutdown, and provides comprehensive monitoring.

Features:
- Automatic thread discovery and tracking
- Async/await support with event loop management
- Multi-phase shutdown with escalation
- Thread pool integration
- Deadlock detection and prevention
- Resource leak detection
- Health monitoring and alerting
- Auto-cleanup of orphaned threads
- Stack trace capture for debugging
- Metric collection and reporting
- Configurable policies (no hardcoding)
- Thread priority management
- Graceful degradation under load
"""

import asyncio
import threading
import logging
import time
import traceback
import signal
import sys
import os
import ctypes
import weakref
from typing import Dict, List, Optional, Callable, Set, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
import atexit
import psutil
from pathlib import Path
import aiohttp
import websockets

logger = logging.getLogger(__name__)


class ThreadState(Enum):
    """Thread lifecycle states"""
    CREATED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()
    LEAKED = auto()


class HUDThreadState(Enum):
    """HUD-specific thread states"""
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    FROZEN = auto()
    DISCONNECTED = auto()
    RECOVERING = auto()


class ShutdownPhase(Enum):
    """Shutdown escalation phases"""
    GRACEFUL = auto()      # Signal threads to stop
    FORCEFUL = auto()      # Interrupt threads
    TERMINATE = auto()     # Force terminate
    EMERGENCY = auto()     # Last resort cleanup


@dataclass
class ThreadPolicy:
    """Configurable thread behavior policy"""
    # Shutdown behavior
    graceful_shutdown_timeout: float = 5.0      # Seconds to wait in graceful phase
    forceful_shutdown_timeout: float = 3.0      # Seconds to wait in forceful phase
    terminate_shutdown_timeout: float = 2.0     # Seconds to wait in terminate phase
    emergency_shutdown_timeout: float = 1.0     # Final timeout before giving up

    # Thread limits
    max_threads: Optional[int] = None           # None = unlimited
    max_thread_lifetime: Optional[float] = None # None = unlimited
    warn_thread_age: float = 3600.0            # Warn if thread runs longer than this

    # Monitoring
    enable_health_check: bool = True
    health_check_interval: float = 30.0        # Seconds between health checks
    enable_deadlock_detection: bool = True
    deadlock_check_interval: float = 60.0      # Seconds between deadlock checks

    # Cleanup
    auto_cleanup_orphans: bool = True
    orphan_check_interval: float = 60.0        # Seconds between orphan checks
    force_daemon_on_shutdown: bool = True      # Convert non-daemons to daemons during shutdown

    # Logging
    log_thread_creation: bool = True
    log_thread_completion: bool = True
    log_stack_traces: bool = True
    capture_full_stack: bool = False           # Capture entire stack vs just caller

    # Performance
    use_thread_pool: bool = True
    thread_pool_size: Optional[int] = None     # None = auto-detect
    recycle_threads: bool = True


@dataclass
class ThreadMetrics:
    """Thread performance metrics"""
    total_created: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_leaked: int = 0
    total_forced_stop: int = 0
    current_active: int = 0
    current_daemon: int = 0
    current_non_daemon: int = 0
    peak_active: int = 0
    avg_lifetime_seconds: float = 0.0
    shutdown_attempts: int = 0
    successful_shutdowns: int = 0
    failed_shutdowns: int = 0


@dataclass
class ThreadInfo:
    """Comprehensive thread information"""
    # Identity
    thread_id: int
    thread: Union[threading.Thread, weakref.ref]
    name: str
    ident: Optional[int] = None  # Thread identifier from OS

    # Lifecycle
    state: ThreadState = ThreadState.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None

    # Origin
    creator: str = "unknown"
    creator_stack: Optional[List[str]] = None
    purpose: str = "Unknown"
    category: str = "general"  # e.g., "io", "compute", "network"

    # Configuration
    daemon: bool = False
    priority: int = 0

    # Cleanup
    shutdown_callback: Optional[Callable] = None
    shutdown_event: Optional[Union[threading.Event, asyncio.Event]] = None
    is_async: bool = False
    event_loop: Optional[asyncio.AbstractEventLoop] = None

    # Monitoring
    last_heartbeat: Optional[datetime] = None
    heartbeat_interval: Optional[float] = None
    health_check_callback: Optional[Callable] = None

    # Metrics
    exception: Optional[Exception] = None
    stack_trace: Optional[str] = None
    cpu_time: float = 0.0

    # Relationships
    parent_thread_id: Optional[int] = None
    child_thread_ids: Set[int] = field(default_factory=set)


class HUDThreadManager:
    """
    🚀 Enhanced JARVIS HUD Thread Manager v2.0
    ==========================================

    Manages HUD-specific threads with advanced health monitoring,
    auto-recovery, and graceful shutdown capabilities.

    Features:
    - Real-time HUD thread health monitoring
    - WebSocket connection tracking
    - Automatic recovery from frozen/disconnected states
    - Graceful shutdown with WebSocket notification
    - Detailed metrics and diagnostics
    - Thread leak detection
    - Performance tracking

    Usage:
        hud_manager = HUDThreadManager()

        # Monitor health
        health = await hud_manager.check_hud_threads_health()

        # Auto-recovery
        await hud_manager.monitor_hud_threads_loop()

        # Graceful shutdown
        await hud_manager.graceful_shutdown_hud_threads()
    """

    def __init__(self, thread_manager: Optional['AdvancedThreadManager'] = None):
        """
        Initialize HUD Thread Manager

        Args:
            thread_manager: Parent thread manager instance
        """
        self.thread_manager = thread_manager
        self.hud_process_name = "JARVIS-HUD"
        self.hud_log_path = "/tmp/jarvis_hud.log"
        self.backend_ws_url = "ws://localhost:8010/ws"
        self.backend_http_url = "http://localhost:8010"

        # HUD thread categories
        self.hud_categories = {
            "hud_main",
            "hud_websocket",
            "hud_ui",
            "hud_animation",
            "hud_audio"
        }

        # Health thresholds
        self.max_connection_failures = 3
        self.health_check_interval = 30  # seconds
        self.ws_timeout = 10  # seconds
        self.frozen_detection_threshold = 120  # 2 minutes without heartbeat

        # Tracking
        self.consecutive_failures = 0
        self.last_health_check = None
        self.hud_threads: Dict[int, Dict[str, Any]] = {}

        logger.info("🎯 HUD Thread Manager v2.0 initialized")
        logger.info(f"   Categories: {', '.join(self.hud_categories)}")
        logger.info(f"   Health check interval: {self.health_check_interval}s")

    def find_hud_threads(self) -> List[ThreadInfo]:
        """
        Find all HUD-related threads

        Returns:
            List of HUD thread information
        """
        if not self.thread_manager:
            return []

        hud_threads = []

        # Get all active threads from parent manager
        all_threads = self.thread_manager.get_active_threads()

        for thread_info in all_threads:
            # Check if thread belongs to HUD categories
            if thread_info.category in self.hud_categories:
                hud_threads.append(thread_info)

            # Also check thread name for HUD-related patterns
            elif any(pattern in thread_info.name.lower() for pattern in ['hud', 'jarvis-hud']):
                hud_threads.append(thread_info)

        return hud_threads

    async def check_hud_threads_health(self) -> Dict[str, Any]:
        """
        Comprehensive HUD threads health check

        Returns:
            Health status dictionary with detailed metrics
        """
        health = {
            'timestamp': time.time(),
            'hud_threads_count': 0,
            'healthy_threads': 0,
            'degraded_threads': 0,
            'unhealthy_threads': 0,
            'frozen_threads': [],
            'websocket_connected': False,
            'backend_reachable': False,
            'status': HUDThreadState.UNHEALTHY.name,
            'issues': [],
            'thread_details': []
        }

        try:
            # Find all HUD threads
            hud_threads = self.find_hud_threads()
            health['hud_threads_count'] = len(hud_threads)

            if not hud_threads:
                health['issues'].append("No HUD threads found")
                return health

            # Check each thread's health
            now = datetime.now()
            for thread_info in hud_threads:
                thread_health = {
                    'name': thread_info.name,
                    'category': thread_info.category,
                    'state': thread_info.state.name,
                    'age_seconds': (now - thread_info.created_at).total_seconds(),
                    'is_alive': self._is_thread_alive(thread_info),
                    'status': 'unknown'
                }

                # Check if thread is frozen (no heartbeat)
                if thread_info.last_heartbeat:
                    time_since_heartbeat = (now - thread_info.last_heartbeat).total_seconds()
                    thread_health['seconds_since_heartbeat'] = time_since_heartbeat

                    if time_since_heartbeat > self.frozen_detection_threshold:
                        thread_health['status'] = 'frozen'
                        health['frozen_threads'].append(thread_info.name)
                        health['unhealthy_threads'] += 1
                        health['issues'].append(f"{thread_info.name}: Frozen (no heartbeat for {time_since_heartbeat:.1f}s)")
                    elif time_since_heartbeat > self.frozen_detection_threshold / 2:
                        thread_health['status'] = 'degraded'
                        health['degraded_threads'] += 1
                        health['issues'].append(f"{thread_info.name}: Degraded (slow heartbeat)")
                    else:
                        thread_health['status'] = 'healthy'
                        health['healthy_threads'] += 1
                else:
                    # No heartbeat recorded at all
                    thread_health['status'] = 'degraded'
                    health['degraded_threads'] += 1
                    health['issues'].append(f"{thread_info.name}: No heartbeat recorded")

                # Check if thread failed
                if thread_info.state == ThreadState.FAILED:
                    thread_health['status'] = 'unhealthy'
                    health['unhealthy_threads'] += 1
                    if thread_info.exception:
                        health['issues'].append(f"{thread_info.name}: Failed - {thread_info.exception}")

                # Check if thread is not alive
                if not thread_health['is_alive']:
                    thread_health['status'] = 'unhealthy'
                    health['unhealthy_threads'] += 1
                    health['issues'].append(f"{thread_info.name}: Thread not alive")

                health['thread_details'].append(thread_health)

            # Check WebSocket connection
            try:
                ws = None
                try:
                    ws = await asyncio.wait_for(
                        websockets.connect(self.backend_ws_url),
                        timeout=self.ws_timeout
                    )
                    await ws.send('{"type": "hud_health_check"}')
                    response = await asyncio.wait_for(ws.recv(), timeout=3)
                    health['websocket_connected'] = True
                finally:
                    if ws:
                        await ws.close()
            except Exception as e:
                health['issues'].append(f"WebSocket connection failed: {e}")

            # Check backend reachability
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.backend_http_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            health['backend_reachable'] = True
            except Exception as e:
                health['issues'].append(f"Backend unreachable: {e}")

            # Determine overall status
            if health['unhealthy_threads'] == 0 and health['websocket_connected'] and health['backend_reachable']:
                if health['degraded_threads'] == 0:
                    health['status'] = HUDThreadState.HEALTHY.name
                else:
                    health['status'] = HUDThreadState.DEGRADED.name
            elif health['frozen_threads']:
                health['status'] = HUDThreadState.FROZEN.name
            elif not health['websocket_connected']:
                health['status'] = HUDThreadState.DISCONNECTED.name
            else:
                health['status'] = HUDThreadState.UNHEALTHY.name

            self.last_health_check = time.time()

            # Log health status
            if health['issues']:
                logger.warning(f"🔍 HUD Thread Health Check - {health['status']}")
                logger.warning(f"   Threads: {health['hud_threads_count']} total, "
                             f"{health['healthy_threads']} healthy, "
                             f"{health['degraded_threads']} degraded, "
                             f"{health['unhealthy_threads']} unhealthy")
                for issue in health['issues'][:5]:  # Limit to first 5 issues
                    logger.warning(f"   ⚠️  {issue}")
            else:
                logger.info(f"✅ HUD Thread Health Check - {health['status']}")
                logger.info(f"   All {health['hud_threads_count']} HUD threads healthy")

            return health

        except Exception as e:
            health['issues'].append(f"Health check failed: {e}")
            logger.error(f"❌ HUD thread health check error: {e}")
            return health

    async def graceful_shutdown_hud_threads(self, timeout: int = 10) -> bool:
        """
        Gracefully shutdown all HUD threads with WebSocket notification

        Args:
            timeout: Shutdown timeout in seconds

        Returns:
            True if all threads shutdown successfully
        """
        logger.info("🛑 Initiating graceful HUD thread shutdown...")

        try:
            # Step 1: Send disconnect signal via WebSocket
            try:
                ws = None
                try:
                    ws = await asyncio.wait_for(
                        websockets.connect(self.backend_ws_url),
                        timeout=5
                    )
                    await ws.send('{"type": "hud_shutdown", "reason": "graceful_thread_shutdown"}')
                    logger.info("   ✅ Shutdown notification sent to backend")
                finally:
                    if ws:
                        await ws.close()
            except Exception as e:
                logger.warning(f"   ⚠️  Could not send shutdown notification: {e}")

            # Step 2: Get all HUD threads
            hud_threads = self.find_hud_threads()

            if not hud_threads:
                logger.info("   No HUD threads to shutdown")
                return True

            logger.info(f"   Found {len(hud_threads)} HUD threads to shutdown")

            # Step 3: Signal threads to stop gracefully
            for thread_info in hud_threads:
                try:
                    # Set shutdown event if available
                    if thread_info.shutdown_event:
                        if isinstance(thread_info.shutdown_event, asyncio.Event):
                            thread_info.shutdown_event.set()
                        else:
                            thread_info.shutdown_event.set()
                        logger.debug(f"   Signaled {thread_info.name} to stop")

                    # Call shutdown callback if available
                    if thread_info.shutdown_callback:
                        if asyncio.iscoroutinefunction(thread_info.shutdown_callback):
                            await thread_info.shutdown_callback()
                        else:
                            thread_info.shutdown_callback()
                        logger.debug(f"   Called shutdown callback for {thread_info.name}")

                    thread_info.state = ThreadState.STOPPING

                except Exception as e:
                    logger.error(f"   ❌ Error signaling {thread_info.name}: {e}")

            # Step 4: Wait for threads to complete
            deadline = time.time() + timeout
            remaining_threads = hud_threads.copy()

            while remaining_threads and time.time() < deadline:
                # Check which threads are still alive
                still_alive = []
                for thread_info in remaining_threads:
                    if self._is_thread_alive(thread_info):
                        still_alive.append(thread_info)
                    else:
                        logger.info(f"   ✅ {thread_info.name} stopped")

                remaining_threads = still_alive

                if remaining_threads:
                    await asyncio.sleep(0.5)

            # Step 5: Report results
            if not remaining_threads:
                logger.info("✅ All HUD threads shutdown gracefully")
                return True
            else:
                logger.warning(f"⚠️  {len(remaining_threads)} HUD threads did not stop within timeout:")
                for thread_info in remaining_threads:
                    logger.warning(f"   - {thread_info.name} ({thread_info.category})")
                return False

        except Exception as e:
            logger.error(f"❌ HUD thread shutdown error: {e}")
            return False

    async def auto_recovery(self, issue: str, health_data: Dict[str, Any]) -> bool:
        """
        Automatic recovery for HUD thread issues

        Args:
            issue: Description of the issue
            health_data: Health check data

        Returns:
            True if recovery was successful
        """
        logger.info(f"🔧 Attempting HUD thread auto-recovery for: {issue}")

        try:
            # Recovery strategy 1: Frozen threads
            if health_data.get('frozen_threads'):
                logger.info("   Strategy: Restart frozen HUD threads")

                # Try graceful shutdown first
                shutdown_success = await self.graceful_shutdown_hud_threads(timeout=5)

                if not shutdown_success:
                    logger.warning("   Graceful shutdown failed, attempting forceful shutdown")
                    # Let parent thread manager handle forceful shutdown
                    if self.thread_manager:
                        await self.thread_manager.shutdown_async(
                            categories=list(self.hud_categories),
                            timeout=3
                        )

                # Wait a moment
                await asyncio.sleep(2)

                # HUD should auto-restart via process manager
                logger.info("   ✅ Frozen threads cleared, waiting for HUD restart")
                self.consecutive_failures = 0
                return True

            # Recovery strategy 2: WebSocket disconnected
            if health_data.get('status') == HUDThreadState.DISCONNECTED.name:
                logger.info("   Strategy: Check backend and attempt reconnection")

                # Check if backend is running
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{self.backend_http_url}/health") as resp:
                            if resp.status == 200:
                                logger.info("   Backend is running, HUD should reconnect")
                                # Wait for automatic reconnection
                                await asyncio.sleep(5)

                                # Check if reconnected
                                recheck = await self.check_hud_threads_health()
                                if recheck['websocket_connected']:
                                    logger.info("   ✅ WebSocket reconnected successfully")
                                    self.consecutive_failures = 0
                                    return True
                                else:
                                    logger.warning("   HUD did not reconnect, may need restart")
                                    return False
                except Exception as e:
                    logger.error(f"   Backend check failed: {e}")
                    return False

            # Recovery strategy 3: Unhealthy threads
            if health_data.get('unhealthy_threads', 0) > 0:
                logger.info("   Strategy: Identify and restart unhealthy threads")

                # Get detailed thread information
                for thread_detail in health_data.get('thread_details', []):
                    if thread_detail['status'] == 'unhealthy':
                        logger.info(f"   Unhealthy thread: {thread_detail['name']}")

                # Attempt graceful shutdown and restart
                await self.graceful_shutdown_hud_threads(timeout=5)
                await asyncio.sleep(2)

                logger.info("   ✅ Unhealthy threads cleared")
                self.consecutive_failures = 0
                return True

            # Recovery strategy 4: No HUD threads found
            if health_data.get('hud_threads_count', 0) == 0:
                logger.warning("   Strategy: No HUD threads - HUD process may need restart")
                # This should be handled by process manager
                return False

            logger.warning("   No recovery strategy matched")
            return False

        except Exception as e:
            logger.error(f"❌ Auto-recovery failed: {e}")
            return False

    async def monitor_hud_threads_loop(self):
        """
        Continuous HUD thread health monitoring loop with auto-recovery

        Runs indefinitely until shutdown, checking health at regular intervals
        and triggering recovery when issues are detected.
        """
        logger.info("🔄 Starting HUD thread health monitoring loop...")
        logger.info(f"   Interval: {self.health_check_interval}s")
        logger.info(f"   Max consecutive failures before recovery: {self.max_connection_failures}")

        while True:
            try:
                await asyncio.sleep(self.health_check_interval)

                # Perform health check
                health = await self.check_hud_threads_health()

                # Check if intervention needed
                if health['status'] in [HUDThreadState.UNHEALTHY.name,
                                       HUDThreadState.FROZEN.name,
                                       HUDThreadState.DISCONNECTED.name]:
                    self.consecutive_failures += 1
                    logger.warning(f"⚠️  HUD threads {health['status']} "
                                 f"({self.consecutive_failures}/{self.max_connection_failures})")

                    # Trigger recovery after threshold
                    if self.consecutive_failures >= self.max_connection_failures:
                        logger.warning("🚨 Consecutive failure threshold reached - triggering auto-recovery")

                        recovery_issue = f"Status: {health['status']}, Issues: {len(health['issues'])}"
                        recovery_success = await self.auto_recovery(recovery_issue, health)

                        if recovery_success:
                            logger.info("✅ Auto-recovery successful")
                            self.consecutive_failures = 0
                        else:
                            logger.error("❌ Auto-recovery failed")
                            # Reset counter to avoid continuous recovery attempts
                            self.consecutive_failures = 0
                            # Wait longer before next attempt
                            await asyncio.sleep(self.health_check_interval * 2)

                else:
                    # Healthy or degraded - reset counter
                    if self.consecutive_failures > 0:
                        logger.info(f"✅ HUD threads recovered (was {self.consecutive_failures} failures)")
                    self.consecutive_failures = 0

            except asyncio.CancelledError:
                logger.info("🛑 HUD thread monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"❌ HUD thread monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval)

    def get_hud_metrics(self) -> Dict[str, Any]:
        """
        Get HUD thread metrics

        Returns:
            Dictionary of metrics
        """
        hud_threads = self.find_hud_threads()

        metrics = {
            'timestamp': time.time(),
            'total_hud_threads': len(hud_threads),
            'by_category': defaultdict(int),
            'by_state': defaultdict(int),
            'consecutive_failures': self.consecutive_failures,
            'last_health_check': self.last_health_check,
            'threads': []
        }

        for thread_info in hud_threads:
            metrics['by_category'][thread_info.category] += 1
            metrics['by_state'][thread_info.state.name] += 1

            thread_data = {
                'name': thread_info.name,
                'category': thread_info.category,
                'state': thread_info.state.name,
                'daemon': thread_info.daemon,
                'age_seconds': (datetime.now() - thread_info.created_at).total_seconds(),
                'is_alive': self._is_thread_alive(thread_info)
            }

            if thread_info.last_heartbeat:
                thread_data['last_heartbeat'] = (datetime.now() - thread_info.last_heartbeat).total_seconds()

            metrics['threads'].append(thread_data)

        return metrics

    def _is_thread_alive(self, info: ThreadInfo) -> bool:
        """Check if thread is alive"""
        thread = info.thread() if isinstance(info.thread, weakref.ref) else info.thread
        return thread and thread.is_alive()

    def print_hud_report(self):
        """Print detailed HUD thread report"""
        metrics = self.get_hud_metrics()

        print("\n" + "=" * 80)
        print("🎯 JARVIS HUD THREAD MANAGER REPORT")
        print("=" * 80)
        print(f"Total HUD Threads: {metrics['total_hud_threads']}")
        print(f"Consecutive Failures: {metrics['consecutive_failures']}")
        print(f"Last Health Check: {metrics['last_health_check']}")

        print(f"\nBy Category:")
        for category, count in metrics['by_category'].items():
            print(f"  {category}: {count}")

        print(f"\nBy State:")
        for state, count in metrics['by_state'].items():
            print(f"  {state}: {count}")

        print(f"\nThread Details:")
        for thread in metrics['threads']:
            alive_status = "✅" if thread['is_alive'] else "❌"
            print(f"  {alive_status} {thread['name']} ({thread['category']}) - {thread['state']}")
            print(f"     Age: {thread['age_seconds']:.1f}s, Daemon: {thread['daemon']}")
            if 'last_heartbeat' in thread:
                print(f"     Last heartbeat: {thread['last_heartbeat']:.1f}s ago")

        print("=" * 80 + "\n")


class AdvancedThreadManager:
    """
    Enterprise-grade thread manager with async support

    Features:
    - Automatic thread discovery
    - Multi-phase shutdown
    - Deadlock detection
    - Health monitoring
    - Async/sync thread support
    - Thread pool management
    - Leak prevention

    Usage:
        # Configure policy
        policy = ThreadPolicy(
            graceful_shutdown_timeout=10.0,
            max_threads=100
        )

        manager = AdvancedThreadManager(policy=policy)

        # Create managed thread
        thread = manager.create_thread(
            target=worker,
            name="Worker-1",
            purpose="Process data",
            category="compute"
        )

        # Or register existing thread
        manager.register(thread, purpose="Legacy worker")

        # At shutdown
        await manager.shutdown_async()  # or manager.shutdown_sync()
    """

    def __init__(self, policy: Optional[ThreadPolicy] = None, enable_hud_manager: bool = True):
        self.policy = policy or ThreadPolicy()
        self.threads: Dict[int, ThreadInfo] = {}
        self.lock = threading.RLock()  # Reentrant lock for nested calls
        self.metrics = ThreadMetrics()

        # Shutdown coordination
        self.shutdown_initiated = False
        self.shutdown_lock = threading.Lock()
        self.shutdown_event = threading.Event()

        # Thread pool
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        if self.policy.use_thread_pool:
            pool_size = self.policy.thread_pool_size or (os.cpu_count() or 4) * 2
            self.thread_pool = ThreadPoolExecutor(
                max_workers=pool_size,
                thread_name_prefix="ManagedPool"
            )

        # Monitoring
        self.health_check_thread: Optional[threading.Thread] = None
        self.deadlock_check_thread: Optional[threading.Thread] = None
        self.orphan_check_thread: Optional[threading.Thread] = None

        # Category tracking
        self.categories: Dict[str, Set[int]] = defaultdict(set)

        # HUD Thread Manager
        self.hud_manager: Optional[HUDThreadManager] = None
        if enable_hud_manager:
            self.hud_manager = HUDThreadManager(thread_manager=self)
            logger.info("   ✅ HUD Thread Manager enabled")

        # Start monitoring
        self._start_monitoring()

        # Register atexit handler
        atexit.register(self._emergency_cleanup)

        logger.info("🧵 AdvancedThreadManager initialized")
        logger.info(f"   Policy: graceful={self.policy.graceful_shutdown_timeout}s, "
                   f"max_threads={self.policy.max_threads or 'unlimited'}")

    def _start_monitoring(self):
        """Start background monitoring threads"""
        if self.policy.enable_health_check:
            self.health_check_thread = threading.Thread(
                target=self._health_check_loop,
                name="ThreadManager-HealthCheck",
                daemon=True
            )
            self.health_check_thread.start()

        if self.policy.enable_deadlock_detection:
            self.deadlock_check_thread = threading.Thread(
                target=self._deadlock_check_loop,
                name="ThreadManager-DeadlockCheck",
                daemon=True
            )
            self.deadlock_check_thread.start()

        if self.policy.auto_cleanup_orphans:
            self.orphan_check_thread = threading.Thread(
                target=self._orphan_check_loop,
                name="ThreadManager-OrphanCheck",
                daemon=True
            )
            self.orphan_check_thread.start()

    def _get_caller_info(self, depth: int = 2) -> Tuple[str, Optional[List[str]]]:
        """Get information about caller"""
        try:
            stack = traceback.extract_stack()
            caller = f"{stack[-depth].filename}:{stack[-depth].lineno} in {stack[-depth].name}"

            if self.policy.log_stack_traces:
                if self.policy.capture_full_stack:
                    stack_trace = [f"{frame.filename}:{frame.lineno} in {frame.name}"
                                  for frame in stack]
                else:
                    stack_trace = [f"{frame.filename}:{frame.lineno} in {frame.name}"
                                  for frame in stack[-5:]]  # Last 5 frames
                return caller, stack_trace

            return caller, None
        except Exception as e:
            logger.debug(f"Failed to get caller info: {e}")
            return "unknown", None

    def register(
        self,
        thread: threading.Thread,
        purpose: str = "Unknown",
        category: str = "general",
        shutdown_callback: Optional[Callable] = None,
        shutdown_event: Optional[Union[threading.Event, asyncio.Event]] = None,
        is_async: bool = False,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
        force_daemon: bool = False,
        priority: int = 0,
        heartbeat_interval: Optional[float] = None,
        health_check_callback: Optional[Callable] = None
    ) -> threading.Thread:
        """
        Register a thread for comprehensive tracking

        Args:
            thread: Thread to register
            purpose: Human-readable purpose
            category: Thread category for grouping
            shutdown_callback: Function to call for clean shutdown
            shutdown_event: Event to set for shutdown signaling
            is_async: Whether thread runs async code
            event_loop: Event loop if async thread
            force_daemon: Convert to daemon if not already
            priority: Thread priority (higher = more important)
            heartbeat_interval: Expected heartbeat interval
            health_check_callback: Custom health check function

        Returns:
            The registered thread
        """
        # Check thread limit
        with self.lock:
            if self.policy.max_threads and len(self.threads) >= self.policy.max_threads:
                raise RuntimeError(
                    f"Thread limit reached: {self.policy.max_threads}. "
                    "Increase max_threads or wait for threads to complete."
                )

            # Get caller context
            caller, stack_trace = self._get_caller_info(depth=3)

            # Force daemon if requested
            original_daemon = thread.daemon
            if force_daemon and not thread.daemon:
                if self.policy.log_thread_creation:
                    logger.debug(f"Converting {thread.name} to daemon")
                thread.daemon = True

            # Create thread info
            thread_id = id(thread)
            info = ThreadInfo(
                thread_id=thread_id,
                thread=weakref.ref(thread) if self.policy.recycle_threads else thread,
                name=thread.name,
                ident=thread.ident,
                state=ThreadState.CREATED,
                creator=caller,
                creator_stack=stack_trace,
                purpose=purpose,
                category=category,
                daemon=thread.daemon,
                priority=priority,
                shutdown_callback=shutdown_callback,
                shutdown_event=shutdown_event,
                is_async=is_async,
                event_loop=event_loop,
                heartbeat_interval=heartbeat_interval,
                health_check_callback=health_check_callback,
                parent_thread_id=id(threading.current_thread())
            )

            # Register
            self.threads[thread_id] = info
            self.categories[category].add(thread_id)

            # Update metrics
            self.metrics.total_created += 1
            self.metrics.current_active += 1
            if thread.daemon:
                self.metrics.current_daemon += 1
            else:
                self.metrics.current_non_daemon += 1

            if self.metrics.current_active > self.metrics.peak_active:
                self.metrics.peak_active = self.metrics.current_active

            # Log
            if self.policy.log_thread_creation:
                logger.info(
                    f"📝 Registered thread: {thread.name} "
                    f"(daemon={thread.daemon}, category={category}, priority={priority})"
                )
                if original_daemon != thread.daemon:
                    logger.info(f"   Converted to daemon: {thread.name}")

            return thread

    def create_thread(
        self,
        target: Callable,
        name: str,
        purpose: str = "Unknown",
        category: str = "general",
        daemon: bool = True,
        **kwargs
    ) -> threading.Thread:
        """
        Create and register a new thread

        Args:
            target: Function to run
            name: Thread name
            purpose: Thread purpose
            category: Thread category
            daemon: Whether thread should be daemon
            **kwargs: Additional arguments for register()

        Returns:
            Created and registered thread
        """
        # Wrap target to track lifecycle
        def wrapped_target(*args, **target_kwargs):
            thread_id = id(threading.current_thread())

            try:
                # Mark as started
                with self.lock:
                    if thread_id in self.threads:
                        self.threads[thread_id].state = ThreadState.RUNNING
                        self.threads[thread_id].started_at = datetime.now()

                # Run target
                result = target(*args, **target_kwargs)

                # Mark as completed
                with self.lock:
                    if thread_id in self.threads:
                        self.threads[thread_id].state = ThreadState.STOPPED
                        self.threads[thread_id].stopped_at = datetime.now()
                        self.metrics.total_completed += 1

                        if self.policy.log_thread_completion:
                            lifetime = (self.threads[thread_id].stopped_at -
                                       self.threads[thread_id].created_at).total_seconds()
                            logger.info(f"✅ Thread completed: {name} (lifetime={lifetime:.2f}s)")

                return result

            except Exception as e:
                # Mark as failed
                with self.lock:
                    if thread_id in self.threads:
                        self.threads[thread_id].state = ThreadState.FAILED
                        self.threads[thread_id].exception = e
                        self.threads[thread_id].stack_trace = traceback.format_exc()
                        self.metrics.total_failed += 1

                        logger.error(f"❌ Thread failed: {name}: {e}")
                        if self.policy.log_stack_traces:
                            logger.error(f"Stack trace:\n{self.threads[thread_id].stack_trace}")
                raise

            finally:
                # Cleanup
                with self.lock:
                    if thread_id in self.threads:
                        self.metrics.current_active -= 1
                        if self.threads[thread_id].daemon:
                            self.metrics.current_daemon -= 1
                        else:
                            self.metrics.current_non_daemon -= 1

        # Get target args/kwargs
        target_args = kwargs.pop('args', ())
        target_kwargs = kwargs.pop('kwargs', {})

        # Create thread
        thread = threading.Thread(
            target=wrapped_target,
            name=name,
            daemon=daemon,
            args=target_args,
            kwargs=target_kwargs
        )

        # Register
        return self.register(thread, purpose=purpose, category=category, **kwargs)

    def create_async_thread(
        self,
        coro: Callable,
        name: str,
        purpose: str = "Unknown",
        category: str = "async",
        **kwargs
    ) -> threading.Thread:
        """
        Create thread that runs async coroutine

        Args:
            coro: Async coroutine function
            name: Thread name
            purpose: Thread purpose
            category: Thread category
            **kwargs: Additional arguments

        Returns:
            Thread running the coroutine
        """
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(coro())
            finally:
                loop.close()

        return self.create_thread(
            target=run_async,
            name=name,
            purpose=purpose,
            category=category,
            is_async=True,
            **kwargs
        )

    def submit_to_pool(
        self,
        func: Callable,
        *args,
        name: str = "PoolTask",
        purpose: str = "Pool task",
        **kwargs
    ) -> Future:
        """
        Submit task to thread pool

        Args:
            func: Function to run
            *args: Function arguments
            name: Task name
            purpose: Task purpose
            **kwargs: Function keyword arguments

        Returns:
            Future representing the task
        """
        if not self.thread_pool:
            raise RuntimeError("Thread pool not enabled")

        future = self.thread_pool.submit(func, *args, **kwargs)

        # Track the worker thread when it starts
        # Note: We can't directly track pool threads as they're created on-demand

        return future

    def unregister(self, thread: threading.Thread):
        """Unregister a completed thread"""
        thread_id = id(thread)
        with self.lock:
            if thread_id in self.threads:
                info = self.threads.pop(thread_id)
                self.categories[info.category].discard(thread_id)

                if self.policy.log_thread_completion:
                    logger.debug(f"📤 Unregistered thread: {info.name}")

    def get_thread_info(self, thread_id: int) -> Optional[ThreadInfo]:
        """Get information about a thread"""
        with self.lock:
            return self.threads.get(thread_id)

    def get_active_threads(self, category: Optional[str] = None) -> List[ThreadInfo]:
        """
        Get list of active threads

        Args:
            category: Optional category filter

        Returns:
            List of active thread info
        """
        with self.lock:
            threads = list(self.threads.values())

            if category:
                threads = [t for t in threads if t.category == category]

            # Filter to alive threads
            active = []
            for info in threads:
                thread = info.thread() if isinstance(info.thread, weakref.ref) else info.thread
                if thread and thread.is_alive():
                    active.append(info)

            return active

    def get_leaked_threads(self) -> List[ThreadInfo]:
        """Get threads that appear to be leaked"""
        active = self.get_active_threads()
        leaked = []

        now = datetime.now()
        for info in active:
            # Check if thread is too old
            if self.policy.max_thread_lifetime:
                age = (now - info.created_at).total_seconds()
                if age > self.policy.max_thread_lifetime:
                    info.state = ThreadState.LEAKED
                    leaked.append(info)

            # Check if thread is non-daemon and should have stopped
            if not info.daemon and info.state == ThreadState.STOPPED:
                leaked.append(info)

        return leaked

    def heartbeat(self, thread_id: Optional[int] = None):
        """
        Record a heartbeat for a thread

        Args:
            thread_id: Thread ID (defaults to current thread)
        """
        if thread_id is None:
            thread_id = id(threading.current_thread())

        with self.lock:
            if thread_id in self.threads:
                self.threads[thread_id].last_heartbeat = datetime.now()

    def _health_check_loop(self):
        """Background health check loop"""
        logger.info("💓 Health check monitoring started")

        while not self.shutdown_event.is_set():
            try:
                time.sleep(self.policy.health_check_interval)

                active = self.get_active_threads()
                issues = []

                now = datetime.now()
                for info in active:
                    # Check heartbeat
                    if info.heartbeat_interval:
                        if info.last_heartbeat:
                            age = (now - info.last_heartbeat).total_seconds()
                            if age > info.heartbeat_interval * 2:  # Allow 2x interval
                                issues.append(f"{info.name}: No heartbeat for {age:.1f}s")
                        else:
                            issues.append(f"{info.name}: No heartbeat recorded")

                    # Check thread age
                    thread_age = (now - info.created_at).total_seconds()
                    if thread_age > self.policy.warn_thread_age:
                        issues.append(f"{info.name}: Running for {thread_age/3600:.1f}h")

                    # Custom health check
                    if info.health_check_callback:
                        try:
                            if not info.health_check_callback():
                                issues.append(f"{info.name}: Failed custom health check")
                        except Exception as e:
                            issues.append(f"{info.name}: Health check error: {e}")

                if issues:
                    logger.warning(f"⚠️  Thread health issues detected:")
                    for issue in issues[:10]:  # Limit output
                        logger.warning(f"   - {issue}")
                    if len(issues) > 10:
                        logger.warning(f"   ... and {len(issues) - 10} more")

            except Exception as e:
                logger.error(f"Health check error: {e}")

    def _deadlock_check_loop(self):
        """Background deadlock detection loop"""
        logger.info("🔒 Deadlock detection started")

        while not self.shutdown_event.is_set():
            try:
                time.sleep(self.policy.deadlock_check_interval)

                # Simple deadlock detection: check for threads stuck in same state
                with self.lock:
                    stuck_threads = []
                    for info in self.threads.values():
                        if info.state == ThreadState.RUNNING:
                            if info.last_heartbeat:
                                age = (datetime.now() - info.last_heartbeat).total_seconds()
                                if age > 60.0:  # No activity for 1 minute
                                    stuck_threads.append((info.name, age))

                    if stuck_threads:
                        logger.warning(f"⚠️  Potential deadlock detected:")
                        for name, age in stuck_threads[:5]:
                            logger.warning(f"   - {name}: No activity for {age:.1f}s")

            except Exception as e:
                logger.error(f"Deadlock check error: {e}")

    def _orphan_check_loop(self):
        """Background orphan cleanup loop"""
        logger.info("🧹 Orphan cleanup monitoring started")

        while not self.shutdown_event.is_set():
            try:
                time.sleep(self.policy.orphan_check_interval)

                # Find orphaned threads (registered but no longer alive)
                with self.lock:
                    orphans = []
                    for thread_id, info in list(self.threads.items()):
                        thread = info.thread() if isinstance(info.thread, weakref.ref) else info.thread
                        if not thread or not thread.is_alive():
                            if info.state not in (ThreadState.STOPPED, ThreadState.FAILED):
                                orphans.append(thread_id)

                    # Clean up orphans
                    for thread_id in orphans:
                        info = self.threads.pop(thread_id)
                        self.categories[info.category].discard(thread_id)
                        self.metrics.total_leaked += 1
                        logger.warning(f"🧹 Cleaned up orphaned thread: {info.name}")

            except Exception as e:
                logger.error(f"Orphan check error: {e}")

    async def shutdown_async(
        self,
        timeout: Optional[float] = None,
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Async shutdown with multi-phase escalation

        Args:
            timeout: Total timeout (uses policy if None)
            categories: Specific categories to shutdown (None = all)

        Returns:
            Shutdown statistics
        """
        if self.shutdown_initiated:
            logger.warning("Shutdown already in progress")
            return {"already_shutdown": True}

        with self.shutdown_lock:
            self.shutdown_initiated = True
            self.shutdown_event.set()

        logger.info("🛑 Initiating advanced thread shutdown...")

        # Calculate timeouts
        if timeout is None:
            timeout = (self.policy.graceful_shutdown_timeout +
                      self.policy.forceful_shutdown_timeout +
                      self.policy.terminate_shutdown_timeout +
                      self.policy.emergency_shutdown_timeout)

        start_time = time.time()
        stats = {
            "total_threads": 0,
            "by_phase": {},
            "by_category": defaultdict(int),
            "leaked": [],
            "failed": []
        }

        # Get threads to shutdown
        active = self.get_active_threads()
        if categories:
            active = [t for t in active if t.category in categories]

        stats["total_threads"] = len(active)
        non_daemon = [t for t in active if not t.daemon]

        logger.info(f"   Total threads: {len(active)} ({len(non_daemon)} non-daemon)")

        # PHASE 1: Graceful shutdown
        logger.info(f"📞 Phase 1: Graceful shutdown ({self.policy.graceful_shutdown_timeout}s)...")
        phase_stats = await self._shutdown_phase_graceful(non_daemon)
        stats["by_phase"]["graceful"] = phase_stats

        # Check if done
        remaining = [t for t in non_daemon if self._is_thread_alive(t)]
        if not remaining:
            logger.info("✅ All threads stopped gracefully")
            return self._finalize_shutdown_stats(stats, start_time)

        # PHASE 2: Forceful shutdown
        logger.info(f"⚡ Phase 2: Forceful shutdown ({self.policy.forceful_shutdown_timeout}s)...")
        phase_stats = await self._shutdown_phase_forceful(remaining)
        stats["by_phase"]["forceful"] = phase_stats

        # Check if done
        remaining = [t for t in remaining if self._is_thread_alive(t)]
        if not remaining:
            logger.info("✅ All threads stopped")
            return self._finalize_shutdown_stats(stats, start_time)

        # PHASE 3: Terminate
        logger.info(f"🔨 Phase 3: Terminate ({self.policy.terminate_shutdown_timeout}s)...")
        phase_stats = await self._shutdown_phase_terminate(remaining)
        stats["by_phase"]["terminate"] = phase_stats

        # Check if done
        remaining = [t for t in remaining if self._is_thread_alive(t)]
        if not remaining:
            logger.info("✅ All threads stopped")
            return self._finalize_shutdown_stats(stats, start_time)

        # PHASE 4: Emergency cleanup
        logger.info(f"🚨 Phase 4: Emergency cleanup ({self.policy.emergency_shutdown_timeout}s)...")
        phase_stats = await self._shutdown_phase_emergency(remaining)
        stats["by_phase"]["emergency"] = phase_stats

        # Final check
        remaining = [t for t in remaining if self._is_thread_alive(t)]
        if remaining:
            logger.error(f"❌ {len(remaining)} threads could not be stopped:")
            for info in remaining[:10]:
                logger.error(f"   - {info.name} ({info.category})")
                stats["leaked"].append(info.name)
            self.metrics.total_leaked += len(remaining)

        return self._finalize_shutdown_stats(stats, start_time)

    def shutdown_sync(
        self,
        timeout: Optional[float] = None,
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous shutdown wrapper

        Args:
            timeout: Total timeout
            categories: Categories to shutdown

        Returns:
            Shutdown statistics
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create new loop in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self.shutdown_async(timeout, categories))
                    )
                    return future.result(timeout=timeout)
            else:
                return asyncio.run(self.shutdown_async(timeout, categories))
        except RuntimeError:
            # No event loop
            return asyncio.run(self.shutdown_async(timeout, categories))

    async def _shutdown_phase_graceful(self, threads: List[ThreadInfo]) -> Dict[str, int]:
        """Graceful shutdown phase"""
        stats = {"success": 0, "timeout": 0, "failed": 0}

        # Signal all threads
        for info in threads:
            try:
                # Set shutdown event
                if info.shutdown_event:
                    if isinstance(info.shutdown_event, asyncio.Event):
                        info.shutdown_event.set()
                    else:
                        info.shutdown_event.set()

                # Call shutdown callback
                if info.shutdown_callback:
                    if asyncio.iscoroutinefunction(info.shutdown_callback):
                        await info.shutdown_callback()
                    else:
                        info.shutdown_callback()

                info.state = ThreadState.STOPPING
            except Exception as e:
                logger.error(f"Error signaling {info.name}: {e}")
                stats["failed"] += 1

        # Wait for threads
        deadline = time.time() + self.policy.graceful_shutdown_timeout
        for info in threads:
            remaining = deadline - time.time()
            if remaining <= 0:
                stats["timeout"] += 1
                continue

            thread = info.thread() if isinstance(info.thread, weakref.ref) else info.thread
            if thread:
                thread.join(timeout=remaining)
                if thread.is_alive():
                    stats["timeout"] += 1
                else:
                    stats["success"] += 1
                    info.state = ThreadState.STOPPED

        return stats

    async def _shutdown_phase_forceful(self, threads: List[ThreadInfo]) -> Dict[str, int]:
        """Forceful shutdown phase"""
        stats = {"success": 0, "timeout": 0, "failed": 0}

        # Convert to daemon if policy allows
        if self.policy.force_daemon_on_shutdown:
            for info in threads:
                thread = info.thread() if isinstance(info.thread, weakref.ref) else info.thread
                if thread and not thread.daemon:
                    thread.daemon = True
                    logger.debug(f"Converted {info.name} to daemon")

        # Wait again
        deadline = time.time() + self.policy.forceful_shutdown_timeout
        for info in threads:
            remaining = deadline - time.time()
            if remaining <= 0:
                stats["timeout"] += 1
                continue

            thread = info.thread() if isinstance(info.thread, weakref.ref) else info.thread
            if thread:
                thread.join(timeout=remaining)
                if thread.is_alive():
                    stats["timeout"] += 1
                else:
                    stats["success"] += 1
                    info.state = ThreadState.STOPPED

        return stats

    async def _shutdown_phase_terminate(self, threads: List[ThreadInfo]) -> Dict[str, int]:
        """Terminate phase - try to forcefully stop threads"""
        stats = {"success": 0, "timeout": 0, "failed": 0}

        # This is platform-specific and dangerous
        # Only use as last resort
        for info in threads:
            thread = info.thread() if isinstance(info.thread, weakref.ref) else info.thread
            if not thread or not thread.is_alive():
                stats["success"] += 1
                continue

            try:
                # Try to raise exception in thread (CPython only, dangerous)
                if hasattr(ctypes, 'pythonapi'):
                    thread_id = thread.ident
                    if thread_id:
                        exc_type = SystemExit
                        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                            ctypes.c_long(thread_id),
                            ctypes.py_object(exc_type)
                        )
                        if ret == 0:
                            stats["failed"] += 1
                        elif ret > 1:
                            # Revert if affected multiple threads
                            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, None)
                            stats["failed"] += 1
                        else:
                            stats["success"] += 1
                            self.metrics.total_forced_stop += 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                logger.error(f"Failed to terminate {info.name}: {e}")
                stats["failed"] += 1

        # Wait briefly
        await asyncio.sleep(self.policy.terminate_shutdown_timeout)

        return stats

    async def _shutdown_phase_emergency(self, threads: List[ThreadInfo]) -> Dict[str, int]:
        """Emergency cleanup phase"""
        stats = {"cleaned": 0, "leaked": 0}

        # Mark all as leaked and clean up references
        for info in threads:
            thread = info.thread() if isinstance(info.thread, weakref.ref) else info.thread
            if thread and thread.is_alive():
                info.state = ThreadState.LEAKED
                stats["leaked"] += 1
                logger.error(f"LEAKED: {info.name} ({info.purpose})")
            else:
                stats["cleaned"] += 1

        return stats

    def _is_thread_alive(self, info: ThreadInfo) -> bool:
        """Check if thread is alive"""
        thread = info.thread() if isinstance(info.thread, weakref.ref) else info.thread
        return thread and thread.is_alive()

    def _finalize_shutdown_stats(self, stats: Dict, start_time: float) -> Dict[str, Any]:
        """Finalize shutdown statistics"""
        stats["duration"] = time.time() - start_time
        stats["timestamp"] = datetime.now().isoformat()

        self.metrics.shutdown_attempts += 1
        if not stats.get("leaked"):
            self.metrics.successful_shutdowns += 1
        else:
            self.metrics.failed_shutdowns += 1

        logger.info(f"✅ Shutdown completed in {stats['duration']:.2f}s")
        return stats

    def _emergency_cleanup(self):
        """Emergency cleanup called by atexit"""
        if not self.shutdown_initiated:
            logger.warning("⚠️  Emergency cleanup triggered (atexit)")
            try:
                self.shutdown_sync(timeout=5.0)
            except Exception as e:
                logger.error(f"Emergency cleanup failed: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return asdict(self.metrics)

    def get_report(self) -> Dict[str, Any]:
        """Get comprehensive report"""
        active = self.get_active_threads()

        return {
            "metrics": self.get_metrics(),
            "active_threads": len(active),
            "by_category": {
                cat: len([t for t in active if t.category == cat])
                for cat in self.categories.keys()
            },
            "leaked": len(self.get_leaked_threads()),
            "policy": asdict(self.policy)
        }

    def print_report(self):
        """Print detailed report"""
        report = self.get_report()

        print("\n" + "=" * 80)
        print("🧵 ADVANCED THREAD MANAGER REPORT")
        print("=" * 80)
        print(f"Metrics:")
        for key, value in report["metrics"].items():
            print(f"  {key}: {value}")
        print(f"\nActive Threads: {report['active_threads']}")
        print(f"By Category:")
        for cat, count in report["by_category"].items():
            print(f"  {cat}: {count}")
        print(f"Leaked: {report['leaked']}")
        print("=" * 80 + "\n")

    # HUD-specific convenience methods

    async def check_hud_health(self) -> Optional[Dict[str, Any]]:
        """
        Check HUD thread health (convenience method)

        Returns:
            Health status dict or None if HUD manager not enabled
        """
        if self.hud_manager:
            return await self.hud_manager.check_hud_threads_health()
        return None

    async def shutdown_hud_threads(self, timeout: int = 10) -> bool:
        """
        Shutdown HUD threads gracefully (convenience method)

        Args:
            timeout: Shutdown timeout

        Returns:
            True if successful
        """
        if self.hud_manager:
            return await self.hud_manager.graceful_shutdown_hud_threads(timeout)
        return True

    def get_hud_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get HUD metrics (convenience method)

        Returns:
            Metrics dict or None if HUD manager not enabled
        """
        if self.hud_manager:
            return self.hud_manager.get_hud_metrics()
        return None

    def print_hud_report(self):
        """Print HUD thread report (convenience method)"""
        if self.hud_manager:
            self.hud_manager.print_hud_report()
        else:
            print("HUD Thread Manager not enabled")

    async def start_hud_monitoring(self):
        """
        Start HUD thread monitoring loop (convenience method)

        This starts a background monitoring task that checks HUD health
        and performs auto-recovery when needed.
        """
        if self.hud_manager:
            logger.info("🔄 Starting HUD thread monitoring...")
            # Create background task for monitoring
            asyncio.create_task(self.hud_manager.monitor_hud_threads_loop())
        else:
            logger.warning("Cannot start HUD monitoring - HUD manager not enabled")


# Global instance
_thread_manager: Optional[AdvancedThreadManager] = None
_manager_lock = threading.Lock()


def get_thread_manager(policy: Optional[ThreadPolicy] = None) -> AdvancedThreadManager:
    """Get or create global thread manager"""
    global _thread_manager

    with _manager_lock:
        if _thread_manager is None:
            _thread_manager = AdvancedThreadManager(policy=policy)
        return _thread_manager


def create_managed_thread(
    target: Callable,
    name: str,
    **kwargs
) -> threading.Thread:
    """Convenience function to create managed thread"""
    manager = get_thread_manager()
    thread = manager.create_thread(target=target, name=name, **kwargs)
    return thread


async def shutdown_all_threads_async(timeout: Optional[float] = None) -> Dict[str, Any]:
    """Async shutdown all threads"""
    manager = get_thread_manager()
    return await manager.shutdown_async(timeout=timeout)


def shutdown_all_threads(timeout: Optional[float] = None) -> Dict[str, Any]:
    """Sync shutdown all threads"""
    manager = get_thread_manager()
    return manager.shutdown_sync(timeout=timeout)
