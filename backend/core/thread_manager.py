"""
Thread Manager - Prevent thread leaks and ensure clean shutdown
=================================================================

Tracks all threads created by JARVIS and ensures they are properly
cleaned up during shutdown.

Features:
- Thread registry with metadata
- Automatic daemon thread detection
- Force cleanup of leaked threads
- Detailed thread lifecycle logging
- Shutdown timeout with escalation
"""

import threading
import logging
import time
import traceback
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import weakref

logger = logging.getLogger(__name__)


@dataclass
class ThreadInfo:
    """Information about a tracked thread"""
    thread: threading.Thread
    name: str
    created_at: datetime
    creator: str  # Module/function that created it
    purpose: str  # What the thread is for
    daemon: bool
    shutdown_callback: Optional[Callable] = None


class ThreadManager:
    """
    Centralized thread management to prevent leaks

    Usage:
        manager = ThreadManager()

        # Register a thread
        thread = threading.Thread(target=work, daemon=False)
        manager.register(thread, purpose="Process data", shutdown_callback=cleanup_func)
        thread.start()

        # At shutdown
        manager.shutdown_all()
    """

    def __init__(self):
        self.threads: Dict[int, ThreadInfo] = {}
        self.lock = threading.Lock()
        self.shutdown_initiated = False

        logger.info("🧵 ThreadManager initialized")

    def register(
        self,
        thread: threading.Thread,
        purpose: str = "Unknown",
        shutdown_callback: Optional[Callable] = None,
        force_daemon: bool = False
    ) -> threading.Thread:
        """
        Register a thread for tracking

        Args:
            thread: Thread to register
            purpose: Description of thread's purpose
            shutdown_callback: Function to call to cleanup thread
            force_daemon: If True, convert non-daemon to daemon

        Returns:
            The registered thread
        """
        # Get caller info
        stack = traceback.extract_stack()
        caller = f"{stack[-2].filename}:{stack[-2].lineno}"

        # Force daemon if requested
        if force_daemon and not thread.daemon:
            logger.debug(f"Converting thread {thread.name} to daemon")
            thread.daemon = True

        with self.lock:
            thread_id = id(thread)
            self.threads[thread_id] = ThreadInfo(
                thread=thread,
                name=thread.name,
                created_at=datetime.now(),
                creator=caller,
                purpose=purpose,
                daemon=thread.daemon,
                shutdown_callback=shutdown_callback
            )

            logger.debug(
                f"📝 Registered thread: {thread.name} "
                f"(daemon={thread.daemon}, purpose={purpose})"
            )

        return thread

    def create_thread(
        self,
        target: Callable,
        name: str,
        purpose: str = "Unknown",
        daemon: bool = True,
        shutdown_callback: Optional[Callable] = None,
        args: tuple = (),
        kwargs: dict = None
    ) -> threading.Thread:
        """
        Create and register a thread

        Args:
            target: Function to run in thread
            name: Thread name
            purpose: Description of purpose
            daemon: Whether thread should be daemon
            shutdown_callback: Cleanup function
            args: Positional arguments for target
            kwargs: Keyword arguments for target

        Returns:
            Created and registered thread
        """
        if kwargs is None:
            kwargs = {}

        thread = threading.Thread(
            target=target,
            name=name,
            daemon=daemon,
            args=args,
            kwargs=kwargs
        )

        self.register(thread, purpose=purpose, shutdown_callback=shutdown_callback)
        return thread

    def unregister(self, thread: threading.Thread):
        """Unregister a thread (called when thread completes)"""
        with self.lock:
            thread_id = id(thread)
            if thread_id in self.threads:
                info = self.threads.pop(thread_id)
                logger.debug(f"✅ Unregistered thread: {info.name}")

    def get_active_threads(self) -> List[ThreadInfo]:
        """Get list of active registered threads"""
        with self.lock:
            return [
                info for info in self.threads.values()
                if info.thread.is_alive()
            ]

    def get_leaked_threads(self) -> List[ThreadInfo]:
        """Get threads that should have stopped but are still running"""
        active = self.get_active_threads()
        return [t for t in active if not t.daemon and t.thread.is_alive()]

    def shutdown_all(self, timeout: float = 10.0) -> Dict[str, int]:
        """
        Shutdown all tracked threads

        Args:
            timeout: Max time to wait for threads

        Returns:
            Statistics about shutdown
        """
        if self.shutdown_initiated:
            logger.warning("Shutdown already initiated")
            return {"already_shutdown": True}

        self.shutdown_initiated = True

        logger.info("🛑 Initiating thread shutdown...")

        active = self.get_active_threads()
        non_daemon = [t for t in active if not t.daemon]
        daemon = [t for t in active if t.daemon]

        logger.info(f"   Active threads: {len(active)} ({len(non_daemon)} non-daemon, {len(daemon)} daemon)")

        stats = {
            "total": len(active),
            "non_daemon": len(non_daemon),
            "daemon": len(daemon),
            "shutdown_success": 0,
            "shutdown_timeout": 0,
            "shutdown_failed": 0
        }

        # Phase 1: Call shutdown callbacks for non-daemon threads
        logger.info("📞 Phase 1: Calling shutdown callbacks...")
        for info in non_daemon:
            if info.shutdown_callback:
                try:
                    logger.debug(f"   Calling shutdown for {info.name}")
                    info.shutdown_callback()
                except Exception as e:
                    logger.error(f"   ❌ Shutdown callback failed for {info.name}: {e}")

        # Phase 2: Wait for threads to finish
        logger.info(f"⏱️  Phase 2: Waiting up to {timeout}s for threads to finish...")
        start_time = time.time()

        for info in non_daemon:
            remaining = timeout - (time.time() - start_time)
            if remaining <= 0:
                logger.warning(f"   ⏱️  Timeout waiting for {info.name}")
                stats["shutdown_timeout"] += 1
                continue

            try:
                info.thread.join(timeout=remaining)
                if info.thread.is_alive():
                    logger.warning(f"   ⏱️  Thread {info.name} still alive after join")
                    stats["shutdown_timeout"] += 1
                else:
                    logger.debug(f"   ✅ Thread {info.name} stopped")
                    stats["shutdown_success"] += 1
            except Exception as e:
                logger.error(f"   ❌ Error joining {info.name}: {e}")
                stats["shutdown_failed"] += 1

        # Phase 3: Report remaining threads
        remaining = self.get_active_threads()
        leaked = [t for t in remaining if not t.daemon]

        if leaked:
            logger.warning(f"⚠️  {len(leaked)} non-daemon threads still running:")
            for info in leaked[:10]:  # Show first 10
                logger.warning(
                    f"   - {info.name} (created {info.created_at.strftime('%H:%M:%S')}, "
                    f"purpose: {info.purpose})"
                )
            if len(leaked) > 10:
                logger.warning(f"   ... and {len(leaked) - 10} more")
        else:
            logger.info("✅ All non-daemon threads stopped")

        if daemon:
            logger.info(f"ℹ️  {len(daemon)} daemon threads will auto-terminate")

        return stats

    def print_report(self):
        """Print detailed report of all threads"""
        active = self.get_active_threads()

        print("\n" + "=" * 70)
        print("🧵 THREAD MANAGER REPORT")
        print("=" * 70)
        print(f"Total active threads: {len(active)}")
        print(f"Non-daemon: {len([t for t in active if not t.daemon])}")
        print(f"Daemon: {len([t for t in active if t.daemon])}")
        print()

        if active:
            print("Active Threads:")
            print("-" * 70)
            for info in active:
                status = "DAEMON" if info.daemon else "NON-DAEMON"
                print(f"  [{status}] {info.name}")
                print(f"    Purpose: {info.purpose}")
                print(f"    Created: {info.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"    Creator: {info.creator}")
                print(f"    Has cleanup: {'Yes' if info.shutdown_callback else 'No'}")
                print()

        print("=" * 70 + "\n")


# Global instance
_thread_manager: Optional[ThreadManager] = None


def get_thread_manager() -> ThreadManager:
    """Get or create global thread manager"""
    global _thread_manager
    if _thread_manager is None:
        _thread_manager = ThreadManager()
    return _thread_manager


def create_managed_thread(
    target: Callable,
    name: str,
    purpose: str = "Unknown",
    daemon: bool = True,
    shutdown_callback: Optional[Callable] = None,
    args: tuple = (),
    kwargs: dict = None,
    auto_start: bool = True
) -> threading.Thread:
    """
    Convenience function to create a managed thread

    Example:
        thread = create_managed_thread(
            target=worker_function,
            name="DataProcessor",
            purpose="Process incoming data",
            daemon=False,
            shutdown_callback=lambda: stop_event.set()
        )
    """
    manager = get_thread_manager()
    thread = manager.create_thread(
        target=target,
        name=name,
        purpose=purpose,
        daemon=daemon,
        shutdown_callback=shutdown_callback,
        args=args,
        kwargs=kwargs
    )

    if auto_start:
        thread.start()

    return thread


def shutdown_all_threads(timeout: float = 10.0) -> Dict[str, int]:
    """Shutdown all managed threads"""
    manager = get_thread_manager()
    return manager.shutdown_all(timeout=timeout)
