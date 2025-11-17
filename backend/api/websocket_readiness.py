#!/usr/bin/env python3
"""
WebSocket Readiness Signaling System
Advanced coordination between backend WebSocket server and HUD client

This module provides:
- Readiness signals (backend → HUD launcher)
- Health check endpoint for HUD to verify connectivity
- Auto-cleanup of stale signal files
- Multi-process safety with file locking
"""

import os
import json
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import fcntl  # For file locking on Unix systems

logger = logging.getLogger(__name__)


class WebSocketReadinessSignal:
    """
    Manages WebSocket readiness signals for HUD coordination

    Uses a file-based signaling system with atomic writes and locks
    to ensure multi-process safety.
    """

    def __init__(self, signal_dir: Optional[Path] = None):
        """
        Initialize readiness signal manager

        Args:
            signal_dir: Directory for signal files (default: /tmp/jarvis)
        """
        self.signal_dir = signal_dir or Path("/tmp/jarvis")  # nosec B108
        self.signal_dir.mkdir(parents=True, exist_ok=True)

        self.signal_file = self.signal_dir / "websocket_ready.json"
        self.lock_file = self.signal_dir / "websocket_ready.lock"

        # Server state
        self.server_info: Dict[str, Any] = {
            "ready": False,
            "host": "localhost",
            "port": 8010,
            "endpoint": "/ws",
            "pid": os.getpid(),
            "started_at": None,
            "last_heartbeat": None,
        }

        logger.info(f"📡 WebSocket Readiness Signal initialized")
        logger.info(f"   Signal file: {self.signal_file}")
        logger.info(f"   Lock file: {self.lock_file}")

    def _acquire_lock(self, timeout: float = 5.0) -> Optional[int]:
        """
        Acquire exclusive lock on signal file

        Args:
            timeout: Maximum time to wait for lock

        Returns:
            File descriptor if lock acquired, None otherwise
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Open lock file (create if doesn't exist)
                fd = os.open(str(self.lock_file), os.O_CREAT | os.O_WRONLY, 0o644)

                # Try to acquire exclusive lock (non-blocking)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

                return fd

            except (IOError, OSError):
                # Lock held by another process, wait a bit
                time.sleep(0.01)
                continue

        logger.warning(f"⚠️  Failed to acquire lock after {timeout}s")
        return None

    def _release_lock(self, fd: int):
        """Release file lock"""
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        except Exception as e:
            logger.warning(f"⚠️  Failed to release lock: {e}")

    def signal_ready(self, host: str = "localhost", port: int = 8010, endpoint: str = "/ws"):
        """
        Signal that WebSocket server is ready

        Args:
            host: WebSocket server host
            port: WebSocket server port
            endpoint: WebSocket endpoint path
        """
        # Acquire lock
        fd = self._acquire_lock()
        if fd is None:
            logger.error("❌ Could not acquire lock to signal readiness")
            return

        try:
            # Update server info
            now = datetime.now().isoformat()
            self.server_info.update({
                "ready": True,
                "host": host,
                "port": port,
                "endpoint": endpoint,
                "started_at": now,
                "last_heartbeat": now,
            })

            # Write signal file atomically
            temp_file = self.signal_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(self.server_info, f, indent=2)

            # Atomic rename
            temp_file.rename(self.signal_file)

            logger.info("=" * 80)
            logger.info("🟢 WebSocket Server READY Signal")
            logger.info("=" * 80)
            logger.info(f"   URL: ws://{host}:{port}{endpoint}")
            logger.info(f"   PID: {self.server_info['pid']}")
            logger.info(f"   Started: {now}")
            logger.info(f"   Signal file: {self.signal_file}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"❌ Failed to write readiness signal: {e}")

        finally:
            # Release lock
            self._release_lock(fd)

    async def start_heartbeat(self, interval: float = 1.0):
        """
        Start heartbeat task to update signal file

        Args:
            interval: Heartbeat interval in seconds
        """
        logger.info(f"💓 Starting WebSocket readiness heartbeat (interval={interval}s)")

        while True:
            try:
                await asyncio.sleep(interval)

                # Update heartbeat timestamp
                fd = self._acquire_lock(timeout=0.5)
                if fd is not None:
                    try:
                        self.server_info["last_heartbeat"] = datetime.now().isoformat()

                        with open(self.signal_file, "w") as f:
                            json.dump(self.server_info, f, indent=2)

                    finally:
                        self._release_lock(fd)

            except asyncio.CancelledError:
                logger.info("💓 Heartbeat task cancelled")
                break
            except Exception as e:
                logger.warning(f"⚠️  Heartbeat update failed: {e}")

    def cleanup(self):
        """Clean up signal files"""
        try:
            if self.signal_file.exists():
                self.signal_file.unlink()
                logger.info(f"🧹 Removed signal file: {self.signal_file}")

            if self.lock_file.exists():
                self.lock_file.unlink()
                logger.info(f"🧹 Removed lock file: {self.lock_file}")

        except Exception as e:
            logger.warning(f"⚠️  Cleanup failed: {e}")


# Global instance
_readiness_signal: Optional[WebSocketReadinessSignal] = None


def get_readiness_signal() -> WebSocketReadinessSignal:
    """Get or create global readiness signal instance"""
    global _readiness_signal

    if _readiness_signal is None:
        _readiness_signal = WebSocketReadinessSignal()

    return _readiness_signal


def signal_websocket_ready(host: str = "localhost", port: int = 8010, endpoint: str = "/ws"):
    """
    Signal that WebSocket server is ready (convenience function)

    Args:
        host: WebSocket server host
        port: WebSocket server port
        endpoint: WebSocket endpoint path
    """
    signal = get_readiness_signal()
    signal.signal_ready(host, port, endpoint)


async def start_readiness_heartbeat(interval: float = 1.0):
    """
    Start heartbeat task (convenience function)

    Args:
        interval: Heartbeat interval in seconds
    """
    signal = get_readiness_signal()
    await signal.start_heartbeat(interval)


def cleanup_readiness_signal():
    """Clean up readiness signal files (convenience function)"""
    signal = get_readiness_signal()
    signal.cleanup()


# Client-side functions for HUD launcher


def is_websocket_ready(max_age_seconds: float = 5.0) -> bool:
    """
    Check if WebSocket server is ready (client-side)

    Args:
        max_age_seconds: Maximum age of heartbeat before considering stale

    Returns:
        True if server is ready and heartbeat is recent
    """
    signal_file = Path("/tmp/jarvis/websocket_ready.json")  # nosec B108

    try:
        if not signal_file.exists():
            return False

        # Read signal file
        with open(signal_file) as f:
            info = json.load(f)

        if not info.get("ready"):
            return False

        # Check heartbeat freshness
        last_heartbeat = info.get("last_heartbeat")
        if last_heartbeat:
            from datetime import datetime
            heartbeat_time = datetime.fromisoformat(last_heartbeat)
            age = (datetime.now() - heartbeat_time).total_seconds()

            if age > max_age_seconds:
                logger.warning(f"⚠️  Stale heartbeat (age={age:.1f}s)")
                return False

        return True

    except Exception as e:
        logger.warning(f"⚠️  Failed to check readiness: {e}")
        return False


def get_websocket_url() -> Optional[str]:
    """
    Get WebSocket URL if server is ready (client-side)

    Returns:
        WebSocket URL or None if not ready
    """
    signal_file = Path("/tmp/jarvis/websocket_ready.json")  # nosec B108

    try:
        if not signal_file.exists():
            return None

        with open(signal_file) as f:
            info = json.load(f)

        if not info.get("ready"):
            return None

        host = info.get("host", "localhost")
        port = info.get("port", 8010)
        endpoint = info.get("endpoint", "/ws")

        return f"ws://{host}:{port}{endpoint}"

    except Exception as e:
        logger.warning(f"⚠️  Failed to get WebSocket URL: {e}")
        return None


async def wait_for_websocket_ready(timeout: float = 30.0, check_interval: float = 0.1) -> bool:
    """
    Wait for WebSocket server to become ready (client-side)

    Args:
        timeout: Maximum time to wait in seconds
        check_interval: How often to check in seconds

    Returns:
        True if server became ready, False if timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        if is_websocket_ready():
            return True

        await asyncio.sleep(check_interval)

    return False
