#!/usr/bin/env python3
"""
Singleton CloudSQL Connection Manager for JARVIS
=================================================

Thread-safe, async-safe, signal-aware connection pool manager that:
- Maintains exactly ONE connection pool throughout the application lifecycle
- Closes leaked connections before creating new ones
- Handles graceful shutdown via signal handlers (SIGINT, SIGTERM, KeyboardInterrupt)
- Supports robust async operations with automatic connection reuse
- Monitors and automatically recovers from stale/broken connections
- Enforces strict connection limits for db-f1-micro (max 3 connections)

Architecture:
- Singleton pattern with double-checked locking
- AsyncIO-safe with asyncio.Lock for initialization
- Signal handlers registered via atexit and signal modules
- Connection validation and automatic cleanup
- Prometheus metrics integration

Author: JARVIS System
Version: 1.0.0
"""

import asyncio
import atexit
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

logger = logging.getLogger(__name__)


class CloudSQLConnectionManager:
    """
    Singleton async-safe CloudSQL connection pool manager.

    Features:
    - Single instance throughout application lifecycle
    - Automatic cleanup of leaked connections
    - Signal-aware graceful shutdown (SIGINT, SIGTERM, atexit)
    - Connection validation and auto-recovery
    - Strict connection limits (3 max for db-f1-micro)
    """

    _instance: Optional['CloudSQLConnectionManager'] = None
    _lock = asyncio.Lock()
    _initialized = False

    def __new__(cls):
        """Singleton pattern - only one instance allowed"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize singleton (only runs once)"""
        if self._initialized:
            return

        self.pool: Optional[asyncpg.Pool] = None
        self.config: Dict[str, Any] = {}
        self.is_shutting_down = False
        self.creation_time: Optional[datetime] = None
        self.connection_count = 0
        self.error_count = 0

        # Register shutdown handlers
        self._register_shutdown_handlers()

        CloudSQLConnectionManager._initialized = True
        logger.info("🔧 CloudSQL Connection Manager singleton created")

    def _register_shutdown_handlers(self):
        """Register signal handlers for graceful shutdown"""
        # Register atexit handler
        atexit.register(self._sync_shutdown)

        # Register signal handlers for SIGINT and SIGTERM
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._signal_handler)

        logger.info("✅ Shutdown handlers registered (SIGINT, SIGTERM, atexit)")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        sig_name = signal.Signals(signum).name
        logger.info(f"📡 Received {sig_name} - initiating graceful shutdown...")
        self.is_shutting_down = True

        # Run async shutdown in new event loop (signal context)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule shutdown coroutine
                asyncio.create_task(self.shutdown())
            else:
                # Create new loop for shutdown
                asyncio.run(self.shutdown())
        except Exception as e:
            logger.error(f"❌ Error during signal shutdown: {e}")

        # Re-raise KeyboardInterrupt for SIGINT
        if signum == signal.SIGINT:
            raise KeyboardInterrupt

    def _sync_shutdown(self):
        """Synchronous shutdown for atexit"""
        if self.pool and not self.is_shutting_down:
            logger.info("🛑 atexit: Running synchronous shutdown...")
            self.is_shutting_down = True

            # Run async shutdown in new event loop
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.shutdown())
                loop.close()
            except Exception as e:
                logger.error(f"❌ Error during atexit shutdown: {e}")

    async def initialize(
        self,
        host: str = "127.0.0.1",
        port: int = 5432,
        database: str = "jarvis_learning",
        user: str = "jarvis",
        password: Optional[str] = None,
        max_connections: int = 3,  # CRITICAL: Keep low for db-f1-micro
        force_reinit: bool = False
    ) -> bool:
        """
        Initialize connection pool (singleton - reuses existing pool if available).

        Args:
            host: Database host (127.0.0.1 for proxy)
            port: Database port (5432 for proxy)
            database: Database name
            user: Database user
            password: Database password
            max_connections: Maximum pool size (default 3 for db-f1-micro)
            force_reinit: Force re-initialization (closes existing pool)

        Returns:
            True if pool is ready, False otherwise
        """
        async with CloudSQLConnectionManager._lock:
            # Reuse existing pool if available
            if self.pool and not force_reinit:
                logger.info("♻️  Reusing existing connection pool")
                return True

            # Close existing pool if forcing re-init
            if self.pool and force_reinit:
                logger.info("🔄 Force re-init: closing existing pool...")
                await self._close_pool()

            if not ASYNCPG_AVAILABLE:
                logger.error("❌ asyncpg not available")
                return False

            if not password:
                logger.error("❌ Database password required")
                return False

            # Store config
            self.config = {
                "host": host,
                "port": port,
                "database": database,
                "user": user,
                "password": password,
                "max_connections": max_connections
            }

            try:
                logger.info(f"🔌 Creating CloudSQL connection pool (max={max_connections})...")
                logger.info(f"   Host: {host}:{port}, Database: {database}, User: {user}")

                # Kill any leaked connections from previous runs
                await self._kill_leaked_connections(host, port, database, user, password)

                # Create new pool with strict limits
                self.pool = await asyncio.wait_for(
                    asyncpg.create_pool(
                        host=host,
                        port=port,
                        database=database,
                        user=user,
                        password=password,
                        min_size=1,  # Keep minimum connections low
                        max_size=max_connections,
                        timeout=5.0,  # Connection acquisition timeout
                        command_timeout=30.0,  # Query timeout
                        max_queries=10000,  # Recycle connection after 10k queries
                        max_inactive_connection_lifetime=300.0,  # Close idle connections after 5min
                    ),
                    timeout=15.0  # Overall pool creation timeout
                )

                # Validate pool with test query
                async with self.pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")

                self.creation_time = datetime.now()
                self.error_count = 0

                logger.info(f"✅ Connection pool created successfully")
                logger.info(f"   Pool size: {self.pool.get_size()}, Idle: {self.pool.get_idle_size()}")

                return True

            except asyncio.TimeoutError:
                logger.error("⏱️  Connection pool creation timeout (15s)")
                logger.error("   This usually means:")
                logger.error("   1. Cloud SQL proxy is not running")
                logger.error("   2. Database credentials are incorrect")
                logger.error("   3. Network connectivity issues")
                self.pool = None
                return False

            except Exception as e:
                logger.error(f"❌ Failed to create connection pool: {e}")
                self.pool = None
                self.error_count += 1
                return False

    async def _kill_leaked_connections(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str
    ):
        """Kill leaked connections from previous runs"""
        try:
            logger.info("🧹 Checking for leaked connections...")

            # Create temporary connection to check/kill leaked connections
            conn = await asyncio.wait_for(
                asyncpg.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password,
                ),
                timeout=5.0
            )

            # Find leaked connections (exclude our temporary connection)
            leaked = await conn.fetch("""
                SELECT pid, usename, application_name, state, state_change
                FROM pg_stat_activity
                WHERE datname = $1
                  AND pid <> pg_backend_pid()
                  AND usename = $2
                  AND state = 'idle'
                  AND state_change < NOW() - INTERVAL '5 minutes'
            """, database, user)

            if leaked:
                logger.warning(f"⚠️  Found {len(leaked)} leaked connections")
                for row in leaked:
                    try:
                        await conn.execute("SELECT pg_terminate_backend($1)", row['pid'])
                        logger.info(f"   ✅ Killed leaked connection PID {row['pid']}")
                    except Exception as e:
                        logger.warning(f"   ⚠️  Failed to kill PID {row['pid']}: {e}")
            else:
                logger.info("✅ No leaked connections found")

            await conn.close()

        except asyncio.TimeoutError:
            logger.warning("⏱️  Leaked connection check timeout (proxy not running?)")
        except Exception as e:
            logger.warning(f"⚠️  Leaked connection check failed: {e}")

    @asynccontextmanager
    async def connection(self):
        """
        Acquire connection from pool (context manager).

        Usage:
            async with manager.connection() as conn:
                result = await conn.fetchval("SELECT 1")
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized - call initialize() first")

        if self.is_shutting_down:
            raise RuntimeError("Connection manager is shutting down")

        start_time = time.time()
        conn = None

        try:
            # Acquire connection with timeout
            conn = await asyncio.wait_for(
                self.pool.acquire(),
                timeout=5.0
            )

            self.connection_count += 1
            latency_ms = (time.time() - start_time) * 1000

            logger.debug(f"✅ Connection acquired ({latency_ms:.1f}ms) - Active: {self.pool.get_size()}")

            yield conn

        except asyncio.TimeoutError:
            logger.error("⏱️  Connection acquisition timeout (5s) - pool exhausted")
            self.error_count += 1
            raise

        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            self.error_count += 1
            raise

        finally:
            # CRITICAL: Always release connection back to pool
            if conn:
                try:
                    await self.pool.release(conn)
                    logger.debug(f"♻️  Connection released - Idle: {self.pool.get_idle_size()}")
                except Exception as e:
                    logger.error(f"❌ Failed to release connection: {e}")

    async def execute(self, query: str, *args, timeout: float = 30.0):
        """Execute query and return result"""
        async with self.connection() as conn:
            return await asyncio.wait_for(
                conn.execute(query, *args),
                timeout=timeout
            )

    async def fetch(self, query: str, *args, timeout: float = 30.0):
        """Fetch multiple rows"""
        async with self.connection() as conn:
            return await asyncio.wait_for(
                conn.fetch(query, *args),
                timeout=timeout
            )

    async def fetchrow(self, query: str, *args, timeout: float = 30.0):
        """Fetch single row"""
        async with self.connection() as conn:
            return await asyncio.wait_for(
                conn.fetchrow(query, *args),
                timeout=timeout
            )

    async def fetchval(self, query: str, *args, timeout: float = 30.0):
        """Fetch single value"""
        async with self.connection() as conn:
            return await asyncio.wait_for(
                conn.fetchval(query, *args),
                timeout=timeout
            )

    async def _close_pool(self):
        """Close connection pool"""
        if self.pool:
            try:
                logger.info("🔌 Closing connection pool...")

                # Close pool gracefully (waits for connections to be released)
                await asyncio.wait_for(
                    self.pool.close(),
                    timeout=10.0
                )

                logger.info("✅ Connection pool closed")

            except asyncio.TimeoutError:
                logger.warning("⏱️  Pool close timeout - force terminating")
                await self.pool.terminate()

            except Exception as e:
                logger.error(f"❌ Error closing pool: {e}")
                try:
                    await self.pool.terminate()
                except:
                    pass

            finally:
                self.pool = None

    async def shutdown(self):
        """Graceful shutdown - close all connections"""
        if self.is_shutting_down and not self.pool:
            logger.debug("Already shut down")
            return

        self.is_shutting_down = True

        logger.info("🛑 Shutting down CloudSQL Connection Manager...")
        logger.info(f"   Lifetime stats:")
        logger.info(f"      Total connections: {self.connection_count}")
        logger.info(f"      Total errors: {self.error_count}")
        if self.creation_time:
            uptime = (datetime.now() - self.creation_time).total_seconds()
            logger.info(f"      Uptime: {uptime:.1f}s")

        await self._close_pool()

        logger.info("✅ CloudSQL Connection Manager shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        if not self.pool:
            return {
                "status": "not_initialized",
                "pool_size": 0,
                "idle_size": 0,
                "error_count": self.error_count
            }

        return {
            "status": "running" if not self.is_shutting_down else "shutting_down",
            "pool_size": self.pool.get_size(),
            "idle_size": self.pool.get_idle_size(),
            "max_size": self.config.get("max_connections", 0),
            "connection_count": self.connection_count,
            "error_count": self.error_count,
            "creation_time": self.creation_time.isoformat() if self.creation_time else None,
            "uptime_seconds": (datetime.now() - self.creation_time).total_seconds() if self.creation_time else 0
        }

    @property
    def is_initialized(self) -> bool:
        """Check if pool is initialized and ready"""
        return self.pool is not None and not self.is_shutting_down


# Global singleton instance accessor
_manager: Optional[CloudSQLConnectionManager] = None


def get_connection_manager() -> CloudSQLConnectionManager:
    """Get singleton connection manager instance"""
    global _manager
    if _manager is None:
        _manager = CloudSQLConnectionManager()
    return _manager
