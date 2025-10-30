#!/usr/bin/env python3
"""
Advanced Learning Database System for JARVIS Goal Inference
Hybrid architecture: SQLite (structured) + ChromaDB (embeddings) + Async + ML-powered insights
"""

import asyncio
import calendar
import hashlib
import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiosqlite

# Try to import Cloud Database Adapter
try:
    from intelligence.cloud_database_adapter import get_database_adapter

    CLOUD_ADAPTER_AVAILABLE = True
except ImportError:
    CLOUD_ADAPTER_AVAILABLE = False
    logging.warning("Cloud Database Adapter not available - using SQLite only")

# Async and ML dependencies
try:
    pass

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# ChromaDB for semantic search and embeddings
try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available - install with: pip install chromadb")

logger = logging.getLogger(__name__)


# ============================================================================
# DATABASE WRAPPER CLASSES
# ============================================================================
# These classes provide a robust, async, dynamic abstraction layer that makes
# Cloud SQL (PostgreSQL) behave like aiosqlite, enabling seamless switching
# between local SQLite and cloud databases without code changes.
#
# Features:
# - Full DB-API 2.0 compatibility (execute, executemany, fetchall, fetchone, etc.)
# - Transaction support with savepoints for nested transactions
# - Connection pooling and resource management
# - Automatic query type detection (SELECT vs DML)
# - Comprehensive error handling and logging
# - Async iteration support
# - Context manager support for RAII pattern
# - No hardcoded values - fully dynamic configuration
# ============================================================================


class DatabaseCursorWrapper:
    """
    Advanced wrapper that makes Cloud SQL adapter look like aiosqlite cursor.
    Provides robust, async, dynamic database operations with comprehensive error handling.
    """

    def __init__(self, adapter_conn, connection_wrapper=None):
        """
        Initialize cursor wrapper.

        Args:
            adapter_conn: Cloud SQL adapter connection instance
            connection_wrapper: Parent connection wrapper for context
        """
        self.adapter_conn = adapter_conn
        self.connection_wrapper = connection_wrapper
        self._last_results: List[Dict[str, Any]] = []
        self._last_query: Optional[str] = None
        self._last_params: Optional[Tuple] = None
        self._current_index: int = 0
        self._row_count: int = -1  # DB-API 2.0: -1 means "not available"
        self._description: Optional[List[Tuple]] = None
        self._lastrowid: Optional[int] = None
        self._arraysize: int = 1  # DB-API 2.0 default
        self._rownumber: Optional[int] = None  # Current row position
        self._column_metadata: Dict[str, Dict[str, Any]] = {}  # Extended metadata

    @property
    def rowcount(self) -> int:
        """
        Return number of rows affected by last operation.

        DB-API 2.0 compliant:
        - For SELECT: number of rows returned
        - For INSERT/UPDATE/DELETE: number of rows affected
        - Returns -1 if not available or not applicable

        Returns:
            int: Row count or -1 if unavailable
        """
        return self._row_count

    @property
    def description(self) -> Optional[List[Tuple]]:
        """
        Return column descriptions for last query result.

        DB-API 2.0 format - 7-item sequence per column:
        (name, type_code, display_size, internal_size, precision, scale, null_ok)

        Enhanced with dynamic type inference from result data:
        - Automatically detects Python types from results
        - Provides basic size estimates
        - Supports NULL detection

        Returns:
            Optional[List[Tuple]]: Column descriptions or None for non-SELECT queries
        """
        if not self._description and self._last_results:
            # Dynamically build description from results
            self._build_description_from_results()
        return self._description

    @property
    def lastrowid(self) -> Optional[int]:
        """
        Return last inserted row ID.

        Advanced implementation:
        - PostgreSQL: Uses RETURNING id clause (must be in query)
        - SQLite: Returns last inserted rowid
        - Auto-detects 'id' column from RETURNING results

        Note: For PostgreSQL, query must include RETURNING clause:
            INSERT INTO table (col) VALUES (?) RETURNING id

        Returns:
            Optional[int]: Last row ID or None if unavailable
        """
        if self._lastrowid is not None:
            return self._lastrowid

        # Try to extract from RETURNING results
        if self._last_results and self._last_query:
            query_upper = self._last_query.strip().upper()
            if "RETURNING" in query_upper and self._last_results:
                # Check for common ID column names
                first_row = self._last_results[0]
                for id_col in ["id", "ID", "Id", "rowid", "ROWID"]:
                    if id_col in first_row:
                        try:
                            self._lastrowid = int(first_row[id_col])
                            return self._lastrowid
                        except (ValueError, TypeError):
                            pass

                # If only one column returned, assume it's the ID
                if len(first_row) == 1:
                    try:
                        self._lastrowid = int(next(iter(first_row.values())))
                        return self._lastrowid
                    except (ValueError, TypeError):
                        pass

        return None

    @property
    def arraysize(self) -> int:
        """
        Number of rows to fetch at a time with fetchmany().
        DB-API 2.0 compliant (default: 1)
        """
        return self._arraysize

    @arraysize.setter
    def arraysize(self, size: int):
        """Set arraysize for fetchmany() operations"""
        if size < 1:
            raise ValueError("arraysize must be >= 1")
        self._arraysize = size

    @property
    def rownumber(self) -> Optional[int]:
        """
        Current 0-based index of cursor in result set.
        None if no query executed or before first row.
        """
        if self._current_index == 0:
            return None
        return self._current_index - 1

    @property
    def connection(self):
        """Return parent connection wrapper (DB-API 2.0 optional)"""
        return self.connection_wrapper

    @property
    def query(self) -> Optional[str]:
        """Return last executed query (extension)"""
        return self._last_query

    @property
    def query_parameters(self) -> Optional[Tuple]:
        """Return last query parameters (extension)"""
        return self._last_params

    def _build_description_from_results(self):
        """
        Dynamically build column description from result data.
        Infers types and sizes from actual values.
        """
        if not self._last_results:
            self._description = None
            return

        first_row = self._last_results[0]
        descriptions = []

        for col_name in first_row.keys():
            # Infer type and properties from all rows
            col_type = None
            max_size = 0
            has_null = False

            for row in self._last_results:
                value = row.get(col_name)

                if value is None:
                    has_null = True
                    continue

                # Detect Python type
                value_type = type(value)
                if col_type is None:
                    col_type = value_type

                # Estimate size
                if isinstance(value, str):
                    max_size = max(max_size, len(value))
                elif isinstance(value, (int, float)):
                    max_size = max(max_size, len(str(value)))
                elif isinstance(value, bytes):
                    max_size = max(max_size, len(value))

            # Build DB-API 2.0 description tuple
            # (name, type_code, display_size, internal_size, precision, scale, null_ok)
            descriptions.append(
                (
                    col_name,  # name
                    col_type,  # type_code (Python type)
                    max_size if max_size > 0 else None,  # display_size
                    None,  # internal_size (not available)
                    None,  # precision (not available)
                    None,  # scale (not available)
                    has_null,  # null_ok
                )
            )

            # Store extended metadata
            self._column_metadata[col_name] = {
                "python_type": col_type,
                "max_size": max_size,
                "nullable": has_null,
                "samples": min(len(self._last_results), 5),
            }

        self._description = descriptions

    async def execute(self, sql: str, parameters: Tuple = ()) -> "DatabaseCursorWrapper":
        """
        Execute SQL with parameters. Handles both SELECT and DML operations.

        Advanced features:
        - Automatic query type detection (SELECT, INSERT, UPDATE, DELETE, etc.)
        - RETURNING clause support for PostgreSQL
        - Rowcount tracking for affected rows
        - Lastrowid extraction from RETURNING results
        - Dynamic column description generation

        Args:
            sql: SQL query string (with %s or $1 style placeholders)
            parameters: Query parameters tuple

        Returns:
            Self for chaining

        Raises:
            Exception: Database errors with detailed logging
        """
        try:
            # Convert %s placeholders to $1, $2, etc. for PostgreSQL/asyncpg
            if "%s" in sql:
                # Simple sequential replacement
                i = 1
                while "%s" in sql:
                    sql = sql.replace("%s", f"${i}", 1)
                    i += 1

            # Reset state for new query
            self._last_query = sql
            self._last_params = parameters
            self._current_index = 0
            self._lastrowid = None
            self._column_metadata.clear()

            # Skip PRAGMA commands for PostgreSQL
            if sql.strip().upper().startswith("PRAGMA"):
                logger.debug(f"Skipping PRAGMA command: {sql}")
                self._last_results = []
                self._row_count = 0
                self._description = None
                return self

            # Detect query type dynamically
            query_upper = sql.strip().upper()
            query_type = self._detect_query_type(query_upper)

            if query_type in ("SELECT", "RETURNING"):
                # Query that returns rows
                results = await self.adapter_conn.fetch(sql, *parameters)
                self._last_results = results if results else []
                self._row_count = len(self._last_results)

                # Build dynamic column descriptions
                if self._last_results:
                    self._build_description_from_results()
                else:
                    self._description = None

                # Extract lastrowid from RETURNING results
                if query_type == "RETURNING" and self._last_results:
                    self._extract_lastrowid()

            elif query_type in ("INSERT", "UPDATE", "DELETE"):
                # DML operation - try to get affected rows count
                try:
                    # For PostgreSQL with asyncpg, we can use execute and get status
                    result = await self.adapter_conn.execute(sql, *parameters)

                    # Try to extract row count from result status
                    # PostgreSQL returns status like "INSERT 0 1" or "UPDATE 5"
                    if hasattr(result, "decode"):
                        result = result.decode("utf-8")

                    if isinstance(result, str):
                        self._row_count = self._parse_rowcount_from_status(result)
                    else:
                        self._row_count = -1  # Not available

                except Exception as e:
                    logger.debug(f"Could not determine rowcount: {e}")
                    await self.adapter_conn.execute(sql, *parameters)
                    self._row_count = -1  # Not available

                self._last_results = []
                self._description = None

            else:
                # Other operations (CREATE, DROP, ALTER, etc.)
                await self.adapter_conn.execute(sql, *parameters)
                self._last_results = []
                self._row_count = -1  # Not applicable
                self._description = None

            return self

        except Exception as e:
            logger.error(f"Error executing query: {sql[:100]}... with params {parameters}: {e}")
            # Reset state on error
            self._row_count = -1
            self._description = None
            self._last_results = []
            raise

    def _detect_query_type(self, query_upper: str) -> str:
        """
        Dynamically detect SQL query type.

        Args:
            query_upper: Uppercase SQL query

        Returns:
            str: Query type ('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'RETURNING', etc.)
        """
        # Check for RETURNING clause (takes precedence)
        if "RETURNING" in query_upper:
            return "RETURNING"

        # Check for standard DML/DDL commands
        query_start = query_upper.split()[0] if query_upper.split() else ""

        if query_start in ("SELECT", "WITH"):  # WITH for CTEs
            return "SELECT"
        elif query_start == "INSERT":
            return "INSERT"
        elif query_start == "UPDATE":
            return "UPDATE"
        elif query_start == "DELETE":
            return "DELETE"
        elif query_start in ("CREATE", "DROP", "ALTER", "TRUNCATE"):
            return "DDL"
        else:
            return "OTHER"

    def _parse_rowcount_from_status(self, status: str) -> int:
        """
        Parse rowcount from PostgreSQL status string.

        Examples:
            "INSERT 0 1" -> 1
            "UPDATE 5" -> 5
            "DELETE 10" -> 10

        Args:
            status: PostgreSQL command status string

        Returns:
            int: Number of affected rows or -1 if unavailable
        """
        try:
            parts = status.split()
            if len(parts) >= 2:
                # Last part is usually the row count
                return int(parts[-1])
        except (ValueError, IndexError):
            pass
        return -1

    def _extract_lastrowid(self):
        """
        Extract last inserted row ID from RETURNING results.
        Checks common ID column names and patterns.
        """
        if not self._last_results:
            return

        first_row = self._last_results[0]

        # Priority order for ID column detection
        id_candidates = ["id", "ID", "Id", "rowid", "ROWID", "RowId", "_id", "_ID", "pk", "PK"]

        # Try known column names first
        for col_name in id_candidates:
            if col_name in first_row:
                try:
                    self._lastrowid = int(first_row[col_name])
                    logger.debug(f"Extracted lastrowid={self._lastrowid} from column '{col_name}'")
                    return
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not convert {col_name} to int: {e}")

        # If RETURNING has exactly one column, assume it's the ID
        if len(first_row) == 1:
            try:
                value = next(iter(first_row.values()))
                self._lastrowid = int(value)
                logger.debug(f"Extracted lastrowid={self._lastrowid} from single RETURNING column")
            except (ValueError, TypeError, StopIteration) as e:
                logger.debug(f"Could not extract lastrowid from single column: {e}")

    async def executemany(self, sql: str, parameters_list: List[Tuple]) -> "DatabaseCursorWrapper":
        """
        Execute SQL with multiple parameter sets.

        Args:
            sql: SQL query string
            parameters_list: List of parameter tuples

        Returns:
            Self for chaining
        """
        try:
            total_affected = 0
            for parameters in parameters_list:
                await self.execute(sql, parameters)
                total_affected += self._row_count

            self._row_count = total_affected
            return self

        except Exception as e:
            logger.error(f"Error in executemany: {e}")
            raise

    async def fetchall(self) -> List[Dict[str, Any]]:
        """
        Fetch all remaining results from last query.

        Returns:
            List of result dictionaries
        """
        try:
            results = self._last_results[self._current_index :]
            self._current_index = len(self._last_results)
            return results

        except Exception as e:
            logger.error(f"Error in fetchall: {e}")
            return []

    async def fetchone(self) -> Optional[Dict[str, Any]]:
        """
        Fetch next result from last query.

        Returns:
            Single result dictionary or None
        """
        try:
            if self._current_index < len(self._last_results):
                result = self._last_results[self._current_index]
                self._current_index += 1
                return result
            return None

        except Exception as e:
            logger.error(f"Error in fetchone: {e}")
            return None

    async def fetchmany(self, size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch multiple results from last query.

        DB-API 2.0 compliant:
        - If size is None, uses cursor.arraysize (default: 1)
        - Returns up to size rows
        - Returns empty list when no more rows

        Args:
            size: Number of rows to fetch (None = use arraysize)

        Returns:
            List of result dictionaries (up to size items)
        """
        try:
            # Use arraysize if size not specified (DB-API 2.0 behavior)
            fetch_size = size if size is not None else self._arraysize

            if fetch_size < 1:
                raise ValueError("fetchmany size must be >= 1")

            results = self._last_results[self._current_index : self._current_index + fetch_size]
            self._current_index += len(results)
            return results

        except Exception as e:
            logger.error(f"Error in fetchmany: {e}")
            return []

    async def fetchval(self) -> Optional[Any]:
        """
        Fetch first column of next result.

        Returns:
            Single value or None
        """
        try:
            row = await self.fetchone()
            if row:
                # Get first value from dict
                return next(iter(row.values())) if row else None
            return None

        except Exception as e:
            logger.error(f"Error in fetchval: {e}")
            return None

    async def scroll(self, value: int, mode: str = "relative"):
        """
        Scroll cursor position (DB-API 2.0 optional extension).

        Args:
            value: Number of rows to move
            mode: 'relative' (default) or 'absolute'

        Raises:
            IndexError: If scrolling beyond result bounds
        """
        if mode == "relative":
            new_index = self._current_index + value
        elif mode == "absolute":
            new_index = value
        else:
            raise ValueError(f"Invalid scroll mode: {mode}. Use 'relative' or 'absolute'")

        if new_index < 0 or new_index > len(self._last_results):
            raise IndexError(f"Scroll out of range: {new_index}")

        self._current_index = new_index

    def setinputsizes(self, sizes):
        """DB-API 2.0 required method (no-op for our implementation)"""

    def setoutputsize(self, size, column=None):
        """DB-API 2.0 required method (no-op for our implementation)"""

    def get_column_metadata(self, column_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get extended column metadata (custom extension).

        Args:
            column_name: Specific column or None for all columns

        Returns:
            Dict with column metadata including types, sizes, nullability
        """
        if column_name:
            return self._column_metadata.get(column_name, {})
        return self._column_metadata.copy()

    async def close(self):
        """Close cursor and release resources"""
        self._last_results.clear()
        self._current_index = 0
        self._row_count = -1
        self._description = None
        self._lastrowid = None
        self._column_metadata.clear()

    def __aiter__(self):
        """Make cursor iterable"""
        return self

    async def __anext__(self) -> Dict[str, Any]:
        """Async iteration support"""
        row = await self.fetchone()
        if row is None:
            raise StopAsyncIteration
        return row

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        return False

    def __repr__(self) -> str:
        """String representation for debugging"""
        status = "closed" if self._row_count == -1 and not self._last_results else "open"
        return (
            f"<DatabaseCursorWrapper status={status} "
            f"rowcount={self._row_count} "
            f"rownumber={self.rownumber}>"
        )

    def __str__(self) -> str:
        """User-friendly string representation"""
        if self._last_query:
            query_preview = (
                self._last_query[:50] + "..." if len(self._last_query) > 50 else self._last_query
            )
            return f"Cursor(query='{query_preview}', rowcount={self._row_count})"
        return "Cursor(no query)"


class DatabaseConnectionWrapper:
    """
    Advanced wrapper that makes Cloud SQL adapter look like aiosqlite connection.
    Provides transaction support, connection pooling, and comprehensive error handling.
    """

    def __init__(self, adapter):
        """
        Initialize connection wrapper.

        Args:
            adapter: Cloud SQL database adapter instance
        """
        self.adapter = adapter
        self.row_factory = None  # For aiosqlite compatibility
        self._in_transaction: bool = False
        self._transaction_savepoint: Optional[str] = None
        self._current_connection = None
        self._connection_lock = asyncio.Lock()
        self._closed: bool = False

    @property
    def is_cloud(self) -> bool:
        """Check if using cloud database backend"""
        return self.adapter.is_cloud if hasattr(self.adapter, "is_cloud") else False

    @property
    def in_transaction(self) -> bool:
        """Check if currently in a transaction"""
        return self._in_transaction

    @asynccontextmanager
    async def cursor(self):
        """
        Return cursor-like object using adapter connection.
        Reuses connection if in transaction, otherwise creates new one.

        Yields:
            DatabaseCursorWrapper: Cursor for executing queries
        """
        if self._closed:
            raise RuntimeError("Connection is closed")

        async with self._connection_lock:
            if self._in_transaction and self._current_connection:
                # Reuse existing transaction connection
                yield DatabaseCursorWrapper(self._current_connection, connection_wrapper=self)
            else:
                # Create new connection for non-transaction queries
                async with self.adapter.connection() as conn:
                    yield DatabaseCursorWrapper(conn, connection_wrapper=self)

    async def execute(self, sql: str, parameters: Tuple = ()) -> "DatabaseCursorWrapper":
        """
        Execute SQL directly without creating cursor.

        Args:
            sql: SQL query string
            parameters: Query parameters

        Returns:
            Cursor wrapper with results
        """
        async with self.cursor() as cur:
            await cur.execute(sql, parameters)
            return cur

    async def executemany(self, sql: str, parameters_list: List[Tuple]) -> "DatabaseCursorWrapper":
        """
        Execute SQL with multiple parameter sets.

        Args:
            sql: SQL query string
            parameters_list: List of parameter tuples

        Returns:
            Cursor wrapper
        """
        async with self.cursor() as cur:
            await cur.executemany(sql, parameters_list)
            return cur

    async def executescript(self, sql_script: str):
        """
        Execute multiple SQL statements (compatibility with aiosqlite).

        Args:
            sql_script: Multiple SQL statements separated by semicolons
        """
        # Split on semicolons and execute each statement
        statements = [s.strip() for s in sql_script.split(";") if s.strip()]
        async with self.cursor() as cur:
            for statement in statements:
                await cur.execute(statement)

    async def begin(self):
        """Start a transaction explicitly"""
        if self._in_transaction:
            logger.warning("Already in transaction, creating savepoint")
            await self._create_savepoint()
            return

        async with self._connection_lock:
            if not self._current_connection:
                # Acquire connection for transaction
                self._current_connection = await self.adapter.connection().__aenter__()

            if self.is_cloud:
                # PostgreSQL: Start transaction
                await self._current_connection.execute("BEGIN")

            self._in_transaction = True
            logger.debug("Transaction started")

    async def commit(self):
        """
        Commit current transaction.
        For Cloud SQL (PostgreSQL): Explicit COMMIT
        For SQLite: Auto-commit mode, this is a no-op
        """
        if not self._in_transaction:
            # Not in explicit transaction, likely auto-commit mode
            return

        async with self._connection_lock:
            try:
                if self.is_cloud and self._current_connection:
                    # PostgreSQL: Explicit commit
                    await self._current_connection.execute("COMMIT")
                    logger.debug("Transaction committed")

                self._in_transaction = False
                self._transaction_savepoint = None

                # Release connection back to pool
                if self._current_connection:
                    await self._release_connection()

            except Exception as e:
                logger.error(f"Error committing transaction: {e}")
                await self.rollback()
                raise

    async def rollback(self):
        """
        Rollback current transaction.
        """
        if not self._in_transaction:
            logger.warning("No active transaction to rollback")
            return

        async with self._connection_lock:
            try:
                if self.is_cloud and self._current_connection:
                    if self._transaction_savepoint:
                        # Rollback to savepoint
                        await self._current_connection.execute(
                            f"ROLLBACK TO SAVEPOINT {self._transaction_savepoint}"
                        )
                        logger.debug(f"Rolled back to savepoint {self._transaction_savepoint}")
                        self._transaction_savepoint = None
                    else:
                        # Full rollback
                        await self._current_connection.execute("ROLLBACK")
                        logger.debug("Transaction rolled back")
                        self._in_transaction = False

                # Release connection
                if not self._transaction_savepoint and self._current_connection:
                    await self._release_connection()

            except Exception as e:
                logger.error(f"Error rolling back transaction: {e}")
                # Force release connection
                await self._release_connection()
                raise

    async def _create_savepoint(self):
        """Create a savepoint for nested transactions"""
        if not self.is_cloud:
            return  # SQLite handles this differently

        savepoint_name = f"sp_{int(time.time() * 1000000)}"
        if self._current_connection:
            await self._current_connection.execute(f"SAVEPOINT {savepoint_name}")
            self._transaction_savepoint = savepoint_name
            logger.debug(f"Created savepoint {savepoint_name}")

    async def _release_connection(self):
        """Release transaction connection back to pool"""
        if self._current_connection:
            try:
                # Exit the context manager for the connection
                await self._current_connection.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error releasing connection: {e}")
            finally:
                self._current_connection = None
                self._in_transaction = False

    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for automatic transaction handling.

        Usage:
            async with conn.transaction():
                await conn.execute("INSERT ...")
                await conn.execute("UPDATE ...")
            # Auto-commits on success, rolls back on exception
        """
        await self.begin()
        try:
            yield self
            await self.commit()
        except Exception:
            await self.rollback()
            raise

    async def close(self):
        """Close connection and release all resources"""
        if self._closed:
            return

        async with self._connection_lock:
            # Rollback any pending transaction
            if self._in_transaction:
                await self.rollback()

            # Release connection
            if self._current_connection:
                await self._release_connection()

            self._closed = True
            logger.debug("Connection wrapper closed")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if exc_type is not None and self._in_transaction:
            # Exception occurred, rollback
            await self.rollback()
        await self.close()
        return False


class PatternType(Enum):
    """Dynamic pattern types - extensible"""

    TEMPORAL = "temporal"  # Time-based patterns
    SEQUENTIAL = "sequential"  # Action sequences
    CONTEXTUAL = "contextual"  # Context-driven patterns
    HYBRID = "hybrid"  # Multi-factor patterns


class ConfidenceBoostStrategy(Enum):
    """Strategies for boosting pattern confidence"""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    ADAPTIVE = "adaptive"  # Learns optimal strategy


@dataclass
class GoalPattern:
    """Represents a learned goal pattern with ML metadata"""

    pattern_id: str
    goal_type: str
    context_embedding: Optional[List[float]]
    action_sequence: List[str]
    confidence: float
    success_rate: float
    occurrence_count: int
    last_seen: datetime
    avg_execution_time: float = 0.0
    std_execution_time: float = 0.0
    decay_factor: float = 0.95  # For time-based decay
    boost_history: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningMetrics:
    """Real-time learning performance metrics"""

    total_patterns: int
    active_patterns: int
    avg_confidence: float
    prediction_accuracy: float
    cache_hit_rate: float
    avg_inference_time_ms: float
    memory_usage_mb: float
    last_updated: datetime


class AdaptiveCache:
    """Smart LRU cache with TTL and size management"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                self.access_count[key] += 1
                self.hits += 1
                return value
            else:
                del self.cache[key]

        self.misses += 1
        return None

    def set(self, key: str, value: Any):
        """Set value in cache with eviction"""
        # Evict if full
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]

        self.cache[key] = (value, time.time())
        self.access_count[key] = 0

    def invalidate(self, key: str):
        """Remove from cache"""
        if key in self.cache:
            del self.cache[key]
            del self.access_count[key]

    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class PatternMatcher:
    """Advanced pattern matching with fuzzy matching and ML"""

    def __init__(self):
        self.embeddings_cache = AdaptiveCache(max_size=500, ttl_seconds=7200)

    async def compute_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """Compute similarity between patterns using multiple signals"""
        similarity_scores = []

        # Context similarity
        if "context" in pattern1 and "context" in pattern2:
            context_sim = self._jaccard_similarity(
                set(pattern1["context"].keys()), set(pattern2["context"].keys())
            )
            similarity_scores.append(context_sim * 0.3)

        # Action sequence similarity
        if "actions" in pattern1 and "actions" in pattern2:
            action_sim = self._sequence_similarity(pattern1["actions"], pattern2["actions"])
            similarity_scores.append(action_sim * 0.4)

        # Temporal similarity
        if "timestamp" in pattern1 and "timestamp" in pattern2:
            time_sim = self._temporal_similarity(pattern1["timestamp"], pattern2["timestamp"])
            similarity_scores.append(time_sim * 0.3)

        return sum(similarity_scores) if similarity_scores else 0.0

    @staticmethod
    def _jaccard_similarity(set1: Set, set2: Set) -> float:
        """Jaccard similarity coefficient"""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _sequence_similarity(seq1: List, seq2: List) -> float:
        """Levenshtein-based sequence similarity"""
        if not seq1 and not seq2:
            return 1.0

        # Simple edit distance
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        max_len = max(m, n)
        return 1.0 - (dp[m][n] / max_len) if max_len > 0 else 0.0

    @staticmethod
    def _temporal_similarity(time1: datetime, time2: datetime) -> float:
        """Temporal similarity based on time difference"""
        diff_seconds = abs((time1 - time2).total_seconds())

        # Decay similarity over time (1 hour window)
        if diff_seconds < 3600:
            return 1.0 - (diff_seconds / 3600)
        elif diff_seconds < 86400:  # 24 hours
            return 0.5 * (1.0 - (diff_seconds / 86400))
        else:
            return 0.0


class JARVISLearningDatabase:
    """
    Advanced hybrid database system for JARVIS learning
    - Async SQLite: Structured data with connection pooling
    - ChromaDB: Embeddings and semantic search
    - Adaptive caching: Smart LRU with TTL
    - ML-powered pattern matching and insights
    - Dynamic schema evolution
    - Real-time metrics and analytics
    """

    def __init__(self, db_path: Optional[Path] = None, config: Optional[Dict] = None):
        """Initialize the advanced learning database"""
        # Configuration with defaults
        self.config = config or {}
        self.cache_size = self.config.get("cache_size", 1000)
        self.cache_ttl = self.config.get("cache_ttl_seconds", 3600)
        self.enable_ml = self.config.get("enable_ml_features", True)
        self.auto_optimize = self.config.get("auto_optimize", True)
        self.batch_size = self.config.get("batch_insert_size", 100)

        # Set up paths
        self.db_dir = db_path or Path.home() / ".jarvis" / "learning"
        self.db_dir.mkdir(parents=True, exist_ok=True)

        self.sqlite_path = self.db_dir / "jarvis_learning.db"
        self.chroma_path = self.db_dir / "chroma_embeddings"

        # Async SQLite connection (will be initialized in async context)
        # Type is Optional during __init__ but guaranteed non-None after initialize()
        self.db: Union[aiosqlite.Connection, "DatabaseConnectionWrapper", None] = None  # type: ignore[assignment]
        self._db_lock = asyncio.Lock()

        # Adaptive caching
        self.pattern_cache = AdaptiveCache(self.cache_size, self.cache_ttl)
        self.goal_cache = AdaptiveCache(self.cache_size, self.cache_ttl)
        self.query_cache = AdaptiveCache(self.cache_size // 2, self.cache_ttl)

        # Pattern matching engine
        self.pattern_matcher = PatternMatcher()

        # Batch processing queues
        self.pending_goals: deque = deque(maxlen=self.batch_size)
        self.pending_actions: deque = deque(maxlen=self.batch_size)
        self.pending_patterns: deque = deque(maxlen=self.batch_size)

        # Performance metrics
        self.metrics = LearningMetrics(
            total_patterns=0,
            active_patterns=0,
            avg_confidence=0.0,
            prediction_accuracy=0.0,
            cache_hit_rate=0.0,
            avg_inference_time_ms=0.0,
            memory_usage_mb=0.0,
            last_updated=datetime.now(),
        )

        # Initialize ChromaDB if available
        self.chroma_client = None
        self.goal_collection = None
        self.pattern_collection = None
        self.context_collection = None

        logger.info(f"Advanced JARVIS Learning Database initializing at {self.db_dir}")

    async def initialize(self):
        """Async initialization - call this after creating instance"""
        # Initialize async SQLite
        await self._init_sqlite()

        # Initialize ChromaDB
        if CHROMADB_AVAILABLE:
            await self._init_chromadb()

        # Load metrics
        await self._load_metrics()

        # Start background tasks
        asyncio.create_task(self._auto_flush_batches())
        asyncio.create_task(self._auto_optimize_task())

        logger.info(f"✅ Advanced Learning Database initialized")
        logger.info(f"   Cache: {self.cache_size} entries, {self.cache_ttl}s TTL")
        logger.info(f"   ML Features: {self.enable_ml}")
        logger.info(f"   Auto-optimize: {self.auto_optimize}")

    def _ensure_db_initialized(self) -> Union[aiosqlite.Connection, "DatabaseConnectionWrapper"]:
        """Assert that database is initialized and return it with proper type."""
        assert self.db is not None, "Database not initialized. Call initialize() first."
        return self.db

    async def _init_sqlite(self):
        """Initialize async database (SQLite or Cloud SQL) with enhanced schema"""
        # Try to use Cloud SQL if available
        if CLOUD_ADAPTER_AVAILABLE:
            try:
                adapter = await get_database_adapter()
                if adapter.is_cloud:
                    logger.info("☁️  Using Cloud SQL (PostgreSQL)")
                    self.db = DatabaseConnectionWrapper(adapter)
                else:
                    logger.info("📂 Using SQLite (Cloud SQL not configured)")
                    self.db = await aiosqlite.connect(str(self.sqlite_path))
                    self.db.row_factory = aiosqlite.Row
            except Exception as e:
                logger.warning(f"⚠️  Cloud SQL adapter failed ({e}), falling back to SQLite")
                self.db = await aiosqlite.connect(str(self.sqlite_path))
                self.db.row_factory = aiosqlite.Row
        else:
            logger.info("📂 Using SQLite (Cloud adapter not available)")
            self.db = await aiosqlite.connect(str(self.sqlite_path))
            self.db.row_factory = aiosqlite.Row

        # Check if using Cloud SQL (PostgreSQL) or SQLite
        is_cloud = isinstance(self.db, DatabaseConnectionWrapper) and self.db.is_cloud

        # Helper function to generate auto-increment syntax
        def auto_increment(column_name: str, col_type: str = "INTEGER") -> str:
            """Generate auto-increment primary key syntax for current database"""
            if is_cloud:
                # PostgreSQL uses SERIAL or BIGSERIAL
                serial_type = "BIGSERIAL" if col_type == "BIGINT" else "SERIAL"
                return f"{column_name} {serial_type} PRIMARY KEY"
            else:
                # SQLite uses AUTOINCREMENT
                return f"{column_name} {col_type} PRIMARY KEY AUTOINCREMENT"

        # Helper function to generate boolean default syntax
        def bool_default(value: int) -> str:
            """Generate boolean default syntax for current database"""
            if is_cloud:
                # PostgreSQL uses TRUE/FALSE
                return "TRUE" if value else "FALSE"
            else:
                # SQLite uses 1/0
                return str(value)

        # Helper function to generate binary data type
        def blob_type() -> str:
            """Generate binary data type for current database"""
            if is_cloud:
                # PostgreSQL uses BYTEA
                return "BYTEA"
            else:
                # SQLite uses BLOB
                return "BLOB"

        db = self._ensure_db_initialized()
        async with db.cursor() as cursor:
            # Enable WAL mode for better concurrency (SQLite only)
            if not is_cloud:
                await cursor.execute("PRAGMA journal_mode=WAL")
                await cursor.execute("PRAGMA synchronous=NORMAL")
                await cursor.execute("PRAGMA cache_size=10000")
                await cursor.execute("PRAGMA temp_store=MEMORY")

            # Goals table with enhanced tracking
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS goals (
                    goal_id TEXT PRIMARY KEY,
                    goal_type TEXT NOT NULL,
                    goal_level TEXT NOT NULL,
                    description TEXT,
                    confidence REAL,
                    progress REAL DEFAULT 0.0,
                    is_completed BOOLEAN DEFAULT {bool_default(0)},
                    created_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    predicted_duration REAL,
                    actual_duration REAL,
                    evidence JSON,
                    context_hash TEXT,
                    embedding_id TEXT,
                    metadata JSON
                )
            """
            )

            # Actions table with performance tracking
            await cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS actions (
                    action_id TEXT PRIMARY KEY,
                    action_type TEXT NOT NULL,
                    target TEXT,
                    goal_id TEXT,
                    confidence REAL,
                    success BOOLEAN,
                    execution_time REAL,
                    timestamp TIMESTAMP,
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    params JSON,
                    result JSON,
                    context_hash TEXT,
                    FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
                )
            """
            )

            # Enhanced patterns table with ML metadata
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    pattern_hash TEXT UNIQUE,
                    pattern_data JSON,
                    confidence REAL,
                    success_rate REAL,
                    occurrence_count INTEGER DEFAULT 1,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    avg_execution_time REAL,
                    std_execution_time REAL,
                    decay_applied BOOLEAN DEFAULT {bool_default(0)},
                    boost_count INTEGER DEFAULT 0,
                    embedding_id TEXT,
                    metadata JSON
                )
            """
            )

            # User preferences with confidence tracking
            await cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    preference_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    confidence REAL,
                    learned_from TEXT,
                    update_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    UNIQUE(category, key)
                )
            """
            )

            # Goal-Action mappings with performance metrics
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS goal_action_mappings (
                    {auto_increment('mapping_id')},
                    goal_type TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    avg_execution_time REAL,
                    std_execution_time REAL,
                    confidence REAL,
                    last_updated TIMESTAMP,
                    prediction_accuracy REAL,
                    UNIQUE(goal_type, action_type)
                )
            """
            )

            # Display patterns with temporal analysis
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS display_patterns (
                    {auto_increment('pattern_id')},
                    display_name TEXT NOT NULL,
                    context JSON,
                    context_hash TEXT,
                    connection_time TIME,
                    day_of_week INTEGER,
                    hour_of_day INTEGER,
                    frequency INTEGER DEFAULT 1,
                    auto_connect BOOLEAN DEFAULT {bool_default(0)},
                    last_seen TIMESTAMP,
                    consecutive_successes INTEGER DEFAULT 0,
                    metadata JSON
                )
            """
            )

            # Learning metrics tracking
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS learning_metrics (
                    {auto_increment('metric_id')},
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    timestamp TIMESTAMP,
                    context JSON
                )
            """
            )

            # Pattern similarity cache
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS pattern_similarity_cache (
                    {auto_increment('cache_id')},
                    pattern1_id TEXT NOT NULL,
                    pattern2_id TEXT NOT NULL,
                    similarity_score REAL,
                    computed_at TIMESTAMP,
                    UNIQUE(pattern1_id, pattern2_id)
                )
            """
            )

            # Context embeddings metadata
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS context_embeddings (
                    embedding_id TEXT PRIMARY KEY,
                    context_hash TEXT UNIQUE,
                    embedding_vector {blob_type()},
                    dimension INTEGER,
                    created_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP
                )
            """
            )

            # ============================================================================
            # 24/7 BEHAVIORAL LEARNING TABLES - Enhanced Workspace Tracking
            # ============================================================================

            # Workspace/Space tracking (Yabai integration)
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS workspace_usage (
                    {auto_increment('usage_id')},
                    space_id INTEGER NOT NULL,
                    space_label TEXT,
                    app_name TEXT NOT NULL,
                    window_title TEXT,
                    window_position JSON,
                    focus_duration_seconds REAL,
                    timestamp TIMESTAMP,
                    day_of_week INTEGER,
                    hour_of_day INTEGER,
                    is_fullscreen BOOLEAN DEFAULT {bool_default(0)},
                    metadata JSON
                )
            """
            )

            # App usage patterns (24/7 tracking)
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS app_usage_patterns (
                    {auto_increment('pattern_id')},
                    app_name TEXT NOT NULL,
                    space_id INTEGER,
                    usage_frequency INTEGER DEFAULT 1,
                    avg_session_duration REAL,
                    total_usage_time REAL,
                    typical_time_of_day INTEGER,
                    typical_day_of_week INTEGER,
                    last_used TIMESTAMP,
                    confidence REAL DEFAULT 0.5,
                    metadata JSON,
                    UNIQUE(app_name, space_id, typical_time_of_day, typical_day_of_week)
                )
            """
            )

            # User workflows (sequential action patterns)
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS user_workflows (
                    {auto_increment('workflow_id')},
                    workflow_name TEXT,
                    action_sequence JSON NOT NULL,
                    space_sequence JSON,
                    app_sequence JSON,
                    frequency INTEGER DEFAULT 1,
                    avg_duration REAL,
                    success_rate REAL DEFAULT 1.0,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    time_of_day_pattern JSON,
                    confidence REAL DEFAULT 0.5,
                    metadata JSON
                )
            """
            )

            # Space transitions (movement between Spaces)
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS space_transitions (
                    {auto_increment('transition_id')},
                    from_space_id INTEGER NOT NULL,
                    to_space_id INTEGER NOT NULL,
                    trigger_app TEXT,
                    trigger_action TEXT,
                    frequency INTEGER DEFAULT 1,
                    avg_time_between_seconds REAL,
                    timestamp TIMESTAMP,
                    hour_of_day INTEGER,
                    day_of_week INTEGER,
                    metadata JSON
                )
            """
            )

            # Behavioral patterns (high-level user habits)
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS behavioral_patterns (
                    {auto_increment('behavior_id')},
                    behavior_type TEXT NOT NULL,
                    behavior_description TEXT,
                    pattern_data JSON NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.5,
                    temporal_pattern JSON,
                    contextual_triggers JSON,
                    first_observed TIMESTAMP,
                    last_observed TIMESTAMP,
                    prediction_accuracy REAL,
                    metadata JSON
                )
            """
            )

            # Temporal patterns (time-based behaviors for leap years too!)
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS temporal_patterns (
                    {auto_increment('temporal_id')},
                    pattern_type TEXT NOT NULL,
                    time_of_day INTEGER,
                    day_of_week INTEGER,
                    day_of_month INTEGER,
                    month_of_year INTEGER,
                    is_leap_year BOOLEAN DEFAULT {bool_default(0)},
                    action_type TEXT NOT NULL,
                    target TEXT,
                    frequency INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.5,
                    last_occurrence TIMESTAMP,
                    metadata JSON
                )
            """
            )

            # Proactive suggestions (what JARVIS should suggest)
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS proactive_suggestions (
                    {auto_increment('suggestion_id')},
                    suggestion_type TEXT NOT NULL,
                    suggestion_text TEXT NOT NULL,
                    trigger_pattern_id TEXT,
                    confidence REAL NOT NULL,
                    times_suggested INTEGER DEFAULT 0,
                    times_accepted INTEGER DEFAULT 0,
                    times_rejected INTEGER DEFAULT 0,
                    acceptance_rate REAL,
                    created_at TIMESTAMP,
                    last_suggested TIMESTAMP,
                    metadata JSON
                )
            """
            )

            # Conversation history (ALL user-JARVIS interactions for learning)
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    {auto_increment('interaction_id')},
                    timestamp TIMESTAMP NOT NULL,
                    session_id TEXT,
                    user_query TEXT NOT NULL,
                    jarvis_response TEXT NOT NULL,
                    response_type TEXT,
                    confidence_score REAL,
                    execution_time_ms REAL,
                    success BOOLEAN DEFAULT {bool_default(1)},
                    user_feedback TEXT,
                    feedback_score INTEGER,
                    was_corrected BOOLEAN DEFAULT {bool_default(0)},
                    correction_text TEXT,
                    context_snapshot JSON,
                    active_apps JSON,
                    current_space TEXT,
                    system_state JSON,
                    embedding_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create index for conversation_history
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timestamp ON conversation_history(timestamp)
            """
            )
            # Create index for conversation_history
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session ON conversation_history(session_id)
            """
            )
            # Create index for conversation_history
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_response_type ON conversation_history(response_type)
            """
            )
            # Create index for conversation_history
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_success ON conversation_history(success)
            """
            )
            # Create index for conversation_history
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_score ON conversation_history(feedback_score)
            """
            )

            # Learning from corrections (when user corrects JARVIS)
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS interaction_corrections (
                    {auto_increment('correction_id')},
                    interaction_id INTEGER NOT NULL,
                    original_response TEXT NOT NULL,
                    corrected_response TEXT NOT NULL,
                    correction_type TEXT NOT NULL,
                    user_explanation TEXT,
                    applied_at TIMESTAMP NOT NULL,
                    learned BOOLEAN DEFAULT {bool_default(0)},
                    pattern_extracted TEXT,
                    metadata JSON,
                    FOREIGN KEY (interaction_id) REFERENCES conversation_history(interaction_id)
                )
            """
            )

            # Create index for interaction_corrections
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_correction_type ON interaction_corrections(correction_type)
            """
            )
            # Create index for interaction_corrections
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_learned ON interaction_corrections(learned)
            """
            )

            # Semantic embeddings for conversation search
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS conversation_embeddings (
                    embedding_id TEXT PRIMARY KEY,
                    interaction_id INTEGER NOT NULL,
                    query_embedding {blob_type()},
                    response_embedding {blob_type()},
                    combined_embedding {blob_type()},
                    embedding_model TEXT DEFAULT 'all-MiniLM-L6-v2',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (interaction_id) REFERENCES conversation_history(interaction_id)
                )
            """
            )

            # Create index for conversation_embeddings
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_interaction ON conversation_embeddings(interaction_id)
            """
            )

            # Voice transcription accuracy tracking
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS voice_transcriptions (
                    {auto_increment('transcription_id')},
                    interaction_id INTEGER,
                    raw_audio_hash TEXT NOT NULL,
                    transcribed_text TEXT NOT NULL,
                    corrected_text TEXT,
                    confidence_score REAL,
                    audio_duration_ms REAL,
                    language_code TEXT DEFAULT 'en-US',
                    accent_detected TEXT,
                    background_noise_level REAL,
                    retry_count INTEGER DEFAULT 0,
                    was_misheard BOOLEAN DEFAULT {bool_default(0)},
                    transcription_engine TEXT DEFAULT 'browser_api',
                    audio_quality_score REAL,
                    timestamp TIMESTAMP NOT NULL,
                    FOREIGN KEY (interaction_id) REFERENCES conversation_history(interaction_id)
                )
            """
            )

            # Create index for voice_transcriptions
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_was_misheard ON voice_transcriptions(was_misheard)
            """
            )
            # Create index for voice_transcriptions
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_confidence ON voice_transcriptions(confidence_score)
            """
            )
            # Create index for voice_transcriptions
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_retry_count ON voice_transcriptions(retry_count)
            """
            )

            # Speaker/Voiceprint recognition (Derek J. Russell)
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS speaker_profiles (
                    {auto_increment('speaker_id')},
                    speaker_name TEXT NOT NULL UNIQUE,
                    voiceprint_embedding {blob_type()},
                    total_samples INTEGER DEFAULT 0,
                    average_pitch_hz REAL,
                    speech_rate_wpm REAL,
                    accent_profile TEXT,
                    common_phrases JSON,
                    vocabulary_preferences JSON,
                    pronunciation_patterns JSON,
                    recognition_confidence REAL DEFAULT 0.0,
                    is_primary_user BOOLEAN DEFAULT {bool_default(0)},
                    security_level TEXT DEFAULT 'standard',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create index for speaker_profiles
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_speaker_name ON speaker_profiles(speaker_name)
            """
            )

            # Voice samples for speaker training (Derek's voice patterns)
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS voice_samples (
                    {auto_increment('sample_id')},
                    speaker_id INTEGER NOT NULL,
                    audio_hash TEXT NOT NULL,
                    audio_fingerprint {blob_type()},
                    mfcc_features {blob_type()},
                    duration_ms REAL NOT NULL,
                    transcription TEXT,
                    pitch_mean REAL,
                    pitch_std REAL,
                    energy_mean REAL,
                    recording_timestamp TIMESTAMP NOT NULL,
                    quality_score REAL,
                    background_noise REAL,
                    used_for_training BOOLEAN DEFAULT {bool_default(1)},
                    FOREIGN KEY (speaker_id) REFERENCES speaker_profiles(speaker_id)
                )
            """
            )

            # Create indexes for voice_samples
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_speaker ON voice_samples(speaker_id)
            """
            )
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_quality ON voice_samples(quality_score)
            """
            )

            # Acoustic adaptation (learning Derek's accent/pronunciation)
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS acoustic_adaptations (
                    {auto_increment('adaptation_id')},
                    speaker_id INTEGER NOT NULL,
                    phoneme_pattern TEXT NOT NULL,
                    expected_pronunciation TEXT NOT NULL,
                    actual_pronunciation TEXT NOT NULL,
                    frequency_count INTEGER DEFAULT 1,
                    confidence REAL,
                    context_words JSON,
                    learned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_observed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (speaker_id) REFERENCES speaker_profiles(speaker_id)
                )
            """
            )

            # Create index for acoustic_adaptations
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_speaker_phoneme ON acoustic_adaptations(speaker_id, phoneme_pattern)
            """
            )

            # Misheard queries (for learning what went wrong)
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS misheard_queries (
                    {auto_increment('misheard_id')},
                    transcription_id INTEGER NOT NULL,
                    what_jarvis_heard TEXT NOT NULL,
                    what_user_meant TEXT NOT NULL,
                    correction_method TEXT,
                    acoustic_similarity_score REAL,
                    phonetic_distance INTEGER,
                    context_clues JSON,
                    learned_pattern TEXT,
                    occurred_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (transcription_id) REFERENCES voice_transcriptions(transcription_id)
                )
            """
            )

            # Create index for misheard_queries
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_similarity ON misheard_queries(acoustic_similarity_score)
            """
            )
            # Create index for misheard_queries
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_occurred ON misheard_queries(occurred_at)
            """
            )

            # Retry patterns (when user has to repeat)
            await cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS query_retries (
                    {auto_increment('retry_id')},
                    original_transcription_id INTEGER NOT NULL,
                    retry_transcription_id INTEGER NOT NULL,
                    retry_number INTEGER NOT NULL,
                    time_between_retries_ms REAL,
                    confidence_improved REAL,
                    retry_reason TEXT,
                    eventually_succeeded BOOLEAN,
                    timestamp TIMESTAMP NOT NULL,
                    FOREIGN KEY (original_transcription_id) REFERENCES voice_transcriptions(transcription_id),
                    FOREIGN KEY (retry_transcription_id) REFERENCES voice_transcriptions(transcription_id)
                )
            """
            )

            # Create index for query_retries
            await cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_retry_number ON query_retries(retry_number)
            """
            )

            # Performance indexes
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_goals_type ON goals(goal_type)")
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_goals_created ON goals(created_at)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_goals_context_hash ON goals(context_hash)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_actions_type ON actions(action_type)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_actions_timestamp ON actions(timestamp)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_actions_context_hash ON actions(context_hash)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_patterns_hash ON patterns(pattern_hash)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_display_patterns_context ON display_patterns(context_hash)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_display_patterns_time ON display_patterns(connection_time, day_of_week)"
            )

            # Indexes for 24/7 behavioral learning tables
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_workspace_usage_space ON workspace_usage(space_id, timestamp)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_workspace_usage_app ON workspace_usage(app_name, timestamp)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_workspace_usage_time ON workspace_usage(hour_of_day, day_of_week)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_app_usage_app ON app_usage_patterns(app_name)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_app_usage_space ON app_usage_patterns(space_id)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_app_usage_time ON app_usage_patterns(typical_time_of_day, typical_day_of_week)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_workflows_frequency ON user_workflows(frequency DESC)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_space_transitions_from ON space_transitions(from_space_id, timestamp)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_space_transitions_to ON space_transitions(to_space_id, timestamp)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_behavioral_patterns_type ON behavioral_patterns(behavior_type)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_temporal_patterns_time ON temporal_patterns(time_of_day, day_of_week)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_temporal_patterns_leap ON temporal_patterns(is_leap_year)"
            )
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_suggestions_confidence ON proactive_suggestions(confidence DESC)"
            )

        await self.db.commit()
        logger.info("SQLite database initialized with enhanced async schema")

    async def _init_chromadb(self):
        """Initialize ChromaDB for embeddings and semantic search"""
        try:
            # Disable ChromaDB telemetry via environment variable (belt and suspenders)
            import os

            os.environ["ANONYMIZED_TELEMETRY"] = "False"

            # Suppress ChromaDB telemetry errors and logging
            import logging as stdlib_logging

            stdlib_logging.getLogger("chromadb.telemetry").setLevel(stdlib_logging.CRITICAL)

            # Monkey-patch ChromaDB telemetry to prevent errors in older versions
            try:
                from chromadb.telemetry.product import posthog

                # Replace capture method with no-op to prevent API mismatch errors
                if hasattr(posthog, "Posthog"):
                    original_capture = (
                        posthog.Posthog.capture if hasattr(posthog.Posthog, "capture") else None
                    )
                    if original_capture:

                        def noop_capture(self, *args, **kwargs):
                            pass

                        posthog.Posthog.capture = noop_capture
            except (ImportError, AttributeError):
                pass  # Telemetry module not available or already disabled

            # Initialize ChromaDB client with persistent storage
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # Create or get collections with metadata
            self.goal_collection = self.chroma_client.get_or_create_collection(
                name="goal_embeddings",
                metadata={
                    "description": "Goal context embeddings for similarity search",
                    "created": datetime.now().isoformat(),
                },
            )

            self.pattern_collection = self.chroma_client.get_or_create_collection(
                name="pattern_embeddings",
                metadata={
                    "description": "Pattern embeddings for matching",
                    "created": datetime.now().isoformat(),
                },
            )

            self.context_collection = self.chroma_client.get_or_create_collection(
                name="context_embeddings",
                metadata={
                    "description": "Context state embeddings for prediction",
                    "created": datetime.now().isoformat(),
                },
            )

            logger.info("ChromaDB initialized for semantic search")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None

    # ==================== Goal Management (Async + Cached) ====================

    async def store_goal(self, goal: Dict[str, Any], batch: bool = False) -> str:
        """Store an inferred goal with batching support"""
        goal_id = goal.get("goal_id", self._generate_id("goal"))

        if batch:
            self.pending_goals.append((goal_id, goal))
            if len(self.pending_goals) >= self.batch_size:
                await self._flush_goal_batch()
            return goal_id

        # Compute context hash for deduplication
        context_hash = self._hash_context(goal.get("evidence", {}))

        async with self._db_lock:
            async with self.db.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT OR REPLACE INTO goals
                    (goal_id, goal_type, goal_level, description, confidence,
                     progress, is_completed, created_at, evidence, context_hash, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        goal_id,
                        goal["goal_type"],
                        goal.get("goal_level", "UNKNOWN"),
                        goal.get("description", ""),
                        goal.get("confidence", 0.0),
                        goal.get("progress", 0.0),
                        goal.get("is_completed", False),
                        goal.get("created_at", datetime.now()),
                        json.dumps(goal.get("evidence", [])),
                        context_hash,
                        json.dumps(goal.get("metadata", {})),
                    ),
                )

            await self.db.commit()

        # Store embedding in ChromaDB if available
        if self.goal_collection and "embedding" in goal:
            await self._store_embedding_async(
                self.goal_collection,
                goal_id,
                goal["embedding"],
                {
                    "goal_type": goal["goal_type"],
                    "confidence": goal.get("confidence", 0.0),
                    "timestamp": datetime.now().isoformat(),
                },
            )

        # Update cache
        self.goal_cache.set(goal_id, goal)

        return goal_id

    async def get_similar_goals(
        self, embedding: List[float], top_k: int = 5, min_confidence: float = 0.5
    ) -> List[Dict]:
        """Find similar goals using semantic search with caching"""
        cache_key = f"similar_goals_{hashlib.md5(str(embedding).encode(), usedforsecurity=False).hexdigest()}_{top_k}"

        cached = self.query_cache.get(cache_key)
        if cached:
            return cached

        if not self.goal_collection:
            return []

        try:
            results = self.goal_collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where={"confidence": {"$gte": min_confidence}} if min_confidence > 0 else None,
            )

            similar_goals = []
            async with self.db.cursor() as cursor:
                for i, goal_id in enumerate(results["ids"][0]):
                    await cursor.execute("SELECT * FROM goals WHERE goal_id = ?", (goal_id,))
                    row = await cursor.fetchone()

                    if row:
                        goal = dict(row)
                        goal["evidence"] = json.loads(goal["evidence"]) if goal["evidence"] else []
                        goal["metadata"] = json.loads(goal["metadata"]) if goal["metadata"] else {}
                        goal["similarity"] = 1.0 - results["distances"][0][i]
                        similar_goals.append(goal)

            self.query_cache.set(cache_key, similar_goals)
            return similar_goals

        except Exception as e:
            logger.error(f"Error finding similar goals: {e}")
            return []

    # ==================== Action Tracking (Async + Metrics) ====================

    async def store_action(self, action: Dict[str, Any], batch: bool = False) -> str:
        """Store an executed action with performance tracking"""
        action_id = action.get("action_id", self._generate_id("action"))

        if batch:
            self.pending_actions.append((action_id, action))
            if len(self.pending_actions) >= self.batch_size:
                await self._flush_action_batch()
            return action_id

        context_hash = self._hash_context(action.get("params", {}))

        async with self._db_lock:
            async with self.db.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO actions
                    (action_id, action_type, target, goal_id, confidence,
                     success, execution_time, timestamp, retry_count, params, result, context_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        action_id,
                        action["action_type"],
                        action.get("target", ""),
                        action.get("goal_id"),
                        action.get("confidence", 0.0),
                        action.get("success", False),
                        action.get("execution_time", 0.0),
                        action.get("timestamp", datetime.now()),
                        action.get("retry_count", 0),
                        json.dumps(action.get("params", {})),
                        json.dumps(action.get("result", {})),
                        context_hash,
                    ),
                )

            await self.db.commit()

        # Update goal-action mapping
        if action.get("goal_id"):
            await self._update_goal_action_mapping(
                action["action_type"],
                action.get("goal_id"),
                action.get("success", False),
                action.get("execution_time", 0.0),
            )

        return action_id

    async def _update_goal_action_mapping(
        self, action_type: str, goal_id: Optional[str], success: bool, execution_time: float
    ):
        """Update goal-action mapping with statistical tracking"""
        if not goal_id:
            return

        async with self.db.cursor() as cursor:
            # Get goal type
            await cursor.execute("SELECT goal_type FROM goals WHERE goal_id = ?", (goal_id,))
            row = await cursor.fetchone()

            if row:
                goal_type = row["goal_type"]

                # Calculate new statistics
                success_inc = 1 if success else 0
                failure_inc = 0 if success else 1

                await cursor.execute(
                    """
                    INSERT INTO goal_action_mappings
                    (goal_type, action_type, success_count, failure_count,
                     avg_execution_time, confidence, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(goal_type, action_type) DO UPDATE SET
                        success_count = success_count + ?,
                        failure_count = failure_count + ?,
                        avg_execution_time = (avg_execution_time * (success_count + failure_count) + ?)
                                            / (success_count + failure_count + 1),
                        confidence = CAST(success_count + ? AS REAL) / (success_count + failure_count + 1),
                        last_updated = ?
                """,
                    (
                        goal_type,
                        action_type,
                        success_inc,
                        failure_inc,
                        execution_time,
                        0.5,
                        datetime.now(),
                        success_inc,
                        failure_inc,
                        execution_time,
                        success_inc,
                        datetime.now(),
                    ),
                )

        await self.db.commit()

    # ==================== Conversation History & Learning ====================

    async def record_interaction(
        self,
        user_query: str,
        jarvis_response: str,
        response_type: Optional[str] = None,
        confidence_score: Optional[float] = None,
        execution_time_ms: Optional[float] = None,
        success: bool = True,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Record every user-JARVIS interaction for learning and improvement.

        Args:
            user_query: What the user said/typed
            jarvis_response: How JARVIS responded
            response_type: Type of response (command, conversation, error, etc.)
            confidence_score: JARVIS's confidence in the response (0-1)
            execution_time_ms: How long it took to respond
            success: Whether the response was successful
            session_id: Unique session identifier
            context: Full context snapshot (apps, spaces, system state, etc.)

        Returns:
            int: interaction_id for reference
        """
        try:
            import uuid

            embedding_id = str(uuid.uuid4())

            async with self.db.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO conversation_history
                    (timestamp, session_id, user_query, jarvis_response, response_type,
                     confidence_score, execution_time_ms, success, context_snapshot,
                     active_apps, current_space, system_state, embedding_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        datetime.now(),
                        session_id,
                        user_query,
                        jarvis_response,
                        response_type,
                        confidence_score,
                        execution_time_ms,
                        1 if success else 0,
                        json.dumps(context or {}),
                        json.dumps(context.get("active_apps", []) if context else []),
                        context.get("current_space") if context else None,
                        json.dumps(context.get("system_state", {}) if context else {}),
                        embedding_id,
                    ),
                )

                await self.db.commit()

                # Get the interaction_id
                interaction_id = cursor.lastrowid

                # Generate embeddings asynchronously (non-blocking)
                if self.enable_ml:
                    asyncio.create_task(
                        self._generate_conversation_embeddings(
                            interaction_id, embedding_id, user_query, jarvis_response
                        )
                    )

                logger.debug(
                    f"Recorded interaction {interaction_id}: '{user_query[:50]}...' -> '{jarvis_response[:50]}...'"
                )

                return interaction_id

        except Exception as e:
            logger.error(f"Failed to record interaction: {e}")
            return -1

    async def record_correction(
        self,
        interaction_id: int,
        corrected_response: str,
        correction_type: str,
        user_explanation: Optional[str] = None,
    ) -> int:
        """
        Record when user corrects JARVIS's response (critical for learning!).

        Args:
            interaction_id: ID of the original interaction
            corrected_response: What the correct response should have been
            correction_type: Type of correction (misunderstanding, wrong_action, etc.)
            user_explanation: User's explanation of what went wrong

        Returns:
            int: correction_id
        """
        try:
            async with self.db.cursor() as cursor:
                # Get original response
                await cursor.execute(
                    "SELECT jarvis_response FROM conversation_history WHERE interaction_id = ?",
                    (interaction_id,),
                )
                row = await cursor.fetchone()

                if not row:
                    logger.error(f"Interaction {interaction_id} not found for correction")
                    return -1

                original_response = row["jarvis_response"]

                # Insert correction
                await cursor.execute(
                    """
                    INSERT INTO interaction_corrections
                    (interaction_id, original_response, corrected_response,
                     correction_type, user_explanation, applied_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        interaction_id,
                        original_response,
                        corrected_response,
                        correction_type,
                        user_explanation,
                        datetime.now(),
                    ),
                )

                # Mark the original interaction as corrected
                await cursor.execute(
                    """
                    UPDATE conversation_history
                    SET was_corrected = 1, correction_text = ?
                    WHERE interaction_id = ?
                """,
                    (corrected_response, interaction_id),
                )

                await self.db.commit()

                correction_id = cursor.lastrowid

                logger.info(
                    f"Recorded correction {correction_id} for interaction {interaction_id}: "
                    f"{correction_type}"
                )

                # Extract learning pattern from correction (ML task)
                asyncio.create_task(self._extract_correction_pattern(correction_id))

                return correction_id

        except Exception as e:
            logger.error(f"Failed to record correction: {e}")
            return -1

    async def add_user_feedback(
        self,
        interaction_id: int,
        feedback_text: Optional[str] = None,
        feedback_score: Optional[int] = None,
    ) -> bool:
        """
        Record user feedback on JARVIS's response (thumbs up/down, rating, comment).

        Args:
            interaction_id: ID of the interaction
            feedback_text: User's feedback comment
            feedback_score: Numeric score (-1, 0, 1 or 1-5 scale)

        Returns:
            bool: Success
        """
        try:
            async with self.db.cursor() as cursor:
                await cursor.execute(
                    """
                    UPDATE conversation_history
                    SET user_feedback = ?, feedback_score = ?
                    WHERE interaction_id = ?
                """,
                    (feedback_text, feedback_score, interaction_id),
                )

                await self.db.commit()

                logger.debug(
                    f"Added feedback to interaction {interaction_id}: score={feedback_score}"
                )

                return True

        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
            return False

    async def _generate_conversation_embeddings(
        self, interaction_id: int, embedding_id: str, query: str, response: str
    ):
        """Generate semantic embeddings for conversation search (async background task)"""
        try:
            # Import embedding model (lazy load)
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")

            # Generate embeddings
            query_embedding = model.encode(query)
            response_embedding = model.encode(response)
            combined_embedding = model.encode(f"{query} {response}")

            # Store embeddings
            async with self.db.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO conversation_embeddings
                    (embedding_id, interaction_id, query_embedding, response_embedding, combined_embedding)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        embedding_id,
                        interaction_id,
                        query_embedding.tobytes(),
                        response_embedding.tobytes(),
                        combined_embedding.tobytes(),
                    ),
                )

                await self.db.commit()

        except ImportError:
            logger.debug("sentence-transformers not installed - skipping embeddings")
        except Exception as e:
            logger.error(f"Failed to generate conversation embeddings: {e}")

    async def _extract_correction_pattern(self, correction_id: int):
        """Extract learning pattern from user correction (ML-powered)"""
        try:
            async with self.db.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT ic.*, ch.user_query, ch.context_snapshot
                    FROM interaction_corrections ic
                    JOIN conversation_history ch ON ic.interaction_id = ch.interaction_id
                    WHERE ic.correction_id = ?
                """,
                    (correction_id,),
                )

                row = await cursor.fetchone()
                if not row:
                    return

                # Extract pattern from correction
                # This is where ML/LLM could analyze what went wrong and create a rule
                # For now, store basic pattern
                pattern_text = f"When user says '{row['user_query']}', respond with '{row['corrected_response']}' not '{row['original_response']}'"

                await cursor.execute(
                    """
                    UPDATE interaction_corrections
                    SET pattern_extracted = ?, learned = 1
                    WHERE correction_id = ?
                """,
                    (pattern_text, correction_id),
                )

                await self.db.commit()

                logger.info(f"Extracted pattern from correction {correction_id}")

        except Exception as e:
            logger.error(f"Failed to extract correction pattern: {e}")

    async def search_similar_conversations(
        self, query: str, top_k: int = 5, min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search past conversations semantically using embeddings.

        Args:
            query: Search query
            top_k: Number of results to return
            min_confidence: Minimum similarity score

        Returns:
            List of similar past interactions with scores
        """
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_embedding = model.encode(query)

            # Get all conversation embeddings
            async with self.db.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT ce.*, ch.user_query, ch.jarvis_response, ch.success, ch.timestamp
                    FROM conversation_embeddings ce
                    JOIN conversation_history ch ON ce.interaction_id = ch.interaction_id
                """
                )

                rows = await cursor.fetchall()

            # Calculate similarity scores
            results = []
            for row in rows:
                combined_embedding = np.frombuffer(row["combined_embedding"], dtype=np.float32)

                # Cosine similarity
                similarity = np.dot(query_embedding, combined_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(combined_embedding)
                )

                if similarity >= min_confidence:
                    results.append(
                        {
                            "interaction_id": row["interaction_id"],
                            "user_query": row["user_query"],
                            "jarvis_response": row["jarvis_response"],
                            "success": row["success"],
                            "timestamp": row["timestamp"],
                            "similarity_score": float(similarity),
                        }
                    )

            # Sort by similarity and return top_k
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:top_k]

        except ImportError:
            logger.warning("sentence-transformers not installed - semantic search unavailable")
            return []
        except Exception as e:
            logger.error(f"Failed to search conversations: {e}")
            return []

    # ==================== Voice Recognition & Speaker Learning ====================

    async def record_voice_transcription(
        self,
        audio_data: bytes,
        transcribed_text: str,
        confidence_score: float,
        audio_duration_ms: float,
        interaction_id: Optional[int] = None,
    ) -> int:
        """
        Record voice transcription for accuracy tracking and learning.
        Tracks misheards, retries, and acoustic patterns for continuous improvement.
        """
        try:
            import hashlib

            audio_hash = hashlib.sha256(audio_data).hexdigest()[:16]

            async with self.db.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO voice_transcriptions
                    (interaction_id, raw_audio_hash, transcribed_text, confidence_score,
                     audio_duration_ms, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        interaction_id,
                        audio_hash,
                        transcribed_text,
                        confidence_score,
                        audio_duration_ms,
                        datetime.now(),
                    ),
                )

                await self.db.commit()
                transcription_id = cursor.lastrowid

                logger.debug(
                    f"🎤 Recorded voice transcription {transcription_id}: '{transcribed_text[:50]}...'"
                )

                return transcription_id

        except Exception as e:
            logger.error(f"Failed to record voice transcription: {e}")
            return -1

    async def record_misheard_query(
        self,
        transcription_id: int,
        what_jarvis_heard: str,
        what_user_meant: str,
        correction_method: str = "user_correction",
    ) -> int:
        """
        Record when JARVIS mishears - critical for learning!
        This trains the acoustic model to handle Derek's accent better.
        """
        try:
            async with self.db.cursor() as cursor:
                # Mark original transcription as misheard
                await cursor.execute(
                    """
                    UPDATE voice_transcriptions
                    SET was_misheard = 1, corrected_text = ?
                    WHERE transcription_id = ?
                """,
                    (what_user_meant, transcription_id),
                )

                # Calculate phonetic distance
                phonetic_distance = self._calculate_phonetic_distance(
                    what_jarvis_heard, what_user_meant
                )

                # Insert misheard record
                await cursor.execute(
                    """
                    INSERT INTO misheard_queries
                    (transcription_id, what_jarvis_heard, what_user_meant,
                     correction_method, phonetic_distance, occurred_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        transcription_id,
                        what_jarvis_heard,
                        what_user_meant,
                        correction_method,
                        phonetic_distance,
                        datetime.now(),
                    ),
                )

                await self.db.commit()
                misheard_id = cursor.lastrowid

                logger.info(
                    f"🎤 MISHEARD: Heard '{what_jarvis_heard}' but user meant '{what_user_meant}' (distance={phonetic_distance})"
                )

                # Trigger acoustic adaptation learning
                asyncio.create_task(self._learn_acoustic_pattern(misheard_id))

                return misheard_id

        except Exception as e:
            logger.error(f"Failed to record misheard query: {e}")
            return -1

    async def get_or_create_speaker_profile(self, speaker_name: str = "Derek J. Russell") -> int:
        """
        Get or create speaker profile for voice recognition.
        Derek J. Russell is the primary user.
        """
        try:
            db = self._ensure_db_initialized()
            # Check if using Cloud SQL (PostgreSQL) or SQLite
            is_cloud = isinstance(db, DatabaseConnectionWrapper) and db.is_cloud

            async with db.cursor() as cursor:
                # Check if profile exists
                if is_cloud:
                    await cursor.execute(
                        "SELECT speaker_id FROM speaker_profiles WHERE speaker_name = %s",
                        (speaker_name,),
                    )
                else:
                    await cursor.execute(
                        "SELECT speaker_id FROM speaker_profiles WHERE speaker_name = ?",
                        (speaker_name,),
                    )
                row = await cursor.fetchone()

                if row:
                    return row["speaker_id"] if isinstance(row, dict) else row[0]

                # Create new profile
                if is_cloud:
                    await cursor.execute(
                        "INSERT INTO speaker_profiles (speaker_name) VALUES (%s) RETURNING speaker_id",
                        (speaker_name,),
                    )
                    row = await cursor.fetchone()
                    speaker_id = row["speaker_id"] if isinstance(row, dict) else row[0]
                else:
                    await cursor.execute(
                        "INSERT INTO speaker_profiles (speaker_name) VALUES (?)",
                        (speaker_name,),
                    )
                    speaker_id = cursor.lastrowid

                await db.commit()

                logger.info(f"👤 Created speaker profile for {speaker_name} (ID: {speaker_id})")

                return speaker_id

        except Exception as e:
            logger.error(f"Failed to get/create speaker profile: {e}")
            return -1

    async def record_voice_sample(
        self,
        speaker_name: str,
        audio_data: bytes,
        transcription: str,
        audio_duration_ms: float,
        quality_score: float = 1.0,
    ) -> int:
        """
        Record voice sample for speaker training (Derek's voiceprint).
        Collects acoustic features for voice recognition and adaptation.
        """
        try:
            import hashlib

            speaker_id = await self.get_or_create_speaker_profile(speaker_name)
            audio_hash = hashlib.sha256(audio_data).hexdigest()[:16]

            # Extract acoustic features (if librosa available)
            mfcc_features = None
            pitch_mean = None
            pitch_std = None
            energy_mean = None

            try:
                import io

                import librosa
                import numpy as np

                # Convert bytes to audio
                audio_array, sr = librosa.load(io.BytesIO(audio_data), sr=16000)

                # Extract MFCC features (13 coefficients)
                mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=13)
                mfcc_features = np.mean(mfccs, axis=1).tobytes()

                # Extract pitch
                pitches, magnitudes = librosa.piptrack(y=audio_array, sr=sr)
                pitch_values = pitches[pitches > 0]
                if len(pitch_values) > 0:
                    pitch_mean = float(np.mean(pitch_values))
                    pitch_std = float(np.std(pitch_values))

                # Extract energy
                energy = librosa.feature.rms(y=audio_array)
                energy_mean = float(np.mean(energy))

                logger.debug(
                    f"🎵 Extracted acoustic features: pitch={pitch_mean:.1f}Hz, energy={energy_mean:.3f}"
                )

            except ImportError:
                logger.debug("librosa not available - storing audio hash only")
            except Exception as e:
                logger.warning(f"Failed to extract acoustic features: {e}")

            async with self.db.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO voice_samples
                    (speaker_id, audio_hash, mfcc_features, duration_ms, transcription,
                     pitch_mean, pitch_std, energy_mean, recording_timestamp, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        speaker_id,
                        audio_hash,
                        mfcc_features,
                        audio_duration_ms,
                        transcription,
                        pitch_mean,
                        pitch_std,
                        energy_mean,
                        datetime.now(),
                        quality_score,
                    ),
                )

                # Update speaker profile stats
                await cursor.execute(
                    """
                    UPDATE speaker_profiles
                    SET total_samples = total_samples + 1,
                        average_pitch_hz = COALESCE(
                            (average_pitch_hz * total_samples + ?) / (total_samples + 1),
                            ?
                        ),
                        last_updated = ?
                    WHERE speaker_id = ?
                """,
                    (pitch_mean, pitch_mean, datetime.now(), speaker_id),
                )

                await self.db.commit()
                sample_id = cursor.lastrowid

                logger.debug(f"🎤 Recorded voice sample {sample_id} for {speaker_name}")

                return sample_id

        except Exception as e:
            logger.error(f"Failed to record voice sample: {e}")
            return -1

    async def record_query_retry(
        self,
        original_transcription_id: int,
        retry_transcription_id: int,
        retry_number: int,
        time_between_ms: float,
    ) -> int:
        """
        Record when user has to repeat a query.
        Critical for identifying problematic words/phrases that need acoustic tuning.
        """
        try:
            async with self.db.cursor() as cursor:
                # Update retry count on both transcriptions
                await cursor.execute(
                    """
                    UPDATE voice_transcriptions
                    SET retry_count = retry_count + 1
                    WHERE transcription_id IN (?, ?)
                """,
                    (original_transcription_id, retry_transcription_id),
                )

                # Insert retry record
                await cursor.execute(
                    """
                    INSERT INTO query_retries
                    (original_transcription_id, retry_transcription_id, retry_number,
                     time_between_retries_ms, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        original_transcription_id,
                        retry_transcription_id,
                        retry_number,
                        time_between_ms,
                        datetime.now(),
                    ),
                )

                await self.db.commit()
                retry_id = cursor.lastrowid

                logger.warning(f"🔁 Query retry #{retry_number} detected (gap={time_between_ms}ms)")

                return retry_id

        except Exception as e:
            logger.error(f"Failed to record query retry: {e}")
            return -1

    async def _learn_acoustic_pattern(self, misheard_id: int):
        """
        Extract acoustic learning pattern from misheard query.
        Adapts to Derek's pronunciation patterns and accent.
        """
        try:
            async with self.db.cursor() as cursor:
                # Get misheard details
                await cursor.execute(
                    """
                    SELECT mq.*, vt.confidence_score, vt.audio_duration_ms
                    FROM misheard_queries mq
                    JOIN voice_transcriptions vt ON mq.transcription_id = vt.transcription_id
                    WHERE mq.misheard_id = ?
                """,
                    (misheard_id,),
                )

                row = await cursor.fetchone()
                if not row:
                    return

                # Extract phoneme patterns
                what_heard = row["what_jarvis_heard"]
                what_meant = row["what_user_meant"]

                # Simple pattern: identify word-level differences
                heard_words = what_heard.lower().split()
                meant_words = what_meant.lower().split()

                # Find mismatched words
                for i, (heard, meant) in enumerate(zip(heard_words, meant_words)):
                    if heard != meant:
                        # Store acoustic adaptation
                        speaker_id = await self.get_or_create_speaker_profile("Derek J. Russell")

                        await cursor.execute(
                            """
                            INSERT INTO acoustic_adaptations
                            (speaker_id, phoneme_pattern, expected_pronunciation, actual_pronunciation, context_words)
                            VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT (speaker_id, phoneme_pattern) DO UPDATE
                            SET frequency_count = frequency_count + 1,
                                last_observed = ?
                        """,
                            (
                                speaker_id,
                                heard,
                                heard,
                                meant,
                                json.dumps({"position": i, "full_query": what_meant}),
                                datetime.now(),
                            ),
                        )

                # Update misheard query with learned pattern
                pattern_text = f"'{what_heard}' -> '{what_meant}'"
                await cursor.execute(
                    """
                    UPDATE misheard_queries
                    SET learned_pattern = ?
                    WHERE misheard_id = ?
                """,
                    (pattern_text, misheard_id),
                )

                await self.db.commit()

                logger.info(f"🧠 Learned acoustic pattern from misheard query: {pattern_text}")

        except Exception as e:
            logger.error(f"Failed to learn acoustic pattern: {e}")

    def _calculate_phonetic_distance(self, str1: str, str2: str) -> int:
        """Calculate Levenshtein distance for phonetic similarity"""
        if str1 == str2:
            return 0

        if len(str1) == 0:
            return len(str2)
        if len(str2) == 0:
            return len(str1)

        # Create distance matrix
        matrix = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]

        # Initialize first row and column
        for i in range(len(str1) + 1):
            matrix[i][0] = i
        for j in range(len(str2) + 1):
            matrix[0][j] = j

        # Calculate distances
        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                cost = 0 if str1[i - 1] == str2[j - 1] else 1
                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,  # deletion
                    matrix[i][j - 1] + 1,  # insertion
                    matrix[i - 1][j - 1] + cost,  # substitution
                )

        return matrix[len(str1)][len(str2)]

    # ==================== Pattern Learning (ML-Powered) ====================

    async def store_pattern(self, pattern: Dict[str, Any], auto_merge: bool = True) -> str:
        """Store pattern with automatic similarity-based merging"""
        pattern_id = pattern.get("pattern_id", self._generate_id("pattern"))
        pattern_hash = self._hash_pattern(pattern)

        # Check cache first
        cached_pattern = self.pattern_cache.get(pattern_hash)
        if cached_pattern:
            # Update occurrence count
            await self._increment_pattern_occurrence(cached_pattern["pattern_id"])
            return cached_pattern["pattern_id"]

        # Check for similar patterns if auto_merge enabled
        if auto_merge and "embedding" in pattern:
            similar = await self._find_similar_patterns(
                pattern["embedding"], pattern["pattern_type"], similarity_threshold=0.85
            )

            if similar:
                # Merge with most similar pattern
                await self._merge_patterns(similar[0]["pattern_id"], pattern)
                return similar[0]["pattern_id"]

        # Store new pattern
        async with self._db_lock:
            async with self.db.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT OR REPLACE INTO patterns
                    (pattern_id, pattern_type, pattern_hash, pattern_data, confidence,
                     success_rate, occurrence_count, first_seen, last_seen, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        pattern_id,
                        pattern["pattern_type"],
                        pattern_hash,
                        json.dumps(pattern.get("pattern_data", {})),
                        pattern.get("confidence", 0.5),
                        pattern.get("success_rate", 0.5),
                        1,
                        datetime.now(),
                        datetime.now(),
                        json.dumps(pattern.get("metadata", {})),
                    ),
                )

            await self.db.commit()

        # Store embedding
        if self.pattern_collection and "embedding" in pattern:
            await self._store_embedding_async(
                self.pattern_collection,
                pattern_id,
                pattern["embedding"],
                {
                    "pattern_type": pattern["pattern_type"],
                    "confidence": pattern.get("confidence", 0.5),
                    "timestamp": datetime.now().isoformat(),
                },
            )

        # Update cache
        self.pattern_cache.set(pattern_hash, {"pattern_id": pattern_id, **pattern})

        return pattern_id

    async def _find_similar_patterns(
        self, embedding: List[float], pattern_type: str, similarity_threshold: float = 0.8
    ) -> List[Dict]:
        """Find similar patterns using embeddings"""
        if not self.pattern_collection:
            return []

        try:
            results = self.pattern_collection.query(
                query_embeddings=[embedding], n_results=5, where={"pattern_type": pattern_type}
            )

            similar_patterns = []
            async with self.db.cursor() as cursor:
                for i, pattern_id in enumerate(results["ids"][0]):
                    similarity = 1.0 - results["distances"][0][i]
                    if similarity >= similarity_threshold:
                        await cursor.execute(
                            "SELECT * FROM patterns WHERE pattern_id = ?", (pattern_id,)
                        )
                        row = await cursor.fetchone()
                        if row:
                            pattern = dict(row)
                            pattern["similarity"] = similarity
                            similar_patterns.append(pattern)

            return similar_patterns

        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return []

    async def _merge_patterns(self, target_pattern_id: str, new_pattern: Dict):
        """Merge new pattern into existing pattern"""
        async with self._db_lock:
            async with self.db.cursor() as cursor:
                # Update occurrence count and confidence
                await cursor.execute(
                    """
                    UPDATE patterns SET
                        occurrence_count = occurrence_count + 1,
                        confidence = (confidence + ?) / 2,
                        success_rate = (success_rate + ?) / 2,
                        last_seen = ?,
                        boost_count = boost_count + 1
                    WHERE pattern_id = ?
                """,
                    (
                        new_pattern.get("confidence", 0.5),
                        new_pattern.get("success_rate", 0.5),
                        datetime.now(),
                        target_pattern_id,
                    ),
                )

            await self.db.commit()

    async def _increment_pattern_occurrence(self, pattern_id: str):
        """Increment pattern occurrence count"""
        async with self._db_lock:
            async with self.db.cursor() as cursor:
                await cursor.execute(
                    """
                    UPDATE patterns SET
                        occurrence_count = occurrence_count + 1,
                        last_seen = ?
                    WHERE pattern_id = ?
                """,
                    (datetime.now(), pattern_id),
                )

            await self.db.commit()

    async def get_pattern_by_type(
        self, pattern_type: str, min_confidence: float = 0.5, limit: int = 10
    ) -> List[Dict]:
        """Get patterns by type with caching"""
        cache_key = f"patterns_{pattern_type}_{min_confidence}_{limit}"

        cached = self.query_cache.get(cache_key)
        if cached:
            return cached

        async with self.db.cursor() as cursor:
            await cursor.execute(
                """
                SELECT * FROM patterns
                WHERE pattern_type = ? AND confidence >= ?
                ORDER BY confidence DESC, occurrence_count DESC
                LIMIT ?
            """,
                (pattern_type, min_confidence, limit),
            )

            rows = await cursor.fetchall()
            patterns = [dict(row) for row in rows]

        self.query_cache.set(cache_key, patterns)
        return patterns

    # ==================== Display Patterns (Enhanced) ====================

    async def learn_display_pattern(self, display_name: str, context: Dict[str, Any]):
        """Learn display connection patterns with temporal analysis"""
        now = datetime.now()
        time_str = now.strftime("%H:%M")
        day_of_week = now.weekday()
        hour_of_day = now.hour
        context_hash = self._hash_context(context)

        async with self._db_lock:
            async with self.db.cursor() as cursor:
                # Check if similar pattern exists
                await cursor.execute(
                    """
                    SELECT * FROM display_patterns
                    WHERE display_name = ?
                    AND hour_of_day = ?
                    AND day_of_week = ?
                """,
                    (display_name, hour_of_day, day_of_week),
                )

                existing = await cursor.fetchone()

                if existing:
                    # Update frequency and consecutive successes
                    await cursor.execute(
                        """
                        UPDATE display_patterns SET
                            frequency = frequency + 1,
                            consecutive_successes = consecutive_successes + 1,
                            last_seen = ?,
                            auto_connect = CASE WHEN frequency >= 3 THEN 1 ELSE auto_connect END,
                            context = ?
                        WHERE pattern_id = ?
                    """,
                        (datetime.now(), json.dumps(context), existing["pattern_id"]),
                    )

                    if existing["frequency"] >= 2:
                        logger.info(
                            f"📊 Display pattern strengthened: {display_name} at {hour_of_day}:00 on day {day_of_week}"
                        )
                else:
                    # Insert new pattern
                    await cursor.execute(
                        """
                        INSERT INTO display_patterns
                        (display_name, context, context_hash, connection_time, day_of_week,
                         hour_of_day, frequency, auto_connect, last_seen, consecutive_successes, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            display_name,
                            json.dumps(context),
                            context_hash,
                            time_str,
                            day_of_week,
                            hour_of_day,
                            1,
                            False,
                            datetime.now(),
                            1,
                            json.dumps({}),
                        ),
                    )

            await self.db.commit()

    async def should_auto_connect_display(
        self, current_context: Optional[Dict] = None
    ) -> Optional[Tuple[str, float]]:
        """Predict display connection with context awareness"""
        now = datetime.now()
        hour_of_day = now.hour
        day_of_week = now.weekday()

        async with self.db.cursor() as cursor:
            # Look for matching patterns with temporal proximity
            await cursor.execute(
                """
                SELECT display_name, frequency, consecutive_successes, auto_connect
                FROM display_patterns
                WHERE hour_of_day = ?
                AND day_of_week = ?
                AND frequency >= 2
                ORDER BY frequency DESC, consecutive_successes DESC
                LIMIT 1
            """,
                (hour_of_day, day_of_week),
            )

            row = await cursor.fetchone()

            if row:
                # Calculate dynamic confidence
                base_confidence = min(row["frequency"] / 10.0, 0.85)
                consecutive_bonus = min(row["consecutive_successes"] * 0.05, 0.10)
                confidence = min(base_confidence + consecutive_bonus, 0.95)

                return row["display_name"], confidence

        return None

    # ==================== User Preferences (Enhanced) ====================

    async def learn_preference(
        self,
        category: str,
        key: str,
        value: Any,
        confidence: float = 0.5,
        learned_from: str = "implicit",
    ):
        """Learn user preference with confidence averaging"""
        async with self._db_lock:
            async with self.db.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO user_preferences
                    (preference_id, category, key, value, confidence,
                     learned_from, created_at, updated_at, update_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(category, key) DO UPDATE SET
                        value = ?,
                        confidence = (confidence * update_count + ?) / (update_count + 1),
                        update_count = update_count + 1,
                        updated_at = ?
                """,
                    (
                        f"{category}_{key}",
                        category,
                        key,
                        str(value),
                        confidence,
                        learned_from,
                        datetime.now(),
                        datetime.now(),
                        1,
                        str(value),
                        confidence,
                        datetime.now(),
                    ),
                )

            await self.db.commit()

    async def get_preference(
        self, category: str, key: str, min_confidence: float = 0.5
    ) -> Optional[Dict]:
        """Get learned preference with confidence threshold"""
        cache_key = f"pref_{category}_{key}"

        cached = self.query_cache.get(cache_key)
        if cached:
            return cached

        async with self.db.cursor() as cursor:
            await cursor.execute(
                """
                SELECT * FROM user_preferences
                WHERE category = ? AND key = ? AND confidence >= ?
            """,
                (category, key, min_confidence),
            )

            row = await cursor.fetchone()
            result = dict(row) if row else None

            if result:
                self.query_cache.set(cache_key, result)

            return result

    # ==================== Analytics & Metrics ====================

    async def get_learning_metrics(self) -> Dict[str, Any]:
        """Get comprehensive real-time learning metrics"""
        metrics = {}

        # Check if using Cloud SQL (PostgreSQL) or SQLite
        is_cloud = isinstance(self.db, DatabaseConnectionWrapper) and self.db.is_cloud

        async with self.db.cursor() as cursor:
            # Goal metrics
            if is_cloud:
                await cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total_goals,
                        AVG(confidence) as avg_confidence,
                        SUM(CASE WHEN is_completed THEN 1 ELSE 0 END) as completed_goals,
                        AVG(actual_duration) as avg_duration
                    FROM goals
                    WHERE created_at >= NOW() - INTERVAL '30 days'
                """
                )
            else:
                await cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total_goals,
                        AVG(confidence) as avg_confidence,
                        SUM(is_completed) as completed_goals,
                        AVG(actual_duration) as avg_duration
                    FROM goals
                    WHERE created_at >= datetime('now', '-30 days')
                """
                )
            metrics["goals"] = dict(await cursor.fetchone())

            # Action metrics
            if is_cloud:
                await cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total_actions,
                        AVG(confidence) as avg_confidence,
                        SUM(CASE WHEN success THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                        AVG(execution_time) as avg_execution_time,
                        AVG(retry_count) as avg_retries
                    FROM actions
                    WHERE timestamp >= NOW() - INTERVAL '30 days'
                """
                )
            else:
                await cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total_actions,
                        AVG(confidence) as avg_confidence,
                        SUM(success) * 100.0 / COUNT(*) as success_rate,
                        AVG(execution_time) as avg_execution_time,
                        AVG(retry_count) as avg_retries
                    FROM actions
                    WHERE timestamp >= datetime('now', '-30 days')
                """
                )
            metrics["actions"] = dict(await cursor.fetchone())

            # Pattern metrics
            await cursor.execute(
                """
                SELECT
                    COUNT(*) as total_patterns,
                    AVG(confidence) as avg_confidence,
                    AVG(success_rate) as avg_success_rate,
                    SUM(occurrence_count) as total_occurrences,
                    AVG(occurrence_count) as avg_occurrences_per_pattern
                FROM patterns
            """
            )
            metrics["patterns"] = dict(await cursor.fetchone())

            # Display pattern metrics
            if is_cloud:
                await cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total_display_patterns,
                        SUM(CASE WHEN auto_connect THEN 1 ELSE 0 END) as auto_connect_enabled,
                        MAX(frequency) as max_frequency,
                        AVG(consecutive_successes) as avg_consecutive_successes
                    FROM display_patterns
                """
                )
            else:
                await cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total_display_patterns,
                        SUM(auto_connect) as auto_connect_enabled,
                        MAX(frequency) as max_frequency,
                        AVG(consecutive_successes) as avg_consecutive_successes
                    FROM display_patterns
                """
                )
            metrics["display_patterns"] = dict(await cursor.fetchone())

            # Goal-action mapping insights
            await cursor.execute(
                """
                SELECT
                    goal_type,
                    action_type,
                    confidence,
                    success_count + failure_count as total_attempts,
                    avg_execution_time
                FROM goal_action_mappings
                WHERE confidence > 0.7
                ORDER BY confidence DESC
                LIMIT 10
            """
            )
            metrics["top_mappings"] = [dict(row) for row in await cursor.fetchall()]

        # Add cache metrics
        metrics["cache_performance"] = {
            "pattern_cache_hit_rate": self.pattern_cache.hit_rate(),
            "goal_cache_hit_rate": self.goal_cache.hit_rate(),
            "query_cache_hit_rate": self.query_cache.hit_rate(),
        }

        # Update internal metrics
        self.metrics.total_patterns = metrics["patterns"]["total_patterns"]
        self.metrics.avg_confidence = metrics["patterns"]["avg_confidence"] or 0.0
        self.metrics.cache_hit_rate = metrics["cache_performance"]["pattern_cache_hit_rate"]
        self.metrics.last_updated = datetime.now()

        return metrics

    async def analyze_patterns(self) -> List[Dict]:
        """Advanced pattern analysis with ML insights"""
        patterns = []

        async with self.db.cursor() as cursor:
            await cursor.execute(
                """
                SELECT *
                FROM patterns
                WHERE occurrence_count >= 3
                ORDER BY success_rate DESC, occurrence_count DESC
                LIMIT 50
            """
            )

            rows = await cursor.fetchall()

            for row in rows:
                pattern = dict(row)

                # Calculate pattern strength score
                strength = (
                    pattern["confidence"] * 0.4
                    + pattern["success_rate"] * 0.4
                    + min(pattern["occurrence_count"] / 100.0, 1.0) * 0.2
                )
                pattern["strength_score"] = strength

                # Time since last seen
                last_seen = datetime.fromisoformat(pattern["last_seen"])
                days_since = (datetime.now() - last_seen).days
                pattern["days_since_last_seen"] = days_since

                # Decay recommendation
                if days_since > 30 and pattern["occurrence_count"] < 5:
                    pattern["should_decay"] = True
                else:
                    pattern["should_decay"] = False

                patterns.append(pattern)

        return patterns

    async def boost_pattern_confidence(
        self,
        pattern_id: str,
        boost: float = 0.05,
        strategy: ConfidenceBoostStrategy = ConfidenceBoostStrategy.ADAPTIVE,
    ):
        """Boost pattern confidence using configurable strategy"""
        async with self._db_lock:
            async with self.db.cursor() as cursor:
                await cursor.execute(
                    "SELECT confidence, boost_count FROM patterns WHERE pattern_id = ?",
                    (pattern_id,),
                )
                row = await cursor.fetchone()

                if row:
                    current_confidence = row["confidence"]
                    boost_count = row["boost_count"]

                    # Apply strategy
                    if strategy == ConfidenceBoostStrategy.LINEAR:
                        new_confidence = min(current_confidence + boost, 1.0)
                    elif strategy == ConfidenceBoostStrategy.EXPONENTIAL:
                        new_confidence = min(current_confidence * (1.0 + boost), 1.0)
                    elif strategy == ConfidenceBoostStrategy.LOGARITHMIC:
                        new_confidence = min(current_confidence + boost / (1 + boost_count), 1.0)
                    else:  # ADAPTIVE
                        # Reduces boost as confidence increases
                        adaptive_boost = boost * (1.0 - current_confidence)
                        new_confidence = min(current_confidence + adaptive_boost, 1.0)

                    await cursor.execute(
                        """
                        UPDATE patterns SET
                            confidence = ?,
                            boost_count = boost_count + 1
                        WHERE pattern_id = ?
                    """,
                        (new_confidence, pattern_id),
                    )

                await self.db.commit()

    # ==================== Maintenance & Optimization ====================

    async def cleanup_old_patterns(self, days: int = 30):
        """Clean up old unused patterns with decay"""
        cutoff_date = datetime.now() - timedelta(days=days)

        async with self._db_lock:
            async with self.db.cursor() as cursor:
                # Decay old patterns instead of deleting
                await cursor.execute(
                    """
                    UPDATE patterns SET
                        confidence = confidence * 0.9,
                        decay_applied = 1
                    WHERE last_seen < ? AND occurrence_count < 3
                """,
                    (cutoff_date,),
                )

                # Delete very old patterns with low occurrence
                await cursor.execute(
                    """
                    DELETE FROM patterns
                    WHERE last_seen < ? AND occurrence_count = 1 AND confidence < 0.3
                """,
                    (cutoff_date,),
                )

                deleted_count = cursor.rowcount

            await self.db.commit()

        logger.info(f"🧹 Cleaned up {deleted_count} old patterns")

    async def optimize(self):
        """Optimize database performance"""
        async with self._db_lock:
            async with self.db.cursor() as cursor:
                await cursor.execute("ANALYZE")
                await cursor.execute("VACUUM")

            await self.db.commit()

        logger.info("✨ Database optimized")

    async def _auto_flush_batches(self):
        """Auto-flush batch queues periodically"""
        while True:
            await asyncio.sleep(5)  # Flush every 5 seconds

            if self.pending_goals:
                await self._flush_goal_batch()

            if self.pending_actions:
                await self._flush_action_batch()

            if self.pending_patterns:
                await self._flush_pattern_batch()

    async def _flush_goal_batch(self):
        """Flush pending goals"""
        if not self.pending_goals:
            return

        goals_to_insert = list(self.pending_goals)
        self.pending_goals.clear()

        async with self._db_lock:
            async with self.db.cursor() as cursor:
                await cursor.executemany(
                    """
                    INSERT OR REPLACE INTO goals
                    (goal_id, goal_type, goal_level, description, confidence,
                     progress, is_completed, created_at, evidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        (
                            goal_id,
                            goal["goal_type"],
                            goal.get("goal_level", "UNKNOWN"),
                            goal.get("description", ""),
                            goal.get("confidence", 0.0),
                            goal.get("progress", 0.0),
                            goal.get("is_completed", False),
                            goal.get("created_at", datetime.now()),
                            json.dumps(goal.get("evidence", [])),
                            json.dumps(goal.get("metadata", {})),
                        )
                        for goal_id, goal in goals_to_insert
                    ],
                )

            await self.db.commit()

    async def _flush_action_batch(self):
        """Flush pending actions"""
        if not self.pending_actions:
            return

        actions_to_insert = list(self.pending_actions)
        self.pending_actions.clear()

        async with self._db_lock:
            async with self.db.cursor() as cursor:
                await cursor.executemany(
                    """
                    INSERT INTO actions
                    (action_id, action_type, target, goal_id, confidence,
                     success, execution_time, timestamp, params, result)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        (
                            action_id,
                            action["action_type"],
                            action.get("target", ""),
                            action.get("goal_id"),
                            action.get("confidence", 0.0),
                            action.get("success", False),
                            action.get("execution_time", 0.0),
                            action.get("timestamp", datetime.now()),
                            json.dumps(action.get("params", {})),
                            json.dumps(action.get("result", {})),
                        )
                        for action_id, action in actions_to_insert
                    ],
                )

            await self.db.commit()

    async def _flush_pattern_batch(self):
        """Flush pending patterns with intelligent deduplication and merging"""
        if not self.pending_patterns:
            return

        patterns_to_process = list(self.pending_patterns)
        self.pending_patterns.clear()

        # Group patterns by hash for intelligent merging
        pattern_groups: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)

        for pattern_id, pattern in patterns_to_process:
            pattern_hash = self._hash_pattern(pattern)
            pattern_groups[pattern_hash].append((pattern_id, pattern))

        async with self._db_lock:
            async with self.db.cursor() as cursor:
                # Process each group
                for pattern_hash, group in pattern_groups.items():
                    if len(group) == 1:
                        # Single pattern - insert directly
                        pattern_id, pattern = group[0]

                        await cursor.execute(
                            """
                            INSERT OR REPLACE INTO patterns
                            (pattern_id, pattern_type, pattern_hash, pattern_data, confidence,
                             success_rate, occurrence_count, first_seen, last_seen, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                pattern_id,
                                pattern["pattern_type"],
                                pattern_hash,
                                json.dumps(pattern.get("pattern_data", {})),
                                pattern.get("confidence", 0.5),
                                pattern.get("success_rate", 0.5),
                                1,
                                datetime.now(),
                                datetime.now(),
                                json.dumps(pattern.get("metadata", {})),
                            ),
                        )

                        # Store embedding if available
                        if self.pattern_collection and "embedding" in pattern:
                            await self._store_embedding_async(
                                self.pattern_collection,
                                pattern_id,
                                pattern["embedding"],
                                {
                                    "pattern_type": pattern["pattern_type"],
                                    "confidence": pattern.get("confidence", 0.5),
                                    "timestamp": datetime.now().isoformat(),
                                },
                            )
                    else:
                        # Multiple patterns with same hash - merge intelligently
                        # Use the first pattern as base
                        base_pattern_id, base_pattern = group[0]

                        # Calculate merged confidence (weighted average)
                        total_confidence = sum(p.get("confidence", 0.5) for _, p in group)
                        avg_confidence = total_confidence / len(group)

                        # Calculate merged success rate
                        total_success_rate = sum(p.get("success_rate", 0.5) for _, p in group)
                        avg_success_rate = total_success_rate / len(group)

                        # Merge metadata intelligently
                        merged_metadata = {}
                        for _, pattern in group:
                            pattern_meta = pattern.get("metadata", {})
                            for key, value in pattern_meta.items():
                                if key not in merged_metadata:
                                    merged_metadata[key] = []
                                if value not in merged_metadata[key]:
                                    merged_metadata[key].append(value)

                        # Flatten single-value lists
                        for key in merged_metadata:
                            if len(merged_metadata[key]) == 1:
                                merged_metadata[key] = merged_metadata[key][0]

                        # Insert merged pattern
                        await cursor.execute(
                            """
                            INSERT OR REPLACE INTO patterns
                            (pattern_id, pattern_type, pattern_hash, pattern_data, confidence,
                             success_rate, occurrence_count, first_seen, last_seen, boost_count, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                base_pattern_id,
                                base_pattern["pattern_type"],
                                pattern_hash,
                                json.dumps(base_pattern.get("pattern_data", {})),
                                avg_confidence,
                                avg_success_rate,
                                len(group),  # Number of merged patterns
                                datetime.now(),
                                datetime.now(),
                                len(group) - 1,  # Boost count from merging
                                json.dumps(merged_metadata),
                            ),
                        )

                        # Store embedding for merged pattern
                        if self.pattern_collection and "embedding" in base_pattern:
                            # Average all embeddings if multiple available
                            embeddings_to_merge = [
                                p.get("embedding") for _, p in group if "embedding" in p
                            ]

                            if embeddings_to_merge and NUMPY_AVAILABLE:
                                # Average embeddings
                                import numpy as np

                                avg_embedding = np.mean(embeddings_to_merge, axis=0).tolist()

                                await self._store_embedding_async(
                                    self.pattern_collection,
                                    base_pattern_id,
                                    avg_embedding,
                                    {
                                        "pattern_type": base_pattern["pattern_type"],
                                        "confidence": avg_confidence,
                                        "merged_count": len(group),
                                        "timestamp": datetime.now().isoformat(),
                                    },
                                )
                            elif embeddings_to_merge:
                                # Use first embedding if numpy not available
                                await self._store_embedding_async(
                                    self.pattern_collection,
                                    base_pattern_id,
                                    embeddings_to_merge[0],
                                    {
                                        "pattern_type": base_pattern["pattern_type"],
                                        "confidence": avg_confidence,
                                        "merged_count": len(group),
                                        "timestamp": datetime.now().isoformat(),
                                    },
                                )

                        # Update cache with merged pattern
                        self.pattern_cache.set(
                            pattern_hash,
                            {
                                "pattern_id": base_pattern_id,
                                "confidence": avg_confidence,
                                "success_rate": avg_success_rate,
                                "occurrence_count": len(group),
                            },
                        )

                        logger.debug(
                            f"🔀 Merged {len(group)} similar patterns into {base_pattern_id}"
                        )

            await self.db.commit()

        logger.debug(
            f"✅ Flushed {len(patterns_to_process)} patterns ({len(pattern_groups)} unique)"
        )

    async def _auto_optimize_task(self):
        """Auto-optimize database periodically"""
        if not self.auto_optimize:
            return

        while True:
            await asyncio.sleep(3600)  # Every hour
            await self.optimize()

    async def _load_metrics(self):
        """Load initial metrics"""
        try:
            metrics = await self.get_learning_metrics()
            logger.info(f"📊 Loaded metrics: {metrics['patterns']['total_patterns']} patterns")
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")

    async def _store_embedding_async(
        self, collection, doc_id: str, embedding: List[float], metadata: Dict
    ):
        """Store embedding asynchronously"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: collection.upsert(
                    ids=[doc_id], embeddings=[embedding], metadatas=[metadata]
                ),
            )
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")

    # ==================== Utility Methods ====================

    @staticmethod
    def _generate_id(prefix: str) -> str:
        """Generate unique ID"""
        timestamp = datetime.now().timestamp()
        return f"{prefix}_{timestamp}_{hashlib.md5(str(timestamp).encode(), usedforsecurity=False).hexdigest()[:8]}"

    @staticmethod
    def _hash_context(context: Dict) -> str:
        """Generate hash for context deduplication"""
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.sha256(context_str.encode()).hexdigest()[:16]

    @staticmethod
    def _hash_pattern(pattern: Dict) -> str:
        """Generate hash for pattern deduplication"""
        pattern_str = json.dumps(
            {"type": pattern.get("pattern_type"), "data": pattern.get("pattern_data")},
            sort_keys=True,
        )
        return hashlib.sha256(pattern_str.encode()).hexdigest()[:16]

    # ========================================================================
    # PHASE 3: BEHAVIORAL LEARNING METHODS
    # ========================================================================

    async def store_workspace_usage(
        self,
        space_id: int,
        app_name: str,
        window_title: Optional[str] = None,
        window_position: Optional[Dict] = None,
        focus_duration: Optional[float] = None,
        is_fullscreen: bool = False,
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        Store workspace usage event

        Args:
            space_id: Space/Desktop ID
            app_name: Application name
            window_title: Window title
            window_position: Window frame {x, y, w, h}
            focus_duration: Focus duration in seconds
            is_fullscreen: Whether window is fullscreen
            metadata: Additional metadata

        Returns:
            usage_id
        """
        now = datetime.now()

        try:
            async with self.db.execute(
                """
                INSERT INTO workspace_usage (
                    space_id, space_label, app_name, window_title,
                    window_position, focus_duration_seconds, timestamp,
                    day_of_week, hour_of_day, is_fullscreen, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    space_id,
                    None,  # space_label can be updated separately
                    app_name,
                    window_title,
                    json.dumps(window_position) if window_position else None,
                    focus_duration,
                    now.isoformat(),
                    now.weekday(),
                    now.hour,
                    is_fullscreen,
                    json.dumps(metadata) if metadata else None,
                ),
            ) as cursor:
                await self.db.commit()
                return cursor.lastrowid

        except Exception as e:
            logger.error(f"Error storing workspace usage: {e}", exc_info=True)
            raise

    async def update_app_usage_pattern(
        self,
        app_name: str,
        space_id: int,
        session_duration: float,
        timestamp: Optional[datetime] = None,
    ):
        """
        Update app usage pattern (incremental learning)

        Args:
            app_name: Application name
            space_id: Space ID
            session_duration: Session duration in seconds
            timestamp: Timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        hour = timestamp.hour
        day = timestamp.weekday()

        try:
            # Check if pattern exists
            async with self.db.execute(
                """
                SELECT pattern_id, usage_frequency, avg_session_duration,
                       total_usage_time, confidence
                FROM app_usage_patterns
                WHERE app_name = ? AND space_id = ?
                  AND typical_time_of_day = ? AND typical_day_of_week = ?
            """,
                (app_name, space_id, hour, day),
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                # Update existing pattern
                pattern_id, freq, avg_dur, total_time, conf = row
                new_freq = freq + 1
                new_total = total_time + session_duration
                new_avg = new_total / new_freq
                new_conf = min(0.99, conf + 0.02)  # Incremental confidence boost

                await self.db.execute(
                    """
                    UPDATE app_usage_patterns
                    SET usage_frequency = ?,
                        avg_session_duration = ?,
                        total_usage_time = ?,
                        last_used = ?,
                        confidence = ?
                    WHERE pattern_id = ?
                """,
                    (new_freq, new_avg, new_total, timestamp.isoformat(), new_conf, pattern_id),
                )

            else:
                # Create new pattern
                await self.db.execute(
                    """
                    INSERT INTO app_usage_patterns (
                        app_name, space_id, usage_frequency,
                        avg_session_duration, total_usage_time,
                        typical_time_of_day, typical_day_of_week,
                        last_used, confidence, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        app_name,
                        space_id,
                        1,
                        session_duration,
                        session_duration,
                        hour,
                        day,
                        timestamp.isoformat(),
                        0.3,
                        json.dumps({}),
                    ),
                )

            await self.db.commit()

        except Exception as e:
            logger.error(f"Error updating app usage pattern: {e}", exc_info=True)

    async def store_user_workflow(
        self,
        workflow_name: str,
        action_sequence: List[Dict],
        space_sequence: Optional[List[int]] = None,
        app_sequence: Optional[List[str]] = None,
        duration: Optional[float] = None,
        success: bool = True,
        confidence: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        Store or update user workflow pattern

        Args:
            workflow_name: Name/ID of workflow
            action_sequence: Sequence of actions
            space_sequence: Sequence of Spaces used
            app_sequence: Sequence of apps used
            duration: Duration in seconds
            success: Whether workflow succeeded
            confidence: Confidence score
            metadata: Additional metadata

        Returns:
            workflow_id
        """
        now = datetime.now()
        hour = now.hour

        try:
            # Check if workflow exists
            async with self.db.execute(
                """
                SELECT workflow_id, frequency, success_rate, time_of_day_pattern
                FROM user_workflows
                WHERE workflow_name = ?
            """,
                (workflow_name,),
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                # Update existing workflow
                wf_id, freq, success_rate, time_pattern = row
                new_freq = freq + 1

                # Update success rate
                new_success_rate = ((success_rate * freq) + (1.0 if success else 0.0)) / new_freq

                # Update time pattern
                time_patterns = json.loads(time_pattern) if time_pattern else []
                time_patterns.append(hour)

                await self.db.execute(
                    """
                    UPDATE user_workflows
                    SET frequency = ?,
                        success_rate = ?,
                        last_seen = ?,
                        time_of_day_pattern = ?,
                        confidence = ?
                    WHERE workflow_id = ?
                """,
                    (
                        new_freq,
                        new_success_rate,
                        now.isoformat(),
                        json.dumps(time_patterns[-20:]),  # Keep last 20
                        min(0.99, confidence + 0.05),
                        wf_id,
                    ),
                )

                return wf_id

            else:
                # Create new workflow
                async with self.db.execute(
                    """
                    INSERT INTO user_workflows (
                        workflow_name, action_sequence, space_sequence,
                        app_sequence, frequency, avg_duration, success_rate,
                        first_seen, last_seen, time_of_day_pattern,
                        confidence, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        workflow_name,
                        json.dumps(action_sequence),
                        json.dumps(space_sequence) if space_sequence else None,
                        json.dumps(app_sequence) if app_sequence else None,
                        1,
                        duration,
                        1.0 if success else 0.0,
                        now.isoformat(),
                        now.isoformat(),
                        json.dumps([hour]),
                        confidence,
                        json.dumps(metadata) if metadata else None,
                    ),
                ) as cursor:
                    await self.db.commit()
                    return cursor.lastrowid

        except Exception as e:
            logger.error(f"Error storing workflow: {e}", exc_info=True)
            raise

    async def store_space_transition(
        self,
        from_space: int,
        to_space: int,
        trigger_app: Optional[str] = None,
        trigger_action: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Store Space transition event

        Args:
            from_space: Source Space ID
            to_space: Target Space ID
            trigger_app: App that triggered transition
            trigger_action: Action that triggered transition
            metadata: Additional metadata
        """
        now = datetime.now()

        try:
            # Check for existing transition pattern
            async with self.db.execute(
                """
                SELECT transition_id, frequency, avg_time_between_seconds
                FROM space_transitions
                WHERE from_space_id = ? AND to_space_id = ?
                  AND hour_of_day = ? AND day_of_week = ?
            """,
                (from_space, to_space, now.hour, now.weekday()),
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                # Update existing
                trans_id, freq, avg_time = row
                await self.db.execute(
                    """
                    UPDATE space_transitions
                    SET frequency = ?,
                        timestamp = ?
                    WHERE transition_id = ?
                """,
                    (freq + 1, now.isoformat(), trans_id),
                )
            else:
                # Create new
                await self.db.execute(
                    """
                    INSERT INTO space_transitions (
                        from_space_id, to_space_id, trigger_app, trigger_action,
                        frequency, timestamp, hour_of_day, day_of_week, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        from_space,
                        to_space,
                        trigger_app,
                        trigger_action,
                        1,
                        now.isoformat(),
                        now.hour,
                        now.weekday(),
                        json.dumps(metadata) if metadata else None,
                    ),
                )

            await self.db.commit()

        except Exception as e:
            logger.error(f"Error storing space transition: {e}", exc_info=True)

    async def store_behavioral_pattern(
        self,
        behavior_type: str,
        description: str,
        pattern_data: Dict,
        confidence: float = 0.5,
        temporal_pattern: Optional[Dict] = None,
        contextual_triggers: Optional[List[str]] = None,
        prediction_accuracy: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        Store high-level behavioral pattern

        Args:
            behavior_type: Type of behavior
            description: Human-readable description
            pattern_data: Pattern data structure
            confidence: Confidence score
            temporal_pattern: Time-based pattern info
            contextual_triggers: List of triggers
            prediction_accuracy: How accurate predictions are
            metadata: Additional metadata

        Returns:
            behavior_id
        """
        now = datetime.now()

        try:
            # Check if behavior exists
            async with self.db.execute(
                """
                SELECT behavior_id, frequency, confidence
                FROM behavioral_patterns
                WHERE behavior_type = ? AND behavior_description = ?
            """,
                (behavior_type, description),
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                # Update existing
                beh_id, freq, old_conf = row
                new_conf = min(0.99, old_conf + 0.03)

                await self.db.execute(
                    """
                    UPDATE behavioral_patterns
                    SET frequency = ?,
                        confidence = ?,
                        last_observed = ?,
                        prediction_accuracy = ?
                    WHERE behavior_id = ?
                """,
                    (freq + 1, new_conf, now.isoformat(), prediction_accuracy, beh_id),
                )

                return beh_id
            else:
                # Create new
                async with self.db.execute(
                    """
                    INSERT INTO behavioral_patterns (
                        behavior_type, behavior_description, pattern_data,
                        frequency, confidence, temporal_pattern,
                        contextual_triggers, first_observed, last_observed,
                        prediction_accuracy, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        behavior_type,
                        description,
                        json.dumps(pattern_data),
                        1,
                        confidence,
                        json.dumps(temporal_pattern) if temporal_pattern else None,
                        json.dumps(contextual_triggers) if contextual_triggers else None,
                        now.isoformat(),
                        now.isoformat(),
                        prediction_accuracy,
                        json.dumps(metadata) if metadata else None,
                    ),
                ) as cursor:
                    await self.db.commit()
                    return cursor.lastrowid

        except Exception as e:
            logger.error(f"Error storing behavioral pattern: {e}", exc_info=True)
            raise

    async def store_temporal_pattern(
        self,
        pattern_type: str,
        action_type: str,
        target: str,
        time_of_day: Optional[int] = None,
        day_of_week: Optional[int] = None,
        day_of_month: Optional[int] = None,
        month_of_year: Optional[int] = None,
        frequency: int = 1,
        confidence: float = 0.5,
        metadata: Optional[Dict] = None,
    ):
        """
        Store temporal (time-based) behavioral pattern

        Args:
            pattern_type: Type of temporal pattern
            action_type: Type of action
            target: Target of action
            time_of_day: Hour (0-23)
            day_of_week: Day (0-6)
            day_of_month: Day (1-31)
            month_of_year: Month (1-12)
            frequency: Occurrence frequency
            confidence: Confidence score
            metadata: Additional metadata
        """
        now = datetime.now()
        is_leap = calendar.isleap(now.year)

        try:
            # Check if pattern exists
            async with self.db.execute(
                """
                SELECT temporal_id, frequency, confidence
                FROM temporal_patterns
                WHERE pattern_type = ? AND action_type = ? AND target = ?
                  AND time_of_day = ? AND day_of_week = ?
            """,
                (pattern_type, action_type, target, time_of_day, day_of_week),
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                # Update existing
                temp_id, freq, old_conf = row
                new_freq = freq + frequency
                new_conf = min(0.99, old_conf + 0.02)

                await self.db.execute(
                    """
                    UPDATE temporal_patterns
                    SET frequency = ?,
                        confidence = ?,
                        last_occurrence = ?
                    WHERE temporal_id = ?
                """,
                    (new_freq, new_conf, now.isoformat(), temp_id),
                )
            else:
                # Create new
                await self.db.execute(
                    """
                    INSERT INTO temporal_patterns (
                        pattern_type, time_of_day, day_of_week,
                        day_of_month, month_of_year, is_leap_year,
                        action_type, target, frequency, confidence,
                        last_occurrence, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        pattern_type,
                        time_of_day,
                        day_of_week,
                        day_of_month,
                        month_of_year,
                        is_leap,
                        action_type,
                        target,
                        frequency,
                        confidence,
                        now.isoformat(),
                        json.dumps(metadata) if metadata else None,
                    ),
                )

            await self.db.commit()

        except Exception as e:
            logger.error(f"Error storing temporal pattern: {e}", exc_info=True)

    async def store_proactive_suggestion(
        self,
        suggestion_type: str,
        suggestion_text: str,
        trigger_pattern_id: Optional[str] = None,
        confidence: float = 0.7,
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        Store proactive suggestion from AI

        Args:
            suggestion_type: Type of suggestion
            suggestion_text: The suggestion itself
            trigger_pattern_id: Pattern that triggered this
            confidence: Confidence score
            metadata: Additional metadata

        Returns:
            suggestion_id
        """
        now = datetime.now()

        try:
            async with self.db.execute(
                """
                INSERT INTO proactive_suggestions (
                    suggestion_type, suggestion_text, trigger_pattern_id,
                    confidence, times_suggested, times_accepted,
                    times_rejected, acceptance_rate, created_at,
                    last_suggested, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    suggestion_type,
                    suggestion_text,
                    trigger_pattern_id,
                    confidence,
                    0,
                    0,
                    0,
                    0.0,
                    now.isoformat(),
                    None,
                    json.dumps(metadata) if metadata else None,
                ),
            ) as cursor:
                await self.db.commit()
                return cursor.lastrowid

        except Exception as e:
            logger.error(f"Error storing suggestion: {e}", exc_info=True)
            raise

    async def update_suggestion_feedback(self, suggestion_id: int, accepted: bool):
        """
        Update suggestion with user feedback

        Args:
            suggestion_id: Suggestion ID
            accepted: Whether user accepted/rejected
        """
        now = datetime.now()

        try:
            # Get current stats
            async with self.db.execute(
                """
                SELECT times_suggested, times_accepted, times_rejected
                FROM proactive_suggestions
                WHERE suggestion_id = ?
            """,
                (suggestion_id,),
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                suggested, accepted_count, rejected_count = row

                if accepted:
                    accepted_count += 1
                else:
                    rejected_count += 1

                total = accepted_count + rejected_count
                acceptance_rate = accepted_count / total if total > 0 else 0.0

                await self.db.execute(
                    """
                    UPDATE proactive_suggestions
                    SET times_accepted = ?,
                        times_rejected = ?,
                        acceptance_rate = ?,
                        last_suggested = ?
                    WHERE suggestion_id = ?
                """,
                    (
                        accepted_count,
                        rejected_count,
                        acceptance_rate,
                        now.isoformat(),
                        suggestion_id,
                    ),
                )

                await self.db.commit()

        except Exception as e:
            logger.error(f"Error updating suggestion feedback: {e}", exc_info=True)

    # Behavioral Query Methods

    async def get_app_usage_patterns(
        self,
        app_name: Optional[str] = None,
        space_id: Optional[int] = None,
        time_of_day: Optional[int] = None,
        min_confidence: float = 0.6,
    ) -> List[Dict]:
        """
        Query app usage patterns

        Args:
            app_name: Filter by app name
            space_id: Filter by Space ID
            time_of_day: Filter by hour
            min_confidence: Minimum confidence threshold

        Returns:
            List of app usage patterns
        """
        query = """
            SELECT * FROM app_usage_patterns
            WHERE confidence >= ?
        """
        params = [min_confidence]

        if app_name:
            query += " AND app_name = ?"
            params.append(app_name)
        if space_id is not None:
            query += " AND space_id = ?"
            params.append(space_id)
        if time_of_day is not None:
            query += " AND typical_time_of_day = ?"
            params.append(time_of_day)

        query += " ORDER BY confidence DESC, usage_frequency DESC LIMIT 50"

        try:
            async with self.db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error querying app patterns: {e}", exc_info=True)
            return []

    async def get_workflows_by_context(
        self,
        app_sequence: Optional[List[str]] = None,
        space_sequence: Optional[List[int]] = None,
        time_of_day: Optional[int] = None,
        min_confidence: float = 0.6,
    ) -> List[Dict]:
        """
        Query workflows by context

        Args:
            app_sequence: Partial app sequence to match
            space_sequence: Partial space sequence to match
            time_of_day: Hour of day
            min_confidence: Minimum confidence

        Returns:
            List of matching workflows
        """
        query = """
            SELECT * FROM user_workflows
            WHERE confidence >= ?
        """
        params = [min_confidence]

        if time_of_day is not None:
            query += " AND json_extract(time_of_day_pattern, '$') LIKE ?"
            params.append(f"%{time_of_day}%")

        query += " ORDER BY confidence DESC, frequency DESC LIMIT 30"

        try:
            async with self.db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                workflows = [dict(row) for row in rows]

                # Filter by sequences if provided
                if app_sequence:
                    workflows = [
                        w
                        for w in workflows
                        if self._sequence_matches(
                            json.loads(w["app_sequence"]) if w["app_sequence"] else [], app_sequence
                        )
                    ]

                return workflows
        except Exception as e:
            logger.error(f"Error querying workflows: {e}", exc_info=True)
            return []

    async def predict_next_action(self, current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Predict next likely actions based on current context

        Args:
            current_context: Current context including:
                - current_app: Current app name
                - current_space: Current Space ID
                - time_of_day: Current hour
                - recent_actions: List of recent actions

        Returns:
            List of predicted actions with confidence scores
        """
        predictions = []

        try:
            now = datetime.now()
            current_app = current_context.get("current_app")
            current_space = current_context.get("current_space")
            time_of_day = current_context.get("time_of_day", now.hour)

            # Check temporal patterns
            async with self.db.execute(
                """
                SELECT action_type, target, confidence, frequency
                FROM temporal_patterns
                WHERE time_of_day = ? AND day_of_week = ?
                ORDER BY confidence DESC, frequency DESC
                LIMIT 10
            """,
                (time_of_day, now.weekday()),
            ) as cursor:
                temporal_rows = await cursor.fetchall()

            for row in temporal_rows:
                predictions.append(
                    {
                        "source": "temporal_pattern",
                        "action_type": row[0],
                        "target": row[1],
                        "confidence": row[2],
                        "reasoning": f"Typical behavior at {time_of_day}:00",
                    }
                )

            # Check app transitions
            if current_app:
                # Find common next apps
                async with self.db.execute(
                    """
                    SELECT app_name, confidence, usage_frequency
                    FROM app_usage_patterns
                    WHERE typical_time_of_day = ?
                      AND space_id = ?
                    ORDER BY confidence DESC
                    LIMIT 5
                """,
                    (time_of_day, current_space),
                ) as cursor:
                    app_rows = await cursor.fetchall()

                for row in app_rows:
                    predictions.append(
                        {
                            "source": "app_usage_pattern",
                            "action_type": "switch_app",
                            "target": row[0],
                            "confidence": row[1],
                            "reasoning": f"Commonly used on Space {current_space} at this time",
                        }
                    )

            # Check workflows
            workflows = await self.get_workflows_by_context(
                time_of_day=time_of_day, min_confidence=0.6
            )

            for workflow in workflows[:3]:
                predictions.append(
                    {
                        "source": "workflow_pattern",
                        "action_type": "workflow",
                        "target": workflow["workflow_name"],
                        "confidence": workflow["confidence"],
                        "reasoning": f"Part of frequent workflow",
                    }
                )

            # Sort by confidence
            predictions.sort(key=lambda x: x["confidence"], reverse=True)

            return predictions[:10]

        except Exception as e:
            logger.error(f"Error predicting next action: {e}", exc_info=True)
            return []

    def _sequence_matches(self, full_sequence: List, partial: List) -> bool:
        """Check if partial sequence appears in full sequence"""
        if not partial:
            return True
        if len(partial) > len(full_sequence):
            return False

        for i in range(len(full_sequence) - len(partial) + 1):
            if full_sequence[i : i + len(partial)] == partial:
                return True
        return False

    async def get_behavioral_insights(self) -> Dict[str, Any]:
        """
        Get comprehensive behavioral insights

        Returns:
            Dictionary of behavioral insights and statistics
        """
        insights = {
            "most_used_apps": [],
            "most_used_spaces": [],
            "common_workflows": [],
            "temporal_habits": [],
            "space_transitions": [],
            "prediction_accuracy": 0.0,
        }

        try:
            # Most used apps
            async with self.db.execute(
                """
                SELECT app_name, space_id, SUM(usage_frequency) as total_use,
                       AVG(confidence) as avg_conf
                FROM app_usage_patterns
                GROUP BY app_name, space_id
                ORDER BY total_use DESC
                LIMIT 10
            """
            ) as cursor:
                rows = await cursor.fetchall()
                insights["most_used_apps"] = [
                    {"app": r[0], "space": r[1], "usage": r[2], "confidence": r[3]} for r in rows
                ]

            # Most used spaces
            async with self.db.execute(
                """
                SELECT space_id, COUNT(*) as usage_count
                FROM workspace_usage
                GROUP BY space_id
                ORDER BY usage_count DESC
                LIMIT 5
            """
            ) as cursor:
                rows = await cursor.fetchall()
                insights["most_used_spaces"] = [
                    {"space_id": r[0], "usage_count": r[1]} for r in rows
                ]

            # Common workflows
            async with self.db.execute(
                """
                SELECT workflow_name, frequency, success_rate, confidence
                FROM user_workflows
                ORDER BY frequency DESC
                LIMIT 10
            """
            ) as cursor:
                rows = await cursor.fetchall()
                insights["common_workflows"] = [
                    {"name": r[0], "frequency": r[1], "success_rate": r[2], "confidence": r[3]}
                    for r in rows
                ]

            # Temporal habits
            async with self.db.execute(
                """
                SELECT time_of_day, day_of_week, action_type,
                       COUNT(*) as occurrences, AVG(confidence) as avg_conf
                FROM temporal_patterns
                GROUP BY time_of_day, day_of_week, action_type
                HAVING occurrences > 2
                ORDER BY occurrences DESC
                LIMIT 20
            """
            ) as cursor:
                rows = await cursor.fetchall()
                insights["temporal_habits"] = [
                    {
                        "hour": r[0],
                        "day": r[1],
                        "action": r[2],
                        "occurrences": r[3],
                        "confidence": r[4],
                    }
                    for r in rows
                ]

            # Space transitions
            async with self.db.execute(
                """
                SELECT from_space_id, to_space_id, trigger_app,
                       SUM(frequency) as total_transitions
                FROM space_transitions
                GROUP BY from_space_id, to_space_id
                ORDER BY total_transitions DESC
                LIMIT 15
            """
            ) as cursor:
                rows = await cursor.fetchall()
                insights["space_transitions"] = [
                    {"from": r[0], "to": r[1], "trigger": r[2], "frequency": r[3]} for r in rows
                ]

            # Overall prediction accuracy
            async with self.db.execute(
                """
                SELECT AVG(acceptance_rate) as avg_acceptance
                FROM proactive_suggestions
                WHERE times_suggested > 0
            """
            ) as cursor:
                row = await cursor.fetchone()
                if row and row[0]:
                    insights["prediction_accuracy"] = row[0]

            return insights

        except Exception as e:
            logger.error(f"Error getting behavioral insights: {e}", exc_info=True)
            return insights

    async def get_all_speaker_profiles(self) -> List[Dict]:
        """
        Get all speaker profiles from database for speaker recognition.

        Returns list of profile dicts with embeddings and metadata.
        """
        try:
            async with self.db.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT speaker_id, speaker_name, voiceprint_embedding,
                           total_samples, average_pitch_hz, recognition_confidence,
                           is_primary_user, security_level, created_at, last_updated
                    FROM speaker_profiles
                """
                )
                rows = await cursor.fetchall()
                logger.info(f"Got {len(rows) if rows else 0} speaker profile rows")
                logger.info(
                    f"Type of rows: {type(rows)}, Type of first row: {type(rows[0]) if rows else 'empty'}"
                )

                if rows and len(rows) > 0:
                    logger.info(f"First row content: {rows[0]}")
                    logger.info(
                        f"First row keys: {list(rows[0].keys()) if hasattr(rows[0], 'keys') else 'no keys'}"
                    )

                profiles = []
                for row in rows:
                    profiles.append(
                        {
                            "speaker_id": row["speaker_id"],
                            "speaker_name": row["speaker_name"],
                            "voiceprint_embedding": row["voiceprint_embedding"],
                            "total_samples": row["total_samples"],
                            "average_pitch_hz": row["average_pitch_hz"],
                            "recognition_confidence": row["recognition_confidence"],
                            "is_primary_user": bool(row["is_primary_user"]),
                            "security_level": row["security_level"] or "standard",
                            "created_at": row["created_at"],
                            "last_updated": row["last_updated"],
                        }
                    )

                return profiles

        except Exception as e:
            logger.error(f"Failed to get speaker profiles: {e}", exc_info=True)
            return []

    async def update_speaker_embedding(
        self,
        speaker_id: int,
        embedding: bytes,
        confidence: float,
        is_primary_user: bool = False,
    ) -> bool:
        """
        Update speaker profile with voice embedding.

        Args:
            speaker_id: Speaker profile ID
            embedding: Voice embedding bytes (numpy array serialized)
            confidence: Recognition confidence (0.0-1.0)
            is_primary_user: Mark as device owner (Derek J. Russell)

        Returns:
            True if successful
        """
        try:
            async with self.db.cursor() as cursor:
                await cursor.execute(
                    """
                    UPDATE speaker_profiles
                    SET voiceprint_embedding = ?,
                        recognition_confidence = ?,
                        is_primary_user = ?,
                        security_level = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE speaker_id = ?
                """,
                    (
                        embedding,
                        confidence,
                        is_primary_user,  # PostgreSQL expects boolean, not int
                        "admin" if is_primary_user else "standard",
                        speaker_id,
                    ),
                )

                await self.db.commit()

                logger.info(
                    f"✅ Updated speaker embedding (ID: {speaker_id}, confidence: {confidence:.2f}, owner: {is_primary_user})"
                )

                return True

        except Exception as e:
            logger.error(f"Failed to update speaker embedding: {e}")
            return False

    async def close(self):
        """Close database connections gracefully"""
        # Flush pending batches
        await self._flush_goal_batch()
        await self._flush_action_batch()
        await self._flush_pattern_batch()

        # Close SQLite
        if self.db:
            await self.db.close()

        logger.info("✅ Learning database closed gracefully")


# Global instance with async initialization
_db_instance = None
_db_lock = asyncio.Lock()


async def get_learning_database(config: Optional[Dict] = None) -> JARVISLearningDatabase:
    """Get or create the global async learning database"""
    global _db_instance

    async with _db_lock:
        if _db_instance is None:
            _db_instance = JARVISLearningDatabase(config=config)
            await _db_instance.initialize()
        return _db_instance


async def test_database():
    """Test the advanced learning database"""
    print("🗄️ Testing Advanced JARVIS Learning Database")
    print("=" * 60)

    db = await get_learning_database()

    # Test storing a goal
    goal = {
        "goal_id": "test_goal_1",
        "goal_type": "meeting_preparation",
        "goal_level": "HIGH",
        "description": "Prepare for team meeting",
        "confidence": 0.92,
        "evidence": [{"source": "calendar", "data": "meeting in 10 min"}],
    }

    goal_id = await db.store_goal(goal)
    print(f"✅ Stored goal: {goal_id}")

    # Test batch storage
    for i in range(5):
        await db.store_goal({"goal_type": "test", "confidence": 0.5 + i * 0.1}, batch=True)
    print(f"✅ Queued 5 goals for batch insert")

    # Test storing an action
    action = {
        "action_id": "test_action_1",
        "action_type": "connect_display",
        "target": "Living Room TV",
        "goal_id": goal_id,
        "confidence": 0.85,
        "success": True,
        "execution_time": 0.45,
    }

    action_id = await db.store_action(action)
    print(f"✅ Stored action: {action_id}")

    # Test learning display pattern
    await db.learn_display_pattern(
        "Living Room TV", {"apps": ["keynote", "calendar"], "time": "09:00"}
    )
    print("✅ Learned display pattern")

    # Test learning preference
    await db.learn_preference("display", "default", "Living Room TV", 0.8)
    print("✅ Learned preference")

    # Get metrics
    metrics = await db.get_learning_metrics()
    print("\n📊 Learning Metrics:")
    print(f"   Total Goals: {metrics['goals']['total_goals']}")
    print(f"   Total Actions: {metrics['actions']['total_actions']}")
    print(f"   Total Patterns: {metrics['patterns']['total_patterns']}")
    print(
        f"   Pattern Cache Hit Rate: {metrics['cache_performance']['pattern_cache_hit_rate']:.2%}"
    )

    # Test pattern analysis
    patterns = await db.analyze_patterns()
    print(f"\n🔍 Analyzed {len(patterns)} patterns")

    await db.close()
    print("\n✅ Advanced database test complete!")


if __name__ == "__main__":
    asyncio.run(test_database())
