#!/usr/bin/env python3
"""
Cloud Database Adapter for JARVIS
Supports both local SQLite and GCP Cloud SQL (PostgreSQL)
Seamless switching between local and cloud databases
"""
import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

# Database drivers
import aiosqlite  # For local SQLite

try:
    import asyncpg  # For PostgreSQL/Cloud SQL
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    logging.warning("asyncpg not available - install with: pip install asyncpg")

try:
    from google.cloud.sql.connector import Connector
    CLOUD_SQL_CONNECTOR_AVAILABLE = True
except ImportError:
    CLOUD_SQL_CONNECTOR_AVAILABLE = False
    logging.warning("Cloud SQL Connector not available - install with: pip install cloud-sql-python-connector[asyncpg]")

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Configuration for database connection"""

    def __init__(self):
        self.db_type = os.getenv('JARVIS_DB_TYPE', 'sqlite')  # 'sqlite' or 'cloudsql'

        # Cloud SQL config
        self.connection_name = os.getenv('JARVIS_DB_CONNECTION_NAME')
        self.db_host = os.getenv('JARVIS_DB_HOST', '127.0.0.1')
        self.db_port = int(os.getenv('JARVIS_DB_PORT', '5432'))
        self.db_name = os.getenv('JARVIS_DB_NAME', 'jarvis_learning')
        self.db_user = os.getenv('JARVIS_DB_USER', 'jarvis')
        self.db_password = os.getenv('JARVIS_DB_PASSWORD', '')

        # Local SQLite config
        self.sqlite_path = Path.home() / ".jarvis" / "learning" / "jarvis_learning.db"

        # Load from config file if exists
        self._load_from_config()

    def _load_from_config(self):
        """Load config from JSON file"""
        config_path = Path.home() / ".jarvis" / "gcp" / "database_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    cloud_sql = config.get('cloud_sql', {})
                    self.connection_name = cloud_sql.get('connection_name', self.connection_name)
                    self.db_host = cloud_sql.get('private_ip', self.db_host)
                    self.db_port = cloud_sql.get('port', self.db_port)
                    self.db_name = cloud_sql.get('database', self.db_name)
                    self.db_user = cloud_sql.get('user', self.db_user)
                    self.db_password = cloud_sql.get('password', self.db_password)
                    logger.info(f"✅ Loaded database config from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

    @property
    def use_cloud_sql(self) -> bool:
        """Check if we should use Cloud SQL"""
        return (
            self.db_type == 'cloudsql' and
            ASYNCPG_AVAILABLE and
            self.db_password and
            self.connection_name
        )


class CloudDatabaseAdapter:
    """
    Adapter that provides unified interface for both SQLite and Cloud SQL
    Automatically chooses backend based on configuration
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.pool: Optional[Any] = None
        self.connector: Optional[Any] = None
        self._local_connection: Optional[aiosqlite.Connection] = None

        logger.info(f"🔧 DatabaseAdapter initialized (type: {self.config.db_type})")

    async def initialize(self):
        """Initialize database connection pool"""
        if self.config.use_cloud_sql:
            await self._init_cloud_sql()
        else:
            await self._init_sqlite()

    async def _init_sqlite(self):
        """Initialize local SQLite connection"""
        self.config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"📂 Using local SQLite: {self.config.sqlite_path}")

    async def _init_cloud_sql(self):
        """Initialize Cloud SQL connection pool"""
        try:
            logger.info(f"☁️  Connecting to Cloud SQL: {self.config.connection_name}")

            if CLOUD_SQL_CONNECTOR_AVAILABLE:
                # Use Cloud SQL Connector (recommended for secure connections)
                self.connector = Connector()

                async def getconn():
                    conn = await self.connector.connect_async(
                        self.config.connection_name,
                        "asyncpg",
                        user=self.config.db_user,
                        password=self.config.db_password,
                        db=self.config.db_name
                    )
                    return conn

                # Create connection pool
                self.pool = await asyncpg.create_pool(
                    min_size=2,
                    max_size=10,
                    command_timeout=60,
                    connection_class=getconn
                )
            else:
                # Direct connection (requires Cloud SQL Proxy running locally)
                self.pool = await asyncpg.create_pool(
                    host=self.config.db_host,
                    port=self.config.db_port,
                    database=self.config.db_name,
                    user=self.config.db_user,
                    password=self.config.db_password,
                    min_size=2,
                    max_size=10,
                    command_timeout=60
                )

            logger.info("✅ Cloud SQL connection pool created")

        except Exception as e:
            logger.error(f"❌ Failed to connect to Cloud SQL: {e}")
            logger.info("📂 Falling back to local SQLite")
            self.pool = None
            await self._init_sqlite()

    @asynccontextmanager
    async def connection(self):
        """Get database connection (context manager)"""
        if self.pool:
            # Cloud SQL (PostgreSQL)
            async with self.pool.acquire() as conn:
                yield CloudSQLConnection(conn)
        else:
            # Local SQLite
            async with aiosqlite.connect(self.config.sqlite_path) as conn:
                yield SQLiteConnection(conn)

    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
            logger.info("✅ Cloud SQL pool closed")

        if self.connector:
            await self.connector.close_async()
            logger.info("✅ Cloud SQL connector closed")

        if self._local_connection:
            await self._local_connection.close()
            logger.info("✅ SQLite connection closed")

    @property
    def is_cloud(self) -> bool:
        """Check if using cloud database"""
        return self.pool is not None


class SQLiteConnection:
    """Wrapper for SQLite connection with unified interface"""

    def __init__(self, conn: aiosqlite.Connection):
        self.conn = conn
        self.conn.row_factory = aiosqlite.Row

    async def execute(self, query: str, *args):
        """Execute query"""
        return await self.conn.execute(query, args)

    async def fetch(self, query: str, *args):
        """Fetch all results"""
        async with self.conn.execute(query, args) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def fetchone(self, query: str, *args):
        """Fetch one result"""
        async with self.conn.execute(query, args) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def fetchval(self, query: str, *args):
        """Fetch single value"""
        async with self.conn.execute(query, args) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None

    async def commit(self):
        """Commit transaction"""
        await self.conn.commit()


class CloudSQLConnection:
    """Wrapper for Cloud SQL (PostgreSQL) connection with unified interface"""

    def __init__(self, conn):
        self.conn = conn

    async def execute(self, query: str, *args):
        """Execute query (convert ? to $1, $2, etc for PostgreSQL)"""
        pg_query = self._convert_placeholders(query)
        return await self.conn.execute(pg_query, *args)

    async def fetch(self, query: str, *args):
        """Fetch all results"""
        pg_query = self._convert_placeholders(query)
        rows = await self.conn.fetch(pg_query, *args)
        return [dict(row) for row in rows]

    async def fetchone(self, query: str, *args):
        """Fetch one result"""
        pg_query = self._convert_placeholders(query)
        row = await self.conn.fetchrow(pg_query, *args)
        return dict(row) if row else None

    async def fetchval(self, query: str, *args):
        """Fetch single value"""
        pg_query = self._convert_placeholders(query)
        return await self.conn.fetchval(pg_query, *args)

    async def commit(self):
        """No-op for PostgreSQL (auto-commit)"""
        pass

    def _convert_placeholders(self, query: str) -> str:
        """Convert SQLite ? placeholders to PostgreSQL $1, $2, etc"""
        result = []
        param_num = 1
        i = 0
        while i < len(query):
            if query[i] == '?':
                result.append(f'${param_num}')
                param_num += 1
            else:
                result.append(query[i])
            i += 1
        return ''.join(result)


# Global adapter instance
_adapter: Optional[CloudDatabaseAdapter] = None


async def get_database_adapter() -> CloudDatabaseAdapter:
    """Get or create global database adapter"""
    global _adapter
    if _adapter is None:
        _adapter = CloudDatabaseAdapter()
        await _adapter.initialize()
    return _adapter


async def close_database_adapter():
    """Close global database adapter"""
    global _adapter
    if _adapter:
        await _adapter.close()
        _adapter = None
