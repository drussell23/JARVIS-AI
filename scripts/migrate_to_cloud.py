#!/usr/bin/env python3
"""
JARVIS Database Migration Script
Migrates local SQLite databases to GCP Cloud SQL (PostgreSQL)
"""
import asyncio
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

import aiosqlite
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    print("❌ asyncpg not installed. Install with: pip install asyncpg")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """Migrates local SQLite to Cloud SQL"""

    def __init__(self):
        self.config = self._load_config()
        self.local_db = Path.home() / ".jarvis" / "learning" / "jarvis_learning.db"

    def _load_config(self) -> dict:
        """Load GCP database config"""
        config_path = Path.home() / ".jarvis" / "gcp" / "database_config.json"
        if not config_path.exists():
            logger.error(f"❌ Config file not found: {config_path}")
            logger.info("Run: ./scripts/setup_gcp_databases.sh first")
            sys.exit(1)

        with open(config_path, 'r') as f:
            return json.load(f)

    async def migrate(self):
        """Run full migration"""
        print("=" * 80)
        print("🚀 JARVIS Database Migration: Local SQLite → Cloud SQL")
        print("=" * 80)
        print()

        # Check local database exists
        if not self.local_db.exists():
            print(f"❌ Local database not found: {self.local_db}")
            print("Nothing to migrate!")
            return

        print(f"📂 Local database: {self.local_db}")
        print(f"☁️  Cloud SQL: {self.config['cloud_sql']['database']}")
        print()

        # Confirm migration (check for --yes flag)
        import sys
        if '--yes' not in sys.argv:
            response = input("⚠️  This will overwrite cloud data. Continue? (y/N): ")
            if response.lower() != 'y':
                print("❌ Migration cancelled")
                return

        print()
        print("🔄 Starting migration...")
        print()

        try:
            # Connect to databases
            print("📡 Connecting to databases...")
            local_conn = await aiosqlite.connect(self.local_db)
            local_conn.row_factory = aiosqlite.Row

            cloud_sql_config = self.config['cloud_sql']
            # Connect via Cloud SQL Proxy (localhost) if available, otherwise use private IP
            cloud_host = '127.0.0.1' if os.path.exists(os.path.expanduser('~/.local/bin/cloud-sql-proxy')) else cloud_sql_config['private_ip']
            cloud_conn = await asyncpg.connect(
                host=cloud_host,
                port=cloud_sql_config['port'],
                database=cloud_sql_config['database'],
                user=cloud_sql_config['user'],
                password=cloud_sql_config['password']
            )

            print("✅ Connected to both databases")
            print()

            # Get table names from SQLite
            async with local_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ) as cursor:
                tables = await cursor.fetchall()

            table_names = [row['name'] for row in tables]
            print(f"📋 Found {len(table_names)} tables to migrate:")
            for table in table_names:
                print(f"   • {table}")
            print()

            # Migrate each table
            for table_name in table_names:
                await self._migrate_table(local_conn, cloud_conn, table_name)

            # Close connections
            await local_conn.close()
            await cloud_conn.close()

            print()
            print("=" * 80)
            print("✅ MIGRATION COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print()
            print("📊 Next steps:")
            print("1. Set environment variable: export JARVIS_DB_TYPE=cloudsql")
            print("2. Start JARVIS and verify it connects to Cloud SQL")
            print("3. Check data: psql -h <ip> -U jarvis -d jarvis_learning")
            print()

        except Exception as e:
            logger.error(f"❌ Migration failed: {e}", exc_info=True)
            sys.exit(1)

    async def _migrate_table(self, local_conn, cloud_conn, table_name: str):
        """Migrate a single table"""
        print(f"🔄 Migrating table: {table_name}")

        # Get table schema from SQLite
        async with local_conn.execute(f"PRAGMA table_info({table_name})") as cursor:
            columns = await cursor.fetchall()

        # Create table in PostgreSQL
        pg_schema = self._convert_schema_to_postgres(table_name, columns)
        print(f"   Creating table schema...")
        print(f"   Schema SQL: {pg_schema[:200]}...")  # Debug: show first 200 chars
        await cloud_conn.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
        await cloud_conn.execute(pg_schema)

        # Get data from SQLite
        async with local_conn.execute(f"SELECT * FROM {table_name}") as cursor:
            rows = await cursor.fetchall()

        if not rows:
            print(f"   ⚠️  Table is empty, skipping data migration")
            return

        print(f"   Migrating {len(rows)} rows...")

        # Prepare insert statement
        col_names = [col['name'] for col in columns]
        placeholders = ', '.join([f'${i+1}' for i in range(len(col_names))])
        insert_sql = f"INSERT INTO {table_name} ({', '.join(col_names)}) VALUES ({placeholders})"

        # Get column info including types for proper conversion
        col_info = {col['name']: col for col in columns}

        # Insert data in batches
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            for row in batch:
                # Convert values to match PostgreSQL types
                values = []
                for col_name in col_names:
                    val = row[col_name]
                    col_type = col_info[col_name]['type'].upper()

                    # Type conversions
                    if val is None:
                        pass  # Keep as None
                    elif col_type == 'TEXT':
                        # Ensure TEXT columns are strings
                        if not isinstance(val, str):
                            val = str(val)
                    elif col_type in ('INTEGER', 'BIGINT'):
                        # Ensure INTEGER columns are ints
                        if isinstance(val, str):
                            val = int(val) if val else None
                    elif col_type in ('REAL', 'DOUBLE PRECISION'):
                        # Ensure REAL columns are floats
                        if isinstance(val, str):
                            val = float(val) if val else None
                    elif col_type == 'BOOLEAN':
                        # Convert to boolean
                        if isinstance(val, (int, str)):
                            val = bool(int(val)) if val not in (None, '') else False
                    elif col_type in ('TIMESTAMP', 'DATETIME'):
                        # Convert timestamp strings to datetime objects
                        if isinstance(val, str):
                            from datetime import datetime as dt
                            try:
                                val = dt.fromisoformat(val.replace(' ', 'T'))
                            except:
                                val = None
                    elif col_type in ('JSON', 'JSONB'):
                        # JSON columns should be strings
                        if val is not None and not isinstance(val, str):
                            import json as json_lib
                            val = json_lib.dumps(val)

                    values.append(val)

                await cloud_conn.execute(insert_sql, *values)

        print(f"   ✅ Migrated {len(rows)} rows")

    def _convert_schema_to_postgres(self, table_name: str, columns: list) -> str:
        """Convert SQLite schema to PostgreSQL"""
        col_defs = []

        for col in columns:
            col_name = col['name']
            col_type = col['type'].upper()

            # Map SQLite types to PostgreSQL
            type_mapping = {
                'INTEGER': 'BIGINT',
                'TEXT': 'TEXT',
                'REAL': 'DOUBLE PRECISION',
                'BLOB': 'BYTEA',
                'NUMERIC': 'NUMERIC',
                'DATETIME': 'TIMESTAMP',
                'TIMESTAMP': 'TIMESTAMP',
                'BOOLEAN': 'BOOLEAN',
                'JSON': 'JSONB'
            }

            pg_type = type_mapping.get(col_type, 'TEXT')

            # Check for primary key
            is_pk = col['pk'] == 1

            if is_pk:
                # PostgreSQL auto-increment
                if pg_type == 'BIGINT':
                    col_def = f"{col_name} BIGSERIAL PRIMARY KEY"
                else:
                    col_def = f"{col_name} {pg_type} PRIMARY KEY"
            else:
                not_null = " NOT NULL" if col['notnull'] else ""

                # Handle default values - need to convert for PostgreSQL
                default = ""
                if col['dflt_value']:
                    dflt_val = col['dflt_value']
                    # Convert boolean defaults
                    if pg_type == 'BOOLEAN' and dflt_val in ('0', '1'):
                        dflt_val = 'false' if dflt_val == '0' else 'true'
                    default = f" DEFAULT {dflt_val}"

                col_def = f"{col_name} {pg_type}{not_null}{default}"

            col_defs.append(col_def)

        col_list = ',\n  '.join(col_defs)
        return f"CREATE TABLE {table_name} (\n  {col_list}\n)"


async def main():
    """Run migration"""
    migrator = DatabaseMigrator()
    await migrator.migrate()


if __name__ == "__main__":
    asyncio.run(main())
