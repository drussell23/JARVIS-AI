#!/usr/bin/env python3
"""
CloudSQL Connection Diagnostics Tool
=====================================

Comprehensive diagnostics for CloudSQL connection management:
1. Check CloudSQL proxy status
2. Count active connections
3. Detect leaked connections
4. Test singleton connection manager
5. Verify cleanup mechanisms

Usage:
    python diagnose_connections.py                    # Run diagnostics
    python diagnose_connections.py --kill-leaked      # Kill leaked connections
    python diagnose_connections.py --emergency        # Emergency cleanup
"""

import argparse
import asyncio
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))


def check_proxy_running():
    """Check if Cloud SQL proxy is running"""
    print("\n" + "=" * 60)
    print("1. Cloud SQL Proxy Status")
    print("=" * 60)

    try:
        result = subprocess.run(
            ["pgrep", "-fl", "cloud-sql-proxy"],
            capture_output=True,
            text=True
        )

        if result.stdout.strip():
            print("✅ Cloud SQL proxy is running:")
            for line in result.stdout.strip().split('\n'):
                print(f"   {line}")
        else:
            print("❌ Cloud SQL proxy is NOT running")
            print("   Start it with: cloud-sql-proxy <connection-name>")

    except Exception as e:
        print(f"❌ Error checking proxy: {e}")


async def count_active_connections():
    """Count active CloudSQL connections"""
    print("\n" + "=" * 60)
    print("2. Active CloudSQL Connections")
    print("=" * 60)

    try:
        import asyncpg
        from backend.core.secret_manager import get_db_password

        password = get_db_password()
        if not password:
            print("❌ Could not get database password from Secret Manager")
            return

        # Try to connect
        try:
            conn = await asyncio.wait_for(
                asyncpg.connect(
                    host='127.0.0.1',
                    port=5432,
                    database='jarvis_learning',
                    user='jarvis',
                    password=password,
                    timeout=5.0
                ),
                timeout=10.0
            )

            # Query active connections
            connections = await conn.fetch("""
                SELECT
                    pid,
                    usename,
                    application_name,
                    client_addr,
                    state,
                    state_change,
                    query_start,
                    NOW() - state_change AS idle_time
                FROM pg_stat_activity
                WHERE datname = 'jarvis_learning'
                ORDER BY state_change DESC
            """)

            print(f"\n📊 Total connections: {len(connections)}")
            print(f"   Max allowed (db-f1-micro): ~25")
            print(f"   Reserved for superuser: ~3")
            print(f"   Available for JARVIS: ~22")
            print()

            if len(connections) > 20:
                print("⚠️  WARNING: High connection count!")
            elif len(connections) > 15:
                print("⚡ Connection count is getting high")
            else:
                print("✅ Connection count is healthy")

            print("\nActive connections:")
            print(f"{'PID':<8} {'User':<12} {'State':<12} {'Idle Time':<15} {'App Name':<20}")
            print("-" * 80)

            for row in connections:
                pid = row['pid']
                user = row['usename']
                state = row['state']
                idle_time = str(row['idle_time']) if row['idle_time'] else 'N/A'
                app_name = row['application_name'] or 'N/A'

                # Highlight problematic connections
                marker = ""
                if 'idle' in state and row['idle_time']:
                    if row['idle_time'].total_seconds() > 300:  # 5 minutes
                        marker = "🔴 LEAKED"
                    elif row['idle_time'].total_seconds() > 60:  # 1 minute
                        marker = "⚠️  OLD"

                print(f"{pid:<8} {user:<12} {state:<12} {idle_time:<15} {app_name:<20} {marker}")

            await conn.close()

        except asyncio.TimeoutError:
            print("❌ Connection timeout - is the proxy running?")
        except Exception as e:
            print(f"❌ Failed to query connections: {e}")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")


async def test_singleton_manager():
    """Test singleton connection manager"""
    print("\n" + "=" * 60)
    print("3. Singleton Connection Manager Test")
    print("=" * 60)

    try:
        from intelligence.cloud_sql_connection_manager import (
            get_connection_manager,
            CloudSQLConnectionManager
        )
        from backend.core.secret_manager import get_db_password

        # Test 1: Singleton pattern
        print("\n📋 Test 1: Singleton Pattern")
        manager1 = get_connection_manager()
        manager2 = get_connection_manager()
        manager3 = CloudSQLConnectionManager()

        if manager1 is manager2 is manager3:
            print("✅ Singleton pattern working - all instances are the same")
        else:
            print("❌ Singleton pattern BROKEN - multiple instances exist!")

        # Test 2: Initialization
        print("\n📋 Test 2: Initialization")
        password = get_db_password()
        if not password:
            print("⚠️  No password - skipping initialization test")
            return

        if manager1.is_initialized:
            print("ℹ️  Manager already initialized - getting stats")
        else:
            print("🔌 Initializing connection manager...")
            success = await manager1.initialize(
                host='127.0.0.1',
                port=5432,
                database='jarvis_learning',
                user='jarvis',
                password=password,
                max_connections=3,
                force_reinit=False
            )

            if success:
                print("✅ Connection manager initialized")
            else:
                print("❌ Connection manager initialization failed")
                return

        # Test 3: Stats
        print("\n📋 Test 3: Connection Pool Stats")
        stats = manager1.get_stats()
        print(f"   Status: {stats['status']}")
        print(f"   Pool size: {stats['pool_size']}")
        print(f"   Idle: {stats['idle_size']}")
        print(f"   Max: {stats['max_size']}")
        print(f"   Total connections: {stats['connection_count']}")
        print(f"   Errors: {stats['error_count']}")

        # Test 4: Connection acquisition
        print("\n📋 Test 4: Connection Acquisition")
        try:
            async with manager1.connection() as conn:
                result = await conn.fetchval("SELECT 1 + 1")
                if result == 2:
                    print("✅ Connection acquisition and query successful")
                else:
                    print(f"❌ Query returned unexpected result: {result}")
        except Exception as e:
            print(f"❌ Connection acquisition failed: {e}")

        # Final stats
        final_stats = manager1.get_stats()
        print(f"\n📊 Final Stats:")
        print(f"   Pool size: {final_stats['pool_size']}, Idle: {final_stats['idle_size']}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def check_process_connections():
    """Check JARVIS processes with database connections"""
    print("\n" + "=" * 60)
    print("4. JARVIS Processes with DB Connections")
    print("=" * 60)

    try:
        import psutil

        jarvis_procs = []

        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'connections']):
            try:
                cmdline = ' '.join(proc.cmdline()).lower()

                if 'jarvis' in cmdline or 'main.py' in cmdline:
                    # Check for connections to port 5432
                    has_db_conn = False
                    try:
                        for conn in proc.connections():
                            if conn.laddr.port == 5432 or conn.raddr.port == 5432:
                                has_db_conn = True
                                break
                    except (psutil.AccessDenied, AttributeError):
                        pass

                    jarvis_procs.append({
                        'pid': proc.pid,
                        'name': proc.name(),
                        'has_db_conn': has_db_conn,
                        'cmdline': cmdline[:80]
                    })

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if jarvis_procs:
            print(f"\nFound {len(jarvis_procs)} JARVIS processes:")
            for p in jarvis_procs:
                marker = "🔌" if p['has_db_conn'] else "  "
                print(f"{marker} PID {p['pid']}: {p['name']}")
                print(f"     {p['cmdline']}")
        else:
            print("✅ No JARVIS processes currently running")

    except ImportError:
        print("❌ psutil not available")
    except Exception as e:
        print(f"❌ Error: {e}")


async def kill_leaked_connections():
    """Kill leaked database connections"""
    print("\n" + "=" * 60)
    print("5. Kill Leaked Connections")
    print("=" * 60)

    try:
        import asyncpg
        from backend.core.secret_manager import get_db_password

        password = get_db_password()
        if not password:
            print("❌ Could not get database password")
            return

        conn = await asyncpg.connect(
            host='127.0.0.1',
            port=5432,
            database='jarvis_learning',
            user='jarvis',
            password=password,
        )

        # Find leaked connections (idle >5 minutes)
        leaked = await conn.fetch("""
            SELECT pid, usename, state, state_change
            FROM pg_stat_activity
            WHERE datname = 'jarvis_learning'
              AND pid <> pg_backend_pid()
              AND usename = 'jarvis'
              AND state = 'idle'
              AND state_change < NOW() - INTERVAL '5 minutes'
        """)

        if leaked:
            print(f"Found {len(leaked)} leaked connections:")
            for row in leaked:
                print(f"   PID {row['pid']}: {row['state']} since {row['state_change']}")

            response = input("\nKill these connections? (y/N): ")
            if response.lower() == 'y':
                killed = 0
                for row in leaked:
                    try:
                        await conn.execute("SELECT pg_terminate_backend($1)", row['pid'])
                        print(f"✅ Killed PID {row['pid']}")
                        killed += 1
                    except Exception as e:
                        print(f"❌ Failed to kill PID {row['pid']}: {e}")
                print(f"\n✅ Killed {killed} connections")
        else:
            print("✅ No leaked connections found")

        await conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    parser = argparse.ArgumentParser(description="CloudSQL Connection Diagnostics")
    parser.add_argument('--kill-leaked', action='store_true', help="Kill leaked connections")
    parser.add_argument('--emergency', action='store_true', help="Emergency cleanup")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("CloudSQL Connection Diagnostics")
    print(f"Time: {datetime.now()}")
    print("=" * 60)

    # Run diagnostics
    check_proxy_running()
    await count_active_connections()
    await test_singleton_manager()
    check_process_connections()

    # Optional actions
    if args.kill_leaked:
        await kill_leaked_connections()

    if args.emergency:
        print("\n" + "=" * 60)
        print("EMERGENCY CLEANUP")
        print("=" * 60)

        response = input("\n⚠️  This will kill ALL JARVIS processes and connections. Continue? (y/N): ")
        if response.lower() == 'y':
            from process_cleanup_manager import emergency_cleanup
            results = emergency_cleanup(force=True)
            print("\n✅ Emergency cleanup complete")

    print("\n" + "=" * 60)
    print("Diagnostics Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
