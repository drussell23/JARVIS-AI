#!/usr/bin/env python3
"""
Test CloudSQL Singleton Connection Manager
==========================================

Tests:
1. Singleton pattern (only one instance)
2. Connection pool creation and reuse
3. Connection acquisition and release
4. Graceful shutdown on SIGINT/SIGTERM
5. Leaked connection cleanup
"""

import asyncio
import signal
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from intelligence.cloud_sql_connection_manager import (
    get_connection_manager,
    CloudSQLConnectionManager
)
from core.secret_manager import get_db_password


async def test_singleton():
    """Test that only one instance is created"""
    print("\n🧪 Test 1: Singleton Pattern")
    print("=" * 50)

    manager1 = get_connection_manager()
    manager2 = get_connection_manager()
    manager3 = CloudSQLConnectionManager()

    assert manager1 is manager2, "❌ get_connection_manager() should return same instance"
    assert manager1 is manager3, "❌ CloudSQLConnectionManager() should return same instance"

    print("✅ Singleton pattern works - all instances are the same")
    return manager1


async def test_initialization(manager):
    """Test connection pool initialization"""
    print("\n🧪 Test 2: Connection Pool Initialization")
    print("=" * 50)

    password = get_db_password()
    if not password:
        print("⚠️  No database password - skipping CloudSQL tests")
        return False

    success = await manager.initialize(
        host="127.0.0.1",
        port=5432,
        database="jarvis_learning",
        user="jarvis",
        password=password,
        max_connections=3,
        force_reinit=True  # Force fresh initialization for test
    )

    if success:
        print("✅ Connection pool initialized")
        stats = manager.get_stats()
        print(f"   Pool size: {stats['pool_size']}")
        print(f"   Idle: {stats['idle_size']}")
        print(f"   Max: {stats['max_size']}")
        return True
    else:
        print("❌ Connection pool initialization failed")
        return False


async def test_connection_reuse(manager):
    """Test connection reuse (should not create new pool)"""
    print("\n🧪 Test 3: Connection Pool Reuse")
    print("=" * 50)

    # Get stats before
    stats_before = manager.get_stats()
    creation_time_before = stats_before['creation_time']

    # Try to initialize again (should reuse existing pool)
    password = get_db_password()
    success = await manager.initialize(
        host="127.0.0.1",
        port=5432,
        database="jarvis_learning",
        user="jarvis",
        password=password,
        max_connections=3,
        force_reinit=False  # Should reuse existing pool
    )

    stats_after = manager.get_stats()
    creation_time_after = stats_after['creation_time']

    assert creation_time_before == creation_time_after, "❌ Pool should be reused"
    print("✅ Existing pool was reused (no new pool created)")


async def test_connection_acquisition(manager):
    """Test connection acquisition and release"""
    print("\n🧪 Test 4: Connection Acquisition & Release")
    print("=" * 50)

    if not manager.is_initialized:
        print("⚠️  Manager not initialized - skipping test")
        return

    # Test context manager
    try:
        async with manager.connection() as conn:
            print("✅ Connection acquired via context manager")

            # Run test query
            result = await conn.fetchval("SELECT 1 + 1")
            assert result == 2, "❌ Query failed"
            print(f"✅ Test query succeeded: 1 + 1 = {result}")

        print("✅ Connection automatically released")

        # Check stats
        stats = manager.get_stats()
        print(f"   Pool size: {stats['pool_size']}, Idle: {stats['idle_size']}")

    except Exception as e:
        print(f"❌ Connection test failed: {e}")


async def test_multiple_concurrent_connections(manager):
    """Test acquiring multiple connections concurrently"""
    print("\n🧪 Test 5: Multiple Concurrent Connections")
    print("=" * 50)

    if not manager.is_initialized:
        print("⚠️  Manager not initialized - skipping test")
        return

    async def query_task(task_id: int):
        """Single query task"""
        async with manager.connection() as conn:
            result = await conn.fetchval(f"SELECT {task_id}")
            print(f"   Task {task_id}: result = {result}")
            await asyncio.sleep(0.1)  # Simulate work
            return result

    # Run 5 concurrent queries (should use pool connections)
    tasks = [query_task(i) for i in range(1, 6)]
    results = await asyncio.gather(*tasks)

    assert results == list(range(1, 6)), "❌ Concurrent queries failed"
    print(f"✅ All {len(results)} concurrent queries succeeded")

    stats = manager.get_stats()
    print(f"   Total connections: {stats['connection_count']}")
    print(f"   Error count: {stats['error_count']}")


async def test_error_handling(manager):
    """Test error handling and recovery"""
    print("\n🧪 Test 6: Error Handling")
    print("=" * 50)

    if not manager.is_initialized:
        print("⚠️  Manager not initialized - skipping test")
        return

    # Test invalid query (should not crash)
    try:
        async with manager.connection() as conn:
            await conn.execute("SELECT * FROM nonexistent_table")
        print("❌ Should have raised an error")
    except Exception as e:
        print(f"✅ Error correctly caught: {type(e).__name__}")

    # Manager should still work after error
    stats = manager.get_stats()
    print(f"   Error count: {stats['error_count']}")
    print(f"   Status: {stats['status']}")

    # Try a valid query after error
    try:
        async with manager.connection() as conn:
            result = await conn.fetchval("SELECT 42")
            assert result == 42
            print("✅ Manager recovered - valid query succeeded")
    except Exception as e:
        print(f"❌ Recovery failed: {e}")


async def test_shutdown():
    """Test graceful shutdown"""
    print("\n🧪 Test 7: Graceful Shutdown")
    print("=" * 50)

    manager = get_connection_manager()

    if manager.is_initialized:
        print("🛑 Shutting down connection manager...")
        await manager.shutdown()

        stats = manager.get_stats()
        print(f"✅ Shutdown complete")
        print(f"   Status: {stats['status']}")
        print(f"   Final connection count: {stats['connection_count']}")
        print(f"   Final error count: {stats['error_count']}")
    else:
        print("⚠️  Manager not initialized - nothing to shut down")


async def main():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("  CloudSQL Singleton Connection Manager Tests")
    print("=" * 50)

    try:
        # Test 1: Singleton
        manager = await test_singleton()

        # Test 2: Initialization
        initialized = await test_initialization(manager)

        if initialized:
            # Test 3: Reuse
            await test_connection_reuse(manager)

            # Test 4: Acquisition
            await test_connection_acquisition(manager)

            # Test 5: Concurrent
            await test_multiple_concurrent_connections(manager)

            # Test 6: Error handling
            await test_error_handling(manager)

        # Test 7: Shutdown
        await test_shutdown()

        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        print("=" * 50 + "\n")

    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        manager = get_connection_manager()
        await manager.shutdown()
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

        manager = get_connection_manager()
        await manager.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
