# Lifecycle Manager Integration Guide

## Overview

The new lifecycle management system ensures:
- **Single connection pool** across entire application
- **Automatic cleanup** on exit (SIGINT, SIGTERM, atexit)
- **No connection leaks** to CloudSQL
- **Graceful shutdown** handling

## Integration Steps for main.py

### 1. Add imports at the top of main.py

```python
from core.lifecycle_manager import (
    get_lifecycle_manager,
    initialize_jarvis_lifecycle,
    shutdown_jarvis_lifecycle
)
```

### 2. Update the main() function initialization

**BEFORE (current code around line 1904):**
```python
# Initialize database (will auto-start Cloud SQL proxy if needed)
db_adapter = await get_database_adapter()
```

**AFTER (replace with):**
```python
# Initialize lifecycle manager (handles all database connections)
lifecycle_manager = await initialize_jarvis_lifecycle()

# Get database adapter from lifecycle manager
db_adapter = lifecycle_manager.db_adapter
```

### 3. Update shutdown handling

**BEFORE (current code around line 1970-1979):**
```python
# Cleanup Cloud SQL database connections
try:
    if db_adapter and db_adapter.is_cloud:
        logger.info("🔐 Closing Cloud SQL database connections...")
        from intelligence.cloud_database_adapter import close_database_adapter

        await close_database_adapter()
        logger.info("✅ Database connections closed")
except Exception as e:
    logger.error(f"Failed to close database connections: {e}")
```

**AFTER (replace with):**
```python
# Shutdown lifecycle manager (handles all cleanup)
try:
    logger.info("🔐 Shutting down lifecycle manager...")
    await shutdown_jarvis_lifecycle()
    logger.info("✅ Lifecycle manager shutdown complete")
except Exception as e:
    logger.error(f"Failed to shutdown lifecycle manager: {e}")
```

### 4. Alternative: Simple Drop-in Replacement

If you want minimal changes, just update the shutdown section:

```python
# At the end of main() shutdown section (around line 1970):

# Cleanup Cloud SQL database connections
try:
    logger.info("🔐 Shutting down database connections...")

    # Use singleton connection manager shutdown
    from intelligence.cloud_sql_connection_manager import get_connection_manager

    conn_manager = get_connection_manager()
    if conn_manager.is_initialized:
        await conn_manager.shutdown()
        logger.info("✅ CloudSQL connection manager shutdown complete")

    # Also close database adapter
    from intelligence.cloud_database_adapter import close_database_adapter
    await close_database_adapter()

except Exception as e:
    logger.error(f"Failed to shutdown database connections: {e}")
```

## Testing the Integration

### 1. Test normal startup/shutdown:
```bash
cd backend
python main.py
# Press Ctrl+C to test graceful shutdown
```

### 2. Test connection manager directly:
```bash
python test_connection_manager.py
```

### 3. Verify connection cleanup:
```bash
# In one terminal:
python main.py

# In another terminal:
lsof -i :5432  # Should show JARVIS connections

# Stop JARVIS (Ctrl+C in first terminal)

# Check again:
lsof -i :5432  # Should show no JARVIS connections (only proxy)
```

## Expected Behavior

### On Startup:
```
🔧 Lifecycle Manager initialized
🛡️  Registering shutdown handlers...
✅ Shutdown handlers registered (SIGINT, SIGTERM, atexit)
🔌 Initializing database connections...
✅ Singleton CloudSQL connection manager loaded
🔌 Creating CloudSQL connection pool (max=3)...
🧹 Checking for leaked connections...
✅ No leaked connections found
✅ Connection pool created successfully
   Pool size: 1, Idle: 1
✅ CloudSQL connection pool initialized
```

### On Shutdown (Ctrl+C):
```
📡 Received SIGINT - initiating graceful shutdown...
🛑 Starting JARVIS graceful shutdown...
🔄 Shutting down hybrid database sync...
✅ Hybrid sync shutdown complete
🔌 Closing database adapter...
✅ Database adapter closed
🔌 Shutting down CloudSQL connection manager...
🔌 Closing connection pool...
✅ Connection pool closed
✅ Connection manager shutdown complete
✅ JARVIS graceful shutdown complete
```

## Connection Leak Prevention

The singleton pattern ensures:

1. **Only ONE pool** exists across:
   - `hybrid_database_sync.py`
   - `cloud_database_adapter.py`
   - Any other database code

2. **Automatic cleanup** on:
   - Normal shutdown
   - Ctrl+C (SIGINT)
   - Kill signal (SIGTERM)
   - Process exit (atexit)
   - Python crashes

3. **Leaked connection cleanup**:
   - Before creating pool, kills connections idle >5min
   - Max 3 connections (safe for db-f1-micro)
   - Automatic release via context managers

## Troubleshooting

### "remaining connection slots are reserved" error

**Old behavior:** Multiple pools created, connections leaked
**New behavior:** Singleton pool, max 3 connections, auto-cleanup

If you still see this:
1. Check for old JARVIS processes: `ps aux | grep jarvis`
2. Kill them: `pkill -9 -f main.py`
3. Run emergency cleanup: `python -c "from process_cleanup_manager import emergency_cleanup; emergency_cleanup()"`
4. Restart JARVIS

### Connection pool not initializing

Check logs for:
- Proxy not running: Auto-starts proxy if needed
- Wrong password: Check Secret Manager
- Firewall issues: Ensure 127.0.0.1:5432 is accessible

## Benefits

✅ **No more connection leaks** - Singleton pattern + automatic cleanup
✅ **Graceful shutdown** - Signal handlers ensure proper cleanup
✅ **Resource efficiency** - Max 3 connections (vs previous ~10-20)
✅ **Crash recovery** - Leaked connection cleanup on startup
✅ **Simple integration** - Drop-in replacement for existing code
