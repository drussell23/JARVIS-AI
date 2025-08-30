#!/usr/bin/env python3
"""
Optimized Server Startup Script
Integrates smart startup manager with main server
"""

import asyncio
import sys
import os
import argparse
import uvicorn
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Import smart startup manager
from smart_startup_manager import startup_manager, smart_startup

async def start_server_with_optimization(host: str = "127.0.0.1", port: int = 8000):
    """Start the server with smart resource management"""
    print("🚀 Starting JARVIS server with optimizations...")
    
    # Run smart startup in background
    startup_task = asyncio.create_task(smart_startup())
    
    # Import main app after smart startup begins
    from main import app
    
    # Configure uvicorn
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        loop="asyncio"
    )
    
    server = uvicorn.Server(config)
    
    # Run server
    await server.serve()
    
    # Cleanup
    startup_task.cancel()
    try:
        await startup_task
    except asyncio.CancelledError:
        pass

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Start JARVIS with optimizations")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["JARVIS_OPTIMIZED"] = "true"
    
    # Run the server
    asyncio.run(start_server_with_optimization(args.host, args.port))

if __name__ == "__main__":
    main()