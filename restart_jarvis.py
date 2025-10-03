#!/usr/bin/env python3
"""
Quick JARVIS Restart Script
===========================
Ensures JARVIS always runs with the latest code.
Usage: python restart_jarvis.py
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from jarvis_reload_manager import JARVISReloadManager


async def restart_jarvis():
    """Restart JARVIS with latest code"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    print("=" * 60)
    print("🔄 JARVIS INTELLIGENT RESTART")
    print("=" * 60)

    manager = JARVISReloadManager()

    # Always detect code changes
    print("\n📊 Checking for code changes...")
    has_changes, changed_files = manager.detect_code_changes()

    if has_changes:
        print(f"✅ Found {len(changed_files)} changed files:")
        for file in changed_files[:5]:
            print(f"   - {file}")
        if len(changed_files) > 5:
            print(f"   ... and {len(changed_files) - 5} more")
    else:
        print("ℹ️  No code changes detected")

    # Check for existing JARVIS
    existing = await manager.find_jarvis_process()
    if existing:
        print(f"\n🛑 Stopping existing JARVIS (PID: {existing.pid})...")
        await manager.stop_jarvis(force=True)
        print("✅ Existing JARVIS stopped")

    # Start new JARVIS
    print(f"\n🚀 Starting JARVIS with latest code...")
    success = await manager.start_jarvis()

    if success:
        print(f"✅ JARVIS started successfully!")
        print(f"📍 Running on port: {manager.config['port']}")
        print(f"🔗 URL: http://localhost:{manager.config['port']}")
        print("\n💡 JARVIS will auto-reload when you make code changes")
        print("Press Ctrl+C to stop")

        # Keep monitoring
        try:
            await manager.monitor_loop()
        except KeyboardInterrupt:
            print("\n👋 Shutting down JARVIS...")
            await manager.stop_jarvis()
    else:
        print("❌ Failed to start JARVIS")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(restart_jarvis())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)