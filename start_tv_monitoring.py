#!/usr/bin/env python3
"""
Start Living Room TV Monitoring
================================

Simple script to monitor for Living Room TV availability.
When detected, JARVIS will prompt: "Would you like to extend to Living Room TV?"

Usage:
    python3 start_tv_monitoring.py

Author: Derek Russell
Date: 2025-10-15
"""

import asyncio
import sys
import signal
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from display.simple_tv_monitor import get_tv_monitor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def main():
    """Main function"""
    print("=" * 70)
    print("🖥️  JARVIS Living Room TV Monitor")
    print("=" * 70)
    print()
    print("📺 Monitoring for: Living Room TV")
    print("⏰ Check interval: Every 10 seconds")
    print()
    print("When your TV becomes available, JARVIS will prompt you to connect.")
    print("Press Ctrl+C to stop monitoring.")
    print()
    print("=" * 70)
    print()

    # Get TV monitor
    monitor = get_tv_monitor("Living Room TV")

    # Start monitoring
    await monitor.start()

    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\n🛑 Stopping TV monitor...")
        asyncio.create_task(monitor.stop())
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping TV monitor...")
        await monitor.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ TV monitoring stopped")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
