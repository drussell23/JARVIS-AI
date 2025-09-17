#!/usr/bin/env python3
"""Test the minimal to full upgrader."""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from minimal_to_full_upgrader import MinimalToFullUpgrader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Test the upgrader functionality."""
    print("Testing Minimal to Full Upgrader...")
    
    # Create upgrader with short check interval for testing
    upgrader = MinimalToFullUpgrader(check_interval=10)
    
    # Check if we're in minimal mode
    is_minimal = await upgrader._check_minimal_mode()
    print(f"Is minimal mode: {is_minimal}")
    
    # Check component readiness
    readiness = await upgrader._check_component_readiness()
    print(f"Component readiness: {readiness}")
    
    # Start the upgrader
    await upgrader.start()
    
    # Keep running for a bit to see if it works
    try:
        await asyncio.sleep(30)
    except KeyboardInterrupt:
        print("\nStopping...")
    
    await upgrader.stop()
    print("Test complete")

if __name__ == "__main__":
    asyncio.run(main())