#!/usr/bin/env python3
"""
Test script to debug display connection flow
"""

import asyncio
import logging
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_display_connection():
    """Test the complete display connection flow"""
    logger.info("Starting display connection test...")

    # Import components
    from display.advanced_display_monitor import get_display_monitor
    from display.vision_ui_navigator import get_vision_navigator
    from api.display_voice_handler import get_display_voice_handler

    # Initialize components
    logger.info("Initializing components...")
    display_monitor = get_display_monitor()
    vision_navigator = get_vision_navigator()
    display_handler = get_display_voice_handler()

    # Initialize connections
    await display_handler.initialize()

    # Test command handling
    test_commands = [
        "Living Room TV",
        "Connect to Living Room TV",
        "living room tv"
    ]

    for command in test_commands:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing command: '{command}'")
        logger.info(f"{'='*50}")

        # Test the display voice handler
        result = await display_handler.handle_command(command)

        logger.info(f"Handler result: {result}")

        if result.get("handled"):
            logger.info(f"✅ Command handled: {result.get('message')}")
            if result.get("success"):
                logger.info("✅ Command successful!")
            else:
                logger.error(f"❌ Command failed: {result.get('message')}")
        else:
            logger.warning("⚠️ Command not handled by display voice handler")

    # Now test the actual display connection flow directly
    logger.info("\n" + "="*50)
    logger.info("Testing direct display connection...")
    logger.info("="*50)

    try:
        # Try to connect directly using the display monitor
        result = await display_monitor.connect_display("living_room_tv")
        logger.info(f"Direct connection result: {result}")

        if result.get("success"):
            logger.info("✅ Direct connection successful!")
        else:
            logger.error(f"❌ Direct connection failed: {result.get('message')}")
    except Exception as e:
        logger.error(f"❌ Exception during direct connection: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_display_connection())