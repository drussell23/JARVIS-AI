#!/usr/bin/env python3
"""
Debug Unlock Flow
=================

Traces the exact flow when processing unlock commands.
"""

import asyncio
import logging

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Reduce noise from some modules
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def test_unlock_flow():
    """Test the unlock command flow with detailed logging"""
    
    logger.info("=== Starting Unlock Flow Debug ===")
    
    # Import modules to trace the flow
    try:
        logger.info("1. Importing unified command processor...")
        from api.unified_command_processor import UnifiedCommandProcessor
        
        logger.info("2. Creating processor instance...")
        processor = UnifiedCommandProcessor()
        
        logger.info("3. Processing 'unlock my screen' command...")
        result = await processor.process_command("unlock my screen")
        
        logger.info(f"4. Result: {result}")
        
    except Exception as e:
        logger.error(f"Error in unlock flow: {e}", exc_info=True)


async def test_voice_unlock_handler():
    """Test the voice unlock handler directly"""
    
    logger.info("=== Testing Voice Unlock Handler Directly ===")
    
    try:
        logger.info("1. Importing voice unlock handler...")
        from api.voice_unlock_handler import handle_voice_unlock_command
        
        logger.info("2. Calling handler with 'unlock my screen'...")
        result = await handle_voice_unlock_command("unlock my screen")
        
        logger.info(f"3. Handler result: {result}")
        
    except Exception as e:
        logger.error(f"Error in voice unlock handler: {e}", exc_info=True)


async def test_voice_unlock_integration():
    """Test the voice unlock integration directly"""
    
    logger.info("=== Testing Voice Unlock Integration ===")
    
    try:
        logger.info("1. Importing voice unlock integration...")
        from api.voice_unlock_integration import handle_voice_unlock_in_jarvis
        
        logger.info("2. Calling integration with 'unlock my screen'...")
        result = await handle_voice_unlock_in_jarvis("unlock my screen")
        
        logger.info(f"3. Integration result: {result}")
        
    except Exception as e:
        logger.error(f"Error in voice unlock integration: {e}", exc_info=True)


async def main():
    """Run all tests"""
    await test_unlock_flow()
    await test_voice_unlock_handler()
    await test_voice_unlock_integration()


if __name__ == "__main__":
    asyncio.run(main())