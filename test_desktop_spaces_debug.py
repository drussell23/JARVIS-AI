#!/usr/bin/env python3
"""
Debug script to test desktop spaces query and capture the actual error
"""

import asyncio
import logging
import sys
import os

# Set up logging to see detailed output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('desktop_spaces_debug.log')
    ]
)

logger = logging.getLogger(__name__)

async def test_desktop_spaces_query():
    """Test the desktop spaces query with full error details"""
    try:
        logger.info("=" * 80)
        logger.info("STARTING DESKTOP SPACES DEBUG TEST")
        logger.info("=" * 80)
        
        # Import the vision command handler
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
        
        from api.vision_command_handler import VisionCommandHandler
        
        logger.info("Creating VisionCommandHandler...")
        handler = VisionCommandHandler()
        
        # Initialize if needed
        if hasattr(handler, 'initialize_intelligence'):
            logger.info("Initializing intelligence...")
            await handler.initialize_intelligence()
        
        # Test query
        test_query = "What's happening across my desktop spaces?"
        logger.info(f"\nTesting query: {test_query}")
        logger.info("-" * 80)
        
        # Call analyze_screen
        result = await handler.analyze_screen(test_query)
        
        logger.info("\n" + "=" * 80)
        logger.info("RESULT:")
        logger.info("=" * 80)
        logger.info(f"Handled: {result.get('handled')}")
        logger.info(f"Response: {result.get('response')}")
        
        if 'error' in result:
            logger.error(f"Error: {result.get('error')}")
            logger.error(f"Error Type: {result.get('error_type')}")
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST COMPLETE")
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error(f"EXCEPTION IN TEST SCRIPT:")
        logger.error(f"{'=' * 80}")
        logger.error(f"Type: {type(e).__name__}")
        logger.error(f"Message: {str(e)}")
        logger.error(f"\nFull traceback:")
        import traceback
        traceback.print_exc()
        logger.error(f"{'=' * 80}")
        return None

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("JARVIS Desktop Spaces Debug Test")
    print("=" * 80)
    print("\nThis script will:")
    print("1. Test the desktop spaces query")
    print("2. Capture full error details")
    print("3. Log everything to desktop_spaces_debug.log")
    print("\nStarting test...\n")
    
    result = asyncio.run(test_desktop_spaces_query())
    
    print("\n" + "=" * 80)
    if result:
        if result.get('handled'):
            print("✅ SUCCESS")
        else:
            print("❌ FAILED")
    else:
        print("❌ EXCEPTION OCCURRED")
    print("=" * 80)
    print("\nCheck desktop_spaces_debug.log for full details")
