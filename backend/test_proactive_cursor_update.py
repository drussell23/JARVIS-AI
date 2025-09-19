#!/usr/bin/env python3
"""
Test Proactive Vision System - Cursor Update Detection
Demonstrates how JARVIS proactively notifies about Cursor updates
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Add backend to path
sys.path.append('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_cursor_update_detection():
    """Test the Cursor update detection scenario"""
    
    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return
        
    try:
        # Import required components
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        from vision.proactive_vision_integration import create_proactive_vision_system
        from chatbots.claude_vision_chatbot import ClaudeVisionChatbot
        
        logger.info("=== PROACTIVE VISION SYSTEM TEST ===")
        logger.info("Scenario: Cursor Update Detection")
        logger.info("")
        
        # Initialize vision analyzer
        logger.info("1. Initializing Claude Vision Analyzer...")
        vision_analyzer = ClaudeVisionAnalyzer(api_key)
        
        # Create proactive vision system
        logger.info("2. Creating Proactive Vision System...")
        proactive_system = await create_proactive_vision_system(vision_analyzer)
        
        # Configure for testing
        proactive_system.update_config({
            'debug_mode': True,
            'test_mode': True,
            'importance_threshold': 0.5  # Lower threshold for testing
        })
        
        # Start monitoring
        logger.info("3. Starting proactive monitoring...")
        await proactive_system.start_proactive_monitoring({
            'activity': 'coding in Cursor',
            'focus_level': 0.6,
            'active_applications': ['Cursor', 'Terminal', 'Chrome']
        })
        
        logger.info("")
        logger.info("✅ Monitoring active. JARVIS is now watching for changes.")
        logger.info("")
        
        # Test Cursor update scenario
        logger.info("4. Testing Cursor update notification...")
        await proactive_system.test_cursor_update_scenario()
        
        # Simulate monitoring for a bit
        logger.info("")
        logger.info("5. Continuing to monitor for 10 seconds...")
        await asyncio.sleep(10)
        
        # Get statistics
        stats = proactive_system.get_system_stats()
        
        logger.info("")
        logger.info("=== MONITORING STATISTICS ===")
        logger.info(f"Notifications sent: {stats['system']['notifications_sent']}")
        logger.info(f"Notifications filtered: {stats['system']['notifications_filtered']}")
        logger.info(f"Filter rate: {stats['system']['filter_rate']:.1%}")
        logger.info(f"Current context: {stats['context']}")
        
        # Stop monitoring
        logger.info("")
        logger.info("6. Stopping monitoring...")
        await proactive_system.stop_proactive_monitoring()
        
        logger.info("")
        logger.info("✅ Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)


async def demonstrate_real_monitoring():
    """Demonstrate real proactive monitoring on actual screen"""
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return
        
    try:
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        from vision.proactive_vision_integration import create_proactive_vision_system
        
        logger.info("=== REAL PROACTIVE MONITORING DEMO ===")
        logger.info("JARVIS will monitor your screen and proactively notify you of:")
        logger.info("- Application updates")
        logger.info("- Error messages")
        logger.info("- Important notifications")
        logger.info("- Status changes")
        logger.info("")
        
        # Initialize
        vision_analyzer = ClaudeVisionAnalyzer(api_key)
        proactive_system = await create_proactive_vision_system(vision_analyzer)
        
        # Configure for real use
        proactive_system.update_config({
            'debug_mode': False,
            'enable_voice': True,
            'importance_threshold': 0.6
        })
        
        # Start monitoring
        await proactive_system.start_proactive_monitoring()
        
        logger.info("Monitoring started. Press Ctrl+C to stop.")
        logger.info("")
        
        # Monitor until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("\nStopping monitoring...")
            
        # Stop and show stats
        await proactive_system.stop_proactive_monitoring()
        
        stats = proactive_system.get_system_stats()
        logger.info("")
        logger.info(f"Session summary: {stats['system']['notifications_sent']} notifications sent")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    # Run test scenario
    print("\nChoose mode:")
    print("1. Test Cursor update scenario")
    print("2. Real monitoring demo")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(test_cursor_update_detection())
    elif choice == "2":
        asyncio.run(demonstrate_real_monitoring())
    else:
        print("Invalid choice")