#!/usr/bin/env python3
"""
Test the integrated proactive vision system in ClaudeVisionAnalyzer
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


async def test_integrated_proactive_vision():
    """Test the integrated proactive vision system"""
    
    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return
        
    try:
        # Import the integrated ClaudeVisionAnalyzer
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        
        logger.info("=== INTEGRATED PROACTIVE VISION TEST ===")
        logger.info("")
        
        # Initialize analyzer with proactive features
        logger.info("1. Initializing ClaudeVisionAnalyzer with proactive features...")
        analyzer = ClaudeVisionAnalyzer(api_key)
        
        # Verify proactive configuration is loaded
        logger.info("2. Checking proactive configuration...")
        if hasattr(analyzer, '_proactive_config'):
            logger.info(f"✅ Proactive config loaded: {analyzer._proactive_config}")
        else:
            logger.error("❌ Proactive config not found")
            return
            
        # Configure proactive settings
        logger.info("3. Updating proactive configuration...")
        analyzer.update_proactive_config({
            'importance_threshold': 0.5,  # Lower for testing
            'notification_style': 'balanced',
            'enable_voice': False,  # Disable voice for testing
            'analysis_interval': 5.0  # Slower for testing
        })
        
        # Start proactive monitoring
        logger.info("4. Starting proactive monitoring...")
        result = await analyzer.start_proactive_monitoring()
        logger.info(f"Start result: {result}")
        
        if not result.get('started'):
            logger.error(f"Failed to start monitoring: {result.get('reason')}")
            return
            
        logger.info("✅ Proactive monitoring started successfully!")
        logger.info("")
        
        # Monitor for a short time
        logger.info("5. Monitoring for 20 seconds...")
        logger.info("   (Make some changes on your screen to test detection)")
        await asyncio.sleep(20)
        
        # Get statistics
        logger.info("6. Getting proactive monitoring statistics...")
        stats = analyzer.get_proactive_stats()
        logger.info(f"Statistics: {stats}")
        
        # Test user response handling
        logger.info("7. Testing user response handling...")
        response = await analyzer.handle_proactive_user_response("What did you notice?")
        logger.info(f"Response: {response}")
        
        # Test follow-up handling
        logger.info("8. Testing follow-up handling...")
        follow_up = await analyzer.handle_proactive_follow_up("Tell me more details")
        logger.info(f"Follow-up response: {follow_up}")
        
        # Stop monitoring
        logger.info("9. Stopping proactive monitoring...")
        stop_result = await analyzer.stop_proactive_monitoring()
        logger.info(f"Stop result: {stop_result}")
        
        logger.info("")
        logger.info("✅ Integration test completed successfully!")
        
        # Verify all integrated methods exist
        logger.info("")
        logger.info("10. Verifying all integrated methods...")
        integrated_methods = [
            'start_proactive_monitoring',
            'stop_proactive_monitoring',
            '_proactive_monitoring_loop',
            '_analyze_proactive_changes',
            '_process_proactive_change',
            '_should_notify_proactive',
            '_apply_communication_style',
            '_send_proactive_message',
            '_record_proactive_notification',
            '_calculate_importance_score',
            '_build_notification_context',
            '_send_progressive_disclosure',
            '_generate_contextual_prefix',
            'handle_proactive_user_response',
            'handle_proactive_follow_up',
            'update_proactive_config',
            'get_proactive_stats'
        ]
        
        missing_methods = []
        for method in integrated_methods:
            if not hasattr(analyzer, method):
                missing_methods.append(method)
                
        if missing_methods:
            logger.error(f"❌ Missing methods: {missing_methods}")
        else:
            logger.info("✅ All integrated methods verified!")
            
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)


async def test_proactive_scenarios():
    """Test specific proactive scenarios"""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return
        
    try:
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        
        logger.info("=== PROACTIVE SCENARIO TESTS ===")
        
        analyzer = ClaudeVisionAnalyzer(api_key)
        
        # Test 1: Communication style adaptation
        logger.info("\nTest 1: Communication Style Adaptation")
        
        # Create a test change object for communication style testing
        from vision.claude_vision_analyzer_main import ScreenChange, Priority, ChangeCategory
        test_change = ScreenChange(
            description="Update available",
            importance=Priority.MEDIUM,
            confidence=0.9,
            category=ChangeCategory.UPDATE,
            suggested_message="new update available",
            location="status bar",
            timestamp=datetime.now(),
            screenshot_hash="test123"
        )
        
        # Test minimal style
        analyzer.update_proactive_config({'notification_style': 'minimal'})
        message = analyzer._apply_communication_style("I noticed there's a new update available", test_change)
        logger.info(f"Minimal style: {message}")
        
        # Test conversational style
        analyzer.update_proactive_config({'notification_style': 'conversational'})
        message = analyzer._apply_communication_style("new update available", test_change)
        logger.info(f"Conversational style: {message}")
        
        # Test 2: Notification context building
        logger.info("\nTest 2: Notification Context Building")
        context = analyzer._build_notification_context()
        logger.info(f"Built context: user_activity={context.user_activity}, time={context.time_of_day}")
        
        # Test 3: Importance scoring
        logger.info("\nTest 3: Importance Scoring")
        
        test_change = ScreenChange(
            description="Cursor has a new update available",
            importance=Priority.MEDIUM,
            confidence=0.9,
            category=ChangeCategory.UPDATE,
            suggested_message="Cursor has a new update available",
            location="status bar",
            timestamp=datetime.now(),
            screenshot_hash="test123"
        )
        
        score = analyzer._calculate_importance_score(test_change)
        logger.info(f"Importance score for update notification: {score:.2f}")
        
        # Test error with high priority
        error_change = ScreenChange(
            description="Error in terminal",
            importance=Priority.HIGH,
            confidence=0.95,
            category=ChangeCategory.ERROR,
            suggested_message="Error detected in terminal",
            location="terminal window",
            timestamp=datetime.now(),
            screenshot_hash="test456"
        )
        
        error_score = analyzer._calculate_importance_score(error_change)
        logger.info(f"Importance score for error: {error_score:.2f}")
        
        logger.info("\n✅ Scenario tests completed!")
        
    except Exception as e:
        logger.error(f"Scenario test failed: {e}", exc_info=True)


if __name__ == "__main__":
    print("\nIntegrated Proactive Vision System Test")
    print("=" * 50)
    
    # Run scenario tests automatically
    asyncio.run(test_proactive_scenarios())