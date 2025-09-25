#!/usr/bin/env python3
"""
Debug Context Intelligence Integration
======================================

Tests if screen state is being checked.
"""

import asyncio
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Filter out noisy loggers
for logger_name in ['websockets', 'asyncio', 'urllib3', 'httpcore']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

async def test_context_flow():
    """Test Context Intelligence flow"""
    print("\n🔍 TESTING CONTEXT INTELLIGENCE FLOW")
    print("=" * 60)
    
    # Import components
    from context_intelligence.core.screen_state import ScreenStateDetector, ScreenState
    from context_intelligence.core.context_manager import ContextManager
    
    # Test 1: Check screen state detection
    print("\n1️⃣ Testing Screen State Detection")
    detector = ScreenStateDetector()
    state = await detector.get_screen_state()
    
    print(f"   Screen state: {state.state.value}")
    print(f"   Screen locked: {state.state == ScreenState.LOCKED}")
    print(f"   Confidence: {state.confidence:.2f}")
    print(f"   Method: {state.detection_method.value}")
    print(f"   Metadata: {state.metadata}")
    
    # Test 2: Test Context Manager
    print("\n2️⃣ Testing Context Manager")
    manager = ContextManager()
    
    # Simulate a command
    test_command = "open safari and search for dogs"
    print(f"   Test command: '{test_command}'")
    
    # Check what would happen
    from context_intelligence.analyzers.intent_analyzer import IntentAnalyzer
    analyzer = IntentAnalyzer()
    intent = await analyzer.analyze(test_command)
    
    print(f"\n   Intent analysis:")
    print(f"   - Type: {intent.type.value}")
    print(f"   - Requires screen: {intent.requires_screen}")
    print(f"   - Confidence: {intent.confidence:.2f}")
    
    # Test 3: Check the actual flow
    print("\n3️⃣ Testing Actual Command Flow")
    
    # Get current screen state
    current_state = await manager.system_monitor.get_states()
    print(f"   Current system state:")
    print(f"   - Screen locked: {current_state.get('screen_locked', 'unknown')}")
    print(f"   - Network: {current_state.get('network_connected', 'unknown')}")
    
    # Check if command would be queued
    if intent.requires_screen and current_state.get('screen_locked'):
        print(f"\n   ✅ Command WOULD be queued (screen is locked)")
    else:
        print(f"\n   ❌ Command would NOT be queued")
        print(f"      - Intent requires screen: {intent.requires_screen}")
        print(f"      - Screen is locked: {current_state.get('screen_locked')}")

async def main():
    """Run tests"""
    try:
        await test_context_flow()
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())