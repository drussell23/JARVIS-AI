#!/usr/bin/env python3
"""
Test Computer Use Integration with JARVIS
==========================================

Demonstrates the full Computer Use API integration with JARVIS voice system.

This script shows:
1. Traditional UAE-based connection (fast, coordinate-based)
2. Computer Use API connection (robust, vision-based)
3. Hybrid approach (intelligent selection)
4. Voice transparency throughout

Usage:
    python test_computer_use_integration.py [device_name] [--force-computer-use]

Examples:
    python test_computer_use_integration.py "Living Room TV"
    python test_computer_use_integration.py "Living Room TV" --force-computer-use

Author: Derek J. Russell
Date: January 2025
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.display.jarvis_computer_use_integration import get_jarvis_computer_use


class MockJARVISVoice:
    """Mock JARVIS voice engine for demonstration"""
    
    def speak(self, text: str):
        """Speak text (print to console in demo)"""
        print(f"\n🔊 [JARVIS]: {text}\n")


async def test_basic_connection(device_name: str, force_computer_use: bool = False):
    """
    Test basic display connection
    
    Args:
        device_name: Name of AirPlay device
        force_computer_use: Force use of Computer Use API
    """
    print("\n" + "=" * 80)
    print("🧪 TEST 1: Basic Display Connection")
    print("=" * 80)
    
    # Create JARVIS Computer Use integration
    integration = get_jarvis_computer_use(
        jarvis_voice_engine=MockJARVISVoice(),
        prefer_computer_use=force_computer_use,
        confidence_threshold=0.7
    )
    
    print(f"\n🎯 Target: {device_name}")
    print(f"   Mode: {'Computer Use API (forced)' if force_computer_use else 'Hybrid (UAE → Computer Use)'}")
    print("\nExecuting connection...\n")
    
    # Execute connection
    result = await integration.connect_to_display(
        device_name=device_name,
        mode="mirror",
        force_computer_use=force_computer_use
    )
    
    # Display results
    print("\n" + "-" * 80)
    print("📊 RESULT")
    print("-" * 80)
    print(f"✅ Success: {result['success']}")
    print(f"📝 Message: {result.get('message', 'N/A')}")
    print(f"🔧 Method Used: {result.get('method', 'unknown')}")
    print(f"⏱️  Duration: {result.get('duration', 0):.2f}s")
    
    if 'stats' in result:
        stats = result['stats']
        print(f"\n📈 Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    if result.get('fallback_used'):
        print("\n⚠️  Note: Fallback to Computer Use was triggered")
    
    return result


async def test_comparison(device_name: str):
    """
    Test comparison: UAE vs Computer Use
    
    Args:
        device_name: Name of AirPlay device
    """
    print("\n" + "=" * 80)
    print("🧪 TEST 2: Method Comparison (UAE vs Computer Use)")
    print("=" * 80)
    
    # Test 1: UAE-first (hybrid)
    print("\n📊 Attempt 1: Hybrid (UAE first, Computer Use fallback)")
    print("-" * 80)
    
    integration_hybrid = get_jarvis_computer_use(
        jarvis_voice_engine=MockJARVISVoice(),
        prefer_computer_use=False,  # UAE first
        confidence_threshold=0.7
    )
    
    result_hybrid = await integration_hybrid.connect_to_display(device_name)
    
    print(f"   Result: {'✅ Success' if result_hybrid['success'] else '❌ Failed'}")
    print(f"   Method: {result_hybrid.get('method', 'unknown')}")
    print(f"   Duration: {result_hybrid.get('duration', 0):.2f}s")
    
    # Wait a moment
    await asyncio.sleep(2)
    
    # Test 2: Computer Use only
    print("\n📊 Attempt 2: Computer Use API (forced)")
    print("-" * 80)
    
    integration_cu = get_jarvis_computer_use(
        jarvis_voice_engine=MockJARVISVoice(),
        prefer_computer_use=True,  # Computer Use only
        confidence_threshold=0.7
    )
    
    result_cu = await integration_cu.connect_to_display(
        device_name=device_name,
        force_computer_use=True
    )
    
    print(f"   Result: {'✅ Success' if result_cu['success'] else '❌ Failed'}")
    print(f"   Method: {result_cu.get('method', 'unknown')}")
    print(f"   Duration: {result_cu.get('duration', 0):.2f}s")
    
    # Comparison
    print("\n" + "=" * 80)
    print("📊 COMPARISON")
    print("=" * 80)
    print(f"Hybrid:        {'✅' if result_hybrid['success'] else '❌'} | "
          f"{result_hybrid.get('duration', 0):.2f}s | "
          f"{result_hybrid.get('method', 'unknown')}")
    print(f"Computer Use:  {'✅' if result_cu['success'] else '❌'} | "
          f"{result_cu.get('duration', 0):.2f}s | "
          f"{result_cu.get('method', 'unknown')}")
    
    return {
        'hybrid': result_hybrid,
        'computer_use': result_cu
    }


async def test_voice_transparency(device_name: str):
    """
    Test voice transparency during connection
    
    Args:
        device_name: Name of AirPlay device
    """
    print("\n" + "=" * 80)
    print("🧪 TEST 3: Voice Transparency")
    print("=" * 80)
    print("\nThis test demonstrates JARVIS providing voice updates")
    print("throughout the connection process.\n")
    
    # Create integration with verbose voice
    integration = get_jarvis_computer_use(
        jarvis_voice_engine=MockJARVISVoice(),
        prefer_computer_use=True,  # Use Computer Use for more verbose output
        confidence_threshold=0.7
    )
    
    print("🎙️  Listen for JARVIS voice updates:\n")
    
    result = await integration.connect_to_display(device_name, force_computer_use=True)
    
    print("\n✅ Voice transparency test complete")
    print(f"   Total voice updates: Many! (check output above)")
    print(f"   Connection result: {'Success' if result['success'] else 'Failed'}")
    
    return result


async def show_integration_stats():
    """Show integration statistics"""
    print("\n" + "=" * 80)
    print("📊 INTEGRATION STATISTICS")
    print("=" * 80)
    
    integration = get_jarvis_computer_use()
    stats = integration.get_stats()
    
    print(f"\n🔧 Configuration:")
    print(f"   Computer Use Enabled: {stats['computer_use_enabled']}")
    print(f"   Prefer Computer Use: {stats['prefer_computer_use']}")
    print(f"   UAE Confidence Threshold: {stats['confidence_threshold']}")
    print(f"   Components Initialized: {stats['components_initialized']}")
    
    if 'hybrid' in stats:
        print(f"\n📈 Hybrid Connector Stats:")
        hybrid = stats['hybrid']
        print(f"   Total Connections: {hybrid.get('total_connections', 0)}")
        print(f"   UAE Attempts: {hybrid.get('uae_attempts', 0)} "
              f"(Success Rate: {hybrid.get('uae_success_rate', 0):.1%})")
        print(f"   Computer Use Attempts: {hybrid.get('computer_use_attempts', 0)} "
              f"(Success Rate: {hybrid.get('computer_use_success_rate', 0):.1%})")
        print(f"   Fallback Triggers: {hybrid.get('fallback_triggers', 0)}")
        print(f"   Learning Events: {hybrid.get('learning_events', 0)}")
        print(f"   Overall Success Rate: {hybrid.get('overall_success_rate', 0):.1%}")
    
    if 'computer_use' in stats:
        print(f"\n🤖 Computer Use Stats:")
        cu = stats['computer_use']
        print(f"   Connections Attempted: {cu.get('connections_attempted', 0)}")
        print(f"   Connections Successful: {cu.get('connections_successful', 0)}")
        print(f"   Tool Calls Made: {cu.get('tool_calls_made', 0)}")
        print(f"   Screenshots Taken: {cu.get('screenshots_taken', 0)}")
        print(f"   Mouse Actions: {cu.get('mouse_actions', 0)}")
        print(f"   Keyboard Actions: {cu.get('keyboard_actions', 0)}")
        print(f"   Total Tokens Used: {cu.get('total_tokens_used', 0)}")


async def main():
    """Main test runner"""
    # Parse arguments
    device_name = "Living Room TV"
    force_computer_use = False
    
    if len(sys.argv) > 1:
        device_name = sys.argv[1]
    
    if "--force-computer-use" in sys.argv:
        force_computer_use = True
    
    print("\n" + "=" * 80)
    print("🎙️  JARVIS COMPUTER USE INTEGRATION TEST SUITE")
    print("=" * 80)
    print(f"\n🎯 Target Device: {device_name}")
    print(f"🔧 Force Computer Use: {force_computer_use}")
    print(f"🔑 API Key: {'✅ Set' if 'ANTHROPIC_API_KEY' in os.environ else '❌ Missing'}")
    
    if 'ANTHROPIC_API_KEY' not in os.environ:
        print("\n⚠️  WARNING: ANTHROPIC_API_KEY not set!")
        print("   Computer Use API will not be available.")
        print("   Only UAE-based connection will work.")
        print("\n   Set it with:")
        print("   export ANTHROPIC_API_KEY='your-api-key'\n")
    
    try:
        # Test 1: Basic connection
        await test_basic_connection(device_name, force_computer_use)
        
        # Only run comparison if API key is set
        if 'ANTHROPIC_API_KEY' in os.environ and not force_computer_use:
            await asyncio.sleep(3)  # Wait before next test
            
            # Test 2: Comparison
            # await test_comparison(device_name)  # Commented out to avoid double connection
            
            await asyncio.sleep(3)
            
            # Test 3: Voice transparency
            # await test_voice_transparency(device_name)  # Commented out for single test
        
        # Show stats
        await show_integration_stats()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 80)
        print("\nIntegration with JARVIS is ready!")
        print("\nTo use in your JARVIS system:")
        print("  from backend.display.jarvis_computer_use_integration import get_jarvis_computer_use")
        print("  integration = get_jarvis_computer_use(jarvis_voice_engine=your_voice_engine)")
        print("  result = await integration.connect_to_display('Living Room TV')")
        print("\n" + "=" * 80 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import os
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    asyncio.run(main())
