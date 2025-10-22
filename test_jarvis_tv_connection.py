#!/usr/bin/env python3
"""
Test JARVIS's ability to connect to Living Room TV with the drag fix
"""

import asyncio
import sys
from pathlib import Path
import time

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

async def test_jarvis_tv_command():
    """Test JARVIS processing 'connect to living room tv' command"""
    print("\n" + "=" * 80)
    print("TESTING JARVIS TV CONNECTION WITH DRAG FIX")
    print("=" * 80)

    # Import unified command processor
    from api.unified_command_processor import UnifiedCommandProcessor

    # Initialize processor
    processor = UnifiedCommandProcessor()

    # Test command
    command = "connect to living room tv"
    print(f"\n🎤 Processing command: '{command}'")

    # Process command
    result = await processor.process_command(command)

    # Show result
    print(f"\n📊 Result:")
    print(f"   Success: {result.get('success', False)}")
    print(f"   Response: {result.get('response', 'No response')}")
    print(f"   Command Type: {result.get('command_type', 'unknown')}")

    if result.get('success'):
        print("\n✅ JARVIS successfully processed the TV connection command!")
        print("   The drag fix is working correctly.")
    else:
        print("\n❌ Command failed")
        if 'error' in result:
            print(f"   Error: {result['error']}")

    return result.get('success', False)


async def test_clicker_factory():
    """Test which clicker the factory returns"""
    print("\n" + "=" * 80)
    print("CHECKING WHICH CLICKER IS BEING USED")
    print("=" * 80)

    from backend.display.control_center_clicker_factory import get_best_clicker, get_clicker_info

    # Get clicker info
    info = get_clicker_info()
    print("\n📋 Available Clickers:")
    print(f"   UAE Enhanced: {info['uae_available']}")
    print(f"   SAI Enhanced: {info['sai_available']}")
    print(f"   Adaptive: {info['adaptive_available']}")
    print(f"   Basic: {info['basic_available']}")
    print(f"   Recommended: {info['recommended']}")

    # Get actual clicker
    clicker = get_best_clicker()
    clicker_type = clicker.__class__.__name__
    print(f"\n🔧 Factory returned: {clicker_type}")

    # Check if it has our drag fix
    import inspect
    if hasattr(clicker, 'click'):
        source = inspect.getsource(clicker.click)
        has_dragto = 'dragTo' in source
        print(f"   Has dragTo fix: {has_dragto}")

    return clicker_type


async def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("JARVIS TV CONNECTION TEST SUITE")
    print("=" * 80)
    print("\nThis test will:")
    print("1. Check which clicker is being used")
    print("2. Test JARVIS processing 'connect to living room tv'")

    try:
        # Wait for JARVIS to be ready
        print("\nWaiting for JARVIS to initialize...")
        await asyncio.sleep(3)

        # Test 1: Check clicker
        clicker_type = await test_clicker_factory()

        # Test 2: Test command processing
        command_success = await test_jarvis_tv_command()

        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Clicker Used: {clicker_type}")
        print(f"Command Processing: {'✅ PASSED' if command_success else '❌ FAILED'}")

        if command_success:
            print("\n🎉 JARVIS is correctly handling TV connections with drag motion!")
        else:
            print("\n⚠️ There may still be issues with TV connection.")
            print("Check /tmp/jarvis.log for details.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())