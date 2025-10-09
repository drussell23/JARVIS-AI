#!/usr/bin/env python3
"""
Complete Screen Lock/Unlock Testing Suite
Tests both lock/unlock functionality and locked screen command detection
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from api.simple_unlock_handler import handle_unlock_command
from system_control.macos_controller import MacOSController


async def test_lock_screen():
    """Test locking the screen"""
    print("\n" + "="*60)
    print("TEST 1: Lock Screen")
    print("="*60)

    result = await handle_unlock_command("lock my screen")
    print(f"✓ Lock command result:")
    print(f"  Success: {result.get('success')}")
    print(f"  Response: {result.get('response')}")
    print(f"  Action: {result.get('action')}")

    # Wait for lock to take effect
    print("\n⏳ Waiting 2 seconds for lock to take effect...")
    await asyncio.sleep(2)

    return result.get('success', False)


async def test_unlock_screen():
    """Test unlocking the screen"""
    print("\n" + "="*60)
    print("TEST 2: Unlock Screen")
    print("="*60)

    result = await handle_unlock_command("unlock my screen")
    print(f"✓ Unlock command result:")
    print(f"  Success: {result.get('success')}")
    print(f"  Response: {result.get('response')}")
    print(f"  Method: {result.get('method', 'N/A')}")

    if 'setup_instructions' in result:
        print(f"  Setup Required: {result['setup_instructions']['description']}")
        print(f"  Command: {result['setup_instructions']['command']}")

    # Wait for unlock to take effect
    if result.get('success'):
        print("\n⏳ Waiting 3 seconds for unlock to take effect...")
        await asyncio.sleep(3)

    return result.get('success', False)


def test_locked_screen_detection():
    """Test that commands are blocked when screen is locked"""
    print("\n" + "="*60)
    print("TEST 3: Locked Screen Command Detection")
    print("="*60)

    controller = MacOSController()

    # Test various commands that should be blocked when locked
    test_commands = [
        ("open_application", lambda: controller.open_application("Safari")),
        ("open_url", lambda: controller.open_url("https://google.com")),
        ("close_application", lambda: controller.close_application("Safari")),
        ("open_new_tab", lambda: controller.open_new_tab()),
        ("open_file", lambda: controller.open_file("~/Desktop/test.txt")),
    ]

    results = []
    for cmd_name, cmd_func in test_commands:
        try:
            success, message = cmd_func()
            results.append({
                'command': cmd_name,
                'success': success,
                'message': message,
                'blocked': 'screen is locked' in message.lower()
            })
            print(f"  {cmd_name}:")
            print(f"    Success: {success}")
            print(f"    Message: {message}")
            print(f"    Blocked correctly: {results[-1]['blocked']}")
        except Exception as e:
            print(f"  {cmd_name}: ERROR - {e}")
            results.append({
                'command': cmd_name,
                'success': False,
                'error': str(e)
            })

    # Check if all commands were properly blocked
    all_blocked = all(r.get('blocked', False) for r in results if 'error' not in r)
    print(f"\n✓ All commands properly blocked when locked: {all_blocked}")

    return all_blocked


async def test_unlock_variations():
    """Test different unlock command variations"""
    print("\n" + "="*60)
    print("TEST 4: Unlock Command Variations")
    print("="*60)

    variations = [
        "unlock my screen",
        "unlock screen",
        "unlock the screen"
    ]

    results = []
    for cmd in variations:
        result = await handle_unlock_command(cmd)
        print(f"\n  Command: '{cmd}'")
        print(f"    Success: {result.get('success')}")
        print(f"    Response: {result.get('response')}")
        results.append(result.get('success', False))

    return all(results)


async def test_lock_variations():
    """Test different lock command variations"""
    print("\n" + "="*60)
    print("TEST 5: Lock Command Variations")
    print("="*60)

    variations = [
        "lock my screen",
        "lock screen",
        "lock the screen"
    ]

    results = []
    for cmd in variations:
        result = await handle_unlock_command(cmd)
        print(f"\n  Command: '{cmd}'")
        print(f"    Success: {result.get('success')}")
        print(f"    Response: {result.get('response')}")
        results.append(result.get('success', False))

        # Wait between locks
        await asyncio.sleep(1)

    return all(results)


async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("🧪 JARVIS SCREEN LOCK/UNLOCK COMPREHENSIVE TEST SUITE")
    print("="*80)

    print("\n⚠️  IMPORTANT: This test will lock and unlock your screen multiple times.")
    print("    Make sure you're ready before proceeding.\n")

    # Test lock variations
    test5_pass = await test_lock_variations()

    # Test unlock variations
    test4_pass = await test_unlock_variations()

    # Test locked screen detection (run when screen is unlocked)
    print("\n⚠️  The next test checks if commands are blocked when screen is locked.")
    print("    First, let's lock the screen...")

    lock_result = await test_lock_screen()

    if lock_result:
        # Screen should be locked now, test detection
        test3_pass = test_locked_screen_detection()

        # Now unlock for the final test
        print("\n⚠️  Now testing unlock functionality...")
        test2_pass = await test_unlock_screen()
    else:
        print("\n⚠️  Lock failed, skipping locked screen detection test")
        test3_pass = False
        test2_pass = False

    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)

    all_tests = [
        ("Lock Screen", lock_result),
        ("Unlock Screen", test2_pass),
        ("Locked Screen Detection", test3_pass),
        ("Unlock Command Variations", test4_pass),
        ("Lock Command Variations", test5_pass),
    ]

    for test_name, passed in all_tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test_name}")

    total_passed = sum(1 for _, p in all_tests if p)
    total_tests = len(all_tests)

    print(f"\n  Total: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed! Lock/unlock system is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
