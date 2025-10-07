#!/usr/bin/env python3
"""
Quick test to check if screen lock detection works for document creation commands
"""

import sys
import asyncio
sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

async def test_screen_lock_detection():
    print("=" * 70)
    print("🔍 TESTING SCREEN LOCK DETECTION FOR DOCUMENT CREATION")
    print("=" * 70)
    print()
    
    # Step 1: Check if screen is locked
    print("STEP 1: Checking current screen lock status...")
    try:
        from context_intelligence.detectors.screen_lock_detector import get_screen_lock_detector
        detector = get_screen_lock_detector()
        is_locked = await detector.is_screen_locked()
        print(f"   {'🔒' if is_locked else '🔓'} Screen is: {'LOCKED' if is_locked else 'UNLOCKED'}")
    except Exception as e:
        print(f"   ❌ Error checking screen lock: {e}")
        return
    
    print()
    
    # Step 2: Test if document commands require screen
    print("STEP 2: Testing if document creation requires screen...")
    test_commands = [
        "write me an essay on robots",
        "create document about AI",
        "draft a paper on climate change"
    ]
    
    for cmd in test_commands:
        try:
            requires_screen = detector._command_requires_screen(cmd)
            print(f"   {'✅' if requires_screen else '❌'} '{cmd}'")
            print(f"      → Requires screen: {requires_screen}")
        except Exception as e:
            print(f"   ❌ Error checking command: {e}")
    
    print()
    
    # Step 3: Test full context check
    print("STEP 3: Full screen context check...")
    test_cmd = "write me an essay on robots"
    try:
        context = await detector.check_screen_context(test_cmd)
        print(f"   Command: '{test_cmd}'")
        print(f"   Screen locked: {context['screen_locked']}")
        print(f"   Requires screen: {context['command_requires_screen']}")
        print(f"   Requires unlock: {context['requires_unlock']}")
        if context['unlock_message']:
            print(f"   Unlock message: '{context['unlock_message']}'")
        
        print()
        if is_locked and context['requires_unlock']:
            print("   ✅ SCREEN LOCK DETECTION WORKING CORRECTLY!")
            print("   📢 JARVIS should say:", context['unlock_message'])
        elif is_locked and not context['requires_unlock']:
            print("   ❌ PROBLEM: Screen is locked but requires_unlock=False")
            print("   🐛 This is why JARVIS isn't announcing the unlock!")
        elif not is_locked:
            print("   ℹ️  Screen is not locked (test with locked screen)")
        
    except Exception as e:
        print(f"   ❌ Error in context check: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_screen_lock_detection())