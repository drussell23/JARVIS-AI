#!/usr/bin/env python3
"""
Simple test: Lock screen → Request essay → Verify unlock flow
"""
import asyncio
import subprocess
import time

async def main():
    print("\n" + "="*60)
    print(" CONTEXT AWARENESS TEST: Essay with Locked Screen")
    print("="*60)
    
    # Step 1: Lock screen
    print("\n[1] Locking screen...")
    subprocess.run(['osascript', '-e', 'tell application "System Events" to key code 12 using {control down, command down}'])
    await asyncio.sleep(2)
    
    # Step 2: Verify lock
    from context_intelligence.detectors.screen_lock_detector import get_screen_lock_detector
    detector = get_screen_lock_detector()
    is_locked = await detector.is_screen_locked()
    print(f"[2] Screen is locked: {is_locked}")
    
    if not is_locked:
        print("⚠️  Screen not locked! Exiting...")
        return
    
    # Step 3: Send command through processor
    print("\n[3] Sending command: 'write me an essay on dolphins'")
    print("    Expected flow:")
    print("    → Detect screen is locked")
    print("    → Speak: 'Your screen is locked. I'll unlock it...'")
    print("    → Unlock screen")
    print("    → Create document")
    print("\n[4] Processing...")
    
    from api.unified_command_processor import UnifiedCommandProcessor
    processor = UnifiedCommandProcessor()
    
    result = await processor.process_command("write me an essay on dolphins")
    
    # Step 4: Check results
    print("\n[5] Results:")
    print(f"    Success: {result.get('success')}")
    print(f"    Response: {result.get('response', result.get('message', 'N/A'))[:100]}")
    
    # Check if screen was unlocked
    is_still_locked = await detector.is_screen_locked()
    print(f"\n[6] Screen is now locked: {is_still_locked}")
    
    if is_locked and not is_still_locked:
        print("\n✅ SUCCESS! Screen was unlocked during command execution")
    elif is_still_locked:
        print("\n⚠️  Screen is still locked - unlock may not have occurred")
    
    return result

if __name__ == "__main__":
    asyncio.run(main())
