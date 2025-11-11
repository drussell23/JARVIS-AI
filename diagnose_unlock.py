#!/usr/bin/env python3
"""
Comprehensive unlock diagnostic tool
Run this with your screen LOCKED to see what happens
"""
import asyncio
import logging
import sys

sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    print("\n" + "="*70)
    print(" JARVIS UNLOCK DIAGNOSTIC TOOL")
    print("="*70)
    
    print("\n🔒 This script will:")
    print("   1. Automatically LOCK your screen")
    print("   2. Wait 3 seconds")
    print("   3. Test the unlock functionality")
    print("   4. Type your password to unlock\n")
    
    input("Press Enter to start the test...")
    
    print("\n" + "="*70)
    print("STARTING DIAGNOSTIC...")
    print("="*70 + "\n")
    
    # Test 1: Check keychain
    print("1. Checking keychain password...")
    import subprocess
    result = subprocess.run(
        ["security", "find-generic-password", "-s", "JARVIS_Screen_Unlock", "-a", "jarvis_user", "-w"],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        pwd_len = len(result.stdout.strip())
        print(f"   ✅ Password found ({pwd_len} characters)")
    else:
        print(f"   ❌ Password NOT found!")
        print(f"   Run: python3 backend/macos_keychain_unlock.py")
        return
    
    # Test 2: Lock the screen automatically
    print("\n2. Locking your screen...")
    from macos_keychain_unlock import MacOSKeychainUnlock
    
    unlock_service = MacOSKeychainUnlock()
    
    # Lock the screen using AppleScript
    lock_script = """
    tell application "System Events"
        keystroke "q" using {command down, control down}
    end tell
    """
    
    lock_result = subprocess.run(
        ["osascript", "-e", lock_script],
        capture_output=True,
        text=True
    )
    
    if lock_result.returncode == 0:
        print("   ✅ Screen lock command sent")
    else:
        print(f"   ⚠️  Screen lock command failed: {lock_result.stderr}")
        print("   Trying alternative method...")
    
    # Wait for lock to take effect
    print("   ⏳ Waiting 3 seconds for screen to lock...")
    await asyncio.sleep(3)
    
    # Verify it's locked
    is_locked = await unlock_service.check_screen_locked()
    
    if is_locked:
        print("   ✅ Screen is now LOCKED!")
    else:
        print("   ⚠️  Screen may not be locked, but continuing anyway...")
        print("   (Lock detection might not work while script is running)")
    
    # Test 3: Test password retrieval
    print("\n3. Testing password retrieval...")
    password = await unlock_service.get_password_from_keychain()
    
    if password:
        print(f"   ✅ Password retrieved ({len(password)} characters)")
    else:
        print("   ❌ Failed to retrieve password")
        return
    
    # Test 4: Check secure_password_typer import
    print("\n4. Testing secure_password_typer import...")
    try:
        from voice_unlock.secure_password_typer import get_secure_typer, TypingConfig
        print("   ✅ secure_password_typer imported successfully")
        
        typer = get_secure_typer()
        print("   ✅ Typer instance created")
    except ImportError as e:
        print(f"   ❌ Failed to import secure_password_typer: {e}")
        print("   Falling back to AppleScript method...")
    except Exception as e:
        print(f"   ❌ Error creating typer: {e}")
    
    # Test 5: Perform actual unlock
    print("\n5. Performing actual unlock...")
    print("   🔓 Watch your screen - the password should be typed automatically!")
    print("   ⏳ Waiting 2 seconds...")
    await asyncio.sleep(2)
    
    print("   🔑 Typing password now...")
    
    result = await unlock_service.unlock_screen(verified_speaker="Derek")
    
    print(f"\n   Result: {result}")
    
    if result['success']:
        print("\n   ✅ UNLOCK SUCCESSFUL!")
        print(f"   Message: {result['message']}")
    else:
        print(f"\n   ❌ UNLOCK FAILED!")
        print(f"   Message: {result['message']}")
        print(f"   Action: {result.get('action')}")
    
    # Test 6: Verify screen is now unlocked
    print("\n6. Verifying unlock...")
    await asyncio.sleep(1)
    
    still_locked = await unlock_service.check_screen_locked()
    
    if not still_locked:
        print("   ✅ Screen is now UNLOCKED!")
    else:
        print("   ❌ Screen is STILL LOCKED!")
        print("   The unlock attempt may have failed.")
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70 + "\n")
    
    if result['success'] and not still_locked:
        print("✅ Everything is working correctly!")
        print("   Your 'unlock my screen' voice command should work.")
    else:
        print("❌ There's an issue with the unlock functionality.")
        print("   Check the logs above for details.")

if __name__ == "__main__":
    asyncio.run(main())
