#!/usr/bin/env python3
"""
Quick Fix: Update JARVIS Screen Unlock Password
No popups, just command line.
"""

import asyncio
import getpass
import subprocess
import sys


async def update_password():
    """Update the password stored in keychain"""
    
    print("\n" + "="*60)
    print("🔧 FIX JARVIS UNLOCK PASSWORD")
    print("="*60)
    print("\nThis will update the password in macOS Keychain.")
    print("Enter your CORRECT macOS login password below.\n")
    
    # Get the correct password
    password = getpass.getpass("Enter your macOS login password: ")
    confirm = getpass.getpass("Confirm password: ")
    
    if password != confirm:
        print("❌ Passwords don't match. Try again.")
        return False
    
    if not password:
        print("❌ Password cannot be empty.")
        return False
    
    print("\n🔄 Updating keychain...")
    
    # Update the keychain entry
    service_name = "JARVIS_Screen_Unlock"
    account_name = "jarvis_user"
    
    # Delete old entry first
    delete_cmd = [
        "security",
        "delete-generic-password",
        "-a", account_name,
        "-s", service_name
    ]
    
    try:
        subprocess.run(delete_cmd, capture_output=True)
        print("   ✅ Removed old password")
    except:
        pass  # Old entry might not exist
    
    # Add new entry
    add_cmd = [
        "security",
        "add-generic-password",
        "-a", account_name,
        "-s", service_name,
        "-w", password,
        "-T", "/usr/bin/security",
        "-U",
        "-l", "JARVIS Screen Unlock"
    ]
    
    result = subprocess.run(add_cmd, capture_output=True)
    
    if result.returncode == 0:
        print("   ✅ New password stored in keychain")
        
        # Test retrieval
        test_cmd = [
            "security",
            "find-generic-password",
            "-a", account_name,
            "-s", service_name,
            "-w"
        ]
        
        test_result = subprocess.run(test_cmd, capture_output=True, text=True)
        
        if test_result.returncode == 0:
            retrieved = test_result.stdout.strip()
            if retrieved == password:
                print("   ✅ Password verified - retrieval working!")
                print("\n" + "="*60)
                print("✅ SUCCESS! Password updated.")
                print("="*60)
                print("\nTry unlocking now:")
                print('  Say: "Jarvis, unlock my screen"')
                print("\nor test with:")
                print("  python backend/macos_keychain_unlock.py")
                print("\n")
                return True
            else:
                print("   ⚠️  Warning: Retrieved password doesn't match")
                print("   Keychain may have issues. Try running:")
                print("   security delete-generic-password -a jarvis_user -s JARVIS_Screen_Unlock")
                return False
        else:
            print("   ⚠️  Could not verify password retrieval")
            return False
    else:
        error = result.stderr.decode()
        print(f"   ❌ Failed to store password: {error}")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(update_password())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
