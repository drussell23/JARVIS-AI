#!/usr/bin/env python3
"""
Simple Command-Line Tool to Update JARVIS Screen Unlock Password
No popups, no GUI - just terminal-based password update
"""
import asyncio
import getpass
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

async def update_password():
    """Update the screen unlock password in Keychain"""
    from macos_keychain_unlock import MacOSKeychainUnlock
    
    print("\n" + "="*60)
    print("🔐 UPDATE JARVIS SCREEN UNLOCK PASSWORD")
    print("="*60)
    print("\nThis will update your screen unlock password in macOS Keychain.")
    print("Enter your CURRENT macOS login password (the correct one).\n")
    
    # Get the correct password from user
    password = getpass.getpass("Enter your macOS login password: ")
    
    if not password:
        print("❌ Password cannot be empty")
        return False
    
    # Confirm password
    password_confirm = getpass.getpass("Confirm password: ")
    
    if password != password_confirm:
        print("❌ Passwords don't match")
        return False
    
    # Update in Keychain
    print("\n🔄 Updating password in Keychain...")
    unlock_service = MacOSKeychainUnlock()
    success = await unlock_service.store_password_in_keychain(password)
    
    if success:
        print("✅ Password updated successfully in Keychain!")
        print(f"   - Service: {unlock_service.service_name}")
        print(f"   - Account: {unlock_service.account_name}")
        print("\n🎯 Testing password retrieval...")
        
        # Test retrieval
        test_pwd = await unlock_service.get_password_from_keychain()
        if test_pwd == password:
            print("✅ Password retrieval working correctly")
            print("\n✅ JARVIS can now unlock your screen with the correct password!")
            return True
        else:
            print("⚠️  Password was stored but retrieval test failed")
            return False
    else:
        print("❌ Failed to update password in Keychain")
        print("\nTroubleshooting:")
        print("1. Make sure you have permission to update Keychain items")
        print("2. Try running: security delete-generic-password -s JARVIS_Screen_Unlock")
        print("3. Then run this script again")
        return False


async def main():
    """Main entry point"""
    try:
        success = await update_password()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
