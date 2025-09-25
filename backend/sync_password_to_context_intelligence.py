#!/usr/bin/env python3
"""
Sync Password from Voice Unlock to Context Intelligence
======================================================

Automatically copies your existing Voice Unlock password to
Context Intelligence so both systems can unlock your screen.
"""

import keyring
import subprocess
import sys


def sync_password():
    """Sync password from Voice Unlock to Context Intelligence"""
    print("🔄 Syncing password from Voice Unlock to Context Intelligence...")
    
    # Get password from Voice Unlock
    try:
        result = subprocess.run([
            'security', 'find-generic-password',
            '-s', 'com.jarvis.voiceunlock',
            '-a', 'unlock_token',
            '-w'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ No password found in Voice Unlock keychain")
            return False
            
        password = result.stdout.strip()
        print("✅ Found password in Voice Unlock keychain")
        
    except Exception as e:
        print(f"❌ Error reading Voice Unlock password: {e}")
        return False
    
    # Store in Context Intelligence keychain
    try:
        keyring.set_password("JARVIS_Screen_Unlock", "jarvis_user", password)
        print("✅ Password copied to Context Intelligence keychain")
        
        # Clear password from memory
        password = None
        
        return True
        
    except Exception as e:
        print(f"❌ Error storing password: {e}")
        return False


def verify_sync():
    """Verify both systems have passwords"""
    from context_intelligence.core.unlock_manager import get_unlock_manager
    
    manager = get_unlock_manager()
    if manager.has_stored_password():
        print("✅ Context Intelligence can now unlock your screen!")
        return True
    else:
        print("❌ Password sync failed")
        return False


def main():
    print("🔐 Password Sync Tool")
    print("="*50)
    print("This will copy your Voice Unlock password to Context Intelligence")
    print("so both systems can unlock your screen.\n")
    
    if sync_password():
        print("\n✅ Success! Verifying...")
        verify_sync()
        print("\n🎉 You're all set! Context Intelligence can now lock/unlock your screen.")
    else:
        print("\n❌ Sync failed. You may need to run:")
        print("./voice_unlock/enable_screen_unlock.sh")


if __name__ == "__main__":
    main()