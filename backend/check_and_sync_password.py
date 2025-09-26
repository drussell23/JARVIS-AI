#!/usr/bin/env python3
"""
Check and Sync Password Between Voice Unlock and Context Intelligence
====================================================================

This script checks if you have a password stored from Voice Unlock
and optionally copies it to Context Intelligence.
"""

import keyring
import subprocess
import sys
import getpass


def check_voice_unlock_password():
    """Check if password exists in Voice Unlock keychain"""
    try:
        # Try to get password from Voice Unlock keychain entry
        result = subprocess.run([
            'security', 'find-generic-password',
            '-s', 'com.jarvis.voiceunlock',
            '-a', 'unlock_token',
            '-w'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, None
    except Exception as e:
        print(f"Error checking Voice Unlock password: {e}")
        return False, None


def check_context_intelligence_password():
    """Check if password exists in Context Intelligence keychain"""
    try:
        password = keyring.get_password("JARVIS_Screen_Unlock", "jarvis_user")
        return password is not None
    except Exception:
        return False


def main():
    print("🔐 Password Check & Sync Tool")
    print("="*50)
    
    # Check Voice Unlock password
    print("\n1. Checking Voice Unlock password...")
    has_voice_unlock, voice_unlock_pw = check_voice_unlock_password()
    
    if has_voice_unlock:
        print("✅ Found password in Voice Unlock keychain")
    else:
        print("❌ No password found in Voice Unlock keychain")
    
    # Check Context Intelligence password
    print("\n2. Checking Context Intelligence password...")
    has_context_intel = check_context_intelligence_password()
    
    if has_context_intel:
        print("✅ Found password in Context Intelligence keychain")
    else:
        print("❌ No password found in Context Intelligence keychain")
    
    # Sync if needed
    if has_voice_unlock and not has_context_intel:
        print("\n🔄 Would you like to copy the Voice Unlock password to Context Intelligence?")
        print("This will allow Context Intelligence to unlock your screen.")
        response = input("Copy password? (y/n): ").lower().strip()
        
        if response == 'y':
            try:
                # Store in Context Intelligence keychain
                keyring.set_password("JARVIS_Screen_Unlock", "jarvis_user", voice_unlock_pw)
                print("✅ Password copied successfully!")
                print("Context Intelligence can now unlock your screen.")
            except Exception as e:
                print(f"❌ Error copying password: {e}")
        else:
            print("Skipped password sync.")
    
    elif has_voice_unlock and has_context_intel:
        print("\n✅ Both systems have passwords stored.")
        print("You're all set!")
    
    elif not has_voice_unlock and has_context_intel:
        print("\n✅ Context Intelligence already has a password.")
        print("You're all set!")
    
    else:
        print("\n⚠️  No passwords found in either system.")
        print("\nYou can store a password by either:")
        print("1. Running: ./voice_unlock/enable_screen_unlock.sh")
        print("2. Or using Python:")
        print('   python -c "from context_intelligence.core.unlock_manager import get_unlock_manager; get_unlock_manager().store_password(getpass.getpass())"')
    
    # Summary
    print("\n" + "="*50)
    print("Summary:")
    print(f"- Voice Unlock password: {'✅ Present' if has_voice_unlock else '❌ Missing'}")
    print(f"- Context Intelligence password: {'✅ Present' if has_context_intel else '❌ Missing'}")
    
    if has_voice_unlock or has_context_intel:
        print("\n✅ Your Context Intelligence System is ready to unlock screens!")


if __name__ == "__main__":
    main()