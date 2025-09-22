#!/usr/bin/env python3
"""
Test Password Fix for Special Characters
========================================

This script tests the AppleScript password escaping fix.
"""

import subprocess
import time

def escape_password_for_applescript(password):
    """Escape special characters in password for AppleScript"""
    # Escape backslashes first, then quotes, then other special characters
    escaped = password.replace('\\', '\\\\')  # Escape backslashes
    escaped = escaped.replace('"', '\\"')     # Escape double quotes  
    escaped = escaped.replace("'", "\\'")     # Escape single quotes
    escaped = escaped.replace('$', '\\$')     # Escape dollar signs
    escaped = escaped.replace('`', '\\`')     # Escape backticks
    return escaped

def test_password_escaping():
    """Test password escaping with the actual stored password"""
    print("🔐 Testing Password Escaping Fix")
    print("================================")
    print()
    
    # Retrieve the actual password from Keychain
    print("🔑 Retrieving password from Keychain...")
    result = subprocess.run([
        'security', 'find-generic-password',
        '-s', 'com.jarvis.voiceunlock',
        '-a', 'unlock_token',
        '-w'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ No password found in Keychain")
        return False
        
    password = result.stdout.strip()
    print(f"✅ Password retrieved (length: {len(password)} characters)")
    print(f"Password contains special chars: {any(c in password for c in '#@!$`\\\"\'')}")
    
    # Test escaping
    escaped = escape_password_for_applescript(password)
    print(f"\n🔧 Original password: {password}")
    print(f"🔧 Escaped password: {escaped}")
    
    # Test script generation
    script = f'''
    tell application "System Events"
        keystroke "{escaped}"
    end tell
    '''
    
    print("\n📝 Generated AppleScript:")
    print(script)
    
    # Test with a safe target (Terminal or TextEdit)
    print("\n⚠️  Testing password typing in 5 seconds...")
    print("Open TextEdit or Terminal to see the result!")
    
    for i in range(5, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    # Execute the script
    result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("\n✅ AppleScript executed successfully!")
        print("Check if the password was typed correctly in your target application.")
        return True
    else:
        print(f"\n❌ AppleScript failed: {result.stderr}")
        return False

def main():
    print("🚀 Voice Unlock Password Fix Test")
    print("==================================")
    print()
    print("This test will:")
    print("1. Retrieve the stored password")
    print("2. Test the new escaping function")
    print("3. Execute a safe AppleScript test")
    print()
    
    success = test_password_escaping()
    
    if success:
        print("\n🎉 Password escaping fix appears to work!")
        print("\nNext steps:")
        print("1. Lock your screen")
        print("2. Try voice unlock: 'Hey JARVIS, unlock my screen'")
        print("3. The password should now be typed correctly")
    else:
        print("\n❌ Password escaping fix needs more work")

if __name__ == "__main__":
    main()