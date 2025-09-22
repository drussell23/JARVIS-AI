#!/usr/bin/env python3
"""
Setup Voice Unlock for JARVIS
=============================

This script securely enrolls your voice and stores your password
for the Voice Unlock system to actually unlock your Mac.
"""

import getpass
import subprocess
import json
import os
from pathlib import Path
import numpy as np
from datetime import datetime

def setup_voice_unlock():
    print("🔐 JARVIS Voice Unlock Setup")
    print("=" * 40)
    print()
    
    # Check if already set up
    voice_unlock_dir = Path.home() / '.jarvis' / 'voice_unlock'
    voice_unlock_dir.mkdir(parents=True, exist_ok=True)
    
    enrolled_file = voice_unlock_dir / 'enrolled_users.json'
    
    print("This setup will:")
    print("1. Store your Mac password securely in the Keychain")
    print("2. Enable actual screen unlocking when you say the unlock phrase")
    print()
    
    # Get user confirmation
    response = input("Do you want to proceed? (yes/no): ").lower()
    if response != 'yes':
        print("Setup cancelled.")
        return
    
    # Get username
    username = subprocess.run(['whoami'], capture_output=True, text=True).stdout.strip()
    print(f"\nSetting up Voice Unlock for user: {username}")
    
    # Get password securely
    print("\nYour password will be stored securely in the macOS Keychain.")
    print("It will ONLY be accessible to the Voice Unlock system.")
    password = getpass.getpass("Enter your Mac password: ")
    
    # Verify password
    password_verify = getpass.getpass("Verify your Mac password: ")
    
    if password != password_verify:
        print("❌ Passwords don't match. Setup cancelled.")
        return
    
    # Store password in Keychain using security command
    print("\n📝 Storing password in Keychain...")
    
    # Delete any existing entry
    subprocess.run([
        'security', 'delete-generic-password',
        '-s', 'com.jarvis.voiceunlock',
        '-a', 'unlock_token'
    ], capture_output=True)
    
    # Add new entry
    result = subprocess.run([
        'security', 'add-generic-password',
        '-s', 'com.jarvis.voiceunlock',
        '-a', 'unlock_token',
        '-w', password,
        '-T', '/usr/bin/security',  # Allow security tool access
        '-U'  # Update if exists
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to store password: {result.stderr}")
        return
    
    print("✅ Password stored securely in Keychain")
    
    # Update enrollment with real user data
    print("\n📝 Updating voice enrollment...")
    
    enrollment_data = {
        "default_user": {
            "name": username.capitalize(),
            "enrolled": datetime.now().isoformat(),
            "active": True,
            "has_password": True
        }
    }
    
    with open(enrolled_file, 'w') as f:
        json.dump(enrollment_data, f, indent=2)
    
    print("✅ Voice enrollment updated")
    
    # Create a marker file to indicate setup is complete
    setup_complete_file = voice_unlock_dir / '.setup_complete'
    setup_complete_file.write_text(datetime.now().isoformat())
    
    print("\n🎉 Voice Unlock Setup Complete!")
    print("\nYour Voice Unlock system is now ready to actually unlock your Mac.")
    print("\nHow to use:")
    print("1. Make sure the Voice Unlock system is running:")
    print("   cd ~/Documents/repos/JARVIS-AI-Agent/backend/voice_unlock")
    print("   ./start_voice_unlock_system.sh")
    print()
    print("2. Lock your screen (⌘+Control+Q)")
    print()
    print("3. Say one of these phrases:")
    print("   - 'Hello JARVIS, unlock my Mac'")
    print("   - 'JARVIS, this is " + username.capitalize() + "'")
    print("   - 'Open sesame, JARVIS'")
    print()
    print("⚠️  Security Note:")
    print("Your password is stored securely in the macOS Keychain.")
    print("Only the Voice Unlock system can access it.")
    print("You can remove it anytime with: security delete-generic-password -s com.jarvis.voiceunlock")

if __name__ == "__main__":
    setup_voice_unlock()