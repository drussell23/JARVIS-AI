#!/usr/bin/env python3
"""
Manual Screen Unlock Test
========================

Test the actual screen unlock mechanism step by step.
"""

import subprocess
import time
import getpass


def wake_display():
    """Wake the display using multiple methods"""
    print("🔆 Waking display...")
    
    # Method 1: caffeinate
    subprocess.run(['caffeinate', '-u', '-t', '1'])
    
    # Method 2: Move mouse
    subprocess.run([
        'osascript', '-e',
        '''
        tell application "System Events"
            set mouse_x to 100
            set mouse_y to 100
        end tell
        '''
    ])
    
    print("✅ Display wake signal sent")
    time.sleep(1)


def type_password(password):
    """Type the password using AppleScript"""
    print("⌨️  Typing password...")
    
    # First, click to ensure we're in password field
    click_script = '''
    tell application "System Events"
        click at {640, 400}
    end tell
    '''
    
    subprocess.run(['osascript', '-e', click_script])
    time.sleep(0.5)
    
    # Type password
    type_script = f'''
    tell application "System Events"
        keystroke "{password}"
        delay 0.5
        key code 36
    end tell
    '''
    
    result = subprocess.run(
        ['osascript', '-e', type_script],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Password typed and Enter pressed")
    else:
        print(f"❌ Error: {result.stderr}")


def main():
    print("🔐 Manual Screen Unlock Test")
    print("============================")
    print()
    print("This will:")
    print("1. Retrieve your password from Keychain")
    print("2. Wake the display")
    print("3. Type the password")
    print("4. Press Enter")
    print()
    
    # Get password from Keychain
    print("🔑 Retrieving password from Keychain...")
    result = subprocess.run([
        'security', 'find-generic-password',
        '-s', 'com.jarvis.voiceunlock',
        '-a', 'unlock_token',
        '-w'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ No password found in Keychain")
        print("Run: ./enable_screen_unlock.sh")
        return
        
    password = result.stdout.strip()
    print("✅ Password retrieved")
    
    # Instructions
    print("\n📝 Instructions:")
    print("1. Lock your screen NOW (⌘+Control+Q)")
    print("2. Wait for the lock screen to appear")
    print("3. Press Enter here to unlock")
    print()
    input("Press Enter when ready...")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"Unlocking in {i}...")
        time.sleep(1)
    
    # Wake and unlock
    wake_display()
    type_password(password)
    
    print("\n🎯 Unlock attempt complete!")
    print("Check if your screen unlocked.")


if __name__ == "__main__":
    main()