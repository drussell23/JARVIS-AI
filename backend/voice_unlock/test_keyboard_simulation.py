#!/usr/bin/env python3
"""
Test Keyboard Simulation for Voice Unlock
=========================================

This script tests if we can simulate keyboard input on the lock screen.
"""

import subprocess
import time
import sys

def test_keyboard_simulation():
    print("🔐 Testing Keyboard Simulation")
    print("==============================")
    print()
    
    # Check if we have accessibility permissions
    print("Checking accessibility permissions...")
    result = subprocess.run([
        'osascript', '-e', 
        'tell application "System Events" to return exists'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ No accessibility permissions!")
        print("Please grant Terminal/iTerm accessibility access in System Preferences")
        return False
    
    print("✅ Accessibility permissions granted")
    
    # Test 1: Simple key press
    print("\n📝 Test 1: Simulating simple keypress...")
    print("This will type 'Hello' in 3 seconds - click somewhere to see it")
    time.sleep(3)
    
    subprocess.run([
        'osascript', '-e',
        '''
        tell application "System Events"
            keystroke "Hello"
        end tell
        '''
    ])
    
    print("✅ Simple keypress test complete")
    
    # Test 2: Password entry simulation
    print("\n🔑 Test 2: Simulating password entry...")
    print("This will simulate typing a password and pressing Enter")
    print("WARNING: This will type 'TestPassword123' in 3 seconds")
    response = input("Continue? (y/n): ")
    
    if response.lower() == 'y':
        time.sleep(3)
        subprocess.run([
            'osascript', '-e',
            '''
            tell application "System Events"
                keystroke "TestPassword123"
                delay 0.5
                key code 36
            end tell
            '''
        ])
        print("✅ Password entry simulation complete")
    
    # Test 3: Check if we can detect lock screen
    print("\n🖥️  Test 3: Checking screen lock detection...")
    
    # Method 1: Check for ScreenSaver process
    result = subprocess.run([
        'ps', 'aux'
    ], capture_output=True, text=True)
    
    if 'ScreenSaver' in result.stdout:
        print("✅ Screen saver/lock screen detected (Method 1)")
    else:
        print("❌ No screen saver/lock screen detected (Method 1)")
    
    # Method 2: Using system_profiler
    result = subprocess.run([
        'system_profiler', 'SPDisplaysDataType'
    ], capture_output=True, text=True)
    
    print(f"Display status: {'Active' if 'Online: Yes' in result.stdout else 'Inactive'}")
    
    return True

def test_with_objective_c():
    """Test using the actual Objective-C unlock manager"""
    print("\n🔧 Test 4: Using Objective-C Unlock Manager...")
    print("This will attempt to retrieve the stored password from Keychain")
    
    # Check if password is stored
    result = subprocess.run([
        'security', 'find-generic-password',
        '-s', 'com.jarvis.voiceunlock',
        '-a', 'unlock_token',
        '-g'
    ], capture_output=True, text=True, stderr=subprocess.STDOUT)
    
    if 'password:' in result.stdout:
        print("✅ Password found in Keychain")
        # Don't print the actual password for security
        print("Password is stored and accessible")
    else:
        print("❌ No password found in Keychain")
        print("Run enable_screen_unlock.sh first")
    
    return 'password:' in result.stdout

def main():
    print("🚀 JARVIS Voice Unlock - Keyboard Simulation Test")
    print("=" * 50)
    print()
    print("This test will verify that we can:")
    print("1. Simulate keyboard input")
    print("2. Detect screen lock status")
    print("3. Access stored credentials")
    print()
    
    # Run tests
    if test_keyboard_simulation():
        print("\n✅ Basic keyboard simulation tests passed")
    
    if test_with_objective_c():
        print("\n✅ Keychain access test passed")
    
    print("\n📋 Next Steps:")
    print("1. Lock your screen (⌘+Control+Q)")
    print("2. Run the Voice Unlock system")
    print("3. Say 'Hello JARVIS, unlock my Mac'")
    print("\nThe system should type your password and unlock the screen.")

if __name__ == "__main__":
    main()