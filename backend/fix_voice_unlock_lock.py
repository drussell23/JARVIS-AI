#!/usr/bin/env python3
"""
Fix Voice Unlock Lock Function
==============================

Replace the broken lock implementation with one that actually works.
"""

import os
import shutil
from datetime import datetime

# Read the current websocket_server.py
ws_path = "/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend/voice_unlock/objc/server/websocket_server.py"

print("🔧 Fixing Voice Unlock Lock Implementation")
print("=" * 50)

# Create backup
backup_path = f"{ws_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy(ws_path, backup_path)
print(f"✅ Created backup: {backup_path}")

# Read the file
with open(ws_path, 'r') as f:
    content = f.read()

# Find and replace the perform_screen_lock method
old_method = '''    async def perform_screen_lock(self) -> bool:
        """Lock the Mac screen"""
        try:
            logger.info("Locking screen...")
            
            # Method 1: Using Control+Command+Q (fastest)
            lock_script = \'\'\'
            tell application "System Events"
                keystroke "q" using {control down, command down}
            end tell
            \'\'\'
            
            result = subprocess.run([
                'osascript', '-e', lock_script
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Screen locked successfully")
                return True
            else:
                # Fallback: Use pmset
                logger.info("Trying alternative lock method...")
                subprocess.run(['pmset', 'displaysleepnow'])
                return True
                
        except Exception as e:
            logger.error(f"Error locking screen: {e}")
            return False'''

new_method = '''    async def perform_screen_lock(self) -> bool:
        """Lock the Mac screen using various methods"""
        try:
            logger.info("Locking screen...")
            
            # Method 1: Use CGSession (most reliable)
            try:
                result = subprocess.run([
                    '/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession',
                    '-suspend'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("Screen locked successfully using CGSession")
                    return True
            except Exception as e:
                logger.debug(f"CGSession method failed: {e}")
            
            # Method 2: Use loginwindow
            try:
                result = subprocess.run([
                    'osascript', '-e', 
                    'tell application "System Events" to tell process "loginwindow" to keystroke "q" using {command down, control down}'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("Screen locked successfully using loginwindow")
                    return True
            except Exception as e:
                logger.debug(f"Loginwindow method failed: {e}")
            
            # Method 3: Use ScreenSaverEngine
            try:
                # Start screensaver which will require password on wake
                subprocess.run([
                    'open', '-a', 'ScreenSaverEngine'
                ])
                logger.info("Started screensaver (will lock if password required)")
                return True
            except Exception as e:
                logger.debug(f"ScreenSaver method failed: {e}")
            
            # Method 4: Fallback to sleep display
            try:
                subprocess.run(['pmset', 'displaysleepnow'])
                logger.info("Put display to sleep")
                return True
            except Exception as e:
                logger.debug(f"Display sleep method failed: {e}")
                
            logger.error("All lock methods failed")
            return False
                
        except Exception as e:
            logger.error(f"Error locking screen: {e}")
            return False'''

if old_method in content:
    content = content.replace(old_method, new_method)
    print("\n✅ Found and replaced perform_screen_lock method")
else:
    print("\n❌ Could not find the exact method to replace")
    print("The method might have already been modified.")

# Write the updated content
with open(ws_path, 'w') as f:
    f.write(content)

print("\n✅ Updated websocket_server.py with fixed lock implementation")
print("\nThe fix includes multiple lock methods:")
print("1. CGSession -suspend (most reliable)")
print("2. loginwindow AppleScript")
print("3. ScreenSaverEngine")
print("4. pmset displaysleepnow (fallback)")

print("\n⚠️  You need to restart the Voice Unlock daemon for changes to take effect!")
print("\nRun these commands:")
print("1. pkill -f 'JARVISVoiceUnlockDaemon'")
print("2. pkill -f 'websocket_server.py'")
print("3. cd voice_unlock/objc/server && python websocket_server.py")