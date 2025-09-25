#!/usr/bin/env python3
"""
Fix Screen Lock Detection
=========================

Replace Voice Unlock daemon check with Context Intelligence detection.
"""

import os
import shutil
from datetime import datetime

print("🔧 Fixing Screen Lock Detection")
print("=" * 50)

# Backup the original file
handler_path = "/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend/api/direct_unlock_handler.py"
backup_path = f"{handler_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copy(handler_path, backup_path)
print(f"✅ Created backup: {backup_path}")

# Read the file
with open(handler_path, 'r') as f:
    content = f.read()

# Find and replace the check_screen_locked_direct function
old_function = '''async def check_screen_locked_direct() -> bool:
    """Check if screen is locked via direct WebSocket"""
    try:
        logger.info("[DIRECT UNLOCK] Checking screen lock status via WebSocket")
        async with websockets.connect(VOICE_UNLOCK_WS_URL) as websocket:
            # Get status
            status_command = {
                "type": "command",
                "command": "get_status"
            }
            await websocket.send(json.dumps(status_command))
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            result = json.loads(response)
            logger.info(f"[DIRECT UNLOCK] Voice unlock status: {result}")
            
            if result.get("type") == "status" and result.get("success"):
                status = result.get("status", {})
                is_locked = status.get("isScreenLocked", False)
                logger.info(f"[DIRECT UNLOCK] Screen locked from daemon: {is_locked}")
                return is_locked
                
        return False
        
    except Exception as e:
        logger.error(f"Error checking screen lock: {e}")
        # Fallback to system check
        return check_screen_locked_system()'''

new_function = '''async def check_screen_locked_direct() -> bool:
    """Check if screen is locked using Context Intelligence"""
    try:
        # Use Context Intelligence screen state detector for accurate detection
        from context_intelligence.core.screen_state import ScreenStateDetector, ScreenState
        
        logger.info("[DIRECT UNLOCK] Checking screen lock via Context Intelligence")
        detector = ScreenStateDetector()
        state = await detector.get_screen_state()
        
        is_locked = state.state == ScreenState.LOCKED
        logger.info(f"[DIRECT UNLOCK] Screen state: {state.state.value} (confidence: {state.confidence:.2f})")
        logger.info(f"[DIRECT UNLOCK] Screen locked: {is_locked}")
        
        # Also try Voice Unlock daemon for comparison (but don't rely on it)
        try:
            async with websockets.connect(VOICE_UNLOCK_WS_URL) as websocket:
                status_command = {"type": "command", "command": "get_status"}
                await websocket.send(json.dumps(status_command))
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                result = json.loads(response)
                daemon_locked = result.get("status", {}).get("isScreenLocked", False)
                logger.debug(f"[DIRECT UNLOCK] Voice Unlock daemon reports: {daemon_locked} (ignored)")
        except:
            pass
        
        return is_locked
        
    except Exception as e:
        logger.error(f"Error checking screen lock: {e}")
        # Fallback to system check
        return check_screen_locked_system()'''

if old_function in content:
    content = content.replace(old_function, new_function)
    print("\n✅ Found and replaced check_screen_locked_direct function")
    
    # Write the updated content
    with open(handler_path, 'w') as f:
        f.write(content)
    
    print("✅ Updated direct_unlock_handler.py")
    print("\nThe fix uses Context Intelligence's accurate screen detection")
    print("instead of Voice Unlock daemon's unreliable status.")
else:
    print("\n❌ Could not find the exact function to replace")
    print("The function might have already been modified.")

print("\n⚠️  You need to restart JARVIS for changes to take effect!")