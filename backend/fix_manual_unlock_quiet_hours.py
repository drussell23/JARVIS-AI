#!/usr/bin/env python3
"""
Fix Manual Unlock During Quiet Hours
====================================

This script modifies the system to allow manual screen unlock commands
to work regardless of quiet hours policy.

The issue: PolicyEngine blocks unlock during quiet hours (10 PM - 7 AM)
The fix: Add high urgency to manual unlock commands to bypass the restriction
"""

import os
import sys
import shutil
from datetime import datetime


def apply_fix():
    """Apply the fix to allow manual unlock during quiet hours"""
    print("🔧 Fixing manual unlock during quiet hours...")
    print("="*60)
    
    # The issue is in the Context Intelligence policy engine
    # But since we're using simple_context_handler_enhanced, 
    # we need a different approach
    
    # Create a new handler for manual unlock requests
    manual_unlock_handler = '''#!/usr/bin/env python3
"""
Manual Unlock Handler
=====================

Handles manual "unlock my screen" commands without policy restrictions
"""

import logging
from typing import Dict, Any

from api.direct_unlock_handler_fixed import unlock_screen_direct

logger = logging.getLogger(__name__)


async def handle_manual_unlock(command: str, websocket=None) -> Dict[str, Any]:
    """Handle manual unlock request directly without policy checks"""
    logger.info("[MANUAL UNLOCK] User requested manual screen unlock")
    
    # Send immediate feedback
    if websocket:
        await websocket.send_json({
            "type": "response",
            "text": "I'll unlock your screen right away, Sir.",
            "command_type": "manual_unlock",
            "speak": True,
            "intermediate": True
        })
    
    # Perform unlock
    success = await unlock_screen_direct("Manual user request - bypass quiet hours")
    
    if success:
        return {
            "success": True,
            "response": "I've unlocked your screen, Sir.",
            "command_type": "manual_unlock"
        }
    else:
        return {
            "success": False,
            "response": "I couldn't unlock the screen, Sir. Please check if the Voice Unlock daemon is running.",
            "command_type": "manual_unlock"
        }
'''
    
    # Write the manual unlock handler
    with open('api/manual_unlock_handler.py', 'w') as f:
        f.write(manual_unlock_handler)
    print("✅ Created manual unlock handler")
    
    # Now modify the unified command processor to use this handler
    processor_path = 'api/unified_command_processor.py'
    
    # Backup the original
    backup_path = f'{processor_path}.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    shutil.copy(processor_path, backup_path)
    print(f"📋 Backed up to {backup_path}")
    
    # Read the file
    with open(processor_path, 'r') as f:
        content = f.read()
    
    # Add import for manual unlock handler after other imports
    import_line = "from api.manual_unlock_handler import handle_manual_unlock"
    if import_line not in content:
        # Find a good place to add the import (after other api imports)
        import_pos = content.find("from api.voice_unlock_handler")
        if import_pos != -1:
            # Find the end of that line
            line_end = content.find('\n', import_pos)
            content = content[:line_end+1] + import_line + '\n' + content[line_end+1:]
            print("✅ Added import for manual unlock handler")
    
    # Now modify the _classify_command method to detect manual unlock
    # Find the voice unlock detection section
    voice_detect_start = content.find("# Voice unlock detection FIRST")
    if voice_detect_start != -1:
        # Add manual unlock detection before voice unlock
        manual_detect = '''        # Manual screen unlock detection (highest priority)
        if command_lower.strip() in ['unlock my screen', 'unlock screen', 'unlock the screen']:
            logger.info(f"[CLASSIFY] Manual unlock command detected: '{command_lower}'")
            return CommandType.META, 0.99  # Use META type with high confidence
        
        '''
        content = content[:voice_detect_start] + manual_detect + content[voice_detect_start:]
        print("✅ Added manual unlock detection")
    
    # Modify the META command handler to handle manual unlock
    meta_handler_pos = content.find("elif command_type == CommandType.META:")
    if meta_handler_pos != -1:
        # Find the handler content
        handler_start = content.find("{", meta_handler_pos)
        if handler_start != -1:
            # Add manual unlock handling at the start of META handler
            manual_handling = '''
                # Handle manual unlock requests
                if command_text.lower().strip() in ['unlock my screen', 'unlock screen', 'unlock the screen']:
                    return await handle_manual_unlock(command_text, websocket)
                '''
            # Find first line after elif
            first_line = content.find("\n", meta_handler_pos) + 1
            next_line = content.find("\n", first_line) + 1
            # Insert after the comment line
            content = content[:next_line] + manual_handling + "\n" + content[next_line:]
            print("✅ Added manual unlock handling in META handler")
    
    # Write the modified file
    with open(processor_path, 'w') as f:
        f.write(content)
    
    print("\n✅ Fix applied successfully!")
    print("\n📝 What this fix does:")
    print("   1. Detects 'unlock my screen' commands as high-priority")
    print("   2. Bypasses any policy restrictions")
    print("   3. Unlocks the screen immediately")
    print("\n🚀 Manual unlock will now work any time of day!")


if __name__ == "__main__":
    apply_fix()