#!/usr/bin/env python3
"""
Fix Manual Unlock to Bypass Policy Engine
=========================================

Updates the voice unlock integration to bypass the Context Intelligence
policy engine for manual "unlock my screen" commands.
"""

import shutil
from datetime import datetime


def apply_fix():
    """Apply the fix to bypass policy for manual unlock commands"""
    print("🔧 Fixing manual unlock to bypass policy restrictions...")
    print("="*60)
    
    # Target file
    target_file = 'api/voice_unlock_integration.py'
    
    # Backup the original
    backup_path = f'{target_file}.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    shutil.copy(target_file, backup_path)
    print(f"📋 Backed up to {backup_path}")
    
    # Read the file
    with open(target_file, 'r') as f:
        content = f.read()
    
    # Find the unlock section
    unlock_section_start = content.find("elif any(phrase in command_lower for phrase in ['unlock my mac', 'unlock my screen'")
    if unlock_section_start == -1:
        print("❌ Could not find unlock section")
        return
    
    # Find where the Voice Unlock fallback ends and Context Intelligence starts
    ci_fallback_start = content.find("# Fallback to Context Intelligence System", unlock_section_start)
    if ci_fallback_start == -1:
        print("❌ Could not find Context Intelligence fallback")
        return
    
    # Find the except block
    except_start = content.find("except Exception as e:", unlock_section_start)
    if except_start == -1:
        print("❌ Could not find except block")
        return
    
    # Replace the Context Intelligence section with direct unlock
    new_unlock_code = '''            
            # For manual unlock commands, bypass policy and use direct unlock
            logger.info("Manual unlock command - bypassing policy restrictions")
            from api.direct_unlock_handler_fixed import unlock_screen_direct
            
            # Send immediate feedback
            if hasattr(voice_unlock_connector, 'websocket') and voice_unlock_connector.websocket:
                await voice_unlock_connector.websocket.send_json({
                    "type": "response",
                    "text": "I'll unlock your screen right away, Sir.",
                    "speak": True,
                    "intermediate": True
                })
            
            # Perform direct unlock
            success = await unlock_screen_direct("Manual unlock command from user")
            
            if success:
                return {
                    'response': 'Screen unlocked successfully, Sir.',
                    'success': True,
                    'method': 'direct_unlock'
                }
            else:
                return {
                    'response': "I couldn't unlock the screen, Sir. Please check if the Voice Unlock daemon is running.",
                    'success': False
                }
                
        '''
    
    # Replace the section
    before = content[:ci_fallback_start]
    after = content[except_start:]
    content = before + new_unlock_code + after
    
    # Write the modified file
    with open(target_file, 'w') as f:
        f.write(content)
    
    print("✅ Fixed voice unlock integration!")
    
    # Now we need to ensure the voice unlock handler is being used for these commands
    # Check the unified command processor routing
    processor_file = 'api/unified_command_processor.py'
    
    print(f"\n🔧 Checking {processor_file} routing...")
    
    with open(processor_file, 'r') as f:
        processor_content = f.read()
    
    # Check if "unlock my screen" is being classified correctly
    if 'unlock my screen' in processor_content and 'CommandType.VOICE_UNLOCK' in processor_content:
        print("✅ Command routing looks correct")
    else:
        print("⚠️  May need to update command classification")
        
        # Add unlock detection to voice unlock patterns
        voice_pattern_pos = processor_content.find("def _detect_voice_unlock_patterns")
        if voice_pattern_pos != -1:
            # Find the direct phrases section
            phrases_pos = processor_content.find("voice_unlock_phrases = [", voice_pattern_pos)
            if phrases_pos != -1:
                # Find the end of the list
                list_end = processor_content.find("]", phrases_pos)
                # Add our phrases
                new_phrases = '''            'unlock my screen',
            'unlock screen',
            'unlock the screen',
            'lock my screen',
            'lock screen',
            'lock the screen','''
                
                # Check if not already there
                if 'unlock my screen' not in processor_content[phrases_pos:list_end]:
                    # Insert before the closing bracket
                    processor_content = processor_content[:list_end] + new_phrases + "\n        " + processor_content[list_end:]
                    
                    # Write back
                    with open(processor_file, 'w') as f:
                        f.write(processor_content)
                    print("✅ Added unlock/lock screen phrases to voice unlock detection")
    
    print("\n✅ Fix applied successfully!")
    print("\n📝 What this fix does:")
    print("   1. Detects manual 'unlock my screen' commands")
    print("   2. Bypasses the Context Intelligence policy engine")
    print("   3. Uses direct unlock without any time restrictions")
    print("   4. Works 24/7, regardless of quiet hours!")
    print("\n🎉 Manual unlock will now work anytime!")


if __name__ == "__main__":
    apply_fix()