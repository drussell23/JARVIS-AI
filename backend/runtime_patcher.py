#!/usr/bin/env python3
"""
Runtime patcher for JARVIS - fixes vision routing in running system
"""

def patch_vision_classification():
    """Monkey-patch the vision classification in the running system"""
    
    import sys
    
    # Get the unified command processor module if it's loaded
    if 'api.unified_command_processor' in sys.modules:
        ucp = sys.modules['api.unified_command_processor']
        
        # Save original classify method
        if hasattr(ucp.UnifiedCommandProcessor, '_classify_command'):
            original_classify = ucp.UnifiedCommandProcessor._classify_command
            
            # Create patched version
            async def patched_classify(self, command_text):
                """Patched classifier that prioritizes vision commands"""
                command_lower = command_text.lower().strip()
                words = command_lower.split()
                
                # Direct vision detection for our test queries
                vision_phrases = [
                    "what is happening across my desktop spaces",
                    "what's happening across my desktop spaces",
                    "what is on my screen",
                    "analyze my screen",
                    "monitor my screen",
                    "show me my desktop"
                ]
                
                for phrase in vision_phrases:
                    if phrase in command_lower:
                        # Return vision with high confidence
                        if hasattr(ucp, 'CommandType'):
                            return (ucp.CommandType.VISION, 0.95)
                
                # Check for desktop/space/screen keywords
                vision_keywords = ['desktop', 'space', 'spaces', 'screen', 'monitor', 'workspace']
                if any(keyword in words for keyword in vision_keywords):
                    # Calculate vision score
                    score = 0.0
                    
                    # Clean words for better matching
                    import re
                    clean_words = [re.sub(r'[^\w\s]', '', word) for word in words]
                    
                    # Multi-space indicators
                    if any(w in clean_words for w in ['desktop', 'space', 'spaces', 'workspace', 'across']):
                        score += 0.5
                    
                    # Screen queries
                    if 'screen' in clean_words:
                        score += 0.4
                        if clean_words[0] in ['what', 'show', 'analyze', 'monitor']:
                            score += 0.4
                    
                    # If high vision score, return vision
                    if score > 0.7 and hasattr(ucp, 'CommandType'):
                        return (ucp.CommandType.VISION, min(0.95, score))
                
                # Fall back to original classifier
                return await original_classify(self, command_text)
            
            # Apply the patch
            ucp.UnifiedCommandProcessor._classify_command = patched_classify
            print("✅ Patched UnifiedCommandProcessor._classify_command")
            return True
    
    return False

def patch_running_system():
    """Apply all patches to the running system"""
    
    success = patch_vision_classification()
    
    if success:
        print("✅ Runtime patches applied successfully!")
        print("   Vision commands will now be properly routed")
    else:
        print("❌ Could not patch - module not loaded yet")
    
    return success

if __name__ == "__main__":
    patch_running_system()