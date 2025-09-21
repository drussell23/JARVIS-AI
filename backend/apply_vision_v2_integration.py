#!/usr/bin/env python3
"""
Apply Vision System v2.0 Integration to JARVIS Voice
"""

def apply_vision_v2_integration():
    """Apply Vision System v2.0 integration to JARVISAgentVoice"""
    
    # Read the current jarvis_agent_voice.py
    with open('voice/jarvis_agent_voice.py', 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'VisionV2Integration' in content:
        print("Vision System v2.0 integration already applied")
        return
    
    # Find the import section
    import_section_end = content.find('logger = logging.getLogger(__name__)')
    if import_section_end == -1:
        print("Could not find logger initialization")
        return
    
    # Add import for Vision v2
    new_import = "\n# Vision System v2.0 Integration\ntry:\n    from voice.vision_v2_integration import VisionV2Integration\n    VISION_V2_AVAILABLE = True\nexcept ImportError:\n    VISION_V2_AVAILABLE = False\n    logger.warning('Vision System v2.0 not available')\n\n"
    
    content = content[:import_section_end] + new_import + content[import_section_end:]
    
    # Find the __init__ method to add Vision v2 initialization
    init_vision_start = content.find("# Initialize vision integration if available")
    if init_vision_start != -1:
        # Add Vision v2 initialization before other vision systems
        vision_v2_init = """        # Try Vision System v2.0 first (newest and best)
        try:
            if VISION_V2_AVAILABLE:
                self.vision_v2 = VisionV2Integration()
                if self.vision_v2.enabled:
                    self.vision_enabled = True
                    self.vision_v2_enabled = True
                    logger.info("Vision System v2.0 initialized - ML-powered vision active")
                else:
                    self.vision_v2_enabled = False
            else:
                self.vision_v2_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Vision System v2.0: {e}")
            self.vision_v2_enabled = False
            
"""
        content = content[:init_vision_start] + vision_v2_init + content[init_vision_start:]
    
    # Update the _handle_vision_command method to use Vision v2
    vision_handler_start = content.find("async def _handle_vision_command(self, text: str) -> str:")
    if vision_handler_start != -1:
        # Find the try block
        try_start = content.find("try:", vision_handler_start)
        if try_start != -1:
            # Insert Vision v2 handling at the beginning of try block
            vision_v2_handler = """
            # Use Vision System v2.0 if available (highest priority)
            if hasattr(self, 'vision_v2_enabled') and self.vision_v2_enabled:
                response = await self.vision_v2.handle_vision_command(text, {
                    'user': self.user_name,
                    'mode': self.command_mode
                })
                return response
            
"""
            # Find the position after "try:"
            try_content_start = content.find("\n", try_start) + 1
            content = content[:try_content_start] + vision_v2_handler + content[try_content_start:]
    
    # Write the updated file
    with open('voice/jarvis_agent_voice.py', 'w') as f:
        f.write(content)
    
    print("Vision System v2.0 integration applied successfully!")
    print("\nTo use Vision System v2.0 with voice commands:")
    print("1. Ensure ANTHROPIC_API_KEY is set")
    print("2. Restart the backend server")
    print("3. Try commands like: 'Hey JARVIS, what do you see on my screen?'")

if __name__ == "__main__":
    apply_vision_v2_integration()