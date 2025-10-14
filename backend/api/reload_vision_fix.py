#!/usr/bin/env python3
"""
Force reload of vision command handler to pick up latest fixes
"""

import importlib
import sys
import logging

logger = logging.getLogger(__name__)

def force_reload_vision_handler():
    """Force reload the vision command handler module to get latest code"""
    try:
        # Remove cached modules
        modules_to_reload = [
            'api.vision_command_handler',
            'vision.multi_space_capture_engine',
        ]
        
        for module_name in modules_to_reload:
            if module_name in sys.modules:
                logger.info(f"Removing cached module: {module_name}")
                del sys.modules[module_name]
        
        # Re-import with fresh code
        from api.vision_command_handler import vision_command_handler
        
        logger.info("✅ Vision command handler reloaded with latest fixes")
        
        # Log the multi-space capture fix status
        from vision.multi_space_capture_engine import MultiSpaceCaptureEngine
        test_engine = MultiSpaceCaptureEngine()
        
        # Test enumeration to confirm fix
        import asyncio
        spaces = asyncio.run(test_engine.enumerate_spaces())
        logger.info(f"✅ Multi-space capture fix confirmed: Found {len(spaces)} spaces")
        
        return vision_command_handler
        
    except Exception as e:
        logger.error(f"Failed to reload vision handler: {e}")
        import traceback
        traceback.print_exc()
        return None

# Auto-run on import
if __name__ != "__main__":
    force_reload_vision_handler()