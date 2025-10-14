#!/usr/bin/env python3
"""
Direct import fix for vision command routing
This patches the running system directly
"""

import sys
import importlib

def force_reload_vision_modules():
    """Force reload all vision-related modules with fixes"""
    
    # List of modules that need reloading
    modules_to_reload = [
        'api.unified_command_processor',
        'api.vision_command_handler',
        'vision.multi_space_capture_engine',
    ]
    
    # Remove from cache
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    # Force reimport with fixes
    from api import unified_command_processor
    from api import vision_command_handler  
    from vision import multi_space_capture_engine
    
    # Reload the modules
    importlib.reload(unified_command_processor)
    importlib.reload(vision_command_handler)
    importlib.reload(multi_space_capture_engine)
    
    print("✅ Modules reloaded with fixes")
    
    # Test the classification
    import asyncio
    
    async def test():
        processor = unified_command_processor.UnifiedCommandProcessor()
        test_query = "What is happening across my desktop spaces?"
        cmd_type, conf = await processor._classify_command(test_query)
        print(f"Test classification: {test_query} -> {cmd_type.value}")
        return cmd_type.value == "vision"
    
    result = asyncio.run(test())
    return result

if __name__ == "__main__":
    success = force_reload_vision_modules()
    if success:
        print("✅ Vision routing fixed!")
    else:
        print("❌ Still having issues")