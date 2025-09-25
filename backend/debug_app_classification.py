#!/usr/bin/env python3
"""Debug app command classification and parsing"""

import asyncio
import sys
sys.path.append('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

from api.unified_command_processor import UnifiedCommandProcessor

async def test_classification():
    """Test how app commands are classified"""
    processor = UnifiedCommandProcessor()
    
    test_commands = [
        "open safari",
        "open Safari", 
        "close safari",
        "Open safari",
        "open music",
        "close weather"
    ]
    
    print("Testing Command Classification")
    print("=" * 50)
    
    for cmd in test_commands:
        print(f"\nCommand: '{cmd}'")
        
        # Test classification
        cmd_type, confidence = await processor._classify_command(cmd)
        print(f"  Type: {cmd_type.value}")
        print(f"  Confidence: {confidence}")
        
        # Test parsing
        parsed = processor._parse_command(cmd)
        print(f"  Parsed: {parsed}")
        
        # Check if safari is in learned apps
        if "safari" in cmd.lower():
            is_learned = processor.pattern_learner.is_learned_app("safari")
            print(f"  Is 'safari' a learned app? {is_learned}")
            
            # Check installed apps cache
            if hasattr(processor, 'dynamic_controller'):
                app_info = processor.dynamic_controller.find_app_by_name("safari")
                if app_info:
                    print(f"  Found in cache: {app_info['name']}")
                else:
                    print("  NOT found in installed apps cache")

if __name__ == "__main__":
    asyncio.run(test_classification())