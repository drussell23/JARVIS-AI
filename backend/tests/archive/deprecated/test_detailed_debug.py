#!/usr/bin/env python3
"""
Detailed debug of "open safari and search for dogs"
"""

import asyncio
import sys
import os
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.unified_command_processor import get_unified_processor

async def test_detailed():
    processor = get_unified_processor()
    
    command = "open safari and search for dogs"
    print(f"Testing command: '{command}'")
    print("=" * 50)
    
    # Test classification
    cmd_type, confidence = await processor._classify_command(command)
    print(f"\n1. Classification: {cmd_type.value} (confidence: {confidence})")
    
    # Test parsing
    parts = processor._parse_compound_parts(command)
    print(f"\n2. Parsed parts: {parts}")
    
    # Test individual part classification
    print("\n3. Individual part analysis:")
    for i, part in enumerate(parts):
        # Test raw classification
        part_type, part_conf = await processor._classify_command(part)
        print(f"   Part {i+1} '{part}': {part_type.value} (confidence: {part_conf})")
        
        # Test enhanced version (simulating context)
        enhanced = processor._enhance_with_context(part, 'safari' if i > 0 else None, None)
        if enhanced != part:
            print(f"   Enhanced to: '{enhanced}'")
            enhanced_type, enhanced_conf = await processor._classify_command(enhanced)
            print(f"   Enhanced classification: {enhanced_type.value} (confidence: {enhanced_conf})")
    
    print("\n4. Processing full command:")
    result = await processor.process_command(command)
    print(f"   Success: {result.get('success')}")
    print(f"   Response: {result.get('response')}")

if __name__ == "__main__":
    asyncio.run(test_detailed())