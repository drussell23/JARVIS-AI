#!/usr/bin/env python3
"""
Force reload all modules and test vision command classification
"""

import sys
import os
import importlib

# Clear all cached modules related to our code
modules_to_clear = []
for module_name in list(sys.modules.keys()):
    if 'api.' in module_name or 'vision.' in module_name or 'unified' in module_name:
        modules_to_clear.append(module_name)

for module_name in modules_to_clear:
    if module_name in sys.modules:
        print(f"Clearing cached module: {module_name}")
        del sys.modules[module_name]

# Now import fresh
import asyncio
from api.unified_command_processor import UnifiedCommandProcessor

async def test_classification():
    processor = UnifiedCommandProcessor()
    
    test_query = "What is happening across my desktop spaces?"
    command_type, confidence = await processor._classify_command(test_query)
    
    print(f"\nClassification Test:")
    print(f"Query: {test_query}")
    print(f"Type: {command_type.value}")
    print(f"Confidence: {confidence:.2f}")
    
    if command_type.value == "vision":
        print("✅ SUCCESS! Query correctly classified as vision")
    else:
        print(f"❌ FAILED! Query classified as {command_type.value} instead of vision")
        
        # Debug info
        words = test_query.lower().split()
        vision_score = processor._calculate_vision_score(words, test_query.lower())
        print(f"Vision score: {vision_score:.2f}")

if __name__ == "__main__":
    asyncio.run(test_classification())