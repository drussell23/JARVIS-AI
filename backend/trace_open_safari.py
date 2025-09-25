#!/usr/bin/env python3
"""Trace the execution of 'open safari' command"""

import asyncio
import sys
import logging
sys.path.append('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

from api.unified_command_processor import UnifiedCommandProcessor

async def trace_command():
    """Trace command execution"""
    processor = UnifiedCommandProcessor()
    
    command = "open safari"
    print(f"Tracing command: '{command}'")
    print("=" * 50)
    
    # Step 1: Classification
    cmd_type, confidence = await processor._classify_command(command)
    print(f"\n1. Classification:")
    print(f"   Type: {cmd_type.value}")
    print(f"   Confidence: {confidence}")
    
    # Step 2: Parse system command
    if hasattr(processor, '_parse_system_command'):
        parsed_type, target, params = processor._parse_system_command(command)
        print(f"\n2. Parsing:")
        print(f"   Command type: {parsed_type}")
        print(f"   Target: {target}")
        print(f"   Params: {params}")
    
    # Step 3: Execute
    print(f"\n3. Executing command...")
    try:
        result = await processor.process_command(command)
        print(f"\nResult:")
        print(f"   Success: {result.get('success')}")
        print(f"   Response: {result.get('response')}")
        print(f"   Full result: {result}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(trace_command())