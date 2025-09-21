#!/usr/bin/env python3
"""
Test the specific scenario: "open safari and search for dogs"
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.unified_command_processor import get_unified_processor

async def test_safari_search():
    processor = get_unified_processor()
    
    command = "open safari and search for dogs"
    print(f"Testing command: '{command}'")
    print("=" * 50)
    
    # Process the command
    result = await processor.process_command(command)
    
    print(f"\nResult:")
    print(f"  Success: {result.get('success')}")
    print(f"  Response: {result.get('response')}")
    print(f"  Command Type: {result.get('command_type')}")
    
    if 'sub_results' in result:
        print(f"\nSub-results:")
        for i, sub in enumerate(result['sub_results']):
            print(f"  Step {i+1}: {sub.get('response')} (success: {sub.get('success')})")
    
    print("\nExpected behavior:")
    print("  1. Open Safari browser")
    print("  2. Search for 'dogs' (open Google search results)")

if __name__ == "__main__":
    asyncio.run(test_safari_search())