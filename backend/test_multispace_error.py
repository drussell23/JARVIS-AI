#!/usr/bin/env python3
"""Test to find the actual error in multi-space queries"""

import asyncio
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Enable ALL logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

async def test_multispace_error():
    """Test the actual multi-space command processing"""
    print("🔍 Testing Multi-Space Command Processing")
    print("=" * 80)
    
    # Test the unified command processor
    try:
        from api.unified_command_processor_pure import PureUnifiedCommandProcessor
        
        processor = PureUnifiedCommandProcessor()
        command = "can you see the Cursor IDE in the other desktop space?"
        
        print(f"\n📝 Processing command: '{command}'")
        result = await processor.process_command(command)
        
        print(f"\n📊 Result:")
        print(f"   Success: {result.get('success')}")
        print(f"   Response: {result.get('response')}")
        print(f"   Command Type: {result.get('command_type')}")
        if 'error' in result:
            print(f"   Error: {result.get('error')}")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_multispace_error())