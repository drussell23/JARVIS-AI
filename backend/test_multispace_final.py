#!/usr/bin/env python3
"""Final test for multi-space functionality"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_multispace_final():
    """Test the complete multi-space flow"""
    print("🔍 Final Multi-Space Test")
    print("=" * 80)
    
    # Use real API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        return
    
    # Test through unified processor (production flow)
    from api.unified_command_processor_pure import get_pure_unified_processor
    
    processor = get_pure_unified_processor(api_key)
    command = "can you see the Cursor IDE in the other desktop space?"
    
    print(f"\n📝 Processing: '{command}'")
    print("⏳ This will capture multiple desktop spaces...")
    
    result = await processor.process_command(command)
    
    print(f"\n📊 Result:")
    print(f"   Success: {result.get('success')}")
    print(f"   Command type: {result.get('command_type')}")
    
    # Check if multi-space was actually used
    if 'response' in result:
        response = result['response']
        # Look for indicators of multi-space analysis
        if any(phrase in response.lower() for phrase in ['desktop', 'space', 'multiple', 'across']):
            print(f"\n✅ Multi-space analysis detected in response!")
        else:
            print(f"\n⚠️  Response might be single-space only")
        
        print(f"\n📄 Response:")
        print("-" * 80)
        print(response)
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(test_multispace_final())