#!/usr/bin/env python3
"""Debug production multi-space issue"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

async def debug_production():
    """Debug the production flow"""
    print("🔍 Debugging Production Multi-Space Issue")
    print("=" * 80)
    
    # Check environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    print(f"\n1️⃣ Environment Check:")
    print(f"   ANTHROPIC_API_KEY set: {api_key is not None}")
    if api_key:
        print(f"   API key length: {len(api_key)}")
        print(f"   API key preview: {api_key[:10]}...")
    
    # Test the production flow
    print("\n2️⃣ Testing Production Flow...")
    
    # Simulate jarvis_voice_api flow
    from api.unified_command_processor_pure import get_pure_unified_processor
    
    # Get processor with API key from environment
    processor = get_pure_unified_processor(api_key)
    
    # Process the command
    command = "can you see the Cursor IDE in the other desktop space?"
    print(f"\n3️⃣ Processing command: '{command}'")
    
    try:
        result = await processor.process_command(command)
        print(f"\n4️⃣ Result:")
        print(f"   Success: {result.get('success')}")
        print(f"   Command type: {result.get('command_type')}")
        print(f"   Response preview: {result.get('response', '')[:100]}...")
        if 'error' in result:
            print(f"   Error: {result.get('error')}")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    # Check vision handler state
    print("\n5️⃣ Vision Handler State:")
    if processor.vision_handler:
        print(f"   Vision handler initialized: True")
        if processor.vision_handler.intelligence:
            print(f"   Intelligence initialized: True")
            print(f"   Multi-space enabled: {processor.vision_handler.intelligence.multi_space_enabled}")
            print(f"   Has Claude client: {processor.vision_handler.intelligence.claude is not None}")
            
            # Check the actual API key in the vision analyzer
            if hasattr(processor.vision_handler, 'vision_analyzer') and processor.vision_handler.vision_analyzer:
                analyzer = processor.vision_handler.vision_analyzer
                if hasattr(analyzer, 'client') and analyzer.client:
                    print(f"   Vision analyzer has client: True")
                    if hasattr(analyzer.client, 'api_key'):
                        key = analyzer.client.api_key
                        print(f"   Client API key length: {len(key) if key else 0}")
                        print(f"   Client API key preview: {key[:10] if key else 'None'}...")

if __name__ == "__main__":
    # Make sure we have an API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: ANTHROPIC_API_KEY not set in environment")
        print("   Setting a test key for debugging...")
        os.environ["ANTHROPIC_API_KEY"] = "test-debug-key"
    
    asyncio.run(debug_production())