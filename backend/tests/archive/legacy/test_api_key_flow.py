#!/usr/bin/env python3
"""Test API key flow in multi-space queries"""

import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_api_key_flow():
    """Test how API key flows through the system"""
    print("🔍 Testing API Key Flow")
    print("=" * 80)
    
    # Set a test API key
    test_key = os.getenv("ANTHROPIC_API_KEY", "test-api-key")
    print(f"Using API key: {test_key[:20]}...")
    
    # Test 1: Direct vision handler
    print("\n1️⃣ Testing direct vision handler...")
    from api.vision_command_handler import vision_command_handler
    
    # Check initial state
    print(f"   Intelligence initialized: {vision_command_handler.intelligence is not None}")
    
    # Initialize
    await vision_command_handler.initialize_intelligence(test_key)
    print(f"   After init: {vision_command_handler.intelligence is not None}")
    
    # Test 2: Unified processor
    print("\n2️⃣ Testing unified processor...")
    from api.unified_command_processor_pure import get_pure_unified_processor
    
    processor = get_pure_unified_processor(test_key)
    await processor._ensure_initialized()
    
    print(f"   Processor initialized: {processor._initialized}")
    print(f"   Vision handler set: {processor.vision_handler is not None}")
    
    # Check if they're the same instance
    print(f"   Same vision handler instance: {processor.vision_handler is vision_command_handler}")
    
    # Test 3: Check intelligence state
    if processor.vision_handler:
        print(f"\n3️⃣ Checking vision handler state in processor...")
        print(f"   Intelligence initialized: {processor.vision_handler.intelligence is not None}")
        if processor.vision_handler.intelligence:
            print(f"   Multi-space enabled: {processor.vision_handler.intelligence.multi_space_enabled}")
            print(f"   Has Claude client: {processor.vision_handler.intelligence.claude is not None}")

if __name__ == "__main__":
    asyncio.run(test_api_key_flow())