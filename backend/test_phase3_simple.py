#!/usr/bin/env python3
"""
Simple test to verify Phase 3 components work
"""

import asyncio
import time
import logging

logging.basicConfig(level=logging.INFO)


async def test_basic_functionality():
    print("\n=== Testing Basic Phase 3 Functionality ===\n")
    
    # Test 1: Transformer Router
    print("1. Testing Transformer Router...")
    try:
        from vision.transformer_command_router import TransformerCommandRouter
        router = TransformerCommandRouter()
        
        # Create simple handler
        async def test_handler(cmd, ctx):
            return f"Handled: {cmd}"
        
        # Register it
        await router.discover_handler(test_handler)
        print("✓ Transformer router initialized and handler registered")
    except Exception as e:
        print(f"✗ Transformer router failed: {e}")
        return False
    
    # Test 2: Continuous Learning Pipeline
    print("\n2. Testing Continuous Learning Pipeline...")
    try:
        from vision.continuous_learning_pipeline import ContinuousLearningPipeline
        pipeline = ContinuousLearningPipeline()
        
        # Record a test event
        await pipeline.record_learning_event(
            event_type='command',
            data={'command': 'test', 'success': True}
        )
        
        status = pipeline.get_learning_status()
        print(f"✓ Learning pipeline running - Model version: {status['model_version']}")
    except Exception as e:
        print(f"✗ Learning pipeline failed: {e}")
        return False
    
    # Test 3: Vision System Integration
    print("\n3. Testing Vision System v2.0 Integration...")
    try:
        from vision.vision_system_v2 import get_vision_system_v2
        system = get_vision_system_v2()
        
        # Test simple command
        start = time.perf_counter()
        response = await system.process_command("test command")
        latency = (time.perf_counter() - start) * 1000
        
        print(f"✓ Vision System processed command in {latency:.1f}ms")
        print(f"  Success: {response.success}")
        print(f"  Version: {system.use_transformer_routing and '3.0 (Transformer)' or '2.0 (Neural)'}")
        
        # Get stats
        stats = await system.get_system_stats()
        print(f"  Phase: {stats.get('phase', 'Unknown')}")
        
    except Exception as e:
        print(f"✗ Vision System failed: {e}")
        return False
    
    # Cleanup
    print("\n4. Testing cleanup...")
    try:
        await system.shutdown()
        print("✓ System shutdown successfully")
    except Exception as e:
        print(f"✗ Shutdown failed: {e}")
    
    return True


async def main():
    print("\n" + "="*50)
    print("Phase 3 Basic Functionality Test")
    print("="*50)
    
    success = await test_basic_functionality()
    
    print("\n" + "="*50)
    if success:
        print("✓ Basic Phase 3 components are working!")
    else:
        print("✗ Some components failed")
    print("="*50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())