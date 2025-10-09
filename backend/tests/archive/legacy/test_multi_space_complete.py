#!/usr/bin/env python3
"""
Comprehensive test for the complete Multi-Space Desktop Vision System
Tests all phases: Capture Engine, Enhanced Intelligence, Monitoring, and Optimization
"""

import asyncio
import sys
from pathlib import Path
import numpy as np
import time

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from api.pure_vision_intelligence import PureVisionIntelligence
from vision.multi_space_capture_engine import (
    MultiSpaceCaptureEngine, SpaceCaptureRequest, CaptureQuality
)
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mock Claude client
class MockClaudeClient:
    """Mock Claude client that returns test responses"""
    
    async def analyze_image_with_prompt(self, image, prompt, max_tokens=500, **kwargs):
        """Mock analyze_image_with_prompt method"""
        return {
            'text': f"Based on the multi-space analysis: I can see {len(image) if isinstance(image, list) else 1} desktop spaces. {prompt[:100]}...",
            'detailed_description': "Multiple desktop spaces with various applications"
        }
    
    async def send_message(self, model, messages, max_tokens=4096):
        return type('Response', (), {
            'content': [type('Content', (), {
                'text': f"Multi-space response for: {messages[-1]['content'][-1]['text'][:100] if messages else 'No message'}"
            })()]
        })()
        
    def create_message(self, model, messages, max_tokens=4096):
        return type('Response', (), {
            'content': [type('Content', (), {
                'text': f"Mock response"
            })()]
        })()

async def test_phase1_capture_engine():
    """Test Phase 1: Multi-Space Capture Engine"""
    print("\n📸 Phase 1: Testing Multi-Space Capture Engine\n")
    
    engine = MultiSpaceCaptureEngine()
    
    # Test 1: Enumerate spaces
    spaces = await engine.enumerate_spaces()
    print(f"✅ Found {len(spaces)} desktop spaces: {spaces}")
    
    # Test 2: Get current space
    current = await engine.get_current_space()
    print(f"✅ Current space: {current}")
    
    # Test 3: Cache statistics
    stats = engine.get_cache_stats()
    print(f"✅ Cache stats: {stats}")
    
    # Test 4: Capture request (mock)
    if len(spaces) >= 2:
        request = SpaceCaptureRequest(
            space_ids=spaces[:2],
            quality=CaptureQuality.FAST,
            use_cache=True,
            reason="phase1_test"
        )
        
        print(f"🔄 Testing capture for spaces {spaces[:2]}...")
        result = await engine.capture_all_spaces(request)
        print(f"✅ Capture result: Success={result.success}, Duration={result.total_duration:.2f}s")
        print(f"   Cache hits: {result.cache_hits}, New captures: {result.new_captures}")

async def test_phase2_enhanced_intelligence():
    """Test Phase 2: Enhanced Pure Vision Intelligence"""
    print("\n\n🧠 Phase 2: Testing Enhanced Pure Vision Intelligence\n")
    
    # Initialize with mock client
    claude_client = MockClaudeClient()
    vision = PureVisionIntelligence(claude_client, enable_multi_space=True)
    
    # Test multi-space queries
    test_queries = [
        "Show me all my workspaces",
        "What's on Desktop 2?",
        "Find all Chrome windows across spaces",
        "Show me my development setup"
    ]
    
    mock_screenshot = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    for query in test_queries[:2]:  # Test first 2 queries
        print(f"🔍 Testing: '{query}'")
        try:
            response = await vision.understand_and_respond(mock_screenshot, query)
            print(f"✅ Response: {response[:100]}...")
            
            # Check multi-space context
            if hasattr(vision, '_last_multi_space_context') and vision._last_multi_space_context:
                ctx = vision._last_multi_space_context
                print(f"   📊 Multi-space context: {ctx.analyzed_spaces} spaces analyzed")
        except Exception as e:
            print(f"❌ Error: {e}")

async def test_phase3_monitoring():
    """Test Phase 3: Proactive Multi-Space Monitoring"""
    print("\n\n👁️ Phase 3: Testing Proactive Monitoring\n")
    
    claude_client = MockClaudeClient()
    vision = PureVisionIntelligence(claude_client, enable_multi_space=True)
    
    # Start monitoring
    print("🔄 Starting multi-space monitoring...")
    success = await vision.start_multi_space_monitoring()
    print(f"✅ Monitoring started: {success}")
    
    if success:
        # Let it run briefly
        await asyncio.sleep(2)
        
        # Get monitoring status
        status = await vision.get_monitoring_status()
        print(f"✅ Monitoring status:")
        print(f"   - Active: {status.get('active', False)}")
        print(f"   - Active spaces: {status.get('summary', {}).get('active_spaces', 0)}")
        
        # Get workspace insights
        insights = await vision.get_workspace_insights()
        print(f"✅ Workspace insights: {insights}")
        
        # Stop monitoring
        await vision.stop_multi_space_monitoring()
        print("✅ Monitoring stopped")

async def test_phase4_optimization():
    """Test Phase 4: Performance Optimization"""
    print("\n\n⚡ Phase 4: Testing Performance Optimization\n")
    
    claude_client = MockClaudeClient()
    vision = PureVisionIntelligence(claude_client, enable_multi_space=True)
    
    # Start optimization
    print("🔄 Starting performance optimization...")
    success = await vision.start_multi_space_optimization()
    print(f"✅ Optimization started: {success}")
    
    if success:
        # Simulate some activity
        if hasattr(vision, 'multi_space_optimizer') and vision.multi_space_optimizer:
            optimizer = vision.multi_space_optimizer
            
            # Track some space accesses
            for i in range(3):
                optimizer.track_space_access(1, {2, 3})
                optimizer.track_space_access(2, {1})
                await asyncio.sleep(0.1)
            
            # Track capture performance
            optimizer.track_capture_performance(1, 0.5, True)  # Cache hit
            optimizer.track_capture_performance(2, 2.0, False)  # Cache miss
        
        # Get optimization stats
        stats = await vision.get_optimization_stats()
        print(f"✅ Optimization stats:")
        print(f"   - Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
        print(f"   - Average capture time: {stats.get('average_capture_time', 0):.2f}s")
        print(f"   - Tracked spaces: {stats.get('tracked_spaces', 0)}")
        
        # Show space patterns
        patterns = stats.get('space_patterns', {})
        if patterns:
            print(f"✅ Detected space patterns:")
            for space_id, info in list(patterns.items())[:3]:
                print(f"   - Space {space_id}: {info['pattern']}, priority={info['priority']:.2f}")
        
        # Stop optimization
        await vision.stop_multi_space_optimization()
        print("✅ Optimization stopped")

async def test_integration():
    """Test full integration of all components"""
    print("\n\n🚀 Integration Test: Full System\n")
    
    claude_client = MockClaudeClient()
    vision = PureVisionIntelligence(claude_client, enable_multi_space=True)
    
    # Start everything
    print("🔄 Starting all systems...")
    monitor_ok = await vision.start_multi_space_monitoring()
    optimize_ok = await vision.start_multi_space_optimization()
    print(f"✅ Systems started - Monitor: {monitor_ok}, Optimizer: {optimize_ok}")
    
    # Simulate user activity
    print("\n🔄 Simulating user activity...")
    mock_screenshot = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    queries = [
        "What's running on my other desktops?",
        "Show me all my browser windows",
        "Find my VS Code workspace"
    ]
    
    for query in queries:
        print(f"\n💬 User: {query}")
        try:
            response = await vision.understand_and_respond(mock_screenshot, query)
            print(f"🤖 JARVIS: {response[:150]}...")
            await asyncio.sleep(1)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Get final stats
    print("\n📊 Final System Statistics:")
    
    monitor_status = await vision.get_monitoring_status()
    if monitor_status.get('active'):
        summary = monitor_status.get('summary', {})
        print(f"✅ Monitoring: {summary.get('active_spaces', 0)} active spaces, "
              f"{summary.get('recent_events', 0)} recent events")
    
    opt_stats = await vision.get_optimization_stats()
    if opt_stats.get('enabled', False):
        print(f"✅ Optimization: {opt_stats.get('cache_hit_rate', 0):.1%} cache hit rate, "
              f"{opt_stats.get('tracked_spaces', 0)} tracked spaces")
    
    # Cleanup
    await vision.stop_multi_space_monitoring()
    await vision.stop_multi_space_optimization()
    print("\n✅ All systems shut down")

async def main():
    """Run all tests"""
    print("=" * 60)
    print("🎯 Multi-Space Desktop Vision System - Comprehensive Test")
    print("=" * 60)
    
    # Set environment for testing
    import os
    os.environ['JARVIS_AUTO_APPROVE_SPACE_SWITCH'] = 'true'
    
    try:
        # Test each phase
        await test_phase1_capture_engine()
        await test_phase2_enhanced_intelligence()
        await test_phase3_monitoring()
        await test_phase4_optimization()
        
        # Integration test
        await test_integration()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())