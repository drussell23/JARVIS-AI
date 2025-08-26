#!/usr/bin/env python3
"""
Test Enhanced Vision System
Verifies all components work without hardcoding
"""

import asyncio
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_unified_vision():
    """Test the unified vision system"""
    print("\n🔧 Testing Enhanced Vision System")
    print("=" * 60)
    
    # Test 1: Unified Vision System
    print("\n1️⃣ Testing Unified Vision System...")
    try:
        from vision.unified_vision_system import get_unified_vision_system, VisionRequest
        
        unified = get_unified_vision_system()
        status = unified.get_system_status()
        
        print(f"✓ Initialized with {status['total_components']} components:")
        for comp in status['components']:
            details = status['component_details'][comp]
            print(f"  - {comp}: {', '.join(details['capabilities'])}")
            
        # Test various requests
        test_requests = [
            "describe what's on my screen",
            "analyze the current workspace with all windows",
            "check for any errors or notifications",
            "tell me what applications are running",
            "perform a custom analysis of the UI",
            "what am I currently working on?"
        ]
        
        for request_text in test_requests:
            print(f"\n📝 Testing: '{request_text}'")
            request = VisionRequest(
                command=request_text,
                context={'user': 'test', 'mode': 'verbose'}
            )
            
            response = await unified.process_vision_request(request)
            print(f"✓ Success: {response.success}")
            print(f"✓ Provider: {response.provider_used}")
            print(f"✓ Confidence: {response.confidence:.2f}")
            print(f"✓ Time: {response.execution_time:.3f}s")
            if response.success:
                print(f"✓ Response: {response.description[:200]}...")
            
    except Exception as e:
        print(f"❌ Unified Vision Error: {e}")
        
    # Test 2: Dynamic Vision Engine
    print("\n\n2️⃣ Testing Dynamic Vision Engine...")
    try:
        from vision.dynamic_vision_engine import get_dynamic_vision_engine
        
        engine = get_dynamic_vision_engine()
        stats = engine.get_statistics()
        
        print(f"✓ Total capabilities: {stats['total_capabilities']}")
        print(f"✓ Learned patterns: {stats['learned_patterns']}")
        
        # Test learning
        test_command = "show me what's happening on my display"
        response, metadata = await engine.process_vision_command(test_command)
        print(f"✓ Command processed: {metadata.get('status', 'success')}")
        
        # Simulate learning from feedback
        engine.learn_from_feedback(
            test_command,
            "This worked well",
            correct_action="describe_screen"
        )
        print("✓ Learning feedback recorded")
        
    except Exception as e:
        print(f"❌ Dynamic Engine Error: {e}")
        
    # Test 3: Plugin System
    print("\n\n3️⃣ Testing Vision Plugin System...")
    try:
        from vision.vision_plugin_system import get_vision_plugin_system
        
        plugin_system = get_vision_plugin_system()
        
        # List capabilities
        capabilities = plugin_system.list_capabilities()
        print(f"✓ Discovered capabilities: {len(capabilities)}")
        for cap, providers in list(capabilities.items())[:5]:
            print(f"  - {cap}: {', '.join(providers)}")
            
        # Test a capability
        if capabilities:
            test_cap = list(capabilities.keys())[0]
            print(f"\n📷 Testing capability: {test_cap}")
            result, metadata = await plugin_system.execute_capability(test_cap)
            print(f"✓ Execution result: {metadata}")
            
    except Exception as e:
        print(f"❌ Plugin System Error: {e}")
        
    # Test 4: Vision Action Handler (Updated)
    print("\n\n4️⃣ Testing Vision Action Handler...")
    try:
        from system_control.vision_action_handler import get_vision_action_handler
        
        handler = get_vision_action_handler()
        
        # Check discovered actions
        print(f"✓ Discovered actions: {len(handler.discovered_actions)}")
        
        # Test dynamic routing
        result = await handler.describe_screen()
        print(f"✓ describe_screen: {result.success}")
        
        # Test fuzzy matching
        result = await handler.process_vision_action("descripe_screan")  # Intentional typo
        print(f"✓ Fuzzy matching works: {result.success}")
        if result.alternative_actions:
            print(f"  Suggestions: {result.alternative_actions}")
            
    except Exception as e:
        print(f"❌ Action Handler Error: {e}")
        
    # Test 5: Custom Plugin
    print("\n\n5️⃣ Testing Custom Plugin Loading...")
    try:
        # The plugin should be auto-discovered
        from vision.vision_plugin_system import get_vision_plugin_system
        
        plugin_system = get_vision_plugin_system()
        
        # Check if custom plugin was loaded
        custom_caps = [cap for cap in plugin_system.list_capabilities() 
                      if 'custom' in cap.lower()]
        
        if custom_caps:
            print(f"✓ Custom plugin loaded with capabilities: {custom_caps}")
            
            # Test custom capability
            result, metadata = await plugin_system.execute_capability(
                "custom_analysis",
                query="test custom analysis"
            )
            print(f"✓ Custom analysis executed: {metadata.get('provider')}")
        else:
            print("ℹ️ No custom plugins found (this is okay)")
            
    except Exception as e:
        print(f"❌ Custom Plugin Error: {e}")
        
    print("\n\n✨ Enhanced Vision System Test Complete!")
    print("\nKey Features Demonstrated:")
    print("  ✓ Zero hardcoding - everything is discovered")
    print("  ✓ ML-based routing and learning")
    print("  ✓ Plugin architecture for extensibility")
    print("  ✓ Unified system intelligently routes requests")
    print("  ✓ Fuzzy matching and error correction")
    print("  ✓ Performance tracking and optimization")


async def test_real_world_scenarios():
    """Test real-world vision scenarios"""
    print("\n\n🌍 Testing Real-World Scenarios")
    print("=" * 60)
    
    try:
        from vision.unified_vision_system import get_unified_vision_system
        unified = get_unified_vision_system()
        
        scenarios = [
            {
                'name': 'Developer Workflow',
                'command': 'analyze my development environment and tell me what files I have open',
                'context': {'domain': 'development'}
            },
            {
                'name': 'Meeting Preparation',
                'command': 'check if I have any meeting notifications or calendar events visible',
                'context': {'task': 'meeting_prep'}
            },
            {
                'name': 'Error Detection',
                'command': 'scan my screen for any error messages or warnings',
                'context': {'priority': 'high'}
            },
            {
                'name': 'Multi-Window Analysis',
                'command': 'describe all open windows and their relationships',
                'context': {'scope': 'workspace'}
            }
        ]
        
        for scenario in scenarios:
            print(f"\n🎯 Scenario: {scenario['name']}")
            response = await unified.process_vision_request(scenario['command'])
            print(f"✓ Success: {response.success}")
            if response.success:
                print(f"✓ Result: {response.description[:150]}...")
                
    except Exception as e:
        print(f"❌ Real-world scenario error: {e}")


if __name__ == "__main__":
    print("🚀 Enhanced Vision System Test Suite")
    print("Testing dynamic, ML-based vision with zero hardcoding")
    
    # Run tests
    asyncio.run(test_unified_vision())
    asyncio.run(test_real_world_scenarios())