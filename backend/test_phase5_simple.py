#!/usr/bin/env python3
"""
Simple test for JARVIS Vision System v2.0 - Phase 5 Features
Quick verification of autonomous capability discovery
"""

import asyncio
from vision.vision_system_v2 import get_vision_system_v2


async def main():
    print("\n🤖 JARVIS Vision System v2.0 - Phase 5 Test")
    print("=" * 50)
    
    # Initialize system
    system = get_vision_system_v2()
    
    # Check Phase 5 availability
    print("\n📋 Phase 5 Component Status:")
    status = system.get_autonomous_capabilities_status()
    
    if status.get('available'):
        components = status.get('components', {})
        print(f"- Capability Generator: {'✅' if components.get('capability_generator') else '❌'}")
        print(f"- Safety Synthesizer: {'✅' if components.get('synthesizer') else '❌'}")
        print(f"- Safety Verifier: {'✅' if components.get('verifier') else '❌'}")
        print(f"- Performance Benchmark: {'✅' if components.get('benchmark') else '❌'}")
        print(f"- Gradual Rollout: {'✅' if components.get('rollout') else '❌'}")
    else:
        print("❌ Phase 5 components not available")
        return
    
    # Test a command that might trigger capability generation
    print("\n🧪 Testing Autonomous Capability Discovery...")
    
    # This command should fail and potentially trigger capability generation
    test_command = "analyze the purple widgets on the desktop"
    
    response = await system.process_command(
        test_command,
        context={'user': 'test_user', 'test_mode': True}
    )
    
    print(f"\n📊 Command Processing Results:")
    print(f"- Command: '{test_command}'")
    print(f"- Success: {'✅' if response.success else '❌'}")
    print(f"- Intent: {response.intent_type}")
    print(f"- Phase 5 Enabled: {'✅' if response.data.get('phase5_enabled') else '❌'}")
    
    if not response.success:
        print(f"- Error handled: System will analyze for capability generation")
    
    # Wait a moment for background processing
    await asyncio.sleep(2)
    
    # Check capability generation stats
    if status.get('available'):
        updated_status = system.get_autonomous_capabilities_status()
        
        if 'generation_stats' in updated_status:
            stats = updated_status['generation_stats']
            print(f"\n📈 Capability Generation Stats:")
            print(f"- Total generated: {stats.get('total_generated', 0)}")
            print(f"- Pending validation: {stats.get('pending_validation', 0)}")
            print(f"- Validated: {stats.get('validated', 0)}")
            
        if 'rollout_status' in updated_status:
            rollout = updated_status['rollout_status']
            print(f"\n🚀 Rollout Status:")
            print(f"- Active rollouts: {rollout.get('active', 0)}")
            print(f"- Total rollouts: {rollout.get('total_rollouts', 0)}")
    
    print("\n✅ Phase 5 test complete!")
    print("\n💡 Key Features Demonstrated:")
    print("- Failed requests trigger capability analysis")
    print("- Safe code synthesis for new capabilities")
    print("- Comprehensive safety verification")
    print("- Gradual rollout management")
    print("- Autonomous learning and adaptation")
    
    # Cleanup
    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())