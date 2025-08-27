#!/usr/bin/env python3
"""
Simple test for JARVIS Vision System v2.0 - Phase 4 Features
Quick verification of continuous learning capabilities
"""

import asyncio
from vision.vision_system_v2 import get_vision_system_v2


async def main():
    print("\n🚀 JARVIS Vision System v2.0 - Phase 4 Test")
    print("=" * 50)
    
    # Initialize system
    system = get_vision_system_v2()
    
    # Check Phase 4 availability
    print("\n📋 Phase 4 Component Status:")
    print(f"- Advanced Learning: {'✅' if system.advanced_learning else '❌'}")
    print(f"- Experience Replay: {'✅' if system.experience_replay else '❌'}")
    
    if not system.advanced_learning:
        print("\n⚠️  Phase 4 components not available. Check imports and dependencies.")
        return
    
    # Test basic functionality
    print("\n🧪 Testing Phase 4 Integration...")
    
    # Process a command
    response = await system.process_command(
        "Can you see what's on my screen?",
        context={'user': 'test_user', 'test_mode': True}
    )
    
    print(f"\n📊 Command Processing Results:")
    print(f"- Success: {'✅' if response.success else '❌'}")
    print(f"- Confidence: {response.confidence:.2%}")
    print(f"- Intent: {response.intent_type}")
    print(f"- Phase 4 Enabled: {'✅' if response.data.get('phase4_enabled') else '❌'}")
    
    # Get learning status
    if system.advanced_learning:
        status = system.advanced_learning.get_status()
        
        print(f"\n🎯 Continuous Learning Status:")
        print(f"- Experience Buffer: {status['experience_replay']['buffer_stats']['current_size']} experiences")
        print(f"- Buffer Utilization: {status['experience_replay']['buffer_stats']['utilization']:.1%}")
        print(f"- Active Learning Tasks: {status['continuous_learning']['active_tasks']}")
        print(f"- Current Learning Rate: {status['learning_rate']:.6f}")
        
        # Meta-learning info
        meta_info = status['meta_learning']
        if 'current_strategy' in meta_info:
            print(f"- Learning Strategy: {meta_info['current_strategy']}")
            
    # Test pattern extraction
    if system.experience_replay:
        print(f"\n🔍 Pattern Extraction:")
        stats = await system.experience_replay.get_statistics()
        print(f"- Patterns Found: {stats['pattern_stats']['total_patterns']}")
        if stats['pattern_stats']['pattern_types']:
            print(f"- Pattern Types: {dict(stats['pattern_stats']['pattern_types'])}")
    
    print("\n✅ Phase 4 test complete!")
    
    # Cleanup
    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())