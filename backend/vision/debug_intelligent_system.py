"""
Debug script to diagnose intelligent routing issues
"""

import asyncio
import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def debug_intelligent_system():
    """Debug the intelligent routing system"""

    print("=" * 80)
    print("JARVIS INTELLIGENT SYSTEM DIAGNOSTIC")
    print("=" * 80)

    # 1. Check if vision handler is initialized
    print("\n1. Checking Vision Command Handler...")
    try:
        # Add parent directories to path
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from api.vision_command_handler import vision_command_handler
        print("✅ Vision handler imported")

        # Check components
        print(f"   - Intelligence: {vision_command_handler.intelligence is not None}")
        print(f"   - Classifier: {vision_command_handler.classifier is not None}")
        print(f"   - Smart Router: {vision_command_handler.smart_router is not None}")
        print(f"   - Context Manager: {vision_command_handler.context_manager is not None}")
        print(f"   - Yabai Detector: {vision_command_handler.yabai_detector is not None}")
        print(f"   - Proactive System: {vision_command_handler.proactive_system is not None}")

    except Exception as e:
        print(f"❌ Failed to import vision handler: {e}")
        return

    # 2. Check if intelligent system modules are available
    print("\n2. Checking Intelligent System Modules...")
    try:
        from vision.intelligent_query_classifier import get_query_classifier
        print("✅ Query Classifier available")
    except ImportError as e:
        print(f"❌ Query Classifier not available: {e}")

    try:
        from vision.smart_query_router import get_smart_router
        print("✅ Smart Router available")
    except ImportError as e:
        print(f"❌ Smart Router not available: {e}")

    try:
        from vision.query_context_manager import get_context_manager
        print("✅ Context Manager available")
    except ImportError as e:
        print(f"❌ Context Manager not available: {e}")

    # 3. Try to initialize intelligence
    print("\n3. Initializing Intelligence System...")
    try:
        # Check if API key is available
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            print(f"✅ API key found: {api_key[:10]}...")
            await vision_command_handler.initialize_intelligence(api_key)
            print("✅ Intelligence initialized")

            # Check again
            print(f"   - Intelligence: {vision_command_handler.intelligence is not None}")
            print(f"   - Classifier: {vision_command_handler.classifier is not None}")
            print(f"   - Smart Router: {vision_command_handler.smart_router is not None}")
        else:
            print("⚠️  No API key found in environment")

    except Exception as e:
        print(f"❌ Failed to initialize intelligence: {e}")
        import traceback
        traceback.print_exc()

    # 4. Check Yabai detector
    print("\n4. Checking Yabai Detector...")
    if vision_command_handler.yabai_detector:
        detector = vision_command_handler.yabai_detector
        print("✅ Yabai detector exists")

        # Check methods
        methods = [
            'get_focused_space',
            'get_all_spaces',
            'get_all_windows',
            'check_availability',
            'get_workspace_data',
            'get_performance_stats'
        ]

        for method in methods:
            has_method = hasattr(detector, method)
            print(f"   - {method}: {'✅' if has_method else '❌'}")
    else:
        print("❌ Yabai detector not initialized")

    # 5. Test a sample query
    print("\n5. Testing Sample Query...")
    try:
        test_query = "What's happening across my desktop spaces?"
        print(f"Query: '{test_query}'")

        result = await vision_command_handler.handle_command(test_query)

        print(f"\nResult:")
        print(f"   - Handled: {result.get('handled')}")
        print(f"   - Intelligent Routing: {result.get('intelligent_routing', False)}")
        print(f"   - Intent: {result.get('intent', 'N/A')}")
        print(f"   - Latency: {result.get('latency_ms', 'N/A')}ms")

        if result.get('intelligent_routing'):
            print("✅ Intelligent routing is ACTIVE")
        else:
            print("❌ Intelligent routing is NOT active")

        print(f"\nResponse preview:")
        response = result.get('response', 'No response')
        print(f"   {response[:200]}...")

    except Exception as e:
        print(f"❌ Query test failed: {e}")
        import traceback
        traceback.print_exc()

    # 6. Check routing stats
    print("\n6. Checking Routing Statistics...")
    try:
        if vision_command_handler.smart_router:
            stats = vision_command_handler.smart_router.get_routing_stats()
            print(f"✅ Router stats:")
            print(f"   - Total queries: {stats.get('total_queries', 0)}")
            print(f"   - Metadata only: {stats.get('metadata_only', 0)}")
            print(f"   - Visual analysis: {stats.get('visual_analysis', 0)}")
            print(f"   - Deep analysis: {stats.get('deep_analysis', 0)}")
        else:
            print("⚠️  No router available")
    except Exception as e:
        print(f"❌ Failed to get stats: {e}")

    # 7. Check context manager
    print("\n7. Checking Context Manager...")
    try:
        if vision_command_handler.context_manager:
            context = vision_command_handler.context_manager.get_context_for_query("")
            print(f"✅ Context manager working")
            print(f"   - Active space: {context.get('active_space')}")
            print(f"   - Total spaces: {context.get('total_spaces')}")
            print(f"   - User pattern: {context.get('user_pattern')}")
        else:
            print("⚠️  No context manager")
    except Exception as e:
        print(f"❌ Context manager error: {e}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(debug_intelligent_system())
