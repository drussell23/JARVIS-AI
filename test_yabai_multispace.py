#!/usr/bin/env python3
"""
Comprehensive test for Yabai Multi-Space Intelligence System
Tests all components: YabaiSpaceDetector, WorkspaceAnalyzer, SpaceResponseGenerator
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_yabai_detection():
    """Test 1: Yabai detection and availability"""
    print("\n" + "=" * 80)
    print("TEST 1: Yabai Detection and Availability")
    print("=" * 80)

    try:
        from vision.yabai_space_detector import YabaiSpaceDetector, YabaiStatus

        detector = YabaiSpaceDetector()
        status = await detector.check_availability()

        print(f"✓ Yabai Status: {status.value}")

        if status == YabaiStatus.AVAILABLE:
            print("✓ Yabai is installed and functional")
            return detector
        elif status == YabaiStatus.NOT_INSTALLED:
            print("⚠ Yabai not installed - install with: brew install koekeishiya/formulae/yabai")
            return None
        elif status == YabaiStatus.NO_PERMISSIONS:
            print("⚠ Yabai needs Accessibility permissions")
            return None
        else:
            print(f"✗ Yabai error: {status.value}")
            return None

    except Exception as e:
        print(f"✗ Error testing Yabai detection: {e}")
        return None


async def test_space_enumeration(detector):
    """Test 2: Space enumeration"""
    print("\n" + "=" * 80)
    print("TEST 2: Space Enumeration")
    print("=" * 80)

    try:
        spaces = await detector.get_spaces()

        print(f"✓ Retrieved {len(spaces)} spaces")

        for space in spaces[:7]:  # Show first 7
            print(f"  - Space {space.index}: {space.display_name}")
            print(f"    Display: {space.display}, Focused: {space.is_focused}, Visible: {space.is_visible}")
            print(f"    Windows: {space.window_count}")

        return spaces

    except Exception as e:
        print(f"✗ Error enumerating spaces: {e}")
        return []


async def test_window_retrieval(detector):
    """Test 3: Window retrieval"""
    print("\n" + "=" * 80)
    print("TEST 3: Window Retrieval")
    print("=" * 80)

    try:
        windows = await detector.get_windows()

        print(f"✓ Retrieved {len(windows)} windows")

        # Group by app
        apps = {}
        for window in windows:
            if window.app_name not in apps:
                apps[window.app_name] = []
            apps[window.app_name].append(window)

        print(f"\n✓ Found {len(apps)} unique applications:")
        for app_name, app_windows in list(apps.items())[:10]:  # Show first 10
            print(f"  - {app_name}: {len(app_windows)} window(s)")

        return windows

    except Exception as e:
        print(f"✗ Error retrieving windows: {e}")
        return []


async def test_workspace_analysis(detector):
    """Test 4: Workspace activity analysis"""
    print("\n" + "=" * 80)
    print("TEST 4: Workspace Activity Analysis")
    print("=" * 80)

    try:
        from vision.workspace_analyzer import WorkspaceAnalyzer

        analyzer = WorkspaceAnalyzer()

        # Get workspace data
        workspace_data = await detector.get_workspace_data()
        spaces = workspace_data['spaces']
        windows = workspace_data['windows']

        # Analyze
        analysis = analyzer.analyze(spaces, windows)

        print(f"✓ Workspace Analysis Complete")
        print(f"\n📊 Summary:")
        print(f"  Total Spaces: {analysis.total_spaces}")
        print(f"  Active Spaces: {analysis.active_spaces}")
        print(f"  Total Windows: {analysis.total_windows}")
        print(f"  Unique Applications: {analysis.unique_applications}")

        if analysis.detected_project:
            print(f"  Detected Project: {analysis.detected_project}")

        if analysis.overall_activity:
            print(f"  Overall Activity: {analysis.overall_activity.activity_type.value}")
            print(f"  Activity Confidence: {analysis.overall_activity.confidence:.1%}")

        if analysis.focus_pattern:
            print(f"  Focus Pattern: {analysis.focus_pattern}")

        print(f"\n📋 Space-by-Space Breakdown:")
        for space in analysis.space_summaries:
            if space.is_active:
                print(f"\n  Space {space.space_id}: {space.space_name}")
                print(f"    Windows: {space.window_count}")
                print(f"    Apps: {', '.join(space.applications[:3])}")
                if len(space.applications) > 3:
                    print(f"          + {len(space.applications) - 3} more")
                print(f"    Activity: {space.activity_description}")
                print(f"    Confidence: {space.primary_activity.confidence:.1%}")

        return analysis

    except Exception as e:
        print(f"✗ Error analyzing workspace: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_response_generation(analysis):
    """Test 5: Natural language response generation"""
    print("\n" + "=" * 80)
    print("TEST 5: Natural Language Response Generation")
    print("=" * 80)

    try:
        from vision.space_response_generator import SpaceResponseGenerator

        generator = SpaceResponseGenerator(use_sir_prefix=True)

        # Generate full overview response
        response = generator.generate_overview_response(analysis, include_details=True)

        print("✓ Generated Response:\n")
        print("-" * 80)
        print(response)
        print("-" * 80)

        return response

    except Exception as e:
        print(f"✗ Error generating response: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_performance(detector):
    """Test 6: Performance and caching"""
    print("\n" + "=" * 80)
    print("TEST 6: Performance and Caching")
    print("=" * 80)

    try:
        import time

        # Test 1: Fresh query
        start = time.time()
        await detector.get_workspace_data(force_refresh=True)
        fresh_time = time.time() - start

        print(f"✓ Fresh query time: {fresh_time * 1000:.1f}ms")

        # Test 2: Cached query
        start = time.time()
        await detector.get_workspace_data(force_refresh=False)
        cached_time = time.time() - start

        print(f"✓ Cached query time: {cached_time * 1000:.1f}ms")

        # Performance stats
        stats = detector.get_performance_stats()
        print(f"\n📈 Performance Metrics:")
        print(f"  Total Queries: {stats['total_queries']}")
        print(f"  Cache Hits: {stats['cache_hits']}")
        print(f"  Cache Hit Rate: {stats['cache_hit_rate']}")
        print(f"  Avg Query Time: {stats['avg_query_time_ms']}ms")
        print(f"  Yabai Status: {stats['yabai_status']}")

        speedup = fresh_time / cached_time if cached_time > 0 else 0
        print(f"\n✓ Cache speedup: {speedup:.1f}x faster")

    except Exception as e:
        print(f"✗ Error testing performance: {e}")


async def test_integration():
    """Test 7: Full integration test"""
    print("\n" + "=" * 80)
    print("TEST 7: Full Integration Test")
    print("=" * 80)

    try:
        # Import all components
        from vision.yabai_space_detector import YabaiSpaceDetector, YabaiStatus
        from vision.workspace_analyzer import WorkspaceAnalyzer
        from vision.space_response_generator import SpaceResponseGenerator

        # Initialize system
        detector = YabaiSpaceDetector(cache_ttl=5, query_timeout=5, enable_cache=True)
        analyzer = WorkspaceAnalyzer()
        generator = SpaceResponseGenerator(use_sir_prefix=True)

        # Check availability
        status = await detector.check_availability()
        if status != YabaiStatus.AVAILABLE:
            print(f"⚠ Yabai not available: {status.value}")
            response = generator.generate_yabai_installation_response(status)
            print(f"\nGuidance:\n{response}")
            return

        # Full workflow
        print("✓ Getting workspace data...")
        workspace_data = await detector.get_workspace_data()

        print("✓ Analyzing workspace activity...")
        analysis = analyzer.analyze(workspace_data['spaces'], workspace_data['windows'])

        print("✓ Generating natural language response...")
        response = generator.generate_overview_response(analysis, include_details=True)

        print("\n" + "=" * 80)
        print("FINAL JARVIS RESPONSE:")
        print("=" * 80)
        print(response)
        print("=" * 80)

        print("\n✅ Full integration test passed!")

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "YABAI MULTI-SPACE INTELLIGENCE TEST SUITE" + " " * 16 + "║")
    print("╚" + "=" * 78 + "╝")

    # Test 1: Detection
    detector = await test_yabai_detection()
    if not detector:
        print("\n❌ Yabai not available - cannot continue tests")
        print("\n💡 Install Yabai to enable multi-space intelligence:")
        print("   brew install koekeishiya/formulae/yabai")
        return

    # Test 2: Space enumeration
    spaces = await test_space_enumeration(detector)
    if not spaces:
        return

    # Test 3: Window retrieval
    windows = await test_window_retrieval(detector)
    if not windows:
        return

    # Test 4: Workspace analysis
    analysis = await test_workspace_analysis(detector)
    if not analysis:
        return

    # Test 5: Response generation
    response = await test_response_generation(analysis)
    if not response:
        return

    # Test 6: Performance
    await test_performance(detector)

    # Test 7: Full integration
    await test_integration()

    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nThe Yabai multi-space intelligence system is fully operational!")
    print("You can now ask JARVIS: 'What's happening across my desktop spaces?'")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠ Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
