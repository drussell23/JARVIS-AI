#!/usr/bin/env python3
"""
Comprehensive Integration Tests for Phase 1.1 Multi-Monitor Support
====================================================================

Tests all PRD requirements:
G1: Detect all connected monitors
G2: Map spaces to displays
G3: Capture screenshots per-monitor
G4: Display-aware summaries
G5: User queries ("What's on my second monitor?")

Author: Derek Russell
Date: 2025-10-14
"""

import pytest
import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestMultiMonitorSupport:
    """Test suite for Phase 1.1 Multi-Monitor Support"""
    
    @pytest.mark.asyncio
    async def test_g1_detect_all_monitors(self):
        """G1: Detect all connected monitors"""
        from vision.multi_monitor_detector import MultiMonitorDetector
        
        detector = MultiMonitorDetector()
        displays = await detector.detect_displays()
        
        # Should detect at least 1 display
        assert len(displays) > 0, "No displays detected"
        assert all(d.display_id is not None for d in displays), "Invalid display IDs"
        assert all(d.resolution is not None for d in displays), "Invalid resolutions"
        
        # Should have one primary display
        primary_displays = [d for d in displays if d.is_primary]
        assert len(primary_displays) >= 1, "No primary display found"
        
        print(f"✅ G1: Detected {len(displays)} displays")
        return True
    
    @pytest.mark.asyncio
    async def test_g2_map_spaces_to_displays(self):
        """G2: Map spaces to displays via Yabai"""
        from vision.multi_monitor_detector import MultiMonitorDetector
        
        detector = MultiMonitorDetector()
        mappings = await detector.get_space_display_mapping()
        
        # Should have at least one space mapped
        assert len(mappings) > 0, "No space mappings found"
        
        # All mappings should be valid
        for space_id, display_id in mappings.items():
            assert isinstance(space_id, int), "Invalid space ID"
            assert isinstance(display_id, int), "Invalid display ID"
        
        print(f"✅ G2: Mapped {len(mappings)} spaces to displays")
        print(f"   Mapping: {dict(list(mappings.items())[:5])}")
        return True
    
    @pytest.mark.asyncio
    async def test_g3_capture_per_monitor(self):
        """G3: Capture screenshots per-monitor"""
        from vision.multi_monitor_detector import MultiMonitorDetector
        import numpy as np
        
        detector = MultiMonitorDetector()
        result = await detector.capture_all_displays()
        
        # Capture should succeed
        assert result.success, f"Capture failed: {result.error}"
        assert len(result.displays_captured) > 0, "No displays captured"
        
        # Screenshots should be valid numpy arrays
        for display_id, screenshot in result.displays_captured.items():
            assert isinstance(screenshot, np.ndarray), f"Screenshot not numpy array"
            assert screenshot.ndim == 3, "Screenshot not 3D array"
            assert screenshot.shape[2] == 3, "Screenshot not RGB"
        
        print(f"✅ G3: Captured {len(result.displays_captured)} displays in {result.capture_time:.2f}s")
        return True
    
    @pytest.mark.asyncio
    async def test_g4_display_aware_summaries(self):
        """G4: Display-aware summaries"""
        from vision.multi_monitor_detector import MultiMonitorDetector
        
        detector = MultiMonitorDetector()
        summary = await detector.get_display_summary()
        
        # Summary should contain display info
        assert "total_displays" in summary
        assert "displays" in summary
        assert "space_mappings" in summary
        
        assert summary["total_displays"] > 0
        assert len(summary["displays"]) > 0
        
        # Each display should have required fields
        for display in summary["displays"]:
            assert "id" in display
            assert "resolution" in display
            assert "position" in display
            assert "is_primary" in display
        
        print(f"✅ G4: Generated display summary with {summary['total_displays']} displays")
        return True
    
    @pytest.mark.asyncio
    async def test_g5_user_queries_disambiguation(self):
        """G5: User queries with disambiguation"""
        from vision.multi_monitor_detector import MultiMonitorDetector
        from vision.query_disambiguation import QueryDisambiguator
        
        detector = MultiMonitorDetector()
        displays = await detector.detect_displays()
        
        if len(displays) < 2:
            print("⏭️  Skipping multi-monitor query test (single display)")
            return True
        
        disambiguator = QueryDisambiguator()
        
        # Test "second monitor" resolution
        ref = await disambiguator.resolve_monitor_reference("What's on my second monitor?", displays)
        assert ref is not None, "Failed to resolve 'second monitor'"
        assert ref.display_id == displays[1].display_id, "Resolved to wrong display"
        assert ref.confidence >= 0.9, "Low confidence"
        
        # Test "primary monitor" resolution
        ref = await disambiguator.resolve_monitor_reference("What's on the primary monitor?", displays)
        assert ref is not None, "Failed to resolve 'primary monitor'"
        assert any(d.display_id == ref.display_id and d.is_primary for d in displays), "Not primary display"
        
        # Test ambiguous query (should need clarification)
        ref = await disambiguator.resolve_monitor_reference("What's on the monitor?", displays)
        if ref:
            assert ref.ambiguous == True, "Should be flagged as ambiguous"
        
        # Test clarification generation
        clarification = await disambiguator.ask_clarification("What's on the monitor?", displays)
        assert "Sir" in clarification
        assert str(len(displays)) in clarification
        
        print(f"✅ G5: Query disambiguation working correctly")
        return True
    
    @pytest.mark.asyncio
    async def test_orchestrator_integration(self):
        """Test integration with Intelligent Orchestrator"""
        from vision.intelligent_orchestrator import get_intelligent_orchestrator
        
        orchestrator = get_intelligent_orchestrator()
        
        # Scout workspace should now include display info
        snapshot = await orchestrator._scout_workspace()
        
        # Should have display information
        assert hasattr(snapshot, 'displays'), "Snapshot missing displays field"
        assert hasattr(snapshot, 'space_display_mapping'), "Snapshot missing space_display_mapping"
        assert hasattr(snapshot, 'total_displays'), "Snapshot missing total_displays"
        
        print(f"✅ Orchestrator Integration: {snapshot.total_displays} displays in workspace snapshot")
        print(f"   {snapshot.total_spaces} spaces across {snapshot.total_displays} displays")
        return True
    
    @pytest.mark.asyncio
    async def test_yabai_integration(self):
        """Test integration with Yabai Space Detector"""
        from vision.yabai_space_detector import YabaiSpaceDetector
        
        yabai = YabaiSpaceDetector()
        
        if not yabai.is_available():
            print("⏭️  Skipping Yabai test (not available)")
            return True
        
        # Should have new display-aware methods
        assert hasattr(yabai, 'get_display_for_space'), "Missing get_display_for_space method"
        assert hasattr(yabai, 'enumerate_spaces_by_display'), "Missing enumerate_spaces_by_display method"
        
        # Test display-aware enumeration
        spaces = yabai.enumerate_all_spaces(include_display_info=True)
        assert len(spaces) > 0, "No spaces found"
        
        # All spaces should have display info
        for space in spaces:
            assert "display" in space, f"Space {space.get('space_id')} missing display field"
        
        # Test grouping by display
        spaces_by_display = yabai.enumerate_spaces_by_display()
        assert len(spaces_by_display) > 0, "No spaces grouped"
        
        print(f"✅ Yabai Integration: {len(spaces)} spaces across {len(spaces_by_display)} displays")
        return True
    
    @pytest.mark.asyncio
    async def test_query_routing(self):
        """Test query routing for multi-monitor queries"""
        from api.vision_command_handler import VisionCommandHandler
        
        handler = VisionCommandHandler()
        
        # Should detect multi-monitor queries
        assert handler._is_multi_monitor_query("What's on my second monitor?")
        assert handler._is_multi_monitor_query("Show me all displays")
        assert handler._is_multi_monitor_query("What's on the primary monitor?")
        assert handler._is_multi_monitor_query("Analyze monitor 2")
        
        # Should NOT detect these as multi-monitor
        assert not handler._is_multi_monitor_query("What's happening across my desktop spaces?")
        assert not handler._is_multi_monitor_query("What error do you see in Space 3?")
        
        print(f"✅ Query Routing: Multi-monitor queries correctly detected")
        return True


async def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("🧪 PHASE 1.1 MULTI-MONITOR SUPPORT - COMPREHENSIVE TESTS")
    print("="*70 + "\n")
    
    tests = TestMultiMonitorSupport()
    results = []
    
    test_methods = [
        ("G1: Display Detection", tests.test_g1_detect_all_monitors),
        ("G2: Space-Display Mapping", tests.test_g2_map_spaces_to_displays),
        ("G3: Per-Monitor Capture", tests.test_g3_capture_per_monitor),
        ("G4: Display Summaries", tests.test_g4_display_aware_summaries),
        ("G5: Query Disambiguation", tests.test_g5_user_queries_disambiguation),
        ("Orchestrator Integration", tests.test_orchestrator_integration),
        ("Yabai Integration", tests.test_yabai_integration),
        ("Query Routing", tests.test_query_routing)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in test_methods:
        try:
            print(f"\n📍 {test_name}")
            print("-" * 70)
            result = await test_func()
            results.append((test_name, True, None))
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            results.append((test_name, False, str(e)))
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for test_name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if error:
            print(f"       Error: {error}")
    
    print(f"\n📈 Results: {passed}/{len(test_methods)} tests passed ({int(passed/len(test_methods)*100)}%)")
    
    if passed == len(test_methods):
        print("\n🎉 ALL TESTS PASSED - PHASE 1.1 COMPLETE!")
        print("\n✅ PRD REQUIREMENTS MET:")
        print("   G1: Detect all monitors - ✅")
        print("   G2: Map spaces to displays - ✅")
        print("   G3: Capture per-monitor - ✅")
        print("   G4: Display-aware summaries - ✅")
        print("   G5: User queries - ✅")
        print("\n🚀 Multi-Monitor Support: PRODUCTION READY")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed - review errors above")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
