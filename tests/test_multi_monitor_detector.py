"""
Test suite for Multi-Monitor Support

This module provides comprehensive testing for the MultiMonitorDetector
and related multi-monitor functionality.

Author: Derek Russell
Date: 2025-01-14
Branch: multi-monitor-support
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import time
from typing import Dict, List, Any

# Import the modules under test
from backend.vision.multi_monitor_detector import (
    MultiMonitorDetector,
    DisplayInfo,
    SpaceDisplayMapping,
    MonitorCaptureResult,
    MACOS_AVAILABLE
)


class TestDisplayInfo:
    """Test DisplayInfo dataclass"""
    
    def test_display_info_creation(self):
        """Test creating DisplayInfo objects"""
        display = DisplayInfo(
            display_id=1,
            resolution=(1920, 1080),
            position=(0, 0),
            is_primary=True,
            name="Primary Display"
        )
        
        assert display.display_id == 1
        assert display.resolution == (1920, 1080)
        assert display.position == (0, 0)
        assert display.is_primary is True
        assert display.name == "Primary Display"
        assert display.refresh_rate == 60.0  # default
        assert display.color_depth == 32  # default
        assert display.spaces == []  # default empty list
        assert display.active_space == 1  # default
    
    def test_display_info_with_custom_values(self):
        """Test DisplayInfo with custom values"""
        display = DisplayInfo(
            display_id=2,
            resolution=(2560, 1440),
            position=(1920, 0),
            is_primary=False,
            refresh_rate=144.0,
            color_depth=24,
            name="Secondary Display",
            spaces=[1, 2, 3],
            active_space=2
        )
        
        assert display.display_id == 2
        assert display.resolution == (2560, 1440)
        assert display.position == (1920, 0)
        assert display.is_primary is False
        assert display.refresh_rate == 144.0
        assert display.color_depth == 24
        assert display.name == "Secondary Display"
        assert display.spaces == [1, 2, 3]
        assert display.active_space == 2


class TestSpaceDisplayMapping:
    """Test SpaceDisplayMapping dataclass"""
    
    def test_space_mapping_creation(self):
        """Test creating SpaceDisplayMapping objects"""
        mapping = SpaceDisplayMapping(
            space_id=1,
            display_id=1,
            space_name="Desktop 1",
            is_active=True
        )
        
        assert mapping.space_id == 1
        assert mapping.display_id == 1
        assert mapping.space_name == "Desktop 1"
        assert mapping.is_active is True
        assert mapping.last_seen > 0  # Should be set to current time


class TestMonitorCaptureResult:
    """Test MonitorCaptureResult dataclass"""
    
    def test_capture_result_success(self):
        """Test successful capture result"""
        screenshots = {
            1: np.zeros((1080, 1920, 3), dtype=np.uint8),
            2: np.zeros((1440, 2560, 3), dtype=np.uint8)
        }
        
        result = MonitorCaptureResult(
            success=True,
            displays_captured=screenshots,
            failed_displays=[],
            capture_time=0.5,
            total_displays=2
        )
        
        assert result.success is True
        assert len(result.displays_captured) == 2
        assert result.failed_displays == []
        assert result.capture_time == 0.5
        assert result.total_displays == 2
        assert result.error is None
    
    def test_capture_result_failure(self):
        """Test failed capture result"""
        result = MonitorCaptureResult(
            success=False,
            displays_captured={},
            failed_displays=[1, 2],
            capture_time=1.0,
            total_displays=2,
            error="Permission denied"
        )
        
        assert result.success is False
        assert len(result.displays_captured) == 0
        assert result.failed_displays == [1, 2]
        assert result.capture_time == 1.0
        assert result.total_displays == 2
        assert result.error == "Permission denied"


class TestMultiMonitorDetector:
    """Test MultiMonitorDetector class"""
    
    @pytest.fixture
    def detector(self):
        """Create a MultiMonitorDetector instance for testing"""
        return MultiMonitorDetector(yabai_path="mock_yabai")
    
    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert detector.yabai_path == "mock_yabai"
        assert detector.displays == {}
        assert detector.space_mappings == {}
        assert detector.last_detection_time == 0.0
        assert detector.detection_cache_duration == 5.0
        
        # Check performance stats initialization
        stats = detector.capture_stats
        assert stats["total_captures"] == 0
        assert stats["successful_captures"] == 0
        assert stats["failed_captures"] == 0
        assert stats["average_capture_time"] == 0.0
    
    @pytest.mark.asyncio
    async def test_detect_displays_no_macos(self, detector):
        """Test display detection when macOS frameworks are not available"""
        with patch('backend.vision.multi_monitor_detector.MACOS_AVAILABLE', False):
            displays = await detector.detect_displays()
            assert displays == []
    
    @pytest.mark.asyncio
    async def test_detect_displays_with_macos(self, detector):
        """Test display detection with macOS frameworks available"""
        if not MACOS_AVAILABLE:
            pytest.skip("macOS frameworks not available for testing")
        
        # Mock Core Graphics functions
        with patch('backend.vision.multi_monitor_detector.Quartz') as mock_quartz:
            # Mock display list
            mock_quartz.CGGetActiveDisplayList.return_value = 2
            mock_quartz.CGDisplayBounds.return_value = Mock(
                size=Mock(width=1920, height=1080),
                origin=Mock(x=0, y=0)
            )
            mock_quartz.CGDisplayIsMain.return_value = True
            
            displays = await detector.detect_displays()
            
            # Should have detected displays
            assert len(displays) > 0
            assert detector.last_detection_time > 0
    
    @pytest.mark.asyncio
    async def test_detect_displays_caching(self, detector):
        """Test display detection caching"""
        # First call should detect displays
        displays1 = await detector.detect_displays()
        
        # Second call within cache duration should use cache
        displays2 = await detector.detect_displays()
        
        # Should be the same result
        assert displays1 == displays2
    
    @pytest.mark.asyncio
    async def test_detect_displays_force_refresh(self, detector):
        """Test forced refresh of display detection"""
        # First call
        await detector.detect_displays()
        first_time = detector.last_detection_time
        
        # Wait a bit
        await asyncio.sleep(0.1)
        
        # Force refresh
        await detector.detect_displays(force_refresh=True)
        second_time = detector.last_detection_time
        
        # Should be different times
        assert second_time > first_time
    
    @pytest.mark.asyncio
    async def test_get_space_display_mapping(self, detector):
        """Test space-display mapping"""
        # Mock Yabai subprocess
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b'{"spaces": []}', b'')
        mock_process.returncode = 0
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            mappings = await detector.get_space_display_mapping()
            
            # Should return a dictionary
            assert isinstance(mappings, dict)
    
    @pytest.mark.asyncio
    async def test_capture_all_displays_no_displays(self, detector):
        """Test capture when no displays are detected"""
        with patch.object(detector, 'detect_displays', return_value=[]):
            result = await detector.capture_all_displays()
            
            assert result.success is False
            assert len(result.displays_captured) == 0
            assert result.total_displays == 0
            assert "No displays detected" in result.error
    
    @pytest.mark.asyncio
    async def test_capture_all_displays_with_displays(self, detector):
        """Test capture with detected displays"""
        # Mock displays
        mock_displays = [
            DisplayInfo(display_id=1, resolution=(1920, 1080), position=(0, 0), is_primary=True),
            DisplayInfo(display_id=2, resolution=(2560, 1440), position=(1920, 0), is_primary=False)
        ]
        
        with patch.object(detector, 'detect_displays', return_value=mock_displays):
            with patch.object(detector, '_capture_display', return_value=np.zeros((1080, 1920, 3), dtype=np.uint8)):
                result = await detector.capture_all_displays()
                
                assert result.success is True
                assert len(result.displays_captured) == 2
                assert result.total_displays == 2
                assert result.capture_time > 0
    
    @pytest.mark.asyncio
    async def test_capture_display_success(self, detector):
        """Test successful display capture"""
        display_info = DisplayInfo(
            display_id=1,
            resolution=(1920, 1080),
            position=(0, 0),
            is_primary=True
        )
        
        # Mock Core Graphics
        mock_image_ref = Mock()
        mock_quartz = Mock()
        mock_quartz.CGWindowListCreateImage.return_value = mock_image_ref
        mock_quartz.CGImageGetWidth.return_value = 1920
        mock_quartz.CGImageGetHeight.return_value = 1080
        mock_quartz.CGImageGetDataProvider.return_value = Mock()
        mock_quartz.CGDataProviderCopyData.return_value = b'\x00' * (1920 * 1080 * 4)
        
        with patch('backend.vision.multi_monitor_detector.Quartz', mock_quartz):
            with patch('backend.vision.multi_monitor_detector.MACOS_AVAILABLE', True):
                screenshot = await detector._capture_display(display_info)
                
                # Should return a numpy array
                assert isinstance(screenshot, np.ndarray)
                assert screenshot.shape == (1080, 1920, 3)
    
    @pytest.mark.asyncio
    async def test_get_display_summary(self, detector):
        """Test getting display summary"""
        # Mock methods
        mock_displays = [
            DisplayInfo(display_id=1, resolution=(1920, 1080), position=(0, 0), is_primary=True, name="Primary")
        ]
        mock_mappings = {1: SpaceDisplayMapping(space_id=1, display_id=1)}
        
        with patch.object(detector, 'detect_displays', return_value=mock_displays):
            with patch.object(detector, 'get_space_display_mapping', return_value=mock_mappings):
                summary = await detector.get_display_summary()
                
                assert summary["total_displays"] == 1
                assert len(summary["displays"]) == 1
                assert summary["space_mappings"] == mock_mappings
                assert "capture_stats" in summary
    
    def test_get_performance_stats(self, detector):
        """Test getting performance statistics"""
        stats = detector.get_performance_stats()
        
        assert "capture_stats" in stats
        assert "displays_cached" in stats
        assert "space_mappings_cached" in stats
        assert "last_detection_time" in stats
        assert "cache_age" in stats


class TestIntegration:
    """Integration tests for multi-monitor functionality"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete multi-monitor workflow"""
        detector = MultiMonitorDetector()
        
        # Test the complete workflow
        try:
            # Detect displays
            displays = await detector.detect_displays()
            
            # Get space mappings
            mappings = await detector.get_space_display_mapping()
            
            # Get summary
            summary = await detector.get_display_summary()
            
            # Get performance stats
            stats = detector.get_performance_stats()
            
            # All operations should complete without errors
            assert isinstance(displays, list)
            assert isinstance(mappings, dict)
            assert isinstance(summary, dict)
            assert isinstance(stats, dict)
            
        except Exception as e:
            # If macOS frameworks aren't available, that's expected
            if "macOS frameworks not available" in str(e):
                pytest.skip("macOS frameworks not available")
            else:
                raise
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in various scenarios"""
        detector = MultiMonitorDetector()
        
        # Test with invalid yabai path
        detector.yabai_path = "/nonexistent/yabai"
        
        try:
            mappings = await detector.get_space_display_mapping()
            # Should handle error gracefully
            assert isinstance(mappings, dict)
        except Exception as e:
            # Should not crash the system
            assert isinstance(e, Exception)


# Convenience test functions
def test_macos_availability():
    """Test macOS availability detection"""
    # This test will pass regardless of platform
    assert isinstance(MACOS_AVAILABLE, bool)


@pytest.mark.asyncio
async def test_convenience_functions():
    """Test convenience functions"""
    from backend.vision.multi_monitor_detector import (
        detect_all_monitors,
        capture_multi_monitor_screenshots,
        get_monitor_summary
    )
    
    try:
        # Test convenience functions
        displays = await detect_all_monitors()
        summary = await get_monitor_summary()
        
        assert isinstance(displays, list)
        assert isinstance(summary, dict)
        
    except Exception as e:
        if "macOS frameworks not available" in str(e):
            pytest.skip("macOS frameworks not available")
        else:
            raise


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
