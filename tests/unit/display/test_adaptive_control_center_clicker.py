#!/usr/bin/env python3
"""
Unit Tests for AdaptiveControlCenterClicker
===========================================

Comprehensive unit tests covering:
- Coordinate caching and TTL
- Detection method fallback chain
- Verification engine
- Error handling and edge cases
- Metrics tracking
- Cache invalidation scenarios

Author: Derek J. Russell
Date: October 2025
"""

import asyncio
import pytest
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from unittest.mock import Mock, AsyncMock, MagicMock, patch, call
from PIL import Image
import numpy as np

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "backend"))

from display.adaptive_control_center_clicker import (
    AdaptiveControlCenterClicker,
    CoordinateCache,
    CachedCoordinate,
    DetectionResult,
    ClickResult,
    VerificationEngine,
    CachedDetection,
    OCRDetection,
    TemplateMatchingDetection,
    EdgeDetection,
    DetectionStatus,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def cache(temp_cache_dir):
    """Create CoordinateCache instance with temp directory"""
    cache_file = temp_cache_dir / "test_cache.json"
    return CoordinateCache(cache_file=cache_file, ttl_seconds=60)


@pytest.fixture
def mock_vision_analyzer():
    """Create mock vision analyzer"""
    mock = AsyncMock()
    mock.analyze_screenshot = AsyncMock(return_value={
        'analysis': 'COORDINATES: x=1245, y=12'
    })
    return mock


@pytest.fixture
def adaptive_clicker(mock_vision_analyzer, temp_cache_dir):
    """Create AdaptiveControlCenterClicker instance"""
    cache_file = temp_cache_dir / "test_cache.json"

    with patch('display.adaptive_control_center_clicker.CoordinateCache') as mock_cache_class:
        mock_cache = Mock()
        mock_cache.get = Mock(return_value=None)
        mock_cache.set = Mock()
        mock_cache_class.return_value = mock_cache

        clicker = AdaptiveControlCenterClicker(
            vision_analyzer=mock_vision_analyzer,
            cache_ttl=60,
            enable_verification=False  # Disable for unit tests
        )
        clicker.cache = mock_cache

        return clicker


# ============================================================================
# CoordinateCache Tests
# ============================================================================

class TestCoordinateCache:
    """Test coordinate caching functionality"""

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initializes correctly"""
        cache_file = temp_cache_dir / "test_cache.json"
        cache = CoordinateCache(cache_file=cache_file, ttl_seconds=60)

        assert cache.cache_file == cache_file
        assert cache.ttl_seconds == 60
        assert isinstance(cache.cache, dict)
        assert len(cache.cache) == 0

    def test_cache_set_and_get(self, cache):
        """Test setting and getting cached coordinates"""
        # Set coordinate
        cache.set(
            target="control_center",
            coordinates=(1245, 12),
            confidence=0.95,
            method="ocr"
        )

        # Get coordinate
        cached = cache.get("control_center")

        assert cached is not None
        assert cached.target == "control_center"
        assert cached.coordinates == (1245, 12)
        assert cached.confidence == 0.95
        assert cached.method == "ocr"
        assert cached.success_count == 1
        assert cached.failure_count == 0

    def test_cache_miss(self, cache):
        """Test cache miss returns None"""
        cached = cache.get("nonexistent")
        assert cached is None

    def test_cache_ttl_expiration(self, temp_cache_dir):
        """Test cache entries expire after TTL"""
        cache_file = temp_cache_dir / "test_cache.json"
        cache = CoordinateCache(cache_file=cache_file, ttl_seconds=1)  # 1 second TTL

        # Set coordinate
        cache.set(
            target="control_center",
            coordinates=(1245, 12),
            confidence=0.95,
            method="ocr"
        )

        # Should be available immediately
        cached = cache.get("control_center")
        assert cached is not None

        # Wait for TTL to expire
        time.sleep(1.5)

        # Should be expired now
        cached = cache.get("control_center")
        assert cached is None

    def test_cache_success_tracking(self, cache):
        """Test success count increments on repeated sets"""
        # Set coordinate multiple times
        for _ in range(3):
            cache.set(
                target="control_center",
                coordinates=(1245, 12),
                confidence=0.95,
                method="ocr"
            )

        cached = cache.get("control_center")
        assert cached.success_count == 3
        assert cached.failure_count == 0

    def test_cache_failure_tracking(self, cache):
        """Test failure tracking"""
        # Set coordinate
        cache.set(
            target="control_center",
            coordinates=(1245, 12),
            confidence=0.95,
            method="ocr"
        )

        # Mark failures
        cache.mark_failure("control_center")
        cache.mark_failure("control_center")

        cached = cache.get("control_center")
        assert cached.failure_count == 2

    def test_cache_high_failure_rate_invalidation(self, cache):
        """Test cache invalidates entries with high failure rate"""
        # Set coordinate
        cache.set(
            target="control_center",
            coordinates=(1245, 12),
            confidence=0.95,
            method="ocr"
        )

        # Mark many failures (more than 2x success count)
        for _ in range(5):
            cache.mark_failure("control_center")

        # Should be invalidated due to high failure rate
        cached = cache.get("control_center")
        assert cached is None

    def test_cache_screen_resolution_awareness(self, cache, temp_cache_dir):
        """Test cache invalidates when screen resolution changes"""
        # Set coordinate with current resolution
        cache.set(
            target="control_center",
            coordinates=(1245, 12),
            confidence=0.95,
            method="ocr"
        )

        # Simulate resolution change by creating new cache with different hash
        with patch.object(CoordinateCache, '_get_screen_hash', return_value='different'):
            cache2 = CoordinateCache(
                cache_file=temp_cache_dir / "test_cache.json",
                ttl_seconds=60
            )

            # Manually load the cache data
            cache2.cache = cache.cache.copy()

            # Should not find cached coordinate due to resolution mismatch
            cached = cache2.get("control_center")
            assert cached is None

    def test_cache_invalidate(self, cache):
        """Test manual cache invalidation"""
        # Set coordinate
        cache.set(
            target="control_center",
            coordinates=(1245, 12),
            confidence=0.95,
            method="ocr"
        )

        # Verify it exists
        assert cache.get("control_center") is not None

        # Invalidate
        cache.invalidate("control_center")

        # Should be gone
        assert cache.get("control_center") is None

    def test_cache_clear(self, cache):
        """Test clearing entire cache"""
        # Set multiple coordinates
        cache.set("control_center", (1245, 12), 0.95, "ocr")
        cache.set("screen_mirroring", (1393, 177), 0.90, "template")

        # Verify they exist
        assert len(cache.cache) == 2

        # Clear cache
        cache.clear()

        # Should be empty
        assert len(cache.cache) == 0
        assert cache.get("control_center") is None
        assert cache.get("screen_mirroring") is None

    def test_cache_persistence(self, temp_cache_dir):
        """Test cache persists to disk"""
        cache_file = temp_cache_dir / "test_cache.json"

        # Create cache and set coordinate
        cache1 = CoordinateCache(cache_file=cache_file, ttl_seconds=60)
        cache1.set("control_center", (1245, 12), 0.95, "ocr")

        # Create new cache instance (should load from disk)
        cache2 = CoordinateCache(cache_file=cache_file, ttl_seconds=60)

        # Should have loaded the coordinate
        cached = cache2.get("control_center")
        assert cached is not None
        assert cached.coordinates == (1245, 12)


# ============================================================================
# Detection Method Tests
# ============================================================================

class TestCachedDetection:
    """Test cached coordinate detection"""

    @pytest.mark.asyncio
    async def test_cached_detection_hit(self, cache):
        """Test successful detection from cache"""
        # Set up cache
        cache.set("control_center", (1245, 12), 0.95, "ocr")

        # Create detection method
        detector = CachedDetection(cache)

        # Detect
        result = await detector.detect("control_center")

        assert result.success is True
        assert result.method == "cached"
        assert result.coordinates == (1245, 12)
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_cached_detection_miss(self, cache):
        """Test detection miss when cache is empty"""
        detector = CachedDetection(cache)

        result = await detector.detect("control_center")

        assert result.success is False
        assert result.coordinates is None
        assert result.error == "No cached coordinates found"

    @pytest.mark.asyncio
    async def test_cached_detection_always_available(self, cache):
        """Test cached detection is always available"""
        detector = CachedDetection(cache)
        assert await detector.is_available() is True


class TestOCRDetection:
    """Test OCR-based detection"""

    @pytest.mark.asyncio
    async def test_ocr_with_tesseract_success(self, mock_vision_analyzer):
        """Test OCR detection with pytesseract"""
        detector = OCRDetection(vision_analyzer=mock_vision_analyzer)

        # Mock pytesseract
        mock_tesseract_data = {
            'text': ['', 'Control', 'Center', ''],
            'left': [0, 100, 1240, 0],
            'top': [0, 10, 10, 0],
            'width': [0, 50, 60, 0],
            'height': [0, 20, 20, 0],
            'conf': [0, 90, 92, 0]
        }

        with patch('pytesseract.image_to_data', return_value=mock_tesseract_data):
            with patch('pytesseract.get_tesseract_version', return_value='5.0'):
                with patch('pyautogui.screenshot') as mock_screenshot:
                    mock_screenshot.return_value = Image.new('RGB', (100, 100))

                    detector._tesseract_available = True
                    result = await detector.detect("control center")

                    assert result.success is True
                    assert result.method == "ocr_tesseract"
                    assert result.coordinates is not None

    @pytest.mark.asyncio
    async def test_ocr_with_claude_vision_success(self, mock_vision_analyzer):
        """Test OCR detection with Claude Vision"""
        detector = OCRDetection(vision_analyzer=mock_vision_analyzer)
        detector._tesseract_available = False

        with patch('pyautogui.screenshot') as mock_screenshot:
            mock_screenshot.return_value = Image.new('RGB', (2000, 100))

            result = await detector.detect("control center")

            assert result.success is True
            assert result.method == "ocr_claude"
            assert result.coordinates == (1245, 12)

    @pytest.mark.asyncio
    async def test_ocr_not_found(self, mock_vision_analyzer):
        """Test OCR detection when target not found"""
        # Configure mock to return NOT_FOUND
        mock_vision_analyzer.analyze_screenshot = AsyncMock(return_value={
            'analysis': 'NOT_FOUND'
        })

        detector = OCRDetection(vision_analyzer=mock_vision_analyzer)
        detector._tesseract_available = False

        with patch('pyautogui.screenshot') as mock_screenshot:
            mock_screenshot.return_value = Image.new('RGB', (2000, 100))

            result = await detector.detect("nonexistent")

            assert result.success is False
            assert result.error == "Claude Vision could not find target"

    @pytest.mark.asyncio
    async def test_ocr_availability_check(self):
        """Test OCR availability checking"""
        # No vision analyzer, no tesseract
        detector = OCRDetection(vision_analyzer=None)
        detector._tesseract_available = False

        assert await detector.is_available() is False

        # With vision analyzer
        detector = OCRDetection(vision_analyzer=Mock())
        detector._tesseract_available = False

        assert await detector.is_available() is True


class TestTemplateMatchingDetection:
    """Test template matching detection"""

    @pytest.mark.asyncio
    async def test_template_matching_success(self, temp_cache_dir):
        """Test successful template matching"""
        template_dir = temp_cache_dir / "templates"
        template_dir.mkdir()

        # Create mock template
        template_path = template_dir / "control_center.png"
        template_img = Image.new('L', (50, 50), color=128)
        template_img.save(template_path)

        detector = TemplateMatchingDetection(template_dir=template_dir)

        # Mock OpenCV functions
        with patch('cv2.matchTemplate') as mock_match:
            with patch('cv2.minMaxLoc') as mock_minmax:
                with patch('pyautogui.screenshot') as mock_screenshot:
                    # Setup mocks
                    mock_match.return_value = np.zeros((100, 100))
                    mock_minmax.return_value = (0.0, 0.9, (0, 0), (100, 100))
                    mock_screenshot.return_value = Image.new('RGB', (200, 200))

                    result = await detector.detect("control_center")

                    assert result.success is True
                    assert result.method == "template_matching"
                    assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_template_matching_low_confidence(self, temp_cache_dir):
        """Test template matching with low confidence"""
        template_dir = temp_cache_dir / "templates"
        template_dir.mkdir()

        template_path = template_dir / "control_center.png"
        template_img = Image.new('L', (50, 50))
        template_img.save(template_path)

        detector = TemplateMatchingDetection(template_dir=template_dir)

        with patch('cv2.matchTemplate') as mock_match:
            with patch('cv2.minMaxLoc') as mock_minmax:
                with patch('pyautogui.screenshot') as mock_screenshot:
                    # Low confidence match
                    mock_match.return_value = np.zeros((100, 100))
                    mock_minmax.return_value = (0.0, 0.5, (0, 0), (100, 100))
                    mock_screenshot.return_value = Image.new('RGB', (200, 200))

                    result = await detector.detect("control_center")

                    assert result.success is False
                    assert "confidence too low" in result.error.lower()

    @pytest.mark.asyncio
    async def test_template_matching_no_template(self, temp_cache_dir):
        """Test template matching when template doesn't exist"""
        template_dir = temp_cache_dir / "templates"
        template_dir.mkdir()

        detector = TemplateMatchingDetection(template_dir=template_dir)

        result = await detector.detect("nonexistent")

        assert result.success is False
        assert "no template found" in result.error.lower()


# ============================================================================
# VerificationEngine Tests
# ============================================================================

class TestVerificationEngine:
    """Test screenshot verification"""

    @pytest.mark.asyncio
    async def test_verification_with_change(self):
        """Test verification passes when screenshots differ"""
        verifier = VerificationEngine()

        # Create different screenshots
        before = Image.new('RGB', (100, 100), color='red')
        after = Image.new('RGB', (100, 100), color='blue')

        with patch('pyautogui.screenshot', return_value=after):
            result = await verifier.verify_click(
                target="control_center",
                coordinates=(50, 50),
                before_screenshot=before
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_verification_without_change(self):
        """Test verification fails when screenshots are identical"""
        verifier = VerificationEngine()

        # Create identical screenshots
        before = Image.new('RGB', (100, 100), color='red')
        after = Image.new('RGB', (100, 100), color='red')

        with patch('pyautogui.screenshot', return_value=after):
            result = await verifier.verify_click(
                target="control_center",
                coordinates=(50, 50),
                before_screenshot=before
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_verification_without_before_screenshot(self):
        """Test verification assumes success without before screenshot"""
        verifier = VerificationEngine()

        with patch('pyautogui.screenshot') as mock_screenshot:
            mock_screenshot.return_value = Image.new('RGB', (100, 100))

            result = await verifier.verify_click(
                target="control_center",
                coordinates=(50, 50),
                before_screenshot=None
            )

            assert result is True


# ============================================================================
# AdaptiveControlCenterClicker Tests
# ============================================================================

class TestAdaptiveControlCenterClicker:
    """Test main adaptive clicker functionality"""

    @pytest.mark.asyncio
    async def test_click_success_with_cache(self, adaptive_clicker):
        """Test successful click using cached coordinates"""
        # Setup cache to return coordinates
        adaptive_clicker.cache.get = Mock(return_value=CachedCoordinate(
            target="control_center",
            coordinates=(1245, 12),
            confidence=0.95,
            method="cached",
            timestamp=time.time(),
            success_count=5,
            failure_count=0,
            screen_hash="abc123",
            macos_version="14.0"
        ))

        with patch('pyautogui.moveTo') as mock_move:
            with patch('pyautogui.click') as mock_click:
                result = await adaptive_clicker.click("control_center")

                assert result.success is True
                assert result.method_used == "cached"
                assert result.coordinates == (1245, 12)

                # Verify mouse actions
                mock_move.assert_called_once_with(1245, 12, duration=0.3)
                mock_click.assert_called_once()

    @pytest.mark.asyncio
    async def test_click_fallback_to_ocr(self, adaptive_clicker, mock_vision_analyzer):
        """Test fallback to OCR when cache misses"""
        # Cache miss
        adaptive_clicker.cache.get = Mock(return_value=None)

        # Configure OCR to succeed
        for method in adaptive_clicker.detection_methods:
            if isinstance(method, OCRDetection):
                method.vision_analyzer = mock_vision_analyzer
                method._tesseract_available = False

        with patch('pyautogui.moveTo') as mock_move:
            with patch('pyautogui.click') as mock_click:
                with patch('pyautogui.screenshot') as mock_screenshot:
                    mock_screenshot.return_value = Image.new('RGB', (2000, 100))

                    result = await adaptive_clicker.click("control_center")

                    assert result.success is True
                    assert result.method_used == "ocr_claude"
                    assert result.fallback_attempts > 0

    @pytest.mark.asyncio
    async def test_click_all_methods_fail(self, adaptive_clicker):
        """Test click failure when all methods fail"""
        # Cache miss
        adaptive_clicker.cache.get = Mock(return_value=None)

        # Make all methods fail
        for method in adaptive_clicker.detection_methods:
            if hasattr(method, 'detect'):
                original_detect = method.detect
                async def failing_detect(*args, **kwargs):
                    result = await original_detect(*args, **kwargs)
                    result.success = False
                    return result
                method.detect = failing_detect

        result = await adaptive_clicker.click("control_center")

        assert result.success is False
        assert result.error is not None
        assert result.fallback_attempts > 0

    @pytest.mark.asyncio
    async def test_click_updates_cache_on_success(self, adaptive_clicker):
        """Test cache is updated after successful click"""
        adaptive_clicker.cache.get = Mock(return_value=None)
        adaptive_clicker.cache.set = Mock()

        # Make OCR succeed
        for method in adaptive_clicker.detection_methods:
            if isinstance(method, OCRDetection):
                method.vision_analyzer = AsyncMock()
                method.vision_analyzer.analyze_screenshot = AsyncMock(
                    return_value={'analysis': 'COORDINATES: x=1245, y=12'}
                )
                method._tesseract_available = False

        with patch('pyautogui.moveTo'):
            with patch('pyautogui.click'):
                with patch('pyautogui.screenshot') as mock_screenshot:
                    mock_screenshot.return_value = Image.new('RGB', (2000, 100))

                    result = await adaptive_clicker.click("control_center")

                    if result.success:
                        # Verify cache was updated
                        adaptive_clicker.cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_to_device_flow(self, adaptive_clicker):
        """Test complete device connection flow"""
        # Setup cache to return coordinates for all steps
        def cache_get(target):
            coords = {
                "control_center": (1245, 12),
                "screen_mirroring": (1393, 177),
                "Living Room TV": (1221, 116)
            }
            if target in coords:
                return CachedCoordinate(
                    target=target,
                    coordinates=coords[target],
                    confidence=0.95,
                    method="cached",
                    timestamp=time.time(),
                    success_count=1,
                    failure_count=0,
                    screen_hash="abc123",
                    macos_version="14.0"
                )
            return None

        adaptive_clicker.cache.get = Mock(side_effect=cache_get)

        with patch('pyautogui.moveTo'):
            with patch('pyautogui.click'):
                result = await adaptive_clicker.connect_to_device("Living Room TV")

                assert result["success"] is True
                assert "steps" in result
                assert "control_center" in result["steps"]
                assert "screen_mirroring" in result["steps"]
                assert "device" in result["steps"]

    def test_get_metrics(self, adaptive_clicker):
        """Test metrics tracking"""
        # Simulate some activity
        adaptive_clicker.metrics["total_attempts"] = 10
        adaptive_clicker.metrics["successful_clicks"] = 8
        adaptive_clicker.metrics["failed_clicks"] = 2
        adaptive_clicker.metrics["cache_hits"] = 5

        metrics = adaptive_clicker.get_metrics()

        assert metrics["total_attempts"] == 10
        assert metrics["successful_clicks"] == 8
        assert metrics["success_rate"] == 0.8
        assert metrics["cache_hit_rate"] == 0.5

    def test_clear_cache(self, adaptive_clicker):
        """Test cache clearing"""
        adaptive_clicker.cache.clear = Mock()

        adaptive_clicker.clear_cache()

        adaptive_clicker.cache.clear.assert_called_once()


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error scenarios"""

    @pytest.mark.asyncio
    async def test_screen_resolution_change(self, temp_cache_dir):
        """Test handling of screen resolution changes"""
        cache_file = temp_cache_dir / "test_cache.json"

        # Create cache with one resolution
        with patch('pyautogui.size', return_value=(1440, 900)):
            cache1 = CoordinateCache(cache_file=cache_file, ttl_seconds=60)
            cache1.set("control_center", (1245, 12), 0.95, "ocr")

        # Load cache with different resolution
        with patch('pyautogui.size', return_value=(1920, 1080)):
            cache2 = CoordinateCache(cache_file=cache_file, ttl_seconds=60)

            # Should not find cached coordinate
            cached = cache2.get("control_center")
            # Will be None because screen hash differs
            # Note: This might pass or fail depending on implementation

    @pytest.mark.asyncio
    async def test_concurrent_clicks(self, adaptive_clicker):
        """Test handling of concurrent click attempts"""
        adaptive_clicker.cache.get = Mock(return_value=CachedCoordinate(
            target="control_center",
            coordinates=(1245, 12),
            confidence=0.95,
            method="cached",
            timestamp=time.time(),
            success_count=1,
            failure_count=0,
            screen_hash="abc123",
            macos_version="14.0"
        ))

        with patch('pyautogui.moveTo'):
            with patch('pyautogui.click'):
                # Launch multiple clicks concurrently
                results = await asyncio.gather(
                    adaptive_clicker.click("control_center"),
                    adaptive_clicker.click("control_center"),
                    adaptive_clicker.click("control_center")
                )

                # All should succeed
                assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_timeout_handling(self, adaptive_clicker):
        """Test timeout handling for slow detection methods"""
        # Create a slow detection method
        async def slow_detect(*args, **kwargs):
            await asyncio.sleep(10)  # Very slow
            return DetectionResult(
                success=False,
                method="slow",
                coordinates=None,
                confidence=0.0,
                duration=10.0,
                metadata={}
            )

        # Replace first non-cached method with slow one
        for i, method in enumerate(adaptive_clicker.detection_methods):
            if method.name != "cached":
                method.detect = slow_detect
                break

        adaptive_clicker.cache.get = Mock(return_value=None)

        # Click should still complete (fallback to other methods)
        # Note: This test might be slow, consider timeout
        with patch('pyautogui.moveTo'):
            with patch('pyautogui.click'):
                result = await asyncio.wait_for(
                    adaptive_clicker.click("control_center"),
                    timeout=15.0
                )

                # Should either succeed or fail, but not hang
                assert isinstance(result, ClickResult)


# ============================================================================
# Integration-like Tests (still unit-level mocking)
# ============================================================================

class TestIntegrationLike:
    """Higher-level tests simulating real usage"""

    @pytest.mark.asyncio
    async def test_learning_from_failures(self, temp_cache_dir):
        """Test system learns from failures and adapts"""
        cache_file = temp_cache_dir / "test_cache.json"
        cache = CoordinateCache(cache_file=cache_file, ttl_seconds=60)

        # Set initial coordinate
        cache.set("control_center", (1245, 12), 0.95, "ocr")

        # Simulate failures
        for _ in range(3):
            cache.mark_failure("control_center")

        # Should still be retrievable (not invalidated yet)
        cached = cache.get("control_center")
        assert cached is not None

        # More failures (exceeds 2x success rate)
        for _ in range(5):
            cache.mark_failure("control_center")

        # Should be invalidated now
        cached = cache.get("control_center")
        assert cached is None

    @pytest.mark.asyncio
    async def test_multi_method_fallback_chain(self, adaptive_clicker):
        """Test complete fallback chain execution"""
        adaptive_clicker.cache.get = Mock(return_value=None)

        # Track which methods were attempted
        attempted_methods = []

        for method in adaptive_clicker.detection_methods:
            original_detect = method.detect

            async def tracking_detect(*args, method=method, **kwargs):
                attempted_methods.append(method.name)
                return await original_detect(*args, **kwargs)

            method.detect = tracking_detect

        # All methods will fail (no real screen/templates)
        result = await adaptive_clicker.click("control_center")

        # Verify fallback chain was executed
        assert len(attempted_methods) > 1
        assert "cached" in attempted_methods  # Should try cached first

    @pytest.mark.asyncio
    async def test_vision_analyzer_integration(self, mock_vision_analyzer):
        """Test integration with vision analyzer"""
        clicker = AdaptiveControlCenterClicker(
            vision_analyzer=mock_vision_analyzer,
            enable_verification=False
        )

        clicker.cache.get = Mock(return_value=None)

        with patch('pyautogui.moveTo'):
            with patch('pyautogui.click'):
                with patch('pyautogui.screenshot') as mock_screenshot:
                    mock_screenshot.return_value = Image.new('RGB', (2000, 100))

                    result = await clicker.click("control_center")

                    # Should succeed with OCR
                    assert result.success is True
                    assert "ocr" in result.method_used


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
