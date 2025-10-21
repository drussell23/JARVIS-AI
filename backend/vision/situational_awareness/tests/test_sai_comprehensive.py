#!/usr/bin/env python3
"""
Comprehensive Test Suite for Situational Awareness Intelligence (SAI)
======================================================================

Tests all SAI components and integration scenarios.

Test Coverage:
- Environment hashing and change detection
- UI element tracking and caching
- Display topology awareness
- Automatic revalidation
- SAI-enhanced Control Center clicker integration
- Multi-monitor scenarios
- Error handling and resilience

Author: Derek J. Russell
Date: October 2025
"""

import asyncio
import pytest
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from backend.vision.situational_awareness import (
    SituationalAwarenessEngine,
    UIElementMonitor,
    SystemUIElementTracker,
    AdaptiveCacheManager,
    EnvironmentHasher,
    MultiDisplayAwareness,
    UIElementDescriptor,
    UIElementPosition,
    EnvironmentalSnapshot,
    ChangeEvent,
    ElementType,
    ChangeType,
    ConfidenceLevel
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_vision_analyzer():
    """Create mock vision analyzer"""
    analyzer = Mock()
    analyzer.analyze_screenshot = AsyncMock(return_value={
        'analysis': 'COORDINATES: x=1236, y=12'
    })
    analyzer.capture_screenshot = AsyncMock(return_value=Mock())
    return analyzer


@pytest.fixture
def sample_element_descriptor():
    """Create sample element descriptor"""
    return UIElementDescriptor(
        element_id="test_element",
        element_type=ElementType.MENU_BAR_ICON,
        display_characteristics={
            'icon_description': 'Test icon',
            'color': 'blue',
            'location': 'top-right'
        }
    )


@pytest.fixture
def sample_element_position():
    """Create sample element position"""
    import time
    return UIElementPosition(
        element_id="test_element",
        coordinates=(100, 200),
        confidence=0.95,
        detection_method="vision",
        timestamp=time.time(),
        display_id=0
    )


# ============================================================================
# Test Environment Hasher
# ============================================================================

class TestEnvironmentHasher:
    """Test environment hashing and change detection"""

    def test_hash_generation(self):
        """Test environment hash generation"""
        hasher = EnvironmentHasher()

        display_topology = {
            'display_count': 1,
            'primary_display_id': 0,
            'displays': [{'width': 1920, 'height': 1080}]
        }

        system_metadata = {
            'os_version': '14.0',
            'active_space': 1
        }

        hash1 = hasher.hash_environment(display_topology, system_metadata)

        assert hash1 is not None
        assert len(hash1) == 12
        assert isinstance(hash1, str)

    def test_hash_consistency(self):
        """Test hash consistency for same environment"""
        hasher = EnvironmentHasher()

        topology = {'display_count': 1, 'displays': []}
        metadata = {'os_version': '14.0'}

        hash1 = hasher.hash_environment(topology, metadata)
        hash2 = hasher.hash_environment(topology, metadata)

        assert hash1 == hash2

    def test_hash_changes_on_environment_change(self):
        """Test hash changes when environment changes"""
        hasher = EnvironmentHasher()

        topology1 = {'display_count': 1, 'displays': []}
        topology2 = {'display_count': 2, 'displays': []}
        metadata = {'os_version': '14.0'}

        hash1 = hasher.hash_environment(topology1, metadata)
        hash2 = hasher.hash_environment(topology2, metadata)

        assert hash1 != hash2

    def test_detect_position_changes(self, sample_element_position):
        """Test detection of element position changes"""
        import time
        hasher = EnvironmentHasher()

        # Create two snapshots with different positions
        old_pos = UIElementPosition(
            element_id="test",
            coordinates=(100, 100),
            confidence=0.9,
            detection_method="vision",
            timestamp=time.time(),
            display_id=0
        )

        new_pos = UIElementPosition(
            element_id="test",
            coordinates=(150, 120),
            confidence=0.9,
            detection_method="vision",
            timestamp=time.time(),
            display_id=0
        )

        old_snapshot = EnvironmentalSnapshot(
            timestamp=time.time(),
            environment_hash="abc123",
            display_topology={},
            active_space=1,
            screen_resolution=(1920, 1080),
            element_positions={'test': old_pos},
            system_metadata={}
        )

        new_snapshot = EnvironmentalSnapshot(
            timestamp=time.time(),
            environment_hash="def456",
            display_topology={},
            active_space=1,
            screen_resolution=(1920, 1080),
            element_positions={'test': new_pos},
            system_metadata={}
        )

        changes = hasher.detect_changes(old_snapshot, new_snapshot)

        assert len(changes) > 0
        position_changes = [c for c in changes if c.change_type == ChangeType.POSITION_CHANGED]
        assert len(position_changes) == 1
        assert position_changes[0].element_id == "test"


# ============================================================================
# Test Adaptive Cache Manager
# ============================================================================

class TestAdaptiveCacheManager:
    """Test intelligent caching system"""

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initialization"""
        cache_file = temp_cache_dir / "test_cache.json"
        cache = AdaptiveCacheManager(cache_file=cache_file, default_ttl=3600)

        assert cache.cache_file == cache_file
        assert cache.default_ttl == 3600
        assert len(cache.position_cache) == 0

    def test_cache_set_and_get(self, temp_cache_dir, sample_element_position):
        """Test caching and retrieval"""
        cache_file = temp_cache_dir / "test_cache.json"
        cache = AdaptiveCacheManager(cache_file=cache_file)

        # Set position
        cache.set("test_element", sample_element_position, "env_hash_123")

        # Get position
        retrieved = cache.get("test_element", "env_hash_123")

        assert retrieved is not None
        assert retrieved.element_id == "test_element"
        assert retrieved.coordinates == sample_element_position.coordinates

    def test_cache_miss(self, temp_cache_dir):
        """Test cache miss"""
        cache_file = temp_cache_dir / "test_cache.json"
        cache = AdaptiveCacheManager(cache_file=cache_file)

        # Try to get non-existent element
        retrieved = cache.get("nonexistent", "env_hash")

        assert retrieved is None
        assert cache.metrics['misses'] == 1

    def test_cache_invalidation(self, temp_cache_dir, sample_element_position):
        """Test cache invalidation"""
        cache_file = temp_cache_dir / "test_cache.json"
        cache = AdaptiveCacheManager(cache_file=cache_file)

        # Set and invalidate
        cache.set("test_element", sample_element_position, "env_hash")
        cache.invalidate("test_element", reason="test")

        # Verify invalidated
        retrieved = cache.get("test_element", "env_hash")
        assert retrieved is None

    def test_cache_persistence(self, temp_cache_dir, sample_element_position):
        """Test cache persistence across instances"""
        cache_file = temp_cache_dir / "test_cache.json"

        # Create first cache instance and save
        cache1 = AdaptiveCacheManager(cache_file=cache_file)
        cache1.set("test_element", sample_element_position, "env_hash")

        # Create second instance and verify data loaded
        cache2 = AdaptiveCacheManager(cache_file=cache_file)
        retrieved = cache2.get("test_element", "env_hash")

        assert retrieved is not None
        assert retrieved.element_id == "test_element"

    def test_lru_eviction(self, temp_cache_dir):
        """Test LRU eviction when cache is full"""
        import time
        cache_file = temp_cache_dir / "test_cache.json"
        cache = AdaptiveCacheManager(cache_file=cache_file, max_cache_size=3)

        # Add 4 elements (should trigger eviction)
        for i in range(4):
            pos = UIElementPosition(
                element_id=f"element_{i}",
                coordinates=(i * 100, i * 100),
                confidence=0.9,
                detection_method="test",
                timestamp=time.time() + i,  # Different timestamps
                display_id=0
            )
            cache.set(f"element_{i}", pos, "env_hash")
            time.sleep(0.01)  # Ensure different timestamps

        # First element should be evicted
        assert cache.get("element_0", "env_hash") is None
        assert len(cache.position_cache) <= 3


# ============================================================================
# Test UI Element Monitor
# ============================================================================

class TestUIElementMonitor:
    """Test vision-based element detection"""

    @pytest.mark.asyncio
    async def test_element_registration(self, mock_vision_analyzer, sample_element_descriptor):
        """Test element registration"""
        monitor = UIElementMonitor(mock_vision_analyzer)

        monitor.register_element(sample_element_descriptor)

        assert "test_element" in monitor.element_registry
        assert monitor.element_registry["test_element"] == sample_element_descriptor

    @pytest.mark.asyncio
    async def test_element_detection(self, mock_vision_analyzer, sample_element_descriptor):
        """Test element position detection"""
        monitor = UIElementMonitor(mock_vision_analyzer)
        monitor.register_element(sample_element_descriptor)

        # Mock screenshot
        from PIL import Image
        mock_screenshot = Image.new('RGB', (1920, 1080), color='white')

        position = await monitor.detect_element("test_element", mock_screenshot)

        assert position is not None
        assert position.element_id == "test_element"
        assert position.coordinates == (1236, 12)  # From mock response

    @pytest.mark.asyncio
    async def test_detection_without_registration(self, mock_vision_analyzer):
        """Test detection fails for unregistered element"""
        monitor = UIElementMonitor(mock_vision_analyzer)

        position = await monitor.detect_element("unregistered", None)

        assert position is None


# ============================================================================
# Test Multi-Display Awareness
# ============================================================================

class TestMultiDisplayAwareness:
    """Test display topology tracking"""

    @pytest.mark.asyncio
    async def test_topology_update(self):
        """Test display topology update"""
        awareness = MultiDisplayAwareness()

        topology = await awareness.update_topology()

        assert topology is not None
        assert 'display_count' in topology
        assert 'displays' in topology
        assert topology['display_count'] > 0

    def test_coordinate_to_display_mapping(self):
        """Test mapping coordinates to display"""
        awareness = MultiDisplayAwareness()
        awareness.display_topology = {
            'display_count': 2,
            'primary_display_id': 0,
            'displays': [
                {'display_id': 0, 'width': 1920, 'height': 1080, 'position': (0, 0)},
                {'display_id': 1, 'width': 1920, 'height': 1080, 'position': (1920, 0)}
            ]
        }

        # Test coordinate on first display
        display = awareness.get_display_for_coordinates(100, 100)
        assert display == 0

        # Test coordinate on second display
        display = awareness.get_display_for_coordinates(2000, 100)
        assert display == 1


# ============================================================================
# Test Situational Awareness Engine
# ============================================================================

class TestSituationalAwarenessEngine:
    """Test main SAI orchestrator"""

    @pytest.mark.asyncio
    async def test_engine_initialization(self, mock_vision_analyzer):
        """Test SAI engine initialization"""
        engine = SituationalAwarenessEngine(
            vision_analyzer=mock_vision_analyzer,
            monitoring_interval=5.0
        )

        assert engine is not None
        assert engine.monitor is not None
        assert engine.tracker is not None
        assert engine.cache is not None
        assert engine.hasher is not None

    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, mock_vision_analyzer):
        """Test monitoring start/stop"""
        engine = SituationalAwarenessEngine(mock_vision_analyzer, monitoring_interval=1.0)

        # Start monitoring
        await engine.start_monitoring()
        assert engine.is_monitoring is True

        # Let it run briefly
        await asyncio.sleep(0.5)

        # Stop monitoring
        await engine.stop_monitoring()
        assert engine.is_monitoring is False

    @pytest.mark.asyncio
    async def test_element_position_retrieval(self, mock_vision_analyzer, sample_element_descriptor):
        """Test getting element position"""
        engine = SituationalAwarenessEngine(mock_vision_analyzer)
        engine.monitor.register_element(sample_element_descriptor)

        position = await engine.get_element_position("test_element", use_cache=False, force_detect=True)

        assert position is not None
        assert position.element_id == "test_element"

    @pytest.mark.asyncio
    async def test_change_callback(self, mock_vision_analyzer):
        """Test change event callbacks"""
        engine = SituationalAwarenessEngine(mock_vision_analyzer)

        callback_triggered = []

        def on_change(change: ChangeEvent):
            callback_triggered.append(change)

        engine.register_change_callback(on_change)

        # Simulate a change
        import time
        change = ChangeEvent(
            change_type=ChangeType.POSITION_CHANGED,
            element_id="test",
            old_value=(100, 100),
            new_value=(150, 150),
            timestamp=time.time(),
            confidence=0.9
        )

        await engine._process_changes([change])

        assert len(callback_triggered) == 1
        assert callback_triggered[0].element_id == "test"

    def test_metrics_collection(self, mock_vision_analyzer):
        """Test metrics collection"""
        engine = SituationalAwarenessEngine(mock_vision_analyzer)

        metrics = engine.get_metrics()

        assert 'monitoring' in metrics
        assert 'cache' in metrics
        assert 'display' in metrics
        assert 'changes' in metrics
        assert 'tracked_elements' in metrics


# ============================================================================
# Integration Tests
# ============================================================================

class TestSAIIntegration:
    """Integration tests for complete SAI system"""

    @pytest.mark.asyncio
    async def test_end_to_end_detection_and_caching(self, mock_vision_analyzer):
        """Test complete flow: detect → cache → retrieve"""
        engine = SituationalAwarenessEngine(mock_vision_analyzer)

        # Register element
        descriptor = UIElementDescriptor(
            element_id="control_center",
            element_type=ElementType.MENU_BAR_ICON,
            display_characteristics={'icon_description': 'Control Center'}
        )
        engine.monitor.register_element(descriptor)

        # First detection (should use vision)
        pos1 = await engine.get_element_position("control_center", use_cache=False)
        assert pos1 is not None

        # Second detection (should use cache)
        pos2 = await engine.get_element_position("control_center", use_cache=True)
        assert pos2 is not None
        assert pos2.coordinates == pos1.coordinates

    @pytest.mark.asyncio
    async def test_environment_change_detection_flow(self, mock_vision_analyzer):
        """Test environment change detection and response"""
        import time
        engine = SituationalAwarenessEngine(mock_vision_analyzer, monitoring_interval=0.5)

        change_events = []

        def record_change(change):
            change_events.append(change)

        engine.register_change_callback(record_change)

        # Create initial snapshot
        snapshot1 = EnvironmentalSnapshot(
            timestamp=time.time(),
            environment_hash="hash1",
            display_topology={'display_count': 1},
            active_space=1,
            screen_resolution=(1920, 1080),
            element_positions={},
            system_metadata={'os_version': '14.0'}
        )

        # Create changed snapshot
        snapshot2 = EnvironmentalSnapshot(
            timestamp=time.time(),
            environment_hash="hash2",  # Different hash
            display_topology={'display_count': 2},  # Display changed
            active_space=1,
            screen_resolution=(1920, 1080),
            element_positions={},
            system_metadata={'os_version': '14.0'}
        )

        engine.current_snapshot = snapshot1

        # Process change
        changes = engine.hasher.detect_changes(snapshot1, snapshot2)
        await engine._process_changes(changes)

        # Verify change detected
        assert len(change_events) > 0


# ============================================================================
# Performance Tests
# ============================================================================

class TestSAIPerformance:
    """Performance and scalability tests"""

    @pytest.mark.asyncio
    async def test_hash_generation_performance(self):
        """Test hash generation speed"""
        import time
        hasher = EnvironmentHasher()

        topology = {'display_count': 1, 'displays': []}
        metadata = {'os_version': '14.0'}

        start = time.time()
        for _ in range(1000):
            hasher.hash_environment(topology, metadata)
        duration = time.time() - start

        # Should be very fast (< 1s for 1000 hashes)
        assert duration < 1.0
        logger.info(f"Hash generation: {duration:.3f}s for 1000 iterations ({1000/duration:.0f} hashes/sec)")

    def test_cache_access_performance(self, temp_cache_dir):
        """Test cache access speed"""
        import time
        cache_file = temp_cache_dir / "perf_cache.json"
        cache = AdaptiveCacheManager(cache_file=cache_file)

        # Add 100 elements
        for i in range(100):
            pos = UIElementPosition(
                element_id=f"element_{i}",
                coordinates=(i, i),
                confidence=0.9,
                detection_method="test",
                timestamp=time.time(),
                display_id=0
            )
            cache.set(f"element_{i}", pos, "env_hash")

        # Measure retrieval speed
        start = time.time()
        for i in range(100):
            cache.get(f"element_{i}", "env_hash")
        duration = time.time() - start

        # Should be very fast (< 0.1s for 100 retrievals)
        assert duration < 0.1
        logger.info(f"Cache retrieval: {duration:.3f}s for 100 retrievals ({100/duration:.0f} ops/sec)")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
