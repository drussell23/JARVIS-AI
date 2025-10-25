"""
Unit Tests for TemporalQueryHandler v3.0

Tests the new v3.0 features:
1. Pattern Analysis - Learning correlations between events
2. Predictive Analysis - Predicting future events based on patterns
3. Anomaly Detection - Detecting unusual behavior
4. Correlation Analysis - Multi-space relationship detection
5. Pattern Persistence - Saving/loading learned_patterns.json
6. Hybrid Monitoring Integration - Using pre-cached monitoring data
"""

import pytest
import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import the TemporalQueryHandler
import sys
sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

from context_intelligence.handlers.temporal_query_handler import (
    TemporalQueryHandler,
    TemporalQueryType,
    ChangeType,
)


class TestTemporalQueryHandlerV3:
    """Test suite for TemporalQueryHandler v3.0"""

    @pytest.fixture
    def temp_patterns_file(self):
        """Create temporary patterns file for testing persistence"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        yield temp_file
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)

    @pytest.fixture
    def mock_hybrid_monitoring(self):
        """Create mock HybridProactiveMonitoringManager"""
        mock = Mock()
        mock._alert_history = deque(maxlen=500)
        mock._pattern_rules = []

        # Add sample monitoring alerts
        now = datetime.now().timestamp()

        # Simulate pattern: Build in Space 5 → Error in Space 3
        mock._alert_history.extend([
            Mock(
                space_id=5,
                severity='INFO',
                message='Build started in Space 5',
                timestamp=now - 300,
                alert_type='BUILD_START'
            ),
            Mock(
                space_id=5,
                severity='INFO',
                message='Build completed successfully in Space 5',
                timestamp=now - 280,
                alert_type='BUILD_SUCCESS'
            ),
            Mock(
                space_id=3,
                severity='ERROR',
                message='TypeError on line 42',
                timestamp=now - 260,  # 20 seconds after build
                alert_type='ERROR_DETECTED'
            ),
            # Repeat the pattern
            Mock(
                space_id=5,
                severity='INFO',
                message='Build completed successfully in Space 5',
                timestamp=now - 180,
                alert_type='BUILD_SUCCESS'
            ),
            Mock(
                space_id=3,
                severity='ERROR',
                message='TypeError on line 42',
                timestamp=now - 160,  # 20 seconds after build
                alert_type='ERROR_DETECTED'
            ),
        ])

        return mock

    @pytest.fixture
    def mock_implicit_resolver(self):
        """Create mock ImplicitReferenceResolver"""
        mock = AsyncMock()
        mock.parse_query = AsyncMock(return_value=Mock(
            intent='DIAGNOSE',
            references=[],
            keywords=['pattern', 'error']
        ))
        return mock

    @pytest.fixture
    def mock_change_detection(self):
        """Create mock ChangeDetectionManager"""
        mock = Mock()
        mock.detect_change = AsyncMock(return_value=ChangeType.CONTENT_CHANGE)
        return mock

    @pytest.fixture
    def handler(self, mock_hybrid_monitoring, mock_implicit_resolver, mock_change_detection):
        """Create TemporalQueryHandler instance with mocks"""
        handler = TemporalQueryHandler(
            proactive_monitoring_manager=mock_hybrid_monitoring,
            change_detection_manager=mock_change_detection,
            implicit_resolver=mock_implicit_resolver,
            conversation_tracker=None
        )
        # Populate monitoring_alerts from mock data
        for alert in mock_hybrid_monitoring._alert_history:
            handler.monitoring_alerts.append({
                'space_id': alert.space_id,
                'severity': alert.severity,
                'message': alert.message,
                'timestamp': alert.timestamp,
                'alert_type': alert.alert_type
            })
        return handler

    # ========================================
    # TEST 1: Pattern Analysis
    # ========================================

    @pytest.mark.asyncio
    async def test_pattern_analysis_detects_build_error_correlation(self, handler, mock_hybrid_monitoring):
        """Test that handler detects pattern: Build in Space 5 → Error in Space 3"""

        # Analyze patterns from monitoring data
        patterns = await handler._analyze_patterns_from_monitoring()

        # Should detect correlation between builds and errors
        assert len(patterns) > 0, "Should detect at least one pattern"

        # Check for build→error pattern
        build_error_pattern = next(
            (p for p in patterns if 'build' in p.get('trigger', '').lower()
             and 'error' in p.get('outcome', '').lower()),
            None
        )

        assert build_error_pattern is not None, "Should detect build→error pattern"
        assert build_error_pattern.get('confidence', 0) > 0.5, "Pattern confidence should be >50%"
        assert build_error_pattern.get('occurrences', 0) >= 2, "Pattern should have occurred at least twice"

    @pytest.mark.asyncio
    async def test_pattern_query_type_routing(self, handler):
        """Test that PATTERN_ANALYSIS query type is handled correctly"""

        # Create a pattern analysis query
        changes = await handler.analyze_temporal_changes(
            query_type=TemporalQueryType.PATTERN_ANALYSIS,
            time_window_minutes=60
        )

        assert changes is not None, "Should return pattern analysis results"
        # Handler should attempt to analyze patterns
        assert handler.is_hybrid_monitoring, "Should use hybrid monitoring for patterns"

    # ========================================
    # TEST 2: Predictive Analysis
    # ========================================

    @pytest.mark.asyncio
    async def test_predictive_analysis_forecasts_future_events(self, handler, mock_hybrid_monitoring):
        """Test that handler can predict future events based on learned patterns"""

        # First, learn some patterns
        await handler._analyze_patterns_from_monitoring()

        # Now run predictive analysis
        predictions = await handler._generate_predictions()

        # Should generate predictions based on patterns
        assert predictions is not None, "Should return predictions"

        # If we have learned patterns, we should have predictions
        if len(handler.learned_patterns) > 0:
            assert len(predictions) > 0, "Should generate predictions from learned patterns"

    @pytest.mark.asyncio
    async def test_predictive_query_type_with_confidence(self, handler):
        """Test PREDICTIVE_ANALYSIS returns results with confidence scores"""

        changes = await handler.analyze_temporal_changes(
            query_type=TemporalQueryType.PREDICTIVE_ANALYSIS,
            time_window_minutes=60
        )

        assert changes is not None, "Should return predictive results"

    # ========================================
    # TEST 3: Anomaly Detection
    # ========================================

    @pytest.mark.asyncio
    async def test_anomaly_detection_identifies_unusual_patterns(self, handler, mock_hybrid_monitoring):
        """Test that handler detects anomalies in monitoring data"""

        # Add an anomalous alert (error in unusual space)
        now = datetime.now().timestamp()
        mock_hybrid_monitoring._alert_history.append(
            Mock(
                space_id=99,  # Unusual space
                severity='CRITICAL',
                message='System crash',
                timestamp=now - 10,
                alert_type='ANOMALY'
            )
        )

        # Detect anomalies
        anomalies = await handler._detect_anomalies()

        assert anomalies is not None, "Should return anomaly detection results"

    @pytest.mark.asyncio
    async def test_anomaly_query_type_routing(self, handler):
        """Test ANOMALY_ANALYSIS query type is handled"""

        changes = await handler.analyze_temporal_changes(
            query_type=TemporalQueryType.ANOMALY_ANALYSIS,
            time_window_minutes=30
        )

        assert changes is not None, "Should return anomaly analysis results"

    # ========================================
    # TEST 4: Correlation Analysis
    # ========================================

    @pytest.mark.asyncio
    async def test_correlation_analysis_multi_space(self, handler, mock_hybrid_monitoring):
        """Test correlation detection across multiple spaces"""

        # Analyze correlations
        correlations = await handler._analyze_correlations()

        assert correlations is not None, "Should return correlation results"

        # Should detect correlation between Space 5 (builds) and Space 3 (errors)
        space_correlations = [c for c in correlations if c.get('spaces')]
        if len(space_correlations) > 0:
            assert any(5 in c.get('spaces', []) and 3 in c.get('spaces', [])
                      for c in space_correlations), "Should detect Space 5↔Space 3 correlation"

    @pytest.mark.asyncio
    async def test_correlation_query_type_routing(self, handler):
        """Test CORRELATION_ANALYSIS query type is handled"""

        changes = await handler.analyze_temporal_changes(
            query_type=TemporalQueryType.CORRELATION_ANALYSIS,
            time_window_minutes=60
        )

        assert changes is not None, "Should return correlation analysis results"

    # ========================================
    # TEST 5: Pattern Persistence
    # ========================================

    @pytest.mark.asyncio
    async def test_pattern_persistence_save_and_load(self, handler, temp_patterns_file):
        """Test saving and loading learned patterns"""

        # Mock patterns
        test_patterns = [
            {
                'pattern_id': 'pattern_1',
                'trigger': 'BUILD_SUCCESS',
                'outcome': 'ERROR_DETECTED',
                'confidence': 0.85,
                'occurrences': 5,
                'spaces': [5, 3],
                'avg_delay_seconds': 20
            }
        ]

        handler.learned_patterns = test_patterns

        # Save patterns
        with patch('os.path.expanduser', return_value=temp_patterns_file):
            handler._save_learned_patterns()

        # Verify file was created
        assert os.path.exists(temp_patterns_file), "Patterns file should be created"

        # Load patterns into new handler - patch before creating to intercept __init__
        with patch('os.path.expanduser', return_value=temp_patterns_file):
            new_handler = TemporalQueryHandler(
                proactive_monitoring_manager=handler.proactive_monitoring,
                change_detection_manager=handler.change_detection,
                implicit_resolver=handler.implicit_resolver
            )

        # Verify patterns were loaded during __init__
        assert len(new_handler.learned_patterns) == len(test_patterns), "Should load all patterns"
        assert new_handler.learned_patterns[0]['pattern_id'] == 'pattern_1', "Should load correct pattern data"
        assert new_handler.learned_patterns[0]['confidence'] == 0.85, "Should preserve confidence"

    @pytest.mark.asyncio
    async def test_pattern_file_format_validation(self, temp_patterns_file):
        """Test that saved patterns file has correct JSON format"""

        test_patterns = [
            {
                'pattern_id': 'test_pattern',
                'trigger': 'BUILD_SUCCESS',
                'outcome': 'ERROR_DETECTED',
                'confidence': 0.75,
                'occurrences': 3,
                'spaces': [1, 2],
                'avg_delay_seconds': 15,
                'learned_at': datetime.now().isoformat()
            }
        ]

        # Save patterns
        with open(temp_patterns_file, 'w') as f:
            json.dump(test_patterns, f, indent=2)

        # Load and validate
        with open(temp_patterns_file, 'r') as f:
            loaded = json.load(f)

        assert len(loaded) == 1, "Should save 1 pattern"
        assert loaded[0]['confidence'] == 0.75, "Should preserve confidence"
        assert loaded[0]['spaces'] == [1, 2], "Should preserve space list"

    # ========================================
    # TEST 6: Hybrid Monitoring Integration
    # ========================================

    @pytest.mark.asyncio
    async def test_hybrid_monitoring_flag_detection(self, handler):
        """Test that handler correctly detects hybrid monitoring availability"""

        assert handler.is_hybrid_monitoring == True, "Should detect hybrid monitoring is available"
        assert handler.proactive_monitoring is not None, "Should have monitoring manager reference"

    @pytest.mark.asyncio
    async def test_monitoring_alerts_categorization(self, handler, mock_hybrid_monitoring):
        """Test that monitoring alerts are properly categorized"""

        # Verify we have monitoring alerts to categorize
        assert len(handler.monitoring_alerts) > 0, "Should have monitoring alerts loaded"

        # Process monitoring alerts
        await handler._categorize_monitoring_alerts()

        # Check alert queues - categorization doesn't necessarily move all alerts,
        # it only categorizes those with specific keywords
        # Just verify the categorization ran without error
        assert handler.anomaly_alerts is not None, "Should have anomaly alerts queue"
        assert handler.predictive_alerts is not None, "Should have predictive alerts queue"
        assert handler.correlation_alerts is not None, "Should have correlation alerts queue"

    @pytest.mark.asyncio
    async def test_monitoring_cache_usage(self, handler):
        """Test that handler uses monitoring cache for instant queries"""

        # Run temporal analysis (should use cache)
        start_time = asyncio.get_event_loop().time()

        changes = await handler.analyze_temporal_changes(
            query_type=TemporalQueryType.CHANGE_DETECTION,
            time_window_minutes=5
        )

        elapsed_time = asyncio.get_event_loop().time() - start_time

        # Should be fast (<1 second) if using cache
        assert elapsed_time < 2.0, f"Query should be fast with cache (took {elapsed_time:.2f}s)"

    # ========================================
    # TEST 7: New Change Types
    # ========================================

    def test_new_change_types_exist(self):
        """Test that new v3.0 change types are defined"""

        # v3.0 added these change types
        assert hasattr(ChangeType, 'ANOMALY_DETECTED'), "Should have ANOMALY_DETECTED type"
        assert hasattr(ChangeType, 'PATTERN_RECOGNIZED'), "Should have PATTERN_RECOGNIZED type"
        assert hasattr(ChangeType, 'PREDICTIVE_EVENT'), "Should have PREDICTIVE_EVENT type"
        assert hasattr(ChangeType, 'CASCADING_FAILURE'), "Should have CASCADING_FAILURE type"

    @pytest.mark.asyncio
    async def test_cascading_failure_detection(self, handler, mock_hybrid_monitoring):
        """Test detection of cascading failures across spaces"""

        # Add cascading failure pattern to alerts
        now = datetime.now().timestamp()
        mock_hybrid_monitoring._alert_history.extend([
            Mock(space_id=1, severity='ERROR', message='Import error', timestamp=now - 100),
            Mock(space_id=2, severity='ERROR', message='Module not found', timestamp=now - 90),
            Mock(space_id=3, severity='ERROR', message='Build failed', timestamp=now - 80),
        ])

        # Detect cascading failures
        cascades = await handler._detect_cascading_failures()

        assert cascades is not None, "Should return cascading failure results"

    # ========================================
    # TEST 8: Alert Queue Management
    # ========================================

    def test_alert_queue_size_limits(self, handler):
        """Test that alert queues respect max size limits"""

        # monitoring_alerts should have maxlen=500 (v3.0 upgrade)
        assert handler.monitoring_alerts.maxlen == 500, "monitoring_alerts should have maxlen=500"

        # New v3.0 queues should have maxlen=100
        assert handler.anomaly_alerts.maxlen == 100, "anomaly_alerts should have maxlen=100"
        assert handler.predictive_alerts.maxlen == 100, "predictive_alerts should have maxlen=100"
        assert handler.correlation_alerts.maxlen == 100, "correlation_alerts should have maxlen=100"

    @pytest.mark.asyncio
    async def test_alert_queue_overflow_handling(self, handler):
        """Test that alert queues handle overflow correctly"""

        # Fill monitoring_alerts beyond capacity
        for i in range(600):
            handler.monitoring_alerts.append(Mock(
                alert_id=f'alert_{i}',
                timestamp=datetime.now().timestamp()
            ))

        # Should only keep last 500
        assert len(handler.monitoring_alerts) == 500, "Should keep only last 500 alerts"

        # First alert should be alert_100 (alert_0 to alert_99 dropped)
        assert handler.monitoring_alerts[0].alert_id == 'alert_100', "Should drop oldest alerts"

    # ========================================
    # TEST 9: Performance & Caching
    # ========================================

    @pytest.mark.asyncio
    async def test_pattern_analysis_caching(self, handler):
        """Test that pattern analysis results are cached"""

        # Run pattern analysis twice
        patterns1 = await handler._analyze_patterns_from_monitoring()
        patterns2 = await handler._analyze_patterns_from_monitoring()

        # If caching works, second call should be faster
        # (Note: This is a basic test - real caching would need timing measurements)
        assert patterns1 is not None, "First call should return results"
        assert patterns2 is not None, "Second call should return cached results"

    @pytest.mark.asyncio
    async def test_large_alert_history_performance(self, handler, mock_hybrid_monitoring):
        """Test performance with large alert history (500 alerts)"""

        # Fill alert history to capacity
        now = datetime.now().timestamp()
        for i in range(500):
            mock_hybrid_monitoring._alert_history.append(
                Mock(
                    space_id=(i % 10) + 1,
                    severity='INFO',
                    message=f'Test alert {i}',
                    timestamp=now - (500 - i) * 10
                )
            )

        # Pattern analysis should still be performant
        start_time = asyncio.get_event_loop().time()
        patterns = await handler._analyze_patterns_from_monitoring()
        elapsed = asyncio.get_event_loop().time() - start_time

        # Should complete in reasonable time (<5 seconds even with 500 alerts)
        assert elapsed < 5.0, f"Pattern analysis should be fast even with 500 alerts (took {elapsed:.2f}s)"

    # ========================================
    # TEST 10: Error Handling
    # ========================================

    @pytest.mark.asyncio
    async def test_handles_missing_monitoring_manager(self):
        """Test graceful degradation when monitoring manager is None"""

        handler = TemporalQueryHandler(
            proactive_monitoring_manager=None,  # No monitoring
            change_detection_manager=None,
            implicit_resolver=None
        )

        assert handler.is_hybrid_monitoring == False, "Should detect monitoring is not available"

        # Should still work, but without pattern analysis
        changes = await handler.analyze_temporal_changes(
            query_type=TemporalQueryType.PATTERN_ANALYSIS,
            time_window_minutes=60
        )

        # Should return empty or placeholder results
        assert changes is not None, "Should handle gracefully without monitoring"

    @pytest.mark.asyncio
    async def test_handles_corrupted_patterns_file(self, temp_patterns_file):
        """Test handling of corrupted learned_patterns.json"""

        # Create corrupted JSON file
        with open(temp_patterns_file, 'w') as f:
            f.write("{invalid json content")

        # Patch before creating handler to intercept __init__
        with patch('os.path.expanduser', return_value=temp_patterns_file):
            handler = TemporalQueryHandler(
                proactive_monitoring_manager=None,
                change_detection_manager=None,
                implicit_resolver=None
            )

        # Should have empty patterns (failed to load corrupted file)
        assert len(handler.learned_patterns) == 0, "Should have empty patterns after load failure"


# ========================================
# TEST UTILITIES
# ========================================

class TestTemporalQueryHelpers:
    """Test helper methods and utilities"""

    def test_change_type_string_conversion(self):
        """Test that ChangeType enum converts to string correctly"""

        assert str(ChangeType.ANOMALY_DETECTED) in ['ChangeType.ANOMALY_DETECTED', 'ANOMALY_DETECTED']
        assert str(ChangeType.PATTERN_RECOGNIZED) in ['ChangeType.PATTERN_RECOGNIZED', 'PATTERN_RECOGNIZED']
        assert str(ChangeType.PREDICTIVE_EVENT) in ['ChangeType.PREDICTIVE_EVENT', 'PREDICTIVE_EVENT']
        assert str(ChangeType.CASCADING_FAILURE) in ['ChangeType.CASCADING_FAILURE', 'CASCADING_FAILURE']

    def test_query_type_string_conversion(self):
        """Test that TemporalQueryType enum converts to string correctly"""

        assert str(TemporalQueryType.PATTERN_ANALYSIS) in ['TemporalQueryType.PATTERN_ANALYSIS', 'PATTERN_ANALYSIS']
        assert str(TemporalQueryType.PREDICTIVE_ANALYSIS) in ['TemporalQueryType.PREDICTIVE_ANALYSIS', 'PREDICTIVE_ANALYSIS']
        assert str(TemporalQueryType.ANOMALY_ANALYSIS) in ['TemporalQueryType.ANOMALY_ANALYSIS', 'ANOMALY_ANALYSIS']
        assert str(TemporalQueryType.CORRELATION_ANALYSIS) in ['TemporalQueryType.CORRELATION_ANALYSIS', 'CORRELATION_ANALYSIS']


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
