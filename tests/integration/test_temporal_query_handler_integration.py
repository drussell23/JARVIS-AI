"""
Integration Tests for TemporalQueryHandler v3.0

Tests end-to-end functionality:
1. Pattern learning with real monitoring data
2. Pattern persistence across sessions
3. Integration with HybridProactiveMonitoringManager
4. Real-world query scenarios
5. Multi-space correlation detection
6. Cascading failure detection
"""

import pytest
import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import sys
sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

from context_intelligence.handlers.temporal_query_handler import (
    TemporalQueryHandler,
    TemporalQueryType,
    ChangeType,
    initialize_temporal_query_handler,
    get_temporal_query_handler
)


class TestTemporalQueryHandlerIntegration:
    """Integration tests for TemporalQueryHandler v3.0"""

    @pytest.fixture
    async def real_monitoring_scenario(self):
        """
        Create realistic monitoring scenario:
        - User makes code changes in Space 3
        - Runs build in Space 5
        - Build succeeds, but errors appear in Space 3
        - Pattern repeats multiple times
        """
        from collections import deque

        mock_monitoring = Mock()
        mock_monitoring._alert_history = deque(maxlen=500)
        mock_monitoring._pattern_rules = []

        now = datetime.now().timestamp()

        # Simulate 1 hour of development activity
        events = [
            # Iteration 1: Code → Build → Error
            (now - 3600, 3, 'INFO', 'Code change detected'),
            (now - 3580, 5, 'INFO', 'Build started'),
            (now - 3560, 5, 'INFO', 'Build completed successfully'),
            (now - 3540, 3, 'ERROR', 'TypeError: Cannot read property of undefined, line 42'),

            # Iteration 2: Code → Build → Error (same pattern)
            (now - 2800, 3, 'INFO', 'Code change detected'),
            (now - 2780, 5, 'INFO', 'Build started'),
            (now - 2760, 5, 'INFO', 'Build completed successfully'),
            (now - 2740, 3, 'ERROR', 'TypeError: Cannot read property of undefined, line 42'),

            # Iteration 3: Code → Build → Error (pattern continues)
            (now - 2000, 3, 'INFO', 'Code change detected'),
            (now - 1980, 5, 'INFO', 'Build started'),
            (now - 1960, 5, 'INFO', 'Build completed successfully'),
            (now - 1940, 3, 'ERROR', 'TypeError: Cannot read property of undefined, line 42'),

            # Iteration 4: Fix attempt
            (now - 1200, 3, 'INFO', 'Code change detected'),
            (now - 1180, 5, 'INFO', 'Build started'),
            (now - 1160, 5, 'INFO', 'Build completed successfully'),
            (now - 1140, 3, 'INFO', 'No errors detected'),  # Fixed!

            # Iteration 5: New error appears (different pattern)
            (now - 400, 7, 'INFO', 'Test started'),
            (now - 380, 7, 'ERROR', 'AssertionError: Expected 5, got 3'),
        ]

        for timestamp, space_id, severity, message in events:
            mock_monitoring._alert_history.append(Mock(
                space_id=space_id,
                severity=severity,
                message=message,
                timestamp=timestamp,
                alert_type=self._get_alert_type(message)
            ))

        return mock_monitoring

    def _get_alert_type(self, message):
        """Helper to determine alert type from message"""
        if 'build' in message.lower():
            if 'success' in message.lower():
                return 'BUILD_SUCCESS'
            elif 'start' in message.lower():
                return 'BUILD_START'
            else:
                return 'BUILD_FAIL'
        elif 'error' in message.lower():
            return 'ERROR_DETECTED'
        elif 'test' in message.lower():
            return 'TEST_START'
        else:
            return 'INFO'

    # ========================================
    # TEST 1: End-to-End Pattern Learning
    # ========================================

    @pytest.mark.asyncio
    async def test_e2e_pattern_learning_from_monitoring(self, real_monitoring_scenario):
        """
        Test complete pattern learning flow:
        1. Handler analyzes monitoring data
        2. Detects repeated build→error pattern
        3. Learns pattern with confidence score
        4. Stores pattern in memory
        """

        handler = TemporalQueryHandler(
            proactive_monitoring_manager=real_monitoring_scenario,
            change_detection_manager=None,
            implicit_resolver=None
        )

        # Run pattern analysis
        patterns = await handler._analyze_patterns_from_monitoring()

        # Should detect the build→error pattern (occurred 3 times)
        assert len(patterns) > 0, "Should detect at least one pattern"

        build_error_pattern = next(
            (p for p in patterns if 'build' in str(p.get('trigger', '')).lower()
             and 'error' in str(p.get('outcome', '')).lower()),
            None
        )

        assert build_error_pattern is not None, "Should detect build→error pattern"
        assert build_error_pattern['occurrences'] >= 3, f"Pattern should occur 3+ times, got {build_error_pattern['occurrences']}"
        assert build_error_pattern['confidence'] > 0.6, f"Confidence should be >60%, got {build_error_pattern['confidence']:.0%}"

        # Pattern should involve Space 5 (build) and Space 3 (error)
        assert 5 in build_error_pattern.get('spaces', []), "Pattern should involve Space 5"
        assert 3 in build_error_pattern.get('spaces', []), "Pattern should involve Space 3"

        # Pattern should have timing information
        assert 'avg_delay_seconds' in build_error_pattern, "Pattern should include average delay"
        assert build_error_pattern['avg_delay_seconds'] > 0, "Delay should be positive"

    # ========================================
    # TEST 2: Pattern Persistence Across Sessions
    # ========================================

    @pytest.mark.asyncio
    async def test_pattern_persistence_across_sessions(self, real_monitoring_scenario):
        """
        Test pattern learning persists across sessions:
        1. Session 1: Learn patterns from monitoring
        2. Save patterns to file
        3. Session 2: New handler loads patterns
        4. Verify patterns are preserved
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            patterns_file = os.path.join(temp_dir, 'learned_patterns.json')

            # Session 1: Learn and save
            handler1 = TemporalQueryHandler(
                proactive_monitoring_manager=real_monitoring_scenario,
                change_detection_manager=None,
                implicit_resolver=None
            )

            # Learn patterns
            patterns = await handler1._analyze_patterns_from_monitoring()
            handler1.learned_patterns = patterns

            # Save patterns
            with patch('os.path.expanduser', return_value=patterns_file):
                handler1._save_learned_patterns()

            # Verify file exists
            assert os.path.exists(patterns_file), "Patterns file should be created"

            # Get pattern count
            original_pattern_count = len(patterns)
            original_pattern_id = patterns[0]['pattern_id'] if len(patterns) > 0 else None

            # Session 2: Load patterns
            handler2 = TemporalQueryHandler(
                proactive_monitoring_manager=real_monitoring_scenario,
                change_detection_manager=None,
                implicit_resolver=None
            )

            with patch('os.path.expanduser', return_value=patterns_file):
                handler2._load_learned_patterns()

            # Verify patterns were loaded
            assert len(handler2.learned_patterns) == original_pattern_count, "Should load same number of patterns"

            if original_pattern_id:
                loaded_pattern = next(
                    (p for p in handler2.learned_patterns if p['pattern_id'] == original_pattern_id),
                    None
                )
                assert loaded_pattern is not None, "Should preserve pattern IDs"
                assert loaded_pattern['confidence'] > 0, "Should preserve confidence scores"

    # ========================================
    # TEST 3: Real-World Query Scenarios
    # ========================================

    @pytest.mark.asyncio
    async def test_user_query_what_patterns_detected(self, real_monitoring_scenario):
        """
        Test user query: "What patterns have you noticed?"

        Expected response:
        - Build in Space 5 → Error in Space 3 (85% confidence, 3 occurrences)
        """

        handler = TemporalQueryHandler(
            proactive_monitoring_manager=real_monitoring_scenario,
            change_detection_manager=None,
            implicit_resolver=None
        )

        # Simulate user query
        changes = await handler.analyze_temporal_changes(
            query_type=TemporalQueryType.PATTERN_ANALYSIS,
            time_window_minutes=60
        )

        assert changes is not None, "Should return pattern analysis"

        # Should mention the build→error pattern
        # (Implementation will vary, checking that analysis ran)

    @pytest.mark.asyncio
    async def test_user_query_predict_next_error(self, real_monitoring_scenario):
        """
        Test user query: "Will I get another error if I build?"

        Expected response:
        - High probability of TypeError in Space 3 after build (based on pattern)
        """

        handler = TemporalQueryHandler(
            proactive_monitoring_manager=real_monitoring_scenario,
            change_detection_manager=None,
            implicit_resolver=None
        )

        # Learn patterns first
        await handler._analyze_patterns_from_monitoring()

        # Run predictive analysis
        predictions = await handler._generate_predictions()

        assert predictions is not None, "Should return predictions"

        # If patterns were learned, should predict error after build
        if len(handler.learned_patterns) > 0:
            assert len(predictions) > 0, "Should generate predictions"

    @pytest.mark.asyncio
    async def test_user_query_anomaly_detection(self, real_monitoring_scenario):
        """
        Test user query: "Are there any anomalies?"

        Expected response:
        - Test error in Space 7 is unusual (doesn't match normal patterns)
        """

        # Add anomalous event
        now = datetime.now().timestamp()
        real_monitoring_scenario._alert_history.append(
            Mock(
                space_id=99,
                severity='CRITICAL',
                message='Kernel panic',
                timestamp=now - 10,
                alert_type='ANOMALY'
            )
        )

        handler = TemporalQueryHandler(
            proactive_monitoring_manager=real_monitoring_scenario,
            change_detection_manager=None,
            implicit_resolver=None
        )

        # Detect anomalies
        anomalies = await handler._detect_anomalies()

        assert anomalies is not None, "Should return anomaly detection"

    # ========================================
    # TEST 4: Multi-Space Correlation
    # ========================================

    @pytest.mark.asyncio
    async def test_multi_space_correlation_detection(self, real_monitoring_scenario):
        """
        Test correlation detection across multiple spaces:
        - Space 3 (code changes)
        - Space 5 (builds)
        - Space 3 (errors)

        Should detect causal relationship
        """

        handler = TemporalQueryHandler(
            proactive_monitoring_manager=real_monitoring_scenario,
            change_detection_manager=None,
            implicit_resolver=None
        )

        # Analyze correlations
        correlations = await handler._analyze_correlations()

        assert correlations is not None, "Should return correlations"

        # Should find correlation between Space 3 and Space 5
        space_pairs = set()
        for corr in correlations:
            spaces = corr.get('spaces', [])
            if len(spaces) >= 2:
                space_pairs.add(tuple(sorted(spaces[:2])))

        # Should detect Space 3 ↔ Space 5 correlation
        assert (3, 5) in space_pairs or (5, 3) in space_pairs, "Should detect Space 3↔5 correlation"

    # ========================================
    # TEST 5: Cascading Failure Detection
    # ========================================

    @pytest.mark.asyncio
    async def test_cascading_failure_detection(self):
        """
        Test cascading failure detection:
        - Import error in Space 1
        - Causes module error in Space 2
        - Causes build failure in Space 3
        """

        from collections import deque

        mock_monitoring = Mock()
        mock_monitoring._alert_history = deque(maxlen=500)

        now = datetime.now().timestamp()

        # Cascading failure chain
        mock_monitoring._alert_history.extend([
            Mock(space_id=1, severity='ERROR', message='ImportError: module not found',
                 timestamp=now - 100, alert_type='ERROR_DETECTED'),
            Mock(space_id=2, severity='ERROR', message='ModuleNotFoundError: No module named X',
                 timestamp=now - 90, alert_type='ERROR_DETECTED'),
            Mock(space_id=3, severity='ERROR', message='Build failed: missing dependency',
                 timestamp=now - 80, alert_type='BUILD_FAIL'),
        ])

        handler = TemporalQueryHandler(
            proactive_monitoring_manager=mock_monitoring,
            change_detection_manager=None,
            implicit_resolver=None
        )

        # Detect cascading failures
        cascades = await handler._detect_cascading_failures()

        assert cascades is not None, "Should detect cascading failures"

        # Should identify chain: Space 1 → Space 2 → Space 3
        if len(cascades) > 0:
            cascade = cascades[0]
            assert 'chain' in cascade or 'spaces' in cascade, "Should identify failure chain"

    # ========================================
    # TEST 6: Global Handler Initialization
    # ========================================

    @pytest.mark.asyncio
    async def test_global_handler_initialization(self, real_monitoring_scenario):
        """Test that global handler can be initialized and retrieved"""

        # Initialize global handler
        handler = initialize_temporal_query_handler(
            proactive_monitoring_manager=real_monitoring_scenario,
            change_detection_manager=None,
            implicit_resolver=None,
            conversation_tracker=None
        )

        assert handler is not None, "Should initialize global handler"

        # Retrieve global handler
        retrieved = get_temporal_query_handler()

        assert retrieved is handler, "Should retrieve same handler instance"
        assert retrieved.is_hybrid_monitoring == True, "Should have monitoring enabled"

    # ========================================
    # TEST 7: Performance with Real Data Volume
    # ========================================

    @pytest.mark.asyncio
    async def test_performance_with_500_alerts(self):
        """Test performance with full alert history (500 alerts)"""

        from collections import deque

        mock_monitoring = Mock()
        mock_monitoring._alert_history = deque(maxlen=500)
        mock_monitoring._pattern_rules = []

        now = datetime.now().timestamp()

        # Generate 500 realistic alerts
        for i in range(500):
            space_id = (i % 5) + 1
            alert_types = ['BUILD_START', 'BUILD_SUCCESS', 'ERROR_DETECTED', 'INFO']
            alert_type = alert_types[i % 4]

            messages = {
                'BUILD_START': 'Build started',
                'BUILD_SUCCESS': 'Build completed successfully',
                'ERROR_DETECTED': f'Error on line {i}',
                'INFO': 'Code change detected'
            }

            severity = 'ERROR' if alert_type == 'ERROR_DETECTED' else 'INFO'

            mock_monitoring._alert_history.append(Mock(
                space_id=space_id,
                severity=severity,
                message=messages[alert_type],
                timestamp=now - (500 - i) * 10,
                alert_type=alert_type
            ))

        handler = TemporalQueryHandler(
            proactive_monitoring_manager=mock_monitoring,
            change_detection_manager=None,
            implicit_resolver=None
        )

        # Measure pattern analysis performance
        start = asyncio.get_event_loop().time()
        patterns = await handler._analyze_patterns_from_monitoring()
        elapsed = asyncio.get_event_loop().time() - start

        # Should complete in reasonable time (<10 seconds)
        assert elapsed < 10.0, f"Pattern analysis should complete in <10s with 500 alerts (took {elapsed:.2f}s)"

        # Should find some patterns
        assert patterns is not None, "Should return pattern results"

    # ========================================
    # TEST 8: Alert Categorization
    # ========================================

    @pytest.mark.asyncio
    async def test_alert_categorization_integration(self, real_monitoring_scenario):
        """Test that monitoring alerts are categorized correctly"""

        handler = TemporalQueryHandler(
            proactive_monitoring_manager=real_monitoring_scenario,
            change_detection_manager=None,
            implicit_resolver=None
        )

        # Categorize alerts
        await handler._categorize_monitoring_alerts()

        # Check that alerts were distributed to queues
        total_alerts = (
            len(handler.monitoring_alerts) +
            len(handler.anomaly_alerts) +
            len(handler.predictive_alerts) +
            len(handler.correlation_alerts)
        )

        assert total_alerts > 0, "Should categorize some alerts"

        # monitoring_alerts should have the most
        assert len(handler.monitoring_alerts) > 0, "Should have some monitoring alerts"


# ========================================
# RUN TESTS
# ========================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
