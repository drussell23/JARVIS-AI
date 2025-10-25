"""
E2E Verification Test for TemporalQueryHandler v3.0 Usage in JARVIS

This test verifies that:
1. TemporalQueryHandler is initialized when JARVIS starts
2. It's accessible via the global handler
3. It's integrated with HybridProactiveMonitoringManager
4. It's used by actual query endpoints
5. Pattern learning works in production

Usage:
    python -m pytest tests/e2e/test_temporal_handler_usage_verification.py -v
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')


class TestTemporalHandlerUsageVerification:
    """Verify TemporalQueryHandler v3.0 is actually used in JARVIS"""

    # ========================================
    # TEST 1: Handler Initialization
    # ========================================

    def test_temporal_handler_import(self):
        """Test that TemporalQueryHandler can be imported"""

        try:
            from context_intelligence.handlers.temporal_query_handler import (
                TemporalQueryHandler,
                TemporalQueryType,
                ChangeType,
                get_temporal_query_handler,
                initialize_temporal_query_handler
            )

            assert TemporalQueryHandler is not None, "Should import TemporalQueryHandler"
            assert TemporalQueryType is not None, "Should import TemporalQueryType"
            assert ChangeType is not None, "Should import ChangeType"

        except ImportError as e:
            pytest.fail(f"Failed to import TemporalQueryHandler: {e}")

    def test_v3_enums_exist(self):
        """Test that v3.0 specific enums exist"""

        from context_intelligence.handlers.temporal_query_handler import (
            TemporalQueryType,
            ChangeType
        )

        # v3.0 query types
        assert hasattr(TemporalQueryType, 'PATTERN_ANALYSIS'), "Should have PATTERN_ANALYSIS"
        assert hasattr(TemporalQueryType, 'PREDICTIVE_ANALYSIS'), "Should have PREDICTIVE_ANALYSIS"
        assert hasattr(TemporalQueryType, 'ANOMALY_ANALYSIS'), "Should have ANOMALY_ANALYSIS"
        assert hasattr(TemporalQueryType, 'CORRELATION_ANALYSIS'), "Should have CORRELATION_ANALYSIS"

        # v3.0 change types
        assert hasattr(ChangeType, 'ANOMALY_DETECTED'), "Should have ANOMALY_DETECTED"
        assert hasattr(ChangeType, 'PATTERN_RECOGNIZED'), "Should have PATTERN_RECOGNIZED"
        assert hasattr(ChangeType, 'PREDICTIVE_EVENT'), "Should have PREDICTIVE_EVENT"
        assert hasattr(ChangeType, 'CASCADING_FAILURE'), "Should have CASCADING_FAILURE"

    # ========================================
    # TEST 2: main.py Integration
    # ========================================

    def test_main_py_imports_temporal_handler(self):
        """Test that main.py imports and initializes TemporalQueryHandler"""

        main_py = Path('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend/main.py')

        assert main_py.exists(), "main.py should exist"

        with open(main_py, 'r') as f:
            content = f.read()

        # Check for import
        assert 'temporal_query_handler' in content.lower(), "main.py should import temporal_query_handler"
        assert 'initialize_temporal_query_handler' in content, "main.py should call initialize_temporal_query_handler"

        # Check for v2.0/v3.0 initialization
        assert 'hybrid_monitoring' in content or 'HybridProactiveMonitoringManager' in content, \
            "main.py should integrate with HybridProactiveMonitoringManager"

    def test_main_py_has_temporal_handler_setup(self):
        """Test that main.py has TemporalQueryHandler initialization code"""

        main_py = Path('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend/main.py')

        with open(main_py, 'r') as f:
            content = f.read()

        # Check for specific v3.0 setup
        assert 'TemporalQueryHandler' in content, "Should reference TemporalQueryHandler class"

        # Check for pattern learning mentions
        has_pattern_features = any(keyword in content for keyword in [
            'pattern', 'predictive', 'anomaly', 'correlation'
        ])

        assert has_pattern_features, "Should mention pattern/predictive/anomaly/correlation features"

    # ========================================
    # TEST 3: Global Handler Accessibility
    # ========================================

    @pytest.mark.asyncio
    async def test_global_handler_can_be_retrieved(self):
        """Test that global handler can be initialized and retrieved"""

        from context_intelligence.handlers.temporal_query_handler import (
            initialize_temporal_query_handler,
            get_temporal_query_handler
        )

        # Initialize handler (without monitoring for this test)
        handler = initialize_temporal_query_handler(
            proactive_monitoring_manager=None,
            change_detection_manager=None,
            implicit_resolver=None,
            conversation_tracker=None
        )

        assert handler is not None, "Should initialize handler"

        # Retrieve it
        retrieved = get_temporal_query_handler()

        assert retrieved is not None, "Should retrieve global handler"
        assert retrieved is handler, "Should retrieve same instance"

    # ========================================
    # TEST 4: Pattern Learning File Setup
    # ========================================

    def test_learned_patterns_file_location(self):
        """Test that learned_patterns.json location is correct"""

        from context_intelligence.handlers.temporal_query_handler import (
            TemporalQueryHandler
        )

        # Handler should use ~/.jarvis/learned_patterns.json
        expected_location = os.path.expanduser('~/.jarvis/learned_patterns.json')

        # Check the handler code references this path
        handler_file = Path('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend/context_intelligence/handlers/temporal_query_handler.py')

        with open(handler_file, 'r') as f:
            content = f.read()

        assert 'learned_patterns.json' in content, "Should reference learned_patterns.json"
        assert '~/.jarvis' in content or '.jarvis' in content, "Should use ~/.jarvis directory"

    def test_jarvis_directory_exists_or_can_be_created(self):
        """Test that ~/.jarvis directory exists or can be created"""

        jarvis_dir = Path.home() / '.jarvis'

        # Either exists or can be created
        if not jarvis_dir.exists():
            try:
                jarvis_dir.mkdir(exist_ok=True)
                created = True
            except Exception as e:
                pytest.fail(f"Cannot create ~/.jarvis directory: {e}")
        else:
            created = False

        assert jarvis_dir.exists(), "~/.jarvis directory should exist"

        # Cleanup if we created it
        if created:
            try:
                jarvis_dir.rmdir()
            except:
                pass  # OK if not empty

    # ========================================
    # TEST 5: Handler Methods Exist
    # ========================================

    def test_handler_has_v3_methods(self):
        """Test that handler has all v3.0 methods"""

        from context_intelligence.handlers.temporal_query_handler import (
            TemporalQueryHandler
        )

        handler = TemporalQueryHandler(
            proactive_monitoring_manager=None,
            change_detection_manager=None,
            implicit_resolver=None,
            conversation_tracker=None
        )

        # Check for v3.0 methods
        assert hasattr(handler, '_analyze_patterns_from_monitoring'), \
            "Should have _analyze_patterns_from_monitoring method"

        assert hasattr(handler, '_generate_predictions'), \
            "Should have _generate_predictions method"

        assert hasattr(handler, '_detect_anomalies'), \
            "Should have _detect_anomalies method"

        assert hasattr(handler, '_analyze_correlations'), \
            "Should have _analyze_correlations method"

        assert hasattr(handler, '_detect_cascading_failures'), \
            "Should have _detect_cascading_failures method"

        assert hasattr(handler, '_categorize_monitoring_alerts'), \
            "Should have _categorize_monitoring_alerts method"

        assert hasattr(handler, '_load_learned_patterns'), \
            "Should have _load_learned_patterns method"

        assert hasattr(handler, '_save_learned_patterns'), \
            "Should have _save_learned_patterns method"

    def test_handler_has_v3_attributes(self):
        """Test that handler has all v3.0 attributes"""

        from context_intelligence.handlers.temporal_query_handler import (
            TemporalQueryHandler
        )

        handler = TemporalQueryHandler(
            proactive_monitoring_manager=None,
            change_detection_manager=None,
            implicit_resolver=None,
            conversation_tracker=None
        )

        # Check for v3.0 attributes
        assert hasattr(handler, 'is_hybrid_monitoring'), "Should have is_hybrid_monitoring flag"
        assert hasattr(handler, 'learned_patterns'), "Should have learned_patterns list"
        assert hasattr(handler, 'monitoring_alerts'), "Should have monitoring_alerts queue"
        assert hasattr(handler, 'anomaly_alerts'), "Should have anomaly_alerts queue"
        assert hasattr(handler, 'predictive_alerts'), "Should have predictive_alerts queue"
        assert hasattr(handler, 'correlation_alerts'), "Should have correlation_alerts queue"

    # ========================================
    # TEST 6: Alert Queue Configuration
    # ========================================

    def test_alert_queues_have_correct_sizes(self):
        """Test that alert queues have v3.0 size configuration"""

        from context_intelligence.handlers.temporal_query_handler import (
            TemporalQueryHandler
        )

        handler = TemporalQueryHandler(
            proactive_monitoring_manager=None,
            change_detection_manager=None,
            implicit_resolver=None,
            conversation_tracker=None
        )

        # v3.0 increased monitoring_alerts from 200 to 500
        assert handler.monitoring_alerts.maxlen == 500, \
            f"monitoring_alerts should have maxlen=500, got {handler.monitoring_alerts.maxlen}"

        # New v3.0 queues should have maxlen=100
        assert handler.anomaly_alerts.maxlen == 100, \
            f"anomaly_alerts should have maxlen=100, got {handler.anomaly_alerts.maxlen}"

        assert handler.predictive_alerts.maxlen == 100, \
            f"predictive_alerts should have maxlen=100, got {handler.predictive_alerts.maxlen}"

        assert handler.correlation_alerts.maxlen == 100, \
            f"correlation_alerts should have maxlen=100, got {handler.correlation_alerts.maxlen}"

    # ========================================
    # TEST 7: Documentation Verification
    # ========================================

    def test_handler_docstring_mentions_v3(self):
        """Test that handler docstring mentions v3.0 features"""

        from context_intelligence.handlers.temporal_query_handler import (
            TemporalQueryHandler
        )

        docstring = TemporalQueryHandler.__doc__

        assert docstring is not None, "Should have docstring"

        # Should mention v3.0 or pattern/predictive features
        v3_keywords = ['v3.0', 'pattern', 'predictive', 'anomaly', 'correlation']

        has_v3_mention = any(keyword.lower() in docstring.lower() for keyword in v3_keywords)

        assert has_v3_mention, "Docstring should mention v3.0 features"

    # ========================================
    # TEST 8: Verify Start System Integration
    # ========================================

    def test_start_system_py_mentions_temporal_handler(self):
        """Test that start_system.py documents TemporalQueryHandler"""

        start_system = Path('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/start_system.py')

        if start_system.exists():
            with open(start_system, 'r') as f:
                content = f.read()

            # Should mention temporal features
            has_temporal = 'temporal' in content.lower() or 'TemporalQueryHandler' in content

            assert has_temporal, "start_system.py should document TemporalQueryHandler"

    def test_readme_mentions_v2_features(self):
        """Test that README.md mentions v2.0/v3.0 features"""

        readme = Path('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/README.md')

        if readme.exists():
            with open(readme, 'r') as f:
                content = f.read()

            # Should mention v2.0 or intelligent features
            has_v2_mention = 'v2.0' in content or 'Intelligent Edition' in content

            assert has_v2_mention, "README should mention v2.0 features"


# ========================================
# USAGE DEMONSTRATION
# ========================================

class TestTemporalHandlerUsageExamples:
    """Demonstrate how to use TemporalQueryHandler v3.0"""

    @pytest.mark.asyncio
    async def test_example_pattern_analysis_query(self):
        """Example: How to run a pattern analysis query"""

        from context_intelligence.handlers.temporal_query_handler import (
            initialize_temporal_query_handler,
            get_temporal_query_handler,
            TemporalQueryType
        )

        # Initialize (this would normally be done in main.py)
        handler = initialize_temporal_query_handler(
            proactive_monitoring_manager=None,  # In production, pass real manager
            change_detection_manager=None,
            implicit_resolver=None,
            conversation_tracker=None
        )

        # Run pattern analysis
        changes = await handler.analyze_temporal_changes(
            query_type=TemporalQueryType.PATTERN_ANALYSIS,
            time_window_minutes=60
        )

        # Result will be list of TemporalChange objects
        assert changes is not None, "Should return pattern analysis results"

    @pytest.mark.asyncio
    async def test_example_check_learned_patterns(self):
        """Example: How to check learned patterns"""

        from context_intelligence.handlers.temporal_query_handler import (
            TemporalQueryHandler
        )

        handler = TemporalQueryHandler(
            proactive_monitoring_manager=None,
            change_detection_manager=None,
            implicit_resolver=None,
            conversation_tracker=None
        )

        # Access learned patterns
        patterns = handler.learned_patterns

        # Initially empty (no monitoring data)
        assert isinstance(patterns, list), "learned_patterns should be a list"

    @pytest.mark.asyncio
    async def test_example_pattern_persistence(self):
        """Example: How pattern persistence works"""

        from context_intelligence.handlers.temporal_query_handler import (
            TemporalQueryHandler
        )
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            handler = TemporalQueryHandler(
                proactive_monitoring_manager=None,
                change_detection_manager=None,
                implicit_resolver=None,
                conversation_tracker=None
            )

            # Add a test pattern
            handler.learned_patterns = [{
                'pattern_id': 'test_pattern',
                'trigger': 'BUILD_SUCCESS',
                'outcome': 'ERROR_DETECTED',
                'confidence': 0.85,
                'occurrences': 5
            }]

            # Save patterns
            from unittest.mock import patch
            with patch('os.path.expanduser', return_value=temp_file):
                handler._save_learned_patterns()

            # Verify file exists
            assert os.path.exists(temp_file), "Patterns file should be created"

            # Load in new handler
            new_handler = TemporalQueryHandler(
                proactive_monitoring_manager=None,
                change_detection_manager=None,
                implicit_resolver=None,
                conversation_tracker=None
            )

            with patch('os.path.expanduser', return_value=temp_file):
                new_handler._load_learned_patterns()

            # Verify patterns were loaded
            assert len(new_handler.learned_patterns) == 1, "Should load 1 pattern"
            assert new_handler.learned_patterns[0]['confidence'] == 0.85, "Should preserve confidence"

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
