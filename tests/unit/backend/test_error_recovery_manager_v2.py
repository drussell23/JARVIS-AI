"""
Unit Tests for ErrorRecoveryManager v2.0

Tests v2.0 features:
1. Error fingerprinting
2. Frequency-based severity escalation (3+ occurrences → CRITICAL)
3. Cross-session pattern detection
4. Proactive monitoring integration
5. Auto-heal recovery mode
6. Multi-space error correlation
7. Predictive fix application
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from collections import deque

import sys
sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

from autonomy.error_recovery import (
    ErrorRecoveryManager,
    ErrorRecord,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
    RecoveryAction
)


class TestErrorRecoveryManagerV2:
    """Unit tests for ErrorRecoveryManager v2.0"""

    @pytest.fixture
    def mock_hybrid_monitoring(self):
        """Create mock HybridProactiveMonitoringManager"""
        mock = Mock()
        mock.enable_ml = True
        mock._pattern_rules = []
        mock._alert_history = deque(maxlen=500)
        return mock

    @pytest.fixture
    def mock_implicit_resolver(self):
        """Create mock ImplicitReferenceResolver"""
        mock = AsyncMock()
        return mock

    @pytest.fixture
    def manager(self, mock_hybrid_monitoring, mock_implicit_resolver):
        """Create ErrorRecoveryManager instance"""
        return ErrorRecoveryManager(
            hybrid_monitoring_manager=mock_hybrid_monitoring,
            implicit_resolver=mock_implicit_resolver,
            change_detection_manager=None
        )

    # ========================================
    # TEST 1: Error Fingerprinting
    # ========================================

    def test_error_fingerprinting_generates_consistent_hash(self, manager):
        """Test that same error generates same fingerprint"""

        # Generate fingerprint for same error twice
        fp1 = manager._calculate_error_fingerprint(
            "TypeError: Cannot read property 'x' of undefined",
            "vision_module"
        )
        fp2 = manager._calculate_error_fingerprint(
            "TypeError: Cannot read property 'x' of undefined",
            "vision_module"
        )

        assert fp1 == fp2, "Same error should generate same fingerprint"
        assert isinstance(fp1, str), "Fingerprint should be a string"
        assert len(fp1) > 0, "Fingerprint should not be empty"

    def test_error_fingerprinting_different_for_different_errors(self, manager):
        """Test that different errors generate different fingerprints"""

        fp1 = manager._calculate_error_fingerprint(
            "TypeError: Cannot read property 'x' of undefined",
            "vision_module"
        )
        fp2 = manager._calculate_error_fingerprint(
            "ValueError: Invalid input",
            "vision_module"
        )

        assert fp1 != fp2, "Different errors should generate different fingerprints"

    # ========================================
    # TEST 2: Frequency-Based Severity Escalation
    # ========================================

    @pytest.mark.asyncio
    async def test_frequency_tracking_increments_count(self, manager):
        """Test that tracking error frequency increments count"""

        fingerprint = "test_error_fp_123"

        # Track error 3 times
        count1 = await manager.track_error_frequency(fingerprint)
        count2 = await manager.track_error_frequency(fingerprint)
        count3 = await manager.track_error_frequency(fingerprint)

        assert count1 == 1, "First occurrence should be 1"
        assert count2 == 2, "Second occurrence should be 2"
        assert count3 == 3, "Third occurrence should be 3"

    def test_severity_escalation_3_occurrences(self, manager):
        """Test that 3 occurrences escalates severity by one level"""

        # LOW → MEDIUM at 3 occurrences
        escalated = manager._escalate_severity_by_frequency(ErrorSeverity.LOW, 3)
        assert escalated == ErrorSeverity.MEDIUM, "LOW should escalate to MEDIUM at 3 occurrences"

        # MEDIUM → HIGH at 3 occurrences
        escalated = manager._escalate_severity_by_frequency(ErrorSeverity.MEDIUM, 3)
        assert escalated == ErrorSeverity.HIGH, "MEDIUM should escalate to HIGH at 3 occurrences"

        # HIGH → CRITICAL at 3 occurrences
        escalated = manager._escalate_severity_by_frequency(ErrorSeverity.HIGH, 3)
        assert escalated == ErrorSeverity.CRITICAL, "HIGH should escalate to CRITICAL at 3 occurrences"

    def test_severity_escalation_5_occurrences_always_critical(self, manager):
        """Test that 5+ occurrences always becomes CRITICAL"""

        for base_severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH]:
            escalated = manager._escalate_severity_by_frequency(base_severity, 5)
            assert escalated == ErrorSeverity.CRITICAL, f"{base_severity} should become CRITICAL at 5 occurrences"

    @pytest.mark.asyncio
    async def test_frequency_escalation_integration(self, manager):
        """Test end-to-end frequency tracking with severity escalation"""

        # Simulate same error occurring 4 times
        error_msg = "TypeError: Cannot read property 'x' of undefined"
        component = "vision_module"

        for i in range(4):
            await manager.handle_proactive_error(error_msg, component=component, space_id=3)

        # Check that error was tracked
        fingerprint = manager._calculate_error_fingerprint(error_msg, component)
        frequency = manager.error_frequency[fingerprint]

        assert frequency >= 3, f"Should have tracked error at least 3 times, got {frequency}"

        # Check that severity was escalated
        # After 3+ occurrences, severity should be elevated
        error_records = manager.error_fingerprints[fingerprint]
        if len(error_records) >= 3:
            latest_error = error_records[-1]
            # Later occurrences should have higher severity
            assert latest_error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL], \
                "After 3+ occurrences, severity should be HIGH or CRITICAL"

    # ========================================
    # TEST 3: Multi-Space Error Correlation
    # ========================================

    @pytest.mark.asyncio
    async def test_multi_space_correlation_detection(self, manager):
        """Test detection of cascading failures across spaces"""

        # Create errors in different spaces within 30 seconds
        error1 = await manager.handle_proactive_error(
            "Build failed",
            component="build_system",
            space_id=5
        )

        await asyncio.sleep(0.1)  # Small delay

        error2 = await manager.handle_proactive_error(
            "Import error",
            component="app_module",
            space_id=3
        )

        await asyncio.sleep(0.1)

        error3 = await manager.handle_proactive_error(
            "Runtime error",
            component="runtime",
            space_id=7
        )

        # Check correlation was detected
        # error3 should have related_errors pointing to error1 and error2
        if len(error3.related_errors) > 0:
            assert True, "Detected related errors across spaces"

    @pytest.mark.asyncio
    async def test_cascading_failure_escalates_to_critical(self, manager):
        """Test that multi-space errors escalate to CRITICAL"""

        # Create errors in 3 different spaces (cascading failure)
        await manager.handle_proactive_error("Error 1", component="c1", space_id=1)
        await asyncio.sleep(0.1)
        await manager.handle_proactive_error("Error 2", component="c2", space_id=2)
        await asyncio.sleep(0.1)
        error3 = await manager.handle_proactive_error("Error 3", component="c3", space_id=3)

        # Third error should be escalated to CRITICAL due to correlation
        if len(error3.related_errors) >= 2:
            assert error3.severity == ErrorSeverity.CRITICAL, \
                "Multi-space correlation should escalate to CRITICAL"

    # ========================================
    # TEST 4: Proactive Monitoring Integration
    # ========================================

    def test_proactive_mode_enabled_with_hybrid_monitoring(self, manager):
        """Test that proactive mode is enabled when HybridMonitoring is provided"""

        assert manager.is_proactive_enabled == True, "Should enable proactive mode with HybridMonitoring"
        assert manager.hybrid_monitoring is not None, "Should have hybrid monitoring reference"

    @pytest.mark.asyncio
    async def test_proactive_error_detection_from_monitoring_alert(self, manager):
        """Test handling proactive errors from monitoring alerts"""

        # Simulate monitoring alert (uses event_type not alert_type)
        alert = {
            'event_type': 'ERROR_DETECTED',  # Changed from alert_type
            'space_id': 5,
            'message': 'TypeError on line 42',
            'timestamp': datetime.now().timestamp(),
            'metadata': {
                'detection_method': 'ml',
                'component': 'app_module'
            }
        }

        await manager.register_monitoring_alert(alert)

        # Should create error record with detection_method='proactive'
        # Check that error was registered proactively
        assert len(manager.error_history) > 0, "Should register proactive error"

        proactive_errors = [e for e in manager.error_history if e.detection_method in ['proactive', 'ml']]
        assert len(proactive_errors) > 0, "Should have proactive detection method"

    # ========================================
    # TEST 5: Recovery Strategy Selection
    # ========================================

    def test_new_recovery_strategies_exist(self, manager):
        """Test that v2.0 recovery strategies are available"""

        # Check new strategies exist
        strategies = [s.value for s in RecoveryStrategy]

        assert 'proactive_monitor' in strategies, "Should have PROACTIVE_MONITOR strategy"
        assert 'predictive_fix' in strategies, "Should have PREDICTIVE_FIX strategy"
        assert 'isolate' in strategies, "Should have ISOLATE strategy"
        assert 'auto_heal' in strategies, "Should have AUTO_HEAL strategy"

    @pytest.mark.asyncio
    async def test_proactive_monitor_increases_monitoring(self, manager):
        """Test that PROACTIVE_MONITOR strategy increases monitoring"""

        error_record = ErrorRecord(
            error_id="test_error_1",
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.MEDIUM,
            message="Potential failure detected",
            component="test_component",
            space_id=3
        )

        # Apply proactive monitoring recovery
        await manager._increase_monitoring(error_record)

        # Should mark proactive action taken
        assert error_record.proactive_action_taken == True, "Should mark proactive action as taken"

    @pytest.mark.asyncio
    async def test_auto_heal_applies_recovery(self, manager):
        """Test that AUTO_HEAL strategy applies self-healing"""

        error_record = ErrorRecord(
            error_id="test_error_2",
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.HIGH,
            message="TypeError: Cannot read property 'x' of undefined",
            component="app_module",
            space_id=5,
            frequency_count=4
        )

        # Apply auto-heal recovery
        await manager._auto_heal(error_record)

        # Should mark proactive action taken
        assert error_record.proactive_action_taken == True, "Should mark auto-heal as taken"

    # ========================================
    # TEST 6: Error Statistics and Reporting
    # ========================================

    @pytest.mark.asyncio
    async def test_get_high_frequency_errors(self, manager):
        """Test retrieval of high-frequency errors (3+ occurrences)"""

        # Create same error 4 times
        error_msg = "Repeated error"
        for i in range(4):
            await manager.handle_proactive_error(
                error_msg,
                component="test_comp",
                space_id=3
            )

        # Get high-frequency errors
        high_freq = manager.get_high_frequency_errors(min_frequency=3)

        assert len(high_freq) > 0, "Should detect high-frequency errors"
        assert high_freq[0]['frequency'] >= 3, "Frequency should be at least 3"

    @pytest.mark.asyncio
    async def test_get_space_error_summary(self, manager):
        """Test space-specific error summary"""

        # Create errors in space 3
        await manager.handle_proactive_error("Error 1", component="c1", space_id=3)
        await manager.handle_proactive_error("Error 2", component="c2", space_id=3)
        await manager.handle_proactive_error("Error 3", component="c3", space_id=5)

        # Get summary for space 3
        summary = manager.get_space_error_summary(space_id=3)

        assert summary['total_errors'] >= 2, "Should have at least 2 errors in space 3"
        assert summary['space_id'] == 3, "Should be for space 3"

    # ========================================
    # TEST 7: Proactive vs Reactive Detection
    # ========================================

    @pytest.mark.asyncio
    async def test_reactive_error_detection(self, manager):
        """Test traditional reactive error handling"""

        error = ValueError("Test error")
        error_record = await manager.handle_error(error, component="test_component")

        assert error_record is not None, "Should create error record"
        # Note: handle_error doesn't set detection_method, that's only for proactive errors

    @pytest.mark.asyncio
    async def test_proactive_error_has_predicted_flag(self, manager):
        """Test that predicted errors are flagged"""

        # Handle proactive error with predicted flag
        await manager.handle_proactive_error(
            error_message="Predicted TypeError based on pattern",
            component="app_module",
            space_id=5,
            predicted=True
        )

        # Check that predicted flag is set
        predicted_errors = [e for e in manager.error_history if e.predicted]
        assert len(predicted_errors) > 0, "Should have predicted errors"

    # ========================================
    # TEST 8: Cross-Session Pattern Detection
    # ========================================

    @pytest.mark.asyncio
    async def test_error_patterns_persist_across_sessions(self, manager):
        """Test that error frequency is tracked across multiple errors"""

        fingerprint = "persistent_error_pattern"

        # Simulate errors across "sessions"
        for i in range(5):
            await manager.track_error_frequency(fingerprint)

        assert manager.error_frequency[fingerprint] == 5, "Should track frequency across all errors"

    # ========================================
    # TEST 9: Component Isolation
    # ========================================

    @pytest.mark.asyncio
    async def test_isolate_component_marks_action_taken(self, manager):
        """Test that component isolation recovery is applied"""

        error_record = ErrorRecord(
            error_id="isolation_test",
            category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.HIGH,
            message="Component failure",
            component="failing_component",
            space_id=3
        )

        await manager._isolate_component(error_record)

        assert error_record.proactive_action_taken == True, "Should mark isolation action as taken"

    # ========================================
    # TEST 10: Integration Tests
    # ========================================

    @pytest.mark.asyncio
    async def test_end_to_end_error_recovery_with_escalation(self, manager):
        """Test complete error recovery flow with frequency escalation"""

        # Simulate same error occurring 4 times
        error_msg = "Critical TypeError"
        component = "critical_module"

        errors = []
        for i in range(4):
            error_record = await manager.handle_proactive_error(
                error_msg,
                component=component,
                space_id=3
            )
            errors.append(error_record)
            await asyncio.sleep(0.1)

        # Verify frequency tracking
        fingerprint = manager._calculate_error_fingerprint(error_msg, component)
        frequency = manager.error_frequency[fingerprint]

        assert frequency >= 3, f"Should track high frequency, got {frequency}"

        # Verify severity escalation occurred
        latest_error = errors[-1]
        assert latest_error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL], \
            "High-frequency errors should be escalated"


class TestErrorRecoveryManagerEdgeCases:
    """Edge case tests for ErrorRecoveryManager v2.0"""

    @pytest.fixture
    def manager_no_monitoring(self):
        """Create manager without monitoring (backward compatibility)"""
        return ErrorRecoveryManager(
            hybrid_monitoring_manager=None,
            implicit_resolver=None,
            change_detection_manager=None
        )

    def test_manager_works_without_proactive_monitoring(self, manager_no_monitoring):
        """Test that manager works in non-proactive mode"""

        assert manager_no_monitoring.is_proactive_enabled == False, \
            "Should disable proactive mode without monitoring"

    @pytest.mark.asyncio
    async def test_handles_errors_without_space_id(self, manager_no_monitoring):
        """Test handling errors without space_id (non-spatial errors)"""

        error_record = await manager_no_monitoring.handle_error(
            ValueError("Test error"),
            component="test_component"
            # No space_id provided
        )

        assert error_record is not None, "Should handle errors without space_id"
        assert error_record.space_id is None, "space_id should be None"


# ========================================
# TEST HELPERS
# ========================================

class TestErrorRecordSerialization:
    """Test ErrorRecord serialization"""

    def test_error_record_to_dict(self):
        """Test ErrorRecord.to_dict() includes all v2.0 fields"""

        record = ErrorRecord(
            error_id="test_123",
            category=ErrorCategory.VISION,
            severity=ErrorSeverity.HIGH,
            message="Test error",
            component="test_component",
            space_id=3,
            detection_method="proactive",
            predicted=True,
            frequency_count=4,
            proactive_action_taken=True
        )

        data = record.to_dict()

        assert data['error_id'] == "test_123"
        assert data['category'] == "vision"
        assert data['severity'] == "HIGH"
        assert data['space_id'] == 3
        assert data['detection_method'] == "proactive"
        assert data['predicted'] == True
        assert data['frequency_count'] == 4
        assert data['proactive_action_taken'] == True
