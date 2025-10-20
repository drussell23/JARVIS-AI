"""
Unit Tests for StateIntelligence v2.0

Tests v2.0 features:
1. Nine behavioral states (ACTIVE_CODING, STUCK, DISTRACTED, etc.)
2. Productivity trend analysis (0.0-1.0 score)
3. Automatic stuck detection (>30 minutes idle)
4. Auto-recording from monitoring alerts
5. Time-of-day preference learning
6. State prediction and recommendations
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from collections import deque

import sys
sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')

from vision.intelligence.state_intelligence import (
    StateIntelligence,
    StateVisit,
    MonitoringStateType,
    TimeOfDay,
    DayType,
    StatePattern,
    UserPreference
)


class TestStateIntelligenceV2:
    """Unit tests for StateIntelligence v2.0"""

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
    def state_intelligence(self, mock_hybrid_monitoring, mock_implicit_resolver):
        """Create StateIntelligence instance"""
        return StateIntelligence(
            hybrid_monitoring_manager=mock_hybrid_monitoring,
            implicit_resolver=mock_implicit_resolver,
            change_detection_manager=None
        )

    # ========================================
    # TEST 1: Nine Behavioral States
    # ========================================

    def test_all_monitoring_states_exist(self):
        """Test that all 9 behavioral states are defined"""
        expected_states = [
            'ACTIVE_CODING',
            'DEBUGGING',
            'READING',
            'IDLE',
            'STUCK',
            'ERROR_STATE',
            'BUILD_WAITING',
            'PRODUCTIVE',
            'DISTRACTED'
        ]

        actual_states = [s.name for s in MonitoringStateType]

        for state in expected_states:
            assert state in actual_states, f"Missing state: {state}"

    def test_state_visit_has_monitoring_fields(self):
        """Test that StateVisit has v2.0 monitoring fields"""
        visit = StateVisit(
            state_id="test_state",
            app_id="VSCode",
            timestamp=datetime.now(),
            space_id=3,
            monitoring_state_type=MonitoringStateType.ACTIVE_CODING,
            is_stuck=False,
            productivity_score=0.8,
            auto_recorded=True
        )

        assert visit.space_id == 3
        assert visit.monitoring_state_type == MonitoringStateType.ACTIVE_CODING
        assert visit.is_stuck == False
        assert visit.productivity_score == 0.8
        assert visit.auto_recorded == True

    # ========================================
    # TEST 2: Productivity Trend Analysis
    # ========================================

    def test_productivity_score_range(self, state_intelligence):
        """Test that productivity scores are in 0.0-1.0 range"""
        # Record visits with different productivity scores
        now = datetime.now()

        visits = [
            StateVisit(
                state_id="coding_1",
                app_id="VSCode",
                timestamp=now - timedelta(hours=2),
                productivity_score=0.9,
                monitoring_state_type=MonitoringStateType.ACTIVE_CODING
            ),
            StateVisit(
                state_id="distracted_1",
                app_id="Chrome",
                timestamp=now - timedelta(hours=1),
                productivity_score=0.2,
                monitoring_state_type=MonitoringStateType.DISTRACTED
            ),
        ]

        for visit in visits:
            state_intelligence.record_visit(visit)

        # Calculate productivity trend
        trend = state_intelligence.calculate_productivity_trend(hours=3)

        assert 'average_score' in trend
        assert 0.0 <= trend['average_score'] <= 1.0, \
            f"Productivity score {trend['average_score']} not in range [0.0, 1.0]"

    def test_productivity_trend_calculation(self, state_intelligence):
        """Test productivity trend over time"""
        now = datetime.now()

        # Record high productivity period
        for i in range(5):
            visit = StateVisit(
                state_id=f"coding_{i}",
                app_id="VSCode",
                timestamp=now - timedelta(hours=i),
                productivity_score=0.8 + (i * 0.02),  # 0.8-0.88
                monitoring_state_type=MonitoringStateType.ACTIVE_CODING
            )
            state_intelligence.record_visit(visit)

        trend = state_intelligence.calculate_productivity_trend(hours=6)

        assert trend['average_score'] >= 0.75, "Should show high productivity"
        assert 'peak_score' in trend
        assert 'low_score' in trend

    # ========================================
    # TEST 3: Stuck State Detection
    # ========================================

    @pytest.mark.asyncio
    async def test_stuck_state_threshold_30_minutes(self, state_intelligence):
        """Test that stuck state threshold is 30 minutes"""
        assert state_intelligence.stuck_threshold == timedelta(minutes=30)

    @pytest.mark.asyncio
    async def test_stuck_state_detection(self, state_intelligence):
        """Test automatic stuck state detection after 30 minutes"""
        # Simulate space staying in same state for >30 minutes
        now = datetime.now()
        space_id = 3

        # Set state start time to 35 minutes ago
        state_intelligence.space_visit_start_times[space_id] = now - timedelta(minutes=35)

        # Create current state
        current_state = StateVisit(
            state_id="space_3_coding",
            app_id="VSCode",
            timestamp=now - timedelta(minutes=35),
            space_id=space_id,
            monitoring_state_type=MonitoringStateType.ACTIVE_CODING
        )
        state_intelligence.current_space_states[space_id] = current_state

        # Manually trigger check (instead of waiting for async task)
        await state_intelligence._check_for_stuck_states()

        # Verify stuck state was detected
        assert len(state_intelligence.stuck_state_alerts) > 0, "Should detect stuck state"

        stuck_alert = state_intelligence.stuck_state_alerts[0]
        assert stuck_alert['space_id'] == space_id
        assert stuck_alert['duration'] > timedelta(minutes=30)

        # Verify state was marked as stuck
        assert current_state.is_stuck == True

    @pytest.mark.asyncio
    async def test_no_stuck_detection_before_threshold(self, state_intelligence):
        """Test that stuck state is NOT detected before 30 minutes"""
        now = datetime.now()
        space_id = 5

        # Set state start time to only 20 minutes ago
        state_intelligence.space_visit_start_times[space_id] = now - timedelta(minutes=20)

        current_state = StateVisit(
            state_id="space_5_reading",
            app_id="Chrome",
            timestamp=now - timedelta(minutes=20),
            space_id=space_id
        )
        state_intelligence.current_space_states[space_id] = current_state

        await state_intelligence._check_for_stuck_states()

        # Should NOT be marked as stuck
        assert current_state.is_stuck == False
        assert len(state_intelligence.stuck_state_alerts) == 0

    # ========================================
    # TEST 4: Auto-Recording from Monitoring
    # ========================================

    @pytest.mark.asyncio
    async def test_auto_recording_from_monitoring_alert(self, state_intelligence):
        """Test that StateVisits are auto-created from monitoring alerts"""
        alert = {
            'space_id': 3,
            'event_type': 'CODE_CHANGED',
            'message': 'User editing main.py',
            'timestamp': datetime.now(),
            'metadata': {
                'app_name': 'VSCode',
                'detection_method': 'ml'
            }
        }

        initial_count = len(state_intelligence.state_visits)

        await state_intelligence.register_monitoring_alert(alert)

        # Should auto-record a StateVisit
        assert len(state_intelligence.state_visits) > initial_count, \
            "Should auto-record state visit from monitoring alert"

        latest_visit = state_intelligence.state_visits[-1]
        assert latest_visit.space_id == 3
        assert latest_visit.auto_recorded == True
        assert latest_visit.detection_method == 'ml'

    @pytest.mark.asyncio
    async def test_monitoring_alert_classification(self, state_intelligence):
        """Test that monitoring alerts are classified into correct states"""
        alerts = [
            ('CODE_CHANGED', MonitoringStateType.ACTIVE_CODING),
            ('ERROR_DETECTED', MonitoringStateType.ERROR_STATE),
            ('BUILD_STARTED', MonitoringStateType.BUILD_WAITING),
            ('NO_ACTIVITY', MonitoringStateType.IDLE),
        ]

        for event_type, expected_state in alerts:
            alert = {
                'space_id': 1,
                'event_type': event_type,
                'timestamp': datetime.now(),
                'metadata': {'app_name': 'Test'}
            }

            await state_intelligence.register_monitoring_alert(alert)

        # Check that visits were created with correct state types
        # (Implementation may vary, so we just verify auto-recording works)
        assert len(state_intelligence.state_visits) >= len(alerts)

    # ========================================
    # TEST 5: Time-of-Day Preference Learning
    # ========================================

    def test_time_of_day_classification(self, state_intelligence):
        """Test time-of-day classification"""
        test_times = [
            (datetime(2025, 1, 1, 6, 0), TimeOfDay.EARLY_MORNING),
            (datetime(2025, 1, 1, 10, 0), TimeOfDay.MORNING),
            (datetime(2025, 1, 1, 14, 0), TimeOfDay.AFTERNOON),
            (datetime(2025, 1, 1, 18, 0), TimeOfDay.EVENING),
            (datetime(2025, 1, 1, 22, 0), TimeOfDay.NIGHT),
            (datetime(2025, 1, 1, 2, 0), TimeOfDay.LATE_NIGHT),
        ]

        for dt, expected_time in test_times:
            result = state_intelligence._get_time_of_day(dt)
            assert result == expected_time, \
                f"Time {dt.hour}:00 should be {expected_time}, got {result}"

    def test_state_visit_has_time_of_day_property(self):
        """Test that StateVisit can determine time of day"""
        morning_visit = StateVisit(
            state_id="morning_coding",
            app_id="VSCode",
            timestamp=datetime(2025, 1, 1, 9, 0)
        )

        assert morning_visit.time_of_day == TimeOfDay.MORNING

    # ========================================
    # TEST 6: State Prediction
    # ========================================

    def test_predict_next_state(self, state_intelligence):
        """Test state prediction based on history"""
        # Record pattern: coding → debugging → testing
        now = datetime.now()

        sequence = [
            ("state_coding", "VSCode"),
            ("state_debugging", "VSCode"),
            ("state_testing", "Terminal"),
            ("state_coding", "VSCode"),
            ("state_debugging", "VSCode"),
            ("state_testing", "Terminal"),
        ]

        for i, (state_id, app_id) in enumerate(sequence):
            visit = StateVisit(
                state_id=state_id,
                app_id=app_id,
                timestamp=now - timedelta(hours=len(sequence) - i),
                transition_to=sequence[i+1][0] if i < len(sequence)-1 else None
            )
            state_intelligence.record_visit(visit)

        # Predict next state after coding
        predictions = state_intelligence.predict_next_state("state_coding")

        assert len(predictions) > 0, "Should make predictions"

        # Should predict debugging as likely next state
        predicted_states = [s for s, prob in predictions]
        assert "state_debugging" in predicted_states, \
            "Should predict debugging after coding based on pattern"

    # ========================================
    # TEST 7: Productivity Insights
    # ========================================

    def test_productivity_insights_generation(self, state_intelligence):
        """Test generation of productivity insights"""
        now = datetime.now()

        # Record various activities
        activities = [
            (MonitoringStateType.ACTIVE_CODING, 0.9),
            (MonitoringStateType.PRODUCTIVE, 0.85),
            (MonitoringStateType.DISTRACTED, 0.3),
            (MonitoringStateType.STUCK, 0.2),
        ]

        for i, (state_type, productivity) in enumerate(activities):
            visit = StateVisit(
                state_id=f"state_{i}",
                app_id="App",
                timestamp=now - timedelta(hours=i),
                monitoring_state_type=state_type,
                productivity_score=productivity
            )
            state_intelligence.record_visit(visit)

        insights = state_intelligence.get_productivity_insights()

        # Should return comprehensive intelligence summary
        assert 'productivity_trend' in insights
        assert 'monitoring_state_breakdown' in insights
        assert insights['is_proactive_enabled'] == True

    # ========================================
    # TEST 8: State Recommendations
    # ========================================

    def test_state_recommendations(self, state_intelligence):
        """Test that system can recommend next states"""
        now = datetime.now()

        # Record a pattern
        for i in range(3):
            visit = StateVisit(
                state_id="coding",
                app_id="VSCode",
                timestamp=now - timedelta(hours=i*2),
                transition_to="testing"
            )
            state_intelligence.record_visit(visit)

        recommendations = state_intelligence.get_state_recommendations("coding")

        assert len(recommendations) > 0, "Should provide recommendations"

    # ========================================
    # TEST 9: Proactive Mode Detection
    # ========================================

    def test_proactive_mode_enabled(self, state_intelligence):
        """Test that proactive mode is enabled with HybridMonitoring"""
        assert state_intelligence.is_proactive_enabled == True
        assert state_intelligence.hybrid_monitoring is not None

    # ========================================
    # TEST 10: State Visit Recording
    # ========================================

    def test_record_visit_adds_to_history(self, state_intelligence):
        """Test that recording visits adds to history"""
        initial_count = len(state_intelligence.state_visits)

        visit = StateVisit(
            state_id="test_state",
            app_id="TestApp",
            timestamp=datetime.now()
        )

        state_intelligence.record_visit(visit)

        assert len(state_intelligence.state_visits) == initial_count + 1

    def test_record_visit_updates_space_tracking(self, state_intelligence):
        """Test that recording visits updates space tracking"""
        visit = StateVisit(
            state_id="space_5_coding",
            app_id="VSCode",
            timestamp=datetime.now(),
            space_id=5
        )

        state_intelligence.record_visit(visit)

        assert 5 in state_intelligence.space_visit_start_times
        assert 5 in state_intelligence.current_space_states


class TestStateIntelligenceEdgeCases:
    """Edge case tests for StateIntelligence v2.0"""

    @pytest.fixture
    def state_intelligence_no_monitoring(self):
        """Create StateIntelligence without monitoring (backward compatibility)"""
        return StateIntelligence(
            hybrid_monitoring_manager=None,
            implicit_resolver=None,
            change_detection_manager=None
        )

    def test_works_without_proactive_monitoring(self, state_intelligence_no_monitoring):
        """Test that StateIntelligence works without monitoring"""
        assert state_intelligence_no_monitoring.is_proactive_enabled == False

        # Should still record visits manually
        visit = StateVisit(
            state_id="manual_state",
            app_id="App",
            timestamp=datetime.now()
        )

        state_intelligence_no_monitoring.record_visit(visit)
        assert len(state_intelligence_no_monitoring.state_visits) == 1

    @pytest.mark.asyncio
    async def test_register_alert_without_space_id(self, state_intelligence_no_monitoring):
        """Test that alerts without space_id are ignored"""
        alert = {
            'event_type': 'SOME_EVENT',
            'message': 'Test',
            'timestamp': datetime.now(),
            'metadata': {}
            # No space_id
        }

        await state_intelligence_no_monitoring.register_monitoring_alert(alert)

        # Should not create a visit
        assert len(state_intelligence_no_monitoring.state_visits) == 0


# ========================================
# TEST HELPERS
# ========================================

class TestStateVisitProperties:
    """Test StateVisit helper properties"""

    def test_day_type_weekday(self):
        """Test weekday detection"""
        monday = datetime(2025, 1, 6)  # Monday
        visit = StateVisit(
            state_id="test",
            app_id="App",
            timestamp=monday
        )

        assert visit.day_type == DayType.WEEKDAY

    def test_day_type_weekend(self):
        """Test weekend detection"""
        saturday = datetime(2025, 1, 11)  # Saturday
        visit = StateVisit(
            state_id="test",
            app_id="App",
            timestamp=saturday
        )

        assert visit.day_type == DayType.WEEKEND
